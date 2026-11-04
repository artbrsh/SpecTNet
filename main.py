# --------------------------------------------------------
# SpecTNet Main (train/validate)
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Adapted from TinyViT
# --------------------------------------------------------

import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from logger import create_logger
from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils import load_checkpoint, save_checkpoint,\
    NativeScalerWithGradNormCount,\
    auto_resume_helper, is_main_process,\
    add_common_args

from engine import train_one_epoch
from validation import validate, run_metrics

try:
    import wandb
except ImportError:
    wandb = None

def parse_option():
    parser = argparse.ArgumentParser(
        'SpecTNet training and evaluation script', add_help=False)
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(args, config):
    if config.THROUGHPUT_MODE:
        if args.only_cpu or not torch.cuda.is_available():
            raise ValueError('GPU metrics require CUDA')
        if dist.get_world_size() != 1:
            raise ValueError('Run GPU metrics with --nproc_per_node=1')
        model = build_model(config)
        checkpoint_path = config.MODEL.RESUME
        if checkpoint_path:
            if checkpoint_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    checkpoint_path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'], strict=True)
            logger.info(f'Metrics checkpoint: {checkpoint_path}')
            del checkpoint
        else:
            logger.info('Metrics use a randomly initialized model.')
        run_metrics(model.cuda(), logger, config)
        return

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config)
    val_num_samples = len(dataset_val)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    local_rank = None
    if not args.only_cpu:
        env_local_rank = int(os.environ.get(
            "LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        n_gpus = torch.cuda.device_count()
        local_rank = (env_local_rank % n_gpus) if n_gpus > 0 else 0
        torch.cuda.set_device(local_rank)

    model = build_model(config)

    if not args.only_cpu:
        model.cuda()

    for p in model.parameters():
        p.data = p.data.contiguous()

    if args.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    logger.info(str(model))

    optimizer = build_optimizer(config, model)

    if not args.only_cpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    loss_scaler = NativeScalerWithGradNormCount(grad_scaler_enabled=config.AMP_ENABLE)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    train_steps_per_epoch = max(1, len(data_loader_train))
    updates_per_epoch = max(1, train_steps_per_epoch // max(1, config.TRAIN.ACCUMULATION_STEPS))
    lr_scheduler = build_scheduler(config, optimizer, updates_per_epoch)

    if mixup_fn is not None:
        # Mixup and CutMix produce soft targets, including CutMix-only runs.
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        
        acc1, acc5, loss = validate(args, config, data_loader_val, model_without_ddp, logger)
        logger.info(
            f"Accuracy of the network on the {val_num_samples} test images: {acc1:.1f}%")
    if config.EVAL_MODE:
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(args, config, model, criterion,
                        data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, logger)

        acc1, acc5, loss = validate(args, config, data_loader_val, model_without_ddp, logger)
        logger.info(
            f"Accuracy of the network on the {val_num_samples} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if is_main_process() and args.use_wandb:
            wandb.log({
                f"val/acc@1": acc1,
                f"val/acc@5": acc5,
                f"val/loss": loss,
                "epoch": epoch,
            })
            wandb.run.summary['epoch'] = epoch
            wandb.run.summary['best_acc@1'] = max_accuracy

        if is_main_process() and config.SAVE_FREQ > 0 and ((epoch + 1) % config.SAVE_FREQ == 0):
            save_checkpoint(
                config=config,
                epoch=epoch,
                model=model_without_ddp,
                max_accuracy=max_accuracy,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                loss_scaler=loss_scaler,
                logger=logger,
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args, config = parse_option()

    # ---- ranks from env (required by init_method='env://') ----
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}", flush=True)
    else:
        rank = -1
        world_size = -1

    # ---- device mapping from distributed environment variables ----
    if args.only_cpu:
        ddp_backend = 'gloo'
        local_rank = None
    else:
        env_local_rank = int(os.environ.get(
            "LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        n_gpus = torch.cuda.device_count()
        local_rank = (env_local_rank % n_gpus) if n_gpus > 0 else 0

        torch.cuda.set_device(local_rank)

        print(
            f"[rank {rank}] env_local_rank={env_local_rank} cuda_device_index={local_rank} "
            f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"current_device={torch.cuda.current_device()} "
            f"n_gpus={torch.cuda.device_count()}",
            flush=True
        )

        ddp_backend = 'nccl'

    # ---- init distributed + explicit device barrier ----
    init_process_group_kwargs = dict(
        backend=ddp_backend,
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    if not args.only_cpu:
        init_process_group_kwargs['device_id'] = torch.device(
            f'cuda:{local_rank}')
    torch.distributed.init_process_group(**init_process_group_kwargs)
    if not args.only_cpu:
        torch.distributed.barrier(device_ids=[local_rank])
    else:
        torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if is_main_process():
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

        if args.use_wandb:
            wandb_output_path = config.OUTPUT
            wandb.init(project="SpecTNet", config=dict(config),
                       dir=wandb_output_path)

    # print config
    logger.info(config.dump())

    main(args, config)
