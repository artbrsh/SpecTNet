# --------------------------------------------------------
# SpecTNet training loop
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Adapted from TinyViT
# Source: TinyViT/main.py (training loop).
# --------------------------------------------------------

from my_meter import AverageMeter
import torch
import time
from timm.utils import accuracy
from utils import is_main_process
import datetime
import torch.distributed as dist


try:
    import wandb
except ImportError:
    wandb = None

NORM_ITER_LEN = 100


def _is_log_step(idx, num_steps, print_freq):
    print_freq = max(1, int(print_freq))
    return idx % print_freq == 0 or idx + 1 == num_steps


def _should_collect_metrics(config, idx, num_steps):
    return (not config.TRAIN.FAST_METRICS) or _is_log_step(idx, num_steps, config.PRINT_FREQ)


def _should_check_nonfinite_loss(config, idx):
    freq = int(getattr(config.TRAIN, "NONFINITE_CHECK_FREQ", 0))
    return freq > 0 and idx % freq == 0


def is_valid_grad_norm(num):
    if num is None:
        return False
    return not bool(torch.isinf(num)) and not bool(torch.isnan(num))


def stop_if_nonfinite_loss(loss, epoch, idx, logger):
    has_nonfinite = ~torch.isfinite(loss.detach()).all()

    if dist.is_available() and dist.is_initialized():
        has_nonfinite = has_nonfinite.to(dtype=torch.int, device=loss.device)
        dist.all_reduce(has_nonfinite, op=dist.ReduceOp.MAX)
        has_nonfinite = bool(has_nonfinite.item())
    else:
        has_nonfinite = bool(has_nonfinite.item())

    if has_nonfinite:
        loss_value = loss.detach().float().mean().item()
        message = (
            f"Stopping training: non-finite loss detected at epoch {epoch}, "
            f"batch {idx}: {loss_value}."
        )
        logger.error(message)
        raise FloatingPointError(message)


def set_bn_state(config, model):
    if config.TRAIN.EVAL_BN_WHEN_TRAINING:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()


def train_one_epoch(args, config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, logger):
    model.train()
    set_bn_state(config, model)

    optimizer.zero_grad(set_to_none=True)
    

    num_steps = max(1, len(data_loader))
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    device = torch.cuda.current_device()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        normal_global_idx = epoch * NORM_ITER_LEN + \
            (idx * NORM_ITER_LEN // num_steps)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            original_targets = targets.argmax(dim=1)
        else:
            original_targets = targets
        
        with torch.amp.autocast("cuda", enabled=config.AMP_ENABLE):
            outputs = model(samples)

        loss = criterion(outputs, targets)
        if _should_check_nonfinite_loss(config, idx):
            stop_if_nonfinite_loss(loss, epoch, idx, logger)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad(set_to_none=True)

            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        collect_metrics = _should_collect_metrics(config, idx, num_steps)
        if collect_metrics:
            loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)
            with torch.no_grad():
                acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))
            acc1_meter.update(acc1.item(), targets.size(0))
            acc5_meter.update(acc5.item(), targets.size(0))

            torch.cuda.synchronize()

            loss_meter.update(loss.item(), targets.size(0))
            if is_valid_grad_norm(grad_norm):
                norm_meter.update(grad_norm)
            scaler_meter.update(loss_scale_value)
            batch_time.update(time.time() - end)

        end = time.time()

        if _is_log_step(idx, num_steps, config.PRINT_FREQ):
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            if is_main_process() and args.use_wandb:
                wandb.log({
                    "train/acc@1": acc1_meter.val,
                    "train/acc@5": acc5_meter.val,
                    "train/loss": loss_meter.val,
                    "train/grad_norm": norm_meter.val,
                    "train/loss_scale": scaler_meter.val,
                    "train/lr": lr,
                }, step=normal_global_idx)
    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
