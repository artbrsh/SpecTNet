# --------------------------------------------------------
# SpecTNet Data Builder
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Adapted from TinyViT
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import Mixup
from timm.data import create_transform

from .sampler import MyDistributedSampler

try:
    from timm.data import TimmDatasetTar
except ImportError:
    # for higher version of timm
    from timm.data import ImageDataset as TimmDatasetTar

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

    config.defrost()
    dataset_train = build_dataset(
        is_train=True, config=config)
    config.freeze()

    print(
        f"local rank {local_rank} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val = build_dataset(is_train=False, config=config)
    print(
        f"local rank {local_rank} / global rank {dist.get_rank()} successfully build val dataset")

    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None

    sampler_train = MyDistributedSampler(
        dataset_train, shuffle=True,
        drop_last=False, padding=True,
    )

    sampler_val = MyDistributedSampler(
        dataset_val, shuffle=False,
        drop_last=False, padding=False,
    )

    train_loader_kwargs = dict(
        dataset=dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )
    val_loader_kwargs = dict(
        dataset=dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )
    if config.DATA.NUM_WORKERS > 0 and config.DATA.PREFETCH_FACTOR > 0:
        train_loader_kwargs["prefetch_factor"] = config.DATA.PREFETCH_FACTOR
        val_loader_kwargs["prefetch_factor"] = config.DATA.PREFETCH_FACTOR
    if config.DATA.NUM_WORKERS > 0:
        train_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["persistent_workers"] = True

    data_loader_train = torch.utils.data.DataLoader(**train_loader_kwargs)
    data_loader_val = torch.utils.data.DataLoader(**val_loader_kwargs)

    # setup mixup / cutmix
    mixup_fn = None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    if not config.DATA.DATA_PATH:
        raise ValueError('Set the dataset path with --data-path or DATA.DATA_PATH')
    transform = build_transform(is_train, config)
    prefix = 'train' if is_train else 'val'

    if config.DATA.DATASET == 'imagenet':
        dataset_tar_t = TimmDatasetTar # load tar dataset
        data_dir = os.path.join(config.DATA.DATA_PATH, f'{prefix}.tar')
        if os.path.exists(data_dir):
            dataset = dataset_tar_t(data_dir, transform=transform)
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)

    else:
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)

    return dataset


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32

    # RGB: mean, std
    rgbs = dict(
        default=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        inception=(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD),
    )
    mean, std = rgbs[config.DATA.MEAN_AND_STD_TYPE]

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                config.DATA.IMG_SIZE, padding=4)

        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(
                    config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(t)
    return transform
