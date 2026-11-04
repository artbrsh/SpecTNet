# --------------------------------------------------------
# SpecTNet Model Builder
# Copyright (c) 2022 Microsoft
# Adapted from TinyViT
# --------------------------------------------------------

from .spectnet import SpecTNet


def build_model(config):
    M = config.MODEL.SPECTNET
    print('config.DATA.IMG_SIZE', config.DATA.IMG_SIZE)
    model = SpecTNet(img_size=config.DATA.IMG_SIZE,
                    in_chans=M.IN_CHANS,
                    num_classes=config.MODEL.NUM_CLASSES,
                    embed_dims=M.EMBED_DIMS,
                    depths=M.DEPTHS,
                    num_heads=M.NUM_HEADS,
                    window_sizes=M.WINDOW_SIZES,
                    expand_ratios=M.EXPAND_RATIOS,
                    drop_rate=config.MODEL.DROP_RATE,
                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                    local_conv_size=M.LOCAL_CONV_SIZE,
                    layer_lr_decay=config.TRAIN.LAYER_LR_DECAY,
                    )
    return model


def prepare_for_inference(model, config):
    """Apply configured deployment transforms after checkpoint loading, in place."""
    merge = config.INFERENCE.MERGE_STAGE1
    if type(merge) is not bool:
        raise ValueError('INFERENCE.MERGE_STAGE1 must be a boolean')
    model.eval()
    if merge:
        target = model.module if hasattr(model, 'module') else model
        target.merge_stage1_for_inference()
    return model
