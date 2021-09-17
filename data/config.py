# config.py

# FCOS_RT
fcos_rt_train_cfg = {
    # network
    'freeze_bn': True,
    # for multi-scale trick
    'img_size': 640,
    'train_size': 640,
    'val_size': 640,
    'random_size_range': [10, 20], #[320, 352, 384, ... 608, 640]
    # scale range
    'scale_range': [[0, 64], [64, 128], [128, 1e5]],
    # train
    'lr': 0.01,
    'max_epoch': 48,
    'lr_step': [24, 36]
}

# FCOS
fcos_train_cfg = {
    # network
    'freeze_bn': True,
    # for multi-scale trick
    'img_size': 1024,
    'train_size': 896,
    'val_size': 896,
    'random_size_range': [10, 20], #[320, 352, 384, ... 608, 640]
    # scale range
    'scale_range': [[0, 64], [64, 128], [128, 256], [256, 512], [512, 1e5]],
    # train
    'lr': 0.01,
    'max_epoch': 12,
    'lr_step': [6, 10]
}