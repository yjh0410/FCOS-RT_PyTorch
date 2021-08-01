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
    'max_iters': 360000,
    'lr_step': [300000, 340000]
}