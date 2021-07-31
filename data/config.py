# config.py

# train
train_cfg = {
    # network
    'backbone': 'r18',
    'freeze_bn': True,
    # for multi-scale trick
    'img_size': 768,
    'train_size': 640,
    'val_size': 640,
    'random_size_range': [3, 6], #[384, 512, 640, 768]
    # scale range
    'scale_range': [[0, 32], [32, 64], [64, 128], [128, 256], [256, 1e5]],
    # 'scale_range': [[0, 64], [64, 128], [128, 256], [256, 512], [512, 1e5]],
    # train
    'lr': 0.01,
    'max_epoch': 12,
    'lr_epoch': [8, 10]
}
