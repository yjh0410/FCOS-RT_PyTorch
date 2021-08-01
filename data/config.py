# config.py

# train
train_cfg = {
    # network
    'freeze_bn': True,
    # for multi-scale trick
    'img_size': 1024,
    'train_size': 768,
    'val_size': 768,
    'random_size_range': [3, 8], #[384, 512, 640, 768, 896, 1024]
    # scale range
    'scale_range': [[0, 48], [48, 96], [96, 192], [192, 384], [384, 1e5]],
    # 'scale_range': [[0, 64], [64, 128], [128, 256], [256, 512], [512, 1e5]],
    # train
    'lr': 0.01,
    'max_iters': 90000,
    'lr_step': [60000, 80000]
}
