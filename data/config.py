# config.py

# train
train_cfg = {
    # network
    'backbone': 'r18',
    # for multi-scale trick
    'img_size': 896,
    'train_size': 768,
    'val_size': 768,
    'random_size_range': [4, 7], #[512, 640, 768, 896]
    # anchor size
    'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                    [30, 61],   [62, 45],   [59, 119],
                    [116, 90],  [156, 198], [373, 326]],
    # train
    'max_epoch': 300,
    'ignore_thresh': 0.5
}
