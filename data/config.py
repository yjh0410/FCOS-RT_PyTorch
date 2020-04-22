# config.py
import os.path

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
MEANS = (104, 117, 123)

# yolo-v2 config
voc_ab = {
    'num_classes': 20,
    'lr_epoch': (150, 200, 250),
    'max_epoch': 250,
    'min_dim': [512, 512],
    'ms_channels':[128, 256, 512],
    'multi_scale': [[320, 320], [352, 352], [384, 384], [416, 416], [448, 448],
                 [480, 480], [512, 512], [544, 544], [576, 576], [608, 608]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco_ab = {
    'num_classes': 80,
    'lr_epoch': (150, 200, 250),
    'max_epoch': 250,
    'min_dim': [512, 512],
    'ms_channels':[128, 256, 512],
    'multi_scale': [[320, 320], [352, 352], [384, 384], [416, 416], [448, 448],
                 [480, 480], [512, 512], [544, 544], [576, 576], [608, 608]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}