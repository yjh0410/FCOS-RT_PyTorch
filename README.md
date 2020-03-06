# FCOS-LITE
This is my first simple attempt to reproduce the famous anchor-free model：FCOS.

For origin FCOS, I think it is too big to deploy on my device. Specifically, FCOS makes a prediction on an 800 x 1024 image where the input image is much big. What's more, its backbone network, resnet-50, is a little big, and its detection head is also slow. Therefore, I want to build a light-weight FCOS: FCOS-LITE.

To overcome above disadvantages, I make the following adjustments:

1. For backbone, I choose resnet-18. Although, I have trained darknet-19, which is also light and effective, resnet-18 is more convenient as PyTorch can easily import pre-trained model from model_zoo.

2. For detection head, I substitute a YOLO-v3-style head for origin FCOS head. Because I can't get a good result with the share-weight head. The whole network structure is as shown:

![Image](https://github.com/yjh0410/FCOS-LITE/blob/master/img_folder/fcos-lite.png)

With those adjustments, I have gotten 69.5 mAP and 200 FPS(on RTX 2080-ti) on VOC2007 test with 320 input size.
But that is not enough, and I'm trying to get better result( faster on low-performance GPU, and almost real-time on CPU). Please wait for my good news !

Before you clone this project, I must emphasize again that my FCOS-LITE is not yet mature and there are still many things to do for improvement which I'm trying to do.

Okay, I think you have known as much as possible. Just have fun !

## Instillation
- Pytorch >= 1.1.0
- Torchvision >= 0.3.0
- python-opencv, python3.6/3.7

## Dataset
As for now, I only train and test on PASCAL VOC2007 and 2012. 

### VOC Dataset
I copy the download files from the following excellent project:
https://github.com/amdegroot/ssd.pytorch

I have uploaded the VOC2007 and VOC2012 to BaiDuYunDisk, so for researchers in China, you can download them from BaiDuYunDisk:

Link：https://pan.baidu.com/s/1tYPGCYGyC0wjpC97H-zzMQ 

Password：4la9

You will get a ```VOCdevkit.zip```, then what you need to do is just to unzip it and put it into ```data/```. After that, the whole path to VOC dataset is ```data/VOCdevkit/VOC2007``` and ```data/VOCdevkit/VOC2012```.

#### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

#### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

## Train
To run:
```Shell
python train_voc.py
```

You can run ```python train_voc.py -h``` to check all optional argument

## Test
To run:
```Shell
python test_voc.py --trained_model [ Please input the path to model dir. ]
```

## Evaluation
To run:
```Shell
python eval_voc.py --train_model [ Please input the path to model dir. ]
```