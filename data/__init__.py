from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT
from .coco2017 import COCODataset, coco_root, coco_class_labels, coco_class_index
from .config import *
import torch
import cv2
import numpy as np


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
        print(sample[1].size())
    return torch.stack(imgs, 0), torch.stack(targets, 0)


def base_transform(img, size, mean, std, boxes=None):
    h0, w0, _ = img.shape
    if h0 > w0:
        # resize
        r = w0 / h0
        img = cv2.resize(img, (int(r * size), size)).astype(np.float32)
        # normalize
        img /= 255.
        img -= mean
        img /= std
        # zero padding
        h, w, _ = img.shape
        img_ = np.zeros([h, h, 3])
        dw = h - w
        left = dw // 2
        img_[:, left:left+w, :] = img
        offset = np.array([[left / h, 0.,  left / h, 0.]])
        scale = np.array([w / h, 1., w / h, 1.])

    elif h0 < w0:
        # resize
        r = h0 / w0
        img = cv2.resize(img, (size, int(r * size))).astype(np.float32)
        # normalize
        img /= 255.
        img -= mean
        img /= std
        # zero padding
        h, w, _ = img.shape
        img_ = np.zeros([w, w, 3])
        dh = w - h
        top = dh // 2
        img_[top:top+h, :, :] = img
        offset = np.array([[0., top / w, 0., top / w]])
        scale = np.array([1., h / w, 1., h / w])

    else:
        # resize
        img = cv2.resize(img, (size, size)).astype(np.float32)
        # normalize
        img /= 255.
        img -= mean
        img /= std
        img_ = img
        offset = np.zeros([1, 4])
        scale = 1.0

    if boxes is not None:
        boxes = boxes * scale + offset
    
    return img_, boxes, scale, offset


class BaseTransform:
    def __init__(self, size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img, boxes=None, labels=None):
        img, boxes, scale, offset = base_transform(img, self.size, self.mean, self.std, boxes)

        return img, boxes, labels, scale, offset
