import os
import numpy as np
import random

import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO


coco_class_labels = ('background',
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                    70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

coco_root = '/home/jxk/object-detection/dataset/COCO/'


class COCODataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, 
                 data_dir='COCO', 
                 img_size=640,
                 transform=None, 
                 json_file='instances_train2017.json',
                 name='train2017', 
                 min_size=1, 
                 debug=False,
                 mosaic=False,
                 mixup=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.data_dir = data_dir
        self.json_file = json_file
        self.coco = COCO(self.data_dir+'annotations/'+self.json_file)
        self.ids = self.coco.getImgIds()
        if debug:
            self.ids = self.ids[1:2]
            print("debug mode...", self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        # augmentation
        self.transform = transform
        self.mosaic = mosaic
        self.mixup = mixup


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        im, gt, h, w, scale, offset = self.pull_item(index)

        return im, gt


    def pull_image(self, index):
        id_ = self.ids[index]
        img_file = os.path.join(self.data_dir, self.name,
                                '{:012}'.format(id_) + '.jpg')
        img = cv2.imread(img_file)

        if self.json_file == 'instances_val5k.json' and img is None:
            img_file = os.path.join(self.data_dir, 'train2017',
                                    '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)

        return img, id_


    def pull_anno(self, index):
        id_ = self.ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)
        
        target = []
        for anno in annotations:
            if 'bbox' in anno:
                xmin = np.max((0, anno['bbox'][0]))
                ymin = np.max((0, anno['bbox'][1]))
                xmax = xmin + anno['bbox'][2]
                ymax = ymin + anno['bbox'][3]
                
                if anno['area'] > 0 and xmax >= xmin and ymax >= ymin:
                    label_ind = anno['category_id']
                    cls_id = self.class_ids.index(label_ind)

                    target.append([xmin, ymin, xmax, ymax, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
            else:
                print('No bbox !!')
        return target


    def load_img_targets(self, index):
        anno_ids = self.coco.getAnnIds(imgIds=[int(index)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        # load image and preprocess
        img_file = os.path.join(self.data_dir, self.name,
                                '{:012}'.format(index) + '.jpg')
        img = cv2.imread(img_file)
        
        if self.json_file == 'instances_val5k.json' and img is None:
            img_file = os.path.join(self.data_dir, 'train2017',
                                    '{:012}'.format(index) + '.jpg')
            img = cv2.imread(img_file)

        assert img is not None

        height, width, channels = img.shape
        
        # COCOAnnotation Transform
        # start here :
        target = []
        for anno in annotations:
            if 'bbox' in anno and anno['area'] > 0:   
                xmin = np.max((0, anno['bbox'][0]))
                ymin = np.max((0, anno['bbox'][1]))
                xmax = np.min((width - 1, xmin + np.max((0, anno['bbox'][2] - 1))))
                ymax = np.min((height - 1, ymin + np.max((0, anno['bbox'][3] - 1))))
                # xmin, ymin, w, h = anno['bbox']
                # xmax = xmin + w
                # ymax = ymin + h
                if xmax > xmin and ymax > ymin:
                    label_ind = anno['category_id']
                    cls_id = self.class_ids.index(label_ind)
                    xmin /= width
                    ymin /= height
                    xmax /= width
                    ymax /= height

                    target.append([xmin, ymin, xmax, ymax, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
            else:
                print('No bbox !!!')
        # end here .

        return img, target, height, width


    def load_mosaic(self, index):
        ids_list_ = self.ids[:index] + self.ids[index+1:]
        # random sample other indexs
        id1 = self.ids[index]
        id2, id3, id4 = random.sample(ids_list_, 3)
        ids = [id1, id2, id3, id4]

        img_lists = []
        tg_lists = []
        # load image and target
        for id_ in ids:
            img_i, target_i, _, _ = self.load_img_targets(id_)
            img_lists.append(img_i)
            tg_lists.append(target_i)

        mosaic_img = np.zeros([self.img_size*2, self.img_size*2, img_i.shape[2]], dtype=np.uint8)
        # mosaic center
        yc, xc = [int(random.uniform(-x, 2*self.img_size + x)) for x in [-self.img_size // 2, -self.img_size // 2]]
        # yc = xc = self.img_size

        mosaic_tg = []
        for i in range(4):
            img_i, target_i = img_lists[i], tg_lists[i]
            h0, w0, _ = img_i.shape

            # resize
            r = self.img_size / max(h0, w0)
            if r != 1: 
                img_i = cv2.resize(img_i, (int(w0 * r), int(h0 * r)))
            h, w, _ = img_i.shape

            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # labels
            target_i = np.array(target_i)
            target_i_ = target_i.copy()
            if len(target_i) > 0:
                # a valid target, and modify it.
                target_i_[:, 0] = (w * (target_i[:, 0]) + padw)
                target_i_[:, 1] = (h * (target_i[:, 1]) + padh)
                target_i_[:, 2] = (w * (target_i[:, 2]) + padw)
                target_i_[:, 3] = (h * (target_i[:, 3]) + padh)     
                
                mosaic_tg.append(target_i_)
        # check target
        if len(mosaic_tg) == 0:
            mosaic_tg = np.zeros([1, 5])
        else:
            mosaic_tg = np.concatenate(mosaic_tg, axis=0)
            # Cutout/Clip targets
            np.clip(mosaic_tg[:, :4], 0, 2 * self.img_size, out=mosaic_tg[:, :4])
            # normalize
            mosaic_tg[:, :4] /= (self.img_size * 2) 

        return mosaic_img, mosaic_tg, self.img_size, self.img_size


    def pull_item(self, index):
        # load mosaic image
        if self.mosaic:
            # mosaic
            img, target, height, width = self.load_mosaic(index)

            # mixup
            if self.mixup and random.random() < 0.5:
                id2 = random.randint(0, len(self.ids)-1)
                img2, target2, height, width = self.load_mosaic(id2)
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                target = np.concatenate((target, target2), axis=0)

        # load a image
        else:
            id_ = self.ids[index]
            img, target, height, width = self.load_img_targets(id_)
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)

        # augment
        img, boxes, labels, scale, offset = self.transform(img, target[:, :4], target[:, 4])
        # to rgb
        img = img[:, :, (2, 1, 0)]
        # to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        # img = img.transpose(2, 0, 1)
        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, target, height, width, scale, offset


if __name__ == "__main__":
    def transform(img, size, mean, std, boxes=None):
        h0, w0, _ = img.shape

        # zero padding
        if h0 > w0:
            r = w0 / h0
            img = cv2.resize(img, (int(r * size), size)).astype(np.float32)
            # normalize
            # img /= 255.
            # img -= mean
            # img /= std
            h, w, _ = img.shape
            img_ = np.zeros([h, h, 3])
            dw = h - w
            left = dw // 2
            img_[:, left:left+w, :] = img
            offset = np.array([[left / h, 0.,  left / h, 0.]])
            scale = np.array([w / h, 1., w / h, 1.])

        elif h0 < w0:
            r = h0 / w0
            img = cv2.resize(img, (size, int(r * size))).astype(np.float32)
            # normalize
            # img /= 255.
            # img -= mean
            # img /= std
            h, w, _ = img.shape
            img_ = np.zeros([w, w, 3])
            dh = w - h
            top = dh // 2
            img_[top:top+h, :, :] = img
            offset = np.array([[0., top / w, 0., top / w]])
            scale = np.array([1., h / w, 1., h / w])

        else:
            img = cv2.resize(img, (size, size)).astype(np.float32)
            # normalize
            # img /= 255.
            # img -= mean
            # img /= std
            img_ = img
            offset = np.zeros([1, 4])
            scale = 1.0

        if boxes is not None:
            boxes_ = boxes * scale + offset
        
        return img_, boxes_, scale, offset


    class BaseTransform:
        def __init__(self, size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
            self.size = size
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)

        def __call__(self, img, boxes=None, labels=None):
            img, boxes, scale, offset = transform(img, self.size, self.mean, self.std, boxes)

            return img, boxes, labels, scale, offset

    img_size = 640
    dataset = COCODataset(
                data_dir=coco_root,
                img_size=img_size,
                transform=BaseTransform(img_size, (0, 0, 0), (1, 1, 1)),
                debug=False,
                mosaic=True,
                mixup=True)
    
    for i in range(1000):
        im, gt, h, w, _, _ = dataset.pull_item(i)
        img = im.permute(1,2,0).numpy()[:, :, (2, 1, 0)].astype(np.uint8)
        cv2.imwrite('-1.jpg', img)
        img = cv2.imread('-1.jpg')

        for box in gt:
            xmin, ymin, xmax, ymax, _ = box
            xmin *= img_size
            ymin *= img_size
            xmax *= img_size
            ymax *= img_size
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
        cv2.imshow('gt', img)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)
