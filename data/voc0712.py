"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random

import xml.etree.ElementTree as ET


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')



class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt if i % 2 == 0 else cur_pt
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [x1, y1, x2, y2, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[x1, y1, x2, y2, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, 
                 data_dir=None,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, 
                 target_transform=VOCAnnotationTransform()
                 ):
        self.root = data_dir
        self.image_set = image_sets
        self.target_transform = target_transform
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        # augmentation
        self.transform = transform


    def __getitem__(self, index):
        img, target = self.pull_item(index)
        return img, target


    def __len__(self):
        return len(self.ids)


    def load_image_annotation(self, img_id):
        # load an image
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape
        # load an annotation
        anno = ET.parse(self._annopath % img_id).getroot()
        if self.target_transform is not None:
            anno = self.target_transform(anno)

        return img, anno, height, width


    def pull_item(self, index):
        img_id = self.ids[index]
        img, anno, height, width = self.load_image_annotation(img_id)
        if len(anno) == 0:
            anno = np.zeros([1, 5])
        else:
            anno = np.array(anno)

        # transform
        target = {'boxes': anno[:, :4],
                  'labels': anno[:, 4],
                  'orig_size': [height, width]}
        img, target = self.transform(img, target)
    
        return img, target


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id


    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt


if __name__ == "__main__":
    from transforms import TrainTransforms, ValTransforms
    img_size = 800
    dataset = VOCDetection(
                data_dir='/mnt/share/ssd2/dataset/VOCdevkit',
                transform=TrainTransforms(img_size))
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]
    rgb_mean = np.array(dataset.transform.mean)
    rgb_std = np.array(dataset.transform.std)
    print('Data length: ', len(dataset))
    for i in range(1000):
        # load an image
        img, target = dataset.pull_item(i)
        img = img.permute(1,2,0).numpy()
        img = (img*rgb_std + rgb_mean) * 255
        # from rgb to bgr
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.uint8).copy()
        # load a target
        cls_gt = target['labels'].tolist()
        box_gt = target['boxes'].tolist()
        for i in range(len(cls_gt)):
            cls_id = int(cls_gt[i])
            cx, cy, bw, bh = box_gt[i]
            x1 = int((cx - bw / 2) * img_size)
            y1 = int((cy - bh / 2) * img_size)
            x2 = int((cx + bw / 2) * img_size)
            y2 = int((cy + bh / 2) * img_size)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cls_name = VOC_CLASSES[cls_id]
            mess = '%s' % (cls_name)
            color = class_colors[cls_id]
            cv2.putText(img, mess, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', img)
        cv2.waitKey(0)
