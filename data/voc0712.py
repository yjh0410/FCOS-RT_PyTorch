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

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

VOC_ROOT = "/home/jxk/object-detection/dataset/VOCdevkit/"


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

    def __call__(self, target, width, height):
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
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
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
                 root,
                 img_size=None,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, 
                 target_transform=VOCAnnotationTransform(),
                 mosaic=False,
                 mixup=False,
                 dataset_name='VOC0712'
                 ):
        self.root = root
        self.img_size = img_size
        self.image_set = image_sets
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        # augmentation
        self.transform = transform
        self.mosaic = mosaic
        self.mixup = mixup

    def __getitem__(self, index):
        im, gt, h, w, scale, offset = self.pull_item(index)

        return im, gt


    def __len__(self):
        return len(self.ids)


    def load_img_targets(self, img_id):
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

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
            img_id = self.ids[index]
            img, target, height, width = self.load_img_targets(img_id)
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
    # dataset
    dataset = VOCDetection(root=VOC_ROOT, 
                           img_size=img_size,
                           image_sets=[('2007', 'trainval')],
                           transform=BaseTransform(img_size, (0, 0, 0), (1, 1, 1)),
                           target_transform=VOCAnnotationTransform(), 
                            mosaic=True,
                            mixup=True)
    for i in range(1000):
        im, gt, h, w, scale, offset = dataset.pull_item(i)
        img = im.permute(1,2,0).numpy()[:, :, (2, 1, 0)].astype(np.uint8)
        cv2.imwrite('-1.jpg', img)
        img = cv2.imread('-1.jpg')
        for box in gt:
            x1, y1, x2, y2, _ = box
            x1 *= img_size
            y1 *= img_size
            x2 *= img_size
            y2 *= img_size
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        cv2.imshow('gt', img)
        cv2.waitKey(0)
