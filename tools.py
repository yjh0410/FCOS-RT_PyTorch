import numpy as np
from data import *
import torch.nn as nn
import torch.nn.functional as F

CLASS_COLOR = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(len(VOC_CLASSES))]

class BCE_focal_loss(nn.Module):
    def __init__(self, gamma=2):
        super(BCE_focal_loss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        loss = (1.0-inputs)**self.gamma * torch.log(inputs + 1e-14) + \
                (inputs)**self.gamma * torch.log(1.0 - inputs + 1e-14)
        loss = -torch.sum(torch.mean(loss, dim=1), dim=-1)
        return loss

def compute_iou(pred_box, gt_box):
    # calculate IoU
    # [l, t, r, b]

    w_gt = gt_box[:, 0, :] + gt_box[:, 2, :]
    h_gt = gt_box[:, 1, :] + gt_box[:, 3, :]
    w_pred = pred_box[:, 0, :] + pred_box[:, 2, :]
    h_pred = pred_box[:, 1, :] + pred_box[:, 3, :]
    S_gt = w_gt * h_gt
    S_pred = w_pred * h_pred
    I_h = torch.min(gt_box[:, 1, :], pred_box[:, 1, :]) + torch.min(gt_box[:, 3, :], pred_box[:, 3, :])
    I_w = torch.min(gt_box[:, 0, :], pred_box[:, 0, :]) + torch.min(gt_box[:, 2, :], pred_box[:, 2, :])
    S_I = I_h * I_w
    U = S_gt + S_pred - S_I + 1e-20
    IoU = S_I / U
    
    return IoU

def gt_creator(input_size, num_classes, stride, scale_thresholds, label_lists=[], name='VOC'):
    batch_size = len(label_lists)
    w = input_size[1]
    h = input_size[0]
    total = sum([(h // s) * (w // s) for s in stride])
    
    # 1 + num_classes = background + class label number
    # 1 = center-ness
    # 4 = l, t, r, b
    # 1 = positive sample
    gt_tensor = np.zeros([batch_size, 1 + num_classes + 1 + 4 + 1, total])

    # generate gt datas
    for batch_index in range(batch_size):
        for gt_box in label_lists[batch_index]:
            gt_class = gt_box[-1]
            xmin, ymin, xmax, ymax = gt_box[:-1]
            xmin *= w
            ymin *= h
            xmax *= w
            ymax *= h
            # check whether it is a dirty data
            if xmax - xmin < 1e-28 or ymax - ymin < 1e-28:
                print("find a dirty data !!!")
                continue

            start_index = 0
            for scale_index, s in enumerate(stride):
                #print(start_index)
                # get a feature map size corresponding the scale
                ws = w // s
                hs = h // s
                for ys in range(hs):
                    for xs in range(ws):
                        # map the location (xs, ys) in f_mp to corresponding location (x, y) in the origin image
                        x = xs * s + s // 2
                        y = ys * s + s // 2
                        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                            l = x - xmin
                            r = xmax - x
                            t = y - ymin
                            b = ymax - y
                            # select a appropriate scale for gt dat
                            M = max(l, t, r, b)
                            
                            if M >= scale_thresholds[scale_index] and M < scale_thresholds[scale_index+1]:
                                index = (ys * ws + xs) + start_index
                                center_ness = np.sqrt((min(l,r) / max(l,r)) * (min(t,b) / max(t,b)))
                                gt_tensor[batch_index, 1 + int(gt_class), index] = 1.0
                                gt_tensor[batch_index, 1 + num_classes, index] = center_ness
                                gt_tensor[batch_index, 1 + num_classes + 1 : -1, index] = torch.tensor([l, t, r, b])
                                gt_tensor[batch_index, -1, index] = 1.0
                start_index += ws * hs
    return gt_tensor

def loss(pred, label, num_classes):
    # define loss functions
    cls_loss_func = BCE_focal_loss()
    ctn_loss_func = nn.MSELoss(reduction='none')
    box_loss_func = nn.BCELoss(reduction='none')

    pred_cls = torch.sigmoid(pred[:, :1 + num_classes, :])
    pred_ctn = torch.sigmoid(pred[:, 1 + num_classes, :])
    pred_box = torch.exp(pred[:, 1 + num_classes + 1:, :])      

    gt_cls = label[:, :1 + num_classes, :].float()
    gt_ctn = label[:, 1 + num_classes, :].float()
    gt_box = label[:, 1 + num_classes + 1 : -1, :].float()
    gt_pos = label[:, -1, :]
    gt_iou = gt_pos.clone()
    N_pos = torch.sum(gt_pos, dim=-1)

    # cls loss
    cls_loss = torch.mean(cls_loss_func(pred_cls, gt_cls) / N_pos)
    
    # ctn loss
    ctn_loss = torch.mean(torch.sum(ctn_loss_func(pred_ctn, gt_ctn), dim=-1) / N_pos)
    
    # box loss
    iou = compute_iou(pred_box, gt_box)
    box_loss = torch.mean(torch.sum(box_loss_func(iou, gt_iou) * gt_pos, dim=-1) / N_pos)

    return cls_loss, ctn_loss, box_loss

if __name__ == "__main__":
    stride = [8, 16, 32]
    scale_thresholds = [0, 64, 128, 1e10]
    input_size = [416, 416]
    num_classes = 20
    total_fmap_size = [input_size[0] // s for s in stride]
    voc_root = 'C:/YJH-HOST/dataset/VOCdevkit/'

    dataset = VOCDetection(voc_root, [('2007', 'trainval'), ('2012', 'trainval')], transform=BaseTransform(input_size, (0, 0, 0)))
    data_loader = torch.utils.data.DataLoader(dataset, 1,
                                  num_workers=0,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)
    batch_iterator = iter(data_loader)
    count = 0
    for images, targets in batch_iterator:
        # origin image
        img = dataset.pull_image(count)
        img = cv2.resize(img, (input_size[1], input_size[0]))
        count += 1
        # gt labels
        targets = [label.tolist() for label in targets]
        gt_tensor = gt_creator(input_size=input_size, num_classes=num_classes, 
                             stride=stride, scale_thresholds=scale_thresholds, 
                             label_lists=targets)

        CLASSES = VOC_CLASSES
        class_color = CLASS_COLOR
        start_index = 0
        for fmap_size, s in zip(total_fmap_size, stride):
            for ys in range(fmap_size):
                for xs in range(fmap_size):
                    x = xs * s + s // 2
                    y = ys * s + s // 2
                    index = (ys * fmap_size + xs) + start_index
                    if gt_tensor[:, -1, index] == 1.0:
                        cls_label = np.argmax(gt_tensor[:, :1+num_classes, index], axis=1) - 1
                        cv2.circle(img, (int(x), int(y)), 5, class_color[int(cls_label)], -1)
            start_index += fmap_size * fmap_size
        cv2.imshow('image', img)
        cv2.waitKey(0)
