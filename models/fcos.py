import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from backbone import *

from utils.modules import Conv
import utils.box_ops as box_ops
from utils.loss import loss


class FCOS(nn.Module):
    def __init__(self, device, img_size, num_classes=80, trainable=False, conf_thresh=0.05, nms_thresh=0.5, bk='r18'):
        super(FCOS, self).__init__()
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.backbone = bk
        self.strides = [8, 16, 32, 64, 128]
        self.grid_cell = self.create_grid(img_size)

        if self.backbone == 'r18':
            print('use backbone: resnet-18', )
            self.backbone = resnet18(pretrained=trainable)
            c3, c4, c5 = 128, 256, 512

        elif self.backbone == 'r50':
            print('use backbone: resnet-50', )
            self.backbone = resnet50(pretrained=trainable)
            c3, c4, c5 = 128, 256, 512

        elif self.backbone == 'r101':
            print('use backbone: resnet-101', )
            self.backbone = resnet101(pretrained=trainable)
            c3, c4, c5 = 128, 256, 512

        # latter layers
        self.latter_1 = nn.Conv2d(c3, 256, kernel_size=1)
        self.latter_2 = nn.Conv2d(c4, 256, kernel_size=1)
        self.latter_3 = nn.Conv2d(c5, 256, kernel_size=1)
        self.latter_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        self.latter_5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)

        # smooth layers
        self.smooth_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # head
        self.cls_head = nn.Sequential(
            Conv(256, 256, k=3, p=1),
            Conv(256, 256, k=3, p=1),
            Conv(256, 256, k=3, p=1),
            Conv(256, 256, k=3, p=1)
        )
        self.reg_head = nn.Sequential(
            Conv(256, 256, k=3, p=1),
            Conv(256, 256, k=3, p=1),
            Conv(256, 256, k=3, p=1),
            Conv(256, 256, k=3, p=1)
        )

        # det
        self.cls_det = nn.Conv2d(256, self.num_classes, kernel_size=1)
        self.reg_det = nn.Conv2d(256, 4, kernel_size=1)
        self.ctn_det = nn.Conv2d(256, 1, kernel_size=1)

        # init weight of cls_pred
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.cls_det.bias, bias_value)


    def create_grid(self, img_size):
        total_grid_xy = []
        w = h = img_size
        for s in self.strides:
            # generate grid cells
            ws, hs = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
            # [1, H*W, 2]
            grid_xy = grid_xy.reshape(1, hs*ws, 2).to(self.device)

            total_grid_xy.append(grid_xy)

        return total_grid_xy


    def set_grid(self, img_size):
        self.img_size = img_size
        self.grid_cell = self.create_grid(img_size)


    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    def forward(self, x, targets=None):
        # backbone
        c3, c4, c5 = self.backbone(x)
        
        # fpn
        p5 = self.latter_3(c5)
        p5_up = F.interpolate(p5, scale_factor=2)
        p5 = self.smooth_3(p5)

        p4 = self.latter_2(c4) + p5_up
        p4_up = F.interpolate(p4, scale_factor=2)
        p4 = self.smooth_2(p4)

        p3 = self.smooth_1(self.latter_1(c3) + p4_up)
        # p5 -> p6, p6 -> p7
        p6 = self.latter_4(p5)
        p7 = self.latter_5(p6)

        features = [p3, p4, p5, p6, p7]
        cls_dets = []
        reg_dets = []
        ctn_dets = []

        # head
        for p in features:
            cls_head = self.cls_head(p)
            reg_head = self.reg_head(p)
            # det
            cls_dets.append(self.cls_det(cls_head))
            reg_dets.append(self.reg_det(reg_head))
            ctn_dets.append(self.ctn_det(reg_head))

        cls_pred = []
        reg_pred = []
        ctn_pred = []
        B = x.size(0)
        C = self.num_classes
        for i, s in enumerate(self.strides):
            # [B, C, H, W] -> [B, H*W, C]
            cls_det = cls_dets[i].permute(0, 2, 3, 1).reshape(B, -1, C)
            # [B, 4, H, W] -> [B, H*W, 4]
            reg_det = reg_dets[i].permute(0, 2, 3, 1).reshape(B, -1, 4)
            reg_det[..., :2] = (self.grid_cell[i] - reg_det[..., :2].exp()) * s # x1y1
            reg_det[..., 2:] = (self.grid_cell[i] + reg_det[..., 2:].exp()) * s # x2y2

            # [B, 1, H, W] -> [B, H*W]
            ctn_det = ctn_dets[i].permute(0, 2, 3, 1).reshape(B, -1, 1)

            cls_pred.append(cls_det)
            reg_pred.append(reg_det)
            ctn_pred.append(ctn_det)
        
        cls_pred = torch.cat(cls_pred, dim=1)  # [B, HW, C]
        reg_pred = torch.cat(reg_pred, dim=1)  # [B, HW, 4]
        ctn_pred = torch.cat(ctn_pred, dim=1)  # [B, HW, 1]

        # train
        if self.trainable:
            # compute giou between pred bboxes and gt bboxes
            x1y1x2y2_pred = (reg_pred / self.img_size).reshape(-1, 4)
            x1y1x2y2_gt = targets[:, :, -5:-1].reshape(-1, 4)

            # giou
            giou_pred = box_ops.giou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)

            # compute loss
            cls_loss, reg_loss, ctn_loss, total_loss = loss(
                                            pred_cls=cls_pred, 
                                            pred_giou=giou_pred,
                                            pred_ctn=ctn_pred,
                                            label=targets, 
                                            num_classes=self.num_classes
                                            )
            
            return cls_loss, reg_loss, ctn_loss, total_loss

        # test
        else:
            with torch.no_grad():
                # batch size = 1
                scores = torch.sqrt(cls_pred.sigmoid() * ctn_pred.sigmoid())[0]
                bboxes = torch.clamp(reg_pred / self.img_size, 0, 1)[0]
                
                # to cpu
                scores = scores.cpu().numpy()
                bboxes = bboxes.cpu().numpy()

                # postprocess
                bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

                return bboxes, scores, cls_inds
