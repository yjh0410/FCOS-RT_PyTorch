import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .resnet import build_backbone

from utils.modules import Conv
import utils.box_ops as box_ops
from utils.loss import loss


class FCOS_RT(nn.Module):
    def __init__(self, 
                 device, 
                 img_size=640, 
                 num_classes=80, 
                 trainable=False, 
                 conf_thresh=0.05, 
                 nms_thresh=0.5, 
                 bk='r18'):
        super(FCOS_RT, self).__init__()
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.strides = [8, 16, 32]
        self.grid_cell = self.create_grid(img_size)

        # backbone
        self.backbone, feature_channels = build_backbone(pretrained=trainable, freeze=trainable, model=bk)
        c3, c4, c5 = feature_channels

        # latter layers
        self.latter_1 = nn.Conv2d(c3, 256, kernel_size=1)
        self.latter_2 = nn.Conv2d(c4, 256, kernel_size=1)
        self.latter_3 = nn.Conv2d(c5, 256, kernel_size=1)

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

        # init weight
        self.init_weight()


    def init_weight(self):
        for m in [self.latter_1, self.latter_2, self.latter_3]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in [self.smooth_1, self.smooth_2, self.smooth_3]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
            # [H, W, 2] -> [HW, 2]
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            # [1, H*W, 2]
            grid_xy = grid_xy[None, :, :].to(self.device)

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
        B = x.size(0)
        C = self.num_classes
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

        features = [p3, p4, p5]

        cls_pred = []
        reg_pred = []
        ctn_pred = []
        # head
        for i, p in enumerate(features):
            cls_head = self.cls_head(p)
            reg_head = self.reg_head(p)
            # [B, C, H, W] -> [B, H*W, C]
            cls_pred_i = self.cls_det(cls_head).permute(0, 2, 3, 1).reshape(B, -1, C)
            # [B, 4, H, W] -> [B, H*W, 4]
            reg_pred_i = self.reg_det(reg_head).permute(0, 2, 3, 1).reshape(B, -1, 4)
            x1y1_pred_i = (self.grid_cell[i] - reg_pred_i[..., :2].exp()) * self.strides[i] # x1y1
            x2y2_pred_i = (self.grid_cell[i] + reg_pred_i[..., 2:].exp()) * self.strides[i] # x2y2
            box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1)
            # [B, 1, H, W] -> [B, H*W, 1]
            ctn_det_i = self.ctn_det(reg_head).permute(0, 2, 3, 1).reshape(B, -1, 1)

            cls_pred.append(cls_pred_i)
            reg_pred.append(box_pred_i)
            ctn_pred.append(ctn_det_i)

        cls_pred = torch.cat(cls_pred, dim=1)  # [B, N, C]
        reg_pred = torch.cat(reg_pred, dim=1)  # [B, N, 4]
        ctn_pred = torch.cat(ctn_pred, dim=1)  # [B, N, 1]

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
                                            target=targets, 
                                            num_classes=self.num_classes)
            
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
