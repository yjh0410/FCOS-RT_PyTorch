import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv2d
from backbone import *
import os
import numpy as np
import tools

class FCOS_LITE(nn.Module):
    def __init__(self, device, input_size, num_classes=20, trainable=False, conf_thresh=0.05, nms_thresh=0.5, hr=False):
        super(FCOS_LITE, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.location_weight =torch.tensor([[-1, -1, 1, 1]]).float().to(device)
        self.stride = [8, 16, 32, 64]
        self.scale_thresholds = [0, 49, 98, 196, 1e10]
        self.pixel_location = self.set_init()
        self.scale = np.array([[input_size[1], input_size[0], input_size[1], input_size[0]]])
        self.scale_torch = torch.tensor(self.scale.copy()).float()

        # backbone resnet-18
        self.backbone = resnet18(pretrained=trainable)
        
        # C_5 -> C_6
        self.conv_3x3_C_6 = Conv2d(512, 1024, 3, padding=1, stride=2)

        # detection head
        # All branches share the self.head
        # 1 is for background label

        # P_6
        self.conv_set_6 = nn.Sequential(
            Conv2d(1024, 512, 1),
            Conv2d(512, 1024, 3, padding=1),
            Conv2d(1024, 512, 1),
        )
        self.pred_6 = nn.Sequential(
            Conv2d(512, 1024, 3, padding=1),
            nn.Conv2d(1024, 1 + self.num_classes + 1 + 4, 1)
        )

        # P_5
        self.conv_set_5 = nn.Sequential(
            Conv2d(512, 256, 1),
            Conv2d(256, 512, 3, padding=1),
            Conv2d(512, 256, 1)
        )
        self.conv_1x1_5 = Conv2d(256, 128, 1)
        self.pred_5 = nn.Sequential(
            Conv2d(256, 512, 3, padding=1),
            nn.Conv2d(512, 1 + self.num_classes + 1 + 4, 1)
        )

        # P_4
        self.conv_set_4 = nn.Sequential(
            Conv2d(384, 128, 1),
            Conv2d(128, 256, 3, padding=1),
            Conv2d(256, 128, 1)
        )
        self.conv_1x1_4 = Conv2d(128, 64, 1)
        self.pred_4 = nn.Sequential(
            Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 1 + self.num_classes + 1 + 4, 1)
        )

        # P_3
        self.conv_set_3 = nn.Sequential(
            Conv2d(192, 64, 1),
            Conv2d(64, 128, 3, padding=1),
            Conv2d(128, 64, 1)
        )
        self.pred_3 = nn.Sequential(
            Conv2d(64, 128, 3, padding=1),
            nn.Conv2d(128, 1 + self.num_classes + 1 + 4, 1)
        )


    def set_init(self):
        total = sum([(self.input_size[0] // s) * (self.input_size[1] // s) for s in self.stride])
        pixel_location = torch.zeros(total, 4).to(self.device)
        start_index = 0
        for index in range(len(self.stride)):
            s = self.stride[index]
            # make a feature map size corresponding the scale
            ws = self.input_size[1] // s
            hs = self.input_size[0] // s
            for ys in range(hs):
                for xs in range(ws):
                    x_y = ys * ws + xs
                    index = x_y + start_index
                    x = xs * s + s / 2
                    y = ys * s + s / 2
                    pixel_location[index, :] = torch.tensor([[x, y, x, y]]).float()
            start_index += ws * hs
        return pixel_location        

    def clip_boxes(self, boxes, im_shape):
        """
        Clip boxes to image boundaries.
        """
        if boxes.shape[0] == 0:
            return boxes

        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes

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
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, all_local, all_conf, im_shape=None):
        """
        bbox_pred: (HxW*anchor_n, 4), bsize = 1
        prob_pred: (HxW*anchor_n, num_classes), bsize = 1
        """
        bbox_pred = all_local
        prob_pred = all_conf

        cls_inds = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
        scores = prob_pred.copy()
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bbox_pred), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_pred[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        if im_shape != None:
            # clip
            bbox_pred = self.clip_boxes(bbox_pred, im_shape)

        return bbox_pred, scores, cls_inds

    def forward(self, x):
        # backbone
        C_3, C_4, C_5 = self.backbone(x)
        C_6 = self.conv_3x3_C_6(C_5)
        B = C_3.shape[0]

        # P_6
        pred_6 = self.pred_6(self.conv_set_6(C_6)).view(B, 1 + self.num_classes + 1 + 4, -1)

        # P_5
        C_5 = self.conv_set_5(C_5)
        C_5_up = F.interpolate(self.conv_1x1_5(C_5), scale_factor=2.0, mode='bilinear', align_corners=True)
        pred_5 = self.pred_5(C_5).view(B, 1 + self.num_classes + 1 + 4, -1)

        # P_4
        C_4 = torch.cat([C_4, C_5_up], dim=1)
        C_4 = self.conv_set_4(C_4)
        C_4_up = F.interpolate(self.conv_1x1_4(C_4), scale_factor=2.0, mode='bilinear', align_corners=True)
        pred_4 = self.pred_4(C_4).view(B, 1 + self.num_classes + 1 + 4, -1)

        # P_3
        C_3 = torch.cat([C_3, C_4_up], dim=1)
        C_3 = self.conv_set_3(C_3)
        pred_3 = self.pred_3(C_3).view(B, 1 + self.num_classes + 1 + 4, -1)

        total_prediction = torch.cat([pred_3, pred_4, pred_5, pred_6], dim=-1).permute(0, 2, 1)
        # test
        if not self.trainable:
            with torch.no_grad():
                # batch size = 1
                # Be careful, the index 0 in all_cls is background !!
                # As index 0 is background, we just ignore it.
                all_cls = torch.sigmoid(total_prediction[0, :, 1:1 + self.num_classes])
                all_ctn = torch.sigmoid(total_prediction[0, :, 1 + self.num_classes : 1 + self.num_classes + 1])
                all_loc = torch.exp(total_prediction[0, :, 1 + self.num_classes + 1 : ]) * self.location_weight + self.pixel_location
                # separate box pred and class conf
                all_cls = all_cls.to('cpu').numpy()
                all_ctn = all_ctn.to('cpu').numpy()
                all_loc = all_loc.to('cpu').numpy()

                bboxes, scores, cls_inds = self.postprocess(all_loc, all_cls)# * all_ctn
                # clip the boxes
                bboxes = self.clip_boxes(bboxes, self.input_size) / self.scale

                # print(len(all_boxes))
                return bboxes, scores, cls_inds

        return total_prediction