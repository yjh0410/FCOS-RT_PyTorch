import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean', gamma=2.0, alpha=0.25):
        super(FocalWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, num_pos):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits, 
                                                     target=targets, 
                                                     reduction="none"
                                                     )
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = ce_loss * ((1.0 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = torch.sum(loss) / num_pos

        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss


def gt_creator(img_size, num_classes, strides, scale_range, label_lists=[]):
    batch_size = len(label_lists)
    w = h = img_size
    gt_tensor = []
    
    # empty gt tensor
    for s in strides:
        gt_tensor.append(np.zeros([batch_size, h//s, w//s, num_classes + 4 + 1]))

    # generate gt datas
    for bi in range(batch_size):
        for gt_label in label_lists[bi]:
            x1, y1, x2, y2 = gt_label[:-1]
            cls_id = int(gt_label[-1])

            # compute the center, width and height
            xc = (x2 + x1) / 2 * w
            yc = (y2 + y1) / 2 * h
            bw = (x2 - x1) * w
            bh = (y2 - y1) * h

            if bw < 1. or bh < 1.:
                # print('A dirty data !!!')
                continue    

            for sr in scale_range:
                for si, s in enumerate(strides):
                    hs, ws = h // s, w // s
                    x1_s, x2_s = x1 * ws, x2 * ws
                    y1_s, y2_s = y1 * hs, y2 * hs
                    xc_s = xc / s
                    yc_s = yc / s

                    gridx = int(xc_s)
                    gridy = int(yc_s)

                    # By default, we only consider the 3x3 neighborhood of the center point
                    for i in range(gridx - 1, gridx + 2):
                        for j in range(gridy - 1, gridy + 2):
                            if (j >= 0 and j < gt_tensor[si].shape[1]) and (i >= 0 and i < gt_tensor[si].shape[2]):
                                t = j - y1_s
                                b = y2_s - j
                                l = i - x1_s
                                r = x2_s - i
                                if min(t, b, l, r) > 0:
                                    if max(t, b, l, r) >= (sr[0]/s) and max(t, b, l, r) < (sr[1]/s):
                                        gt_tensor[si][bi, j, i, cls_id] = 1.0
                                        gt_tensor[si][bi, j, i, num_classes:num_classes + 4] = np.array([x1, y1, x2, y2])
                                        gt_tensor[si][bi, j, i, num_classes + 4] = np.sqrt(min(l, r) / max(l, r) * \
                                                                                           min(t, b) / max(t, b))
                                
    return gt_tensor


def loss(pred_cls, pred_giou, pred_ctn, label, num_classes):
    # create loss_f
    cls_loss_function = FocalWithLogitsLoss(reduction='mean')
    ctn_loss_function = nn.BCELoss(reduction='none')
    
    # groundtruth    
    gt_cls = label[..., :num_classes]
    gt_ctn = label[..., -1]
    gt_pos = (gt_ctn > 0.).float()
    num_pos = gt_pos.sum()

    # cls loss
    cls_loss = cls_loss_function(logits=pred_cls, targets=gt_cls, num_pos=num_pos)
        
    # reg loss
    reg_loss = ((1. - pred_giou) * gt_pos).sum() / num_pos

    # ctn loss
    ctn_loss = (ctn_loss_function(pred_ctn.sigmoid(), gt_ctn) * gt_pos).sum() / num_pos

    # total loss
    total_loss = cls_loss + reg_loss + ctn_loss

    return cls_loss, reg_loss, ctn_loss, total_loss


if __name__ == "__main__":
    pass