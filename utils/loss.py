import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean', gamma=2.0, alpha=0.25):
        super(FocalWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
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
            batch_size = logits.size(0)
            pos_inds = (targets == 1.0).float()
            # [B, H*W, C] -> [B,]
            num_pos = pos_inds.sum([1, 2]).clamp(1)
            loss = loss.sum([1, 2])
    
            loss = (loss / num_pos).sum() / batch_size

        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss


def loss(pred_cls, pred_giou, pred_ctn, label, num_classes):
    # create loss_f
    cls_loss_function = FocalWithLogitsLoss(reduction='mean')
    ctn_loss_function = nn.BCELoss(reduction='none')
    
    # groundtruth    
    gt_cls = label[..., :num_classes]
    gt_ctn = label[..., -1]
    gt_pos = (gt_ctn > 0.).float()
    num_pos = gt_pos.sum(-1, keepdim=True).clamp(1)

    batch_size = pred_cls.size(0)
    # cls loss
    cls_loss = cls_loss_function(pred_cls, gt_cls)
        
    # reg loss
    reg_loss = ((1. - pred_giou) * gt_pos / num_pos).sum() / batch_size

    # ctn loss
    ctn_loss = (ctn_loss_function(pred_ctn.sigmoid(), gt_ctn) * gt_pos / num_pos).sum() / batch_size

    # total loss
    total_loss = cls_loss + reg_loss + ctn_loss

    return cls_loss, reg_loss, ctn_loss, total_loss


if __name__ == "__main__":
    pass