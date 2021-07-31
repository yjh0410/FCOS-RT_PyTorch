import torch
import torch.nn as nn
import math
from copy import deepcopy


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, p=0, s=1, d=1, g=1, act=True, bias=False):
        super(Conv, self).__init__()
        if act:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                nn.BatchNorm2d(out_ch)
            )
        self.init_weight()
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.convs(x)

class ModelEMA(object):
    def __init__(self, model, decay=0.9999, updates=0):
        # create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000.))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
