import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, p=0, s=1, d=1, g=1, act='relu', bias=False):
        super(Conv, self).__init__()
        if act is not None:
            if act == 'relu':
                self.convs = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            elif act == 'leaky':
                self.convs = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.1, inplace=True)
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
