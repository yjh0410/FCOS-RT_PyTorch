"""https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/dla.py"""

""" Deep Layer Aggregation and DLA w/ Res2Net
DLA original adapted from Official Pytorch impl at:
DLA Paper: `Deep Layer Aggregation` - https://arxiv.org/abs/1707.06484
Res2Net additions from: https://github.com/gasvn/Res2Net/
Res2Net Paper: `Res2Net: A New Multi-scale Backbone Architecture` - https://arxiv.org/abs/1904.01169
"""
import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['DLA']


model_urls = {
    'dla34': 'http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth',
    'dla60': 'http://dl.yf.io/dla/models/imagenet/dla60-24839fc4.pth',
    'dla102': 'http://dl.yf.io/dla/models/imagenet/dla102-d94d9790.pth',
    'dla169': 'http://dl.yf.io/dla/models/imagenet/dla169-0914e092.pth'
}


class DlaBasic(nn.Module):
    """DLA Basic"""

    def __init__(self, inplanes, planes, stride=1, dilation=1, **_):
        super(DlaBasic, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, shortcut=None):
        if shortcut is None:
            shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut
        out = self.relu(out)

        return out


class DlaBottleneck(nn.Module):
    """DLA/DLA-X Bottleneck"""
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, dilation=1, cardinality=1, base_width=64):
        super(DlaBottleneck, self).__init__()
        self.stride = stride
        mid_planes = int(math.floor(outplanes * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion

        self.conv1 = nn.Conv2d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(
            mid_planes, mid_planes, kernel_size=3, stride=stride, padding=dilation,
            bias=False, dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, shortcut=None):
        if shortcut is None:
            shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)

        return out


class DlaBottle2neck(nn.Module):
    """ Res2Net/Res2NeXT DLA Bottleneck
    Adapted from https://github.com/gasvn/Res2Net/blob/master/dla.py
    """
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, dilation=1, scale=4, cardinality=8, base_width=4):
        super(DlaBottle2neck, self).__init__()
        self.is_first = stride > 1
        self.scale = scale
        mid_planes = int(math.floor(outplanes * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion
        self.width = mid_planes

        self.conv1 = nn.Conv2d(inplanes, mid_planes * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes * scale)

        num_scale_convs = max(1, scale - 1)
        convs = []
        bns = []
        for _ in range(num_scale_convs):
            convs.append(nn.Conv2d(
                mid_planes, mid_planes, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation, groups=cardinality, bias=False))
            bns.append(nn.BatchNorm2d(mid_planes))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        if self.is_first:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        self.conv3 = nn.Conv2d(mid_planes * scale, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, shortcut=None):
        if shortcut is None:
            shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        spo = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            sp = spx[i] if i == 0 or self.is_first else sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            spo.append(sp)
        if self.scale > 1:
            spo.append(self.pool(spx[-1]) if self.is_first else spx[-1])
        out = torch.cat(spo, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)

        return out


class DlaRoot(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut):
        super(DlaRoot, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.shortcut:
            x += children[0]
        x = self.relu(x)

        return x


class DlaTree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 dilation=1, cardinality=1, base_width=64,
                 level_root=False, root_dim=0, root_kernel_size=1, root_shortcut=False):
        super(DlaTree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else nn.Identity()
        self.project = nn.Identity()
        cargs = dict(dilation=dilation, cardinality=cardinality, base_width=base_width)
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, **cargs)
            self.tree2 = block(out_channels, out_channels, 1, **cargs)
            if in_channels != out_channels:
                # NOTE the official impl/weights have  project layers in levels > 1 case that are never
                # used, I've moved the project layer here to avoid wasted params but old checkpoints will
                # need strict=False while loading.
                self.project = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_channels))
        else:
            cargs.update(dict(root_kernel_size=root_kernel_size, root_shortcut=root_shortcut))
            self.tree1 = DlaTree(
                levels - 1, block, in_channels, out_channels, stride, root_dim=0, **cargs)
            self.tree2 = DlaTree(
                levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, **cargs)
        if levels == 1:
            self.root = DlaRoot(root_dim, out_channels, root_kernel_size, root_shortcut)
        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

    def forward(self, x, shortcut=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x)
        shortcut = self.project(bottom)
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, shortcut)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, 
                 levels, 
                 channels, 
                 output_stride=32, 
                 in_chans=3,
                 cardinality=1, 
                 base_width=64, 
                 block=DlaBottle2neck, 
                 shortcut_root=False,):
        super(DLA, self).__init__()
        self.channels = channels
        self.cardinality = cardinality
        self.base_width = base_width
        assert output_stride == 32  # FIXME support dilation

        self.base_layer = nn.Sequential(
            nn.Conv2d(in_chans, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        cargs = dict(cardinality=cardinality, base_width=base_width, root_shortcut=shortcut_root)
        self.level2 = DlaTree(levels[2], block, channels[1], channels[2], 2, level_root=False, **cargs)
        self.level3 = DlaTree(levels[3], block, channels[2], channels[3], 2, level_root=True, **cargs)
        self.level4 = DlaTree(levels[4], block, channels[3], channels[4], 2, level_root=True, **cargs)
        self.level5 = DlaTree(levels[5], block, channels[4], channels[5], 2, level_root=True, **cargs)
        self.feature_info = [
            dict(num_chs=channels[0], reduction=1, module='level0'),  # rare to have a meaningful stride 1 level
            dict(num_chs=channels[1], reduction=2, module='level1'),
            dict(num_chs=channels[2], reduction=4, module='level2'),
            dict(num_chs=channels[3], reduction=8, module='level3'),
            dict(num_chs=channels[4], reduction=16, module='level4'),
            dict(num_chs=channels[5], reduction=32, module='level5'),
        ]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.base_layer(x)
        x = self.level0(x)
        c1 = self.level1(x)
        c2 = self.level2(c1)
        c3 = self.level3(c2)
        c4 = self.level4(c3)
        c5 = self.level5(c4)

        return c3, c4, c5


def dla34(pretrained=False, **kwargs):  # DLA-34
    model = DLA(
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        block=DlaBasic
    )
    if pretrained:
        print('loading pre-trained dla34 ...')
        model.load_state_dict(model_zoo.load_url(model_urls['dla34']), strict=False)

    return model


def dla60(pretrained=False, **kwargs):  # DLA-60
    model = DLA(
        levels=(1, 1, 1, 2, 3, 1),
        channels=(16, 32, 128, 256, 512, 1024),
    )
    if pretrained:
        print('loading pre-trained dla60 ...')
        model.load_state_dict(model_zoo.load_url(model_urls['dla60']), strict=False)

    return model


def dla102(pretrained=False, **kwargs):  # DLA-102
    model = DLA(
        levels=[1, 1, 1, 3, 4, 1],
        channels=[16, 32, 128, 256, 512, 1024],
        block=DlaBottleneck,
        shortcut_root=True
    )
    if pretrained:
        print('loading pre-trained dla102 ...')
        model.load_state_dict(model_zoo.load_url(model_urls['dla102']), strict=False)
    
    return model


def dla169(pretrained=False, **kwargs):  # DLA-169
    model = DLA(
        levels=[1, 1, 2, 3, 5, 1],
        channels=[16, 32, 128, 256, 512, 1024],
        block=DlaBottleneck,
        shortcut_root=True
    )
    if pretrained:
        print('loading pre-trained dla169 ...')
        model.load_state_dict(model_zoo.load_url(model_urls['dla169']), strict=False)
    
    return model
  