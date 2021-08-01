import torch
import torch.nn as nn
import numpy as np
import os


__all__ = ['darknet19', 'darknet53']


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class resblock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch//2, 1),
                Conv_BN_LeakyReLU(ch//2, ch, 3, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


class DarkNet_19(nn.Module):
    def __init__(self):
        print("Initializing the darknet19 network ......")
        
        super(DarkNet_19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            nn.MaxPool2d((2,2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d((2,2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d((2,2), 2)
        )

        # output : stride = 8, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
        )

        # output : stride = 16, c = 512
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
        )
        
        # output : stride = 32, c = 1024
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )


    def forward(self, x):
        x = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(c2)
        c3 = self.conv_4(c3)
        c4 = self.conv_5(self.maxpool_4(c3))
        c5 = self.conv_6(self.maxpool_5(c4))

        return c3, c4, c5


class DarkNet_53(nn.Module):
    """
    DarkNet-53.
    """
    def __init__(self):
        super(DarkNet_53, self).__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, padding=1),
            Conv_BN_LeakyReLU(32, 64, 3, padding=1, stride=2),
            resblock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, padding=1, stride=2),
            resblock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, padding=1, stride=2),
            resblock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, padding=1, stride=2),
            resblock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, padding=1, stride=2),
            resblock(1024, nblocks=4)
        )


    def forward(self, x, targets=None):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)


        return c3, c4, c5


def darknet19(pretrained=False, hr=False, **kwargs):
    """Constructs a darknet-19 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_19()
    if pretrained:
        print('Loading the pretrained model ...')
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        if hr:
            print('Loading the hi-res darknet19-448 ...')
            model.load_state_dict(torch.load(path_to_dir + '/weights/darknet19_hr_75.52_92.73.pth', map_location='cuda'), strict=False)
        else:
            print('Loading the darknet19 ...')
            model.load_state_dict(torch.load(path_to_dir + '/weights/darknet19_72.96.pth', map_location='cuda'), strict=False)
    return model


def darknet53(pretrained=False, hr=False, **kwargs):
    """Constructs a darknet-53 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_53()
    if pretrained:
        print('Loading the pretrained model ...')
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        if hr:
            print('Loading the hi-res darknet53-448 ...')
            model.load_state_dict(torch.load(path_to_dir + '/weights/darknet53/darknet53_hr_77.76.pth', map_location='cuda'), strict=False)
        else:
            print('Loading the darknet53 ...')
            model.load_state_dict(torch.load(path_to_dir + '/weights/darknet53/darknet53_75.42.pth', map_location='cuda'), strict=False)
    return model
