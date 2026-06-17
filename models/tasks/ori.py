from typing import Optional
import torch
from torch import nn
from torchvision import models
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation, Conv2dNormActivation
from functools import partial
import torch 
from torch import nn,Tensor
from typing import *

if __name__=="__main__":
    from ..backbones.omniSeam.omnipose import get_omnipose
    from ..backbones.omniSeam.hrnet import get_hrnet
    from ..backbones.resnet_dcn_3_ch import get_resnet_ch_3
else:
    from models.backbones.omniSeam.omnipose import get_omnipose
    from models.backbones.omniSeam.hrnet import get_hrnet
    from models.backbones.resnet_dcn_3_ch import get_resnet_ch_3

class Omnipose_mult(nn.Module):
    def __init__(self,type='common'):
        super().__init__()
        self.backbone = get_omnipose(type)
        last_conv_channels = self.backbone.get_last_channel()
        cls_cnt = 5
        self.hm = nn.Conv2d(last_conv_channels,cls_cnt,kernel_size=3,stride=1,padding=1)
        self.off = nn.Conv2d(last_conv_channels,4,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        x = self.backbone(x)
        hm = self.hm(x)
        off = self.off(x)
        return (hm,off)
    
class Omnipose_mod_mult(nn.Module):
    def __init__(self,type='common',with_gauss_filter=True):
        super().__init__()
        print("ori:{}".format(type))
        self.backbone = get_omnipose(type,with_gauss_filter=with_gauss_filter)
        last_conv_channels = self.backbone.get_last_channel()
        print("last_channel:{}".format(last_conv_channels))
        cls_cnt = 5
        if type == "2stage_32":
            planes = 32
        else:
            planes = 48
        # reduction = 6
        self.hm = nn.Sequential(nn.Conv2d(last_conv_channels, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(),
                                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(),
                                nn.Conv2d(planes, cls_cnt, kernel_size=1, stride=1))
        self.off = nn.Sequential(nn.Conv2d(last_conv_channels, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(),
                                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(),
                                nn.Conv2d(planes, 4, kernel_size=1, stride=1))
        self.ori = nn.Sequential(nn.Conv2d(last_conv_channels, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(),
                                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(),
                                nn.Conv2d(planes, 4, kernel_size=1, stride=1))

        # self.hm = nn.Conv2d(last_conv_channels,cls_cnt,kernel_size=3,stride=1,padding=1)
        # self.off = nn.Conv2d(last_conv_channels,4,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        x = self.backbone(x)
        hm = self.hm(x)
        off = self.off(x)
        ori = self.ori(x)
        return (hm,off,ori)

class HRNet(nn.Module):
    def __init__(self,type='common',with_gauss_filter=True):
        super().__init__()
        self.backbone = get_hrnet(type,with_gauss_filter=with_gauss_filter)
        # last_conv_channels = self.backbone.get_last_channel()
        cls_cnt = 5
        planes = 48
        reduction = 0
        self.hm = nn.Sequential(nn.Conv2d(planes+reduction, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(),
                                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(),
                                nn.Conv2d(planes, cls_cnt, kernel_size=1, stride=1))
        self.off = nn.Sequential(nn.Conv2d(planes+reduction, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(),
                                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(),
                                nn.Conv2d(planes, 4, kernel_size=1, stride=1))
        self.ori = nn.Sequential(nn.Conv2d(planes+reduction, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(),
                                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(),
                                nn.Conv2d(planes, 4, kernel_size=1, stride=1))

        # self.hm = nn.Conv2d(last_conv_channels,cls_cnt,kernel_size=3,stride=1,padding=1)
        # self.off = nn.Conv2d(last_conv_channels,4,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        x = self.backbone(x)
        hm = self.hm(x)
        off = self.off(x)
        ori = self.ori(x)
        return (hm,off,ori)
    
class MyModelResNet_head_MB_mult_innerdecoder(nn.Module):
    def __init__(self,layer_num):
        super().__init__()
        cls_cnt = 5
        self.backbone = get_resnet_ch_3(layer_num,decoder=True)
        if layer_num < 50:
            last_conv_channels = 64# self.backbone.get_last_channel()
        else:
            last_conv_channels = 64# self.backbone.get_last_channel()
        self.hm = nn.Conv2d(last_conv_channels,cls_cnt,kernel_size=3,stride=1,padding=1)
        self.off = nn.Conv2d(last_conv_channels,4,kernel_size=3,stride=1,padding=1)
        self.ori = nn.Conv2d(last_conv_channels,4,kernel_size=3,stride=1,padding=1)
    def forward(self,x:Tensor):
        x = self.backbone(x)
        hm = self.hm(x)
        off = self.off(x)
        ori = self.ori(x)
        return (hm,off,ori) 
    
