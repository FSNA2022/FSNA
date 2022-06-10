# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import ipdb
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from fsdet.layers import Conv2d, ShapeSpec, get_norm
from fsdet.utils.registry import Registry
from fsdet.modeling.backbone.resnet import BasicBlock, BottleneckBlock
from functools import partial
from .sync_batchnorm import SynchronizedBatchNorm2d

ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")
ROI_BOX_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""
class All_att(nn.Module):
    def __init__(self, channel):
        super(All_att, self).__init__()
        self.inter_channel = channel
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.linear_0 = nn.Conv1d(channel, channel, 1, bias=False)
    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # ipdb.set_trace()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1).cuda()   #经过1*1卷积降低通道数   Q
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous().cuda()   #K
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous().cuda()       #V
        self.mv = nn.Linear(x_g.shape[2], x_g.shape[2], bias=False).cuda()
        x_g = self.mv(x_g).cuda()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi).cuda()   #Q*K
        self.mk = nn.Linear(mul_theta_phi.shape[2], mul_theta_phi.shape[2], bias=False).cuda()
        mul_theta_phi = self.mk(mul_theta_phi).cuda()
        mul_theta_phi = self.softmax(mul_theta_phi).cuda()
        # print(mul_theta_phi[0,:,0])
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g).cuda()
        # ipdb.set_trace()
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w).cuda()
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g).cuda()
        out = mask + x
        return out

class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel 
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)   # Q
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()   #K
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()       #V
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)   #Q*K
        mul_theta_phi = self.softmax(mul_theta_phi)
        # print(mul_theta_phi[0,:,0])
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # ipdb.set_trace()
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

norm_layer = partial(SynchronizedBatchNorm2d, momentum=0.1)
class Total_Att(nn.Module):
    def __init__(self, channel):
        super(Total_Att, self).__init__()
        self.inter_channel = channel
        self.q = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.k = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.v = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear_0 = nn.Conv1d(channel, 64, 1, bias=False)
        self.linear_1 = nn.Conv1d(64, channel, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel, 1, bias=False), norm_layer(channel))
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        query = self.q(x).view(b, -1, h*w).permute(0, 2, 1).contiguous().cuda()
        key = self.k(x).view(b, -1, h*w).cuda()
        value = self.v(x).view(b, c, -1).cuda()
        energy = torch.matmul(query, key).cuda()  #b*n*n
        # self.linner = nn.Conv1d(energy.shape[2], energy.shape[2], 1, bias=False).cuda()
        # energy = self.linner(energy).cuda()
        ######################
        # value = self.linear_0(value)  效果不好
        attn = self.softmax(energy).cuda()  #b*n*n
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))
        out = torch.matmul(attn, value.permute(0, 2, 1)).cuda()  #b*n*c
        out = self.linear_0(out.permute(0, 2, 1)).cuda()    #Mv
        out = self.linear_1(out).cuda()
        out1 = out.view(b, c, h, w)
        out1 = self.conv2(out1)
        # out1 = self.conv_mask(out1)
        out2 = out1 + x
        # print("gammaaaaaaaaaaaaaaaaaaaaaaaaa:",self.gamma)
        return out2

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
            self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    ############################

 
    def forward(self, x):
    #     # ipdb.set_trace()
        x_cls = x
        attn = Total_Att(channel=256).cuda()
        x_cls = attn(x_cls)
        for layer in self.conv_norm_relus:
           # x = layer(x)
            x_cls = layer(x_cls)
        if len(self.fcs):
            if x.dim() > 2:
               # x = torch.flatten(x, start_dim=1)
                x_cls = torch.flatten(x_cls, start_dim=1)
            for layer in self.fcs:
               # x = F.relu(layer(x))
                x_cls = F.relu(layer(x_cls))
        return x_cls  # torch.Size([2560, 1024])

 
   # def forward(self,x):
   #     x_cls=x
   #     self_attn=NonLocalBlock(channel=256).cuda()
   #     x_cls = self_attn(x_cls)
   #     for layer in self.conv_norm_relus:
   #         x_cls = layer(x_cls)
   #     if len(self.fcs):
   #         if x.dim()>2:
   #             x_cls=torch.flatten(x_cls,start_dim=1)
   #         for layer in self.fcs:
   #             x_cls=F.relu(layer(x_cls))
   #     return x_cls

    @property
    def output_size(self):
        return self._output_size


@ROI_BOX_HEAD_REGISTRY.register()
class FastRcnnNovelHead(nn.Module):
    #     """
    #     A head that has separate 1024 fc for regression and classification branch
    #     """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        sub_fc_dim = cfg.MODEL.ROI_BOX_HEAD.SUB_FC_DIM
        norm = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        box_feat_shape = (input_shape.channels, input_shape.height, input_shape.width)

        self.fc_main = nn.Linear(np.prod(box_feat_shape), fc_dim)
        self.fc_reg = nn.Linear(fc_dim, sub_fc_dim)
        self.fc_cls = nn.Linear(fc_dim, sub_fc_dim)

        self._output_size = sub_fc_dim

        for layer in [self.fc_main, self.fc_reg, self.fc_cls]:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        main_feat = F.relu(self.fc_main(x))
        loc_feat = F.relu(self.fc_reg(main_feat))
        cls_feat = F.relu(self.fc_cls(main_feat))
        return loc_feat, cls_feat

    @property
    def output_size(self):
        return self._output_size


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNDoubleHead(nn.Module):
    """
    Double Head as described in https://arxiv.org/pdf/1904.06493.pdf
    The Conv Head composed of 1 (BasicBlock) + x (BottleneckBlock) and average pooling
    for bbox regression. From config: num_conv = 1 + x
    The FC Head composed of 2 fc layers for classification.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self.convs = []
        for k in range(num_conv):
            if k == 0:
                # import pdb; pdb.set_trace()
                conv = BasicBlock(input_shape.channels, conv_dim, norm=norm)
                # for name, param in conv.named_parameters():
                #     print(name, param.requires_grad)

                # bottleneck_channels = conv_dim // 4
                # conv = BottleneckBlock(input_shape.channels, conv_dim,
                #                        bottleneck_channels=bottleneck_channels, norm=norm)
                # import pdb; pdb.set_trace()
                # for name, param in conv.named_parameters():
                #     print(name, param)
            else:
                bottleneck_channels = conv_dim // 4
                conv = BottleneckBlock(conv_dim, conv_dim,
                                       bottleneck_channels=bottleneck_channels, norm=norm)
            self.add_module("conv{}".format(k + 1), conv)
            self.convs.append(conv)
        # this is a @property, see line 153, will be used as input_size for box_predictor
        # here when this function return, self._output_size = fc_dim (=1024)
        self._output_size = input_shape.channels * input_shape.height * input_shape.width
        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(self._output_size, fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        # init has already been done in BasicBlock and BottleneckBlock
        # for layer in self.conv_norm_relus:
        #     weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        loc_feat = x
        for layer in self.convs:
            loc_feat = layer(loc_feat)

        loc_feat = F.adaptive_avg_pool2d(loc_feat, (1, 1))
        loc_feat = torch.flatten(loc_feat, start_dim=1)

        cls_feat = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            cls_feat = F.relu(layer(cls_feat))
        return loc_feat, cls_feat

    @property
    def output_size(self):
        return self._output_size


def build_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)
