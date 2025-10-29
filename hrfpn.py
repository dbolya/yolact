#!/usr/bin/env python
# coding=utf-8
"""
# This module provide
# Authors: mmdetection, fixed by zhaojing31@baidu.com
# Date: 2020/04/02
"""

import torch

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.cnn.weight_init import caffe2_xavier_init
from torch.utils.checkpoint import checkpoint


class HRFPN(nn.Module):
    """HRFPN (High Resolution Feature Pyrmamids)

    arXiv: https://arxiv.org/abs/1904.04514

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        stride (int): stride of 3x3 convolutional layers
    """

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 num_outs=5,
                 pooling_type='AVG',
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False,
                 stride=1):
        super(HRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
#         print('enter hrfpn')
#         print('self.num_ins:', self.num_ins)
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(self.in_channels),
                      out_channels=self.out_channels,
                      kernel_size=1),
        )
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_convs.append(nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                padding=1
            ))

        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        assert len(inputs[0]) == self.num_ins
        ###add my own
        new_inputs = inputs[0]
        outs = [new_inputs[0]]
        for i in range(1, self.num_ins):
            ####add my own
            _, _, h, w = outs[0].size()
            outs.append(
                F.interpolate(new_inputs[i], size=(h, w), mode='bilinear', align_corners=False))
        out = torch.cat(outs, dim=1)
        if out.requires_grad and self.with_cp:
            out = checkpoint(self.reduction_conv, out)
        else:
            out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, self.num_outs):
            outs.append(self.pooling(out, kernel_size=2 ** i, stride=2 ** i))
        outputs = []

        for i in range(self.num_outs):
            if outs[i].requires_grad and self.with_cp:
                tmp_out = checkpoint(self.fpn_convs[i], outs[i])
            else:
                tmp_out = self.fpn_convs[i](outs[i])
            outputs.append(tmp_out)
        return tuple(outputs)
