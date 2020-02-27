#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

import _ext as _backend


class _DCNv2(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        mask,
        weight,
        bias,
        stride,
        padding,
        dilation,
        deformable_groups,
        # use_amp,
    ):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.deformable_groups = deformable_groups
        # ctx.use_amp = use_amp
        ctx.use_amp = True
        use_amp = True

        if use_amp:
            input = input.float()
            offset = bias.float()
            mask = mask.float()
            weight = weight.float()
            bias = bias.float()

        output = _backend.dcn_v2_forward(
            input,
            weight,
            bias,
            offset,
            mask,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.deformable_groups,
        )
        ctx.save_for_backward(input, offset, mask, weight, bias)
        if use_amp:
            return output.half()
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        print("in backward")
        input, offset, mask, weight, bias = ctx.saved_tensors
        if ctx.use_amp:
            grad_output = grad_output.float()
        (
            grad_input,
            grad_offset,
            grad_mask,
            grad_weight,
            grad_bias,
        ) = _backend.dcn_v2_backward(
            input,
            weight,
            bias,
            offset,
            mask,
            grad_output,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.deformable_groups,
        )

        if ctx.use_amp:
            grad_input = grad_input.half()
            grad_offset = grad_offset.half()
            grad_mask = grad_mask.half()
            grad_weight = grad_weight.half()
            grad_bias = grad_bias.half()
        return (
            grad_input,
            grad_offset,
            grad_mask,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            # None,
        )


dcn_v2_conv = _DCNv2.apply


class DCNv2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
        # use_amp=False,
    ):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups
        # self.use_amp = use_amp
        self.use_amp = True

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert (
            2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
            == offset.shape[1]
        )
        assert (
            self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
            == mask.shape[1]
        )
        return dcn_v2_conv(
            input,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
            # self.use_amp,
        )
        # add if amp here


class DCN(DCNv2):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
        # use_amp=False,
    ):
        super(DCN, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            deformable_groups,
            # use_amp,
        )
        # self.use_amp = use_amp
        self.use_amp = True
        channels_ = (
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        )
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(
            input,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
            # self.use_amp,
        )


class _DCNv2Pooling(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        rois,
        offset,
        spatial_scale,
        pooled_size,
        output_dim,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=0.0,
        # use_amp=False,
    ):
        ctx.spatial_scale = spatial_scale
        ctx.no_trans = int(no_trans)
        ctx.output_dim = output_dim
        ctx.group_size = group_size
        ctx.pooled_size = pooled_size
        ctx.part_size = pooled_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std
        # ctx.use_amp = use_amp
        ctx.use_amp = True

        if use_amp:
            input = input.float()
            rois = rois.float()
            offset = offset.float()

        output, output_count = _backend.dcn_v2_psroi_pooling_forward(
            input,
            rois,
            offset,
            ctx.no_trans,
            ctx.spatial_scale,
            ctx.output_dim,
            ctx.group_size,
            ctx.pooled_size,
            ctx.part_size,
            ctx.sample_per_part,
            ctx.trans_std,
        )
        ctx.save_for_backward(input, rois, offset, output_count)
        if use_amp:
            return output.half()
        else:
            return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, offset, output_count = ctx.saved_tensors
        grad_input, grad_offset = _backend.dcn_v2_psroi_pooling_backward(
            grad_output,
            input,
            rois,
            offset,
            output_count,
            ctx.no_trans,
            ctx.spatial_scale,
            ctx.output_dim,
            ctx.group_size,
            ctx.pooled_size,
            ctx.part_size,
            ctx.sample_per_part,
            ctx.trans_std,
        )

        if ctx.use_amp:
            grad_input = grad_input.half()
            grad_offset = grad_offset.half()

        return (
            grad_input,
            None,
            grad_offset,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            # None,
        )


dcn_v2_pooling = _DCNv2Pooling.apply


class DCNv2Pooling(nn.Module):
    def __init__(
        self,
        spatial_scale,
        pooled_size,
        output_dim,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=0.0,
        # use_amp=False,
    ):
        super(DCNv2Pooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std
        # self.use_amp = use_amp
        self.use_amp = True

    def forward(self, input, rois, offset):
        assert input.shape[1] == self.output_dim
        if self.no_trans:
            offset = input.new()
        return dcn_v2_pooling(
            input,
            rois,
            offset,
            self.spatial_scale,
            self.pooled_size,
            self.output_dim,
            self.no_trans,
            self.group_size,
            self.part_size,
            self.sample_per_part,
            self.trans_std,
            # self.use_amp,
        )


class DCNPooling(DCNv2Pooling):
    def __init__(
        self,
        spatial_scale,
        pooled_size,
        output_dim,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=0.0,
        deform_fc_dim=1024,
        # use_amp=False,
    ):
        super(DCNPooling, self).__init__(
            spatial_scale,
            pooled_size,
            output_dim,
            no_trans,
            group_size,
            part_size,
            sample_per_part,
            trans_std,
            # use_amp,
        )
        # self.use_amp = use_amp
        self.use_amp = True

        self.deform_fc_dim = deform_fc_dim

        if not no_trans:
            self.offset_mask_fc = nn.Sequential(
                nn.Linear(
                    self.pooled_size * self.pooled_size * self.output_dim,
                    self.deform_fc_dim,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.pooled_size * self.pooled_size * 3),
            )
            self.offset_mask_fc[4].weight.data.zero_()
            self.offset_mask_fc[4].bias.data.zero_()

    def forward(self, input, rois):
        offset = input.new()

        if not self.no_trans:

            # do roi_align first
            n = rois.shape[0]
            roi = dcn_v2_pooling(
                input,
                rois,
                offset,
                self.spatial_scale,
                self.pooled_size,
                self.output_dim,
                True,  # no trans
                self.group_size,
                self.part_size,
                self.sample_per_part,
                self.trans_std,
                # self.use_amp,
            )

            # build mask and offset
            offset_mask = self.offset_mask_fc(roi.view(n, -1))
            offset_mask = offset_mask.view(n, 3, self.pooled_size, self.pooled_size)
            o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)

            # do pooling with offset and mask
            return (
                dcn_v2_pooling(
                    input,
                    rois,
                    offset,
                    self.spatial_scale,
                    self.pooled_size,
                    self.output_dim,
                    self.no_trans,
                    self.group_size,
                    self.part_size,
                    self.sample_per_part,
                    self.trans_std,
                    # self.use_amp,
                )
                * mask
            )
        # only roi_align
        return dcn_v2_pooling(
            input,
            rois,
            offset,
            self.spatial_scale,
            self.pooled_size,
            self.output_dim,
            self.no_trans,
            self.group_size,
            self.part_size,
            self.sample_per_part,
            self.trans_std,
            # self.use_amp,
        )

