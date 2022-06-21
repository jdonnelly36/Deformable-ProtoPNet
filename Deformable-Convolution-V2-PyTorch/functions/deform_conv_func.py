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

import DCN

class DeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias,
                stride, padding, dilation, group, deformable_groups, im2col_step, zero_padding=True):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.group = group
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.zero_padding = zero_padding
        output = DCN.deform_conv_forward(input, weight, bias,
                                         offset,
                                         ctx.kernel_size[0], ctx.kernel_size[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.group,
                                         ctx.deformable_groups,
                                         ctx.im2col_step,
                                         ctx.zero_padding)
        # Want to also save x_deformed_sqrt
        ctx.save_for_backward(input, offset, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight, bias = ctx.saved_tensors
        '''print("\n(Pair A) DCN input is given as: ", input)
        print("\n(Pair B) DCN offset is given as: ", offset)
        print("\n(Pair C) DCN weight is given as: ", weight)
        print("\n(Pair D) DCN bias is given as: ", bias)'''
        
        print("\n(Pair H) DCN grad output is given as: ", grad_output)
        # Can't just use grad_output * 1/2 * x^(-1/2) because interpolation comes after sqrt func
        grad_input, grad_offset, grad_weight, grad_bias = \
            DCN.deform_conv_backward(input, weight,
                                     bias,
                                     offset,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1],
                                     ctx.stride[0], ctx.stride[1],
                                     ctx.padding[0], ctx.padding[1],
                                     ctx.dilation[0], ctx.dilation[1],
                                     ctx.group,
                                     ctx.deformable_groups,
                                     ctx.im2col_step,
                                     ctx.zero_padding)
        print("\nDCN grad offset shape: ", grad_offset.shape)
        print("\n(Pair E) Deformable convolution grad input:", grad_input)
        print("\n(Pair H) Deformable convolution grad weight:", grad_weight)
        print("\n(Pair G) Deformable convolution grad offset:", grad_offset)
        return grad_input, grad_offset, grad_weight, grad_bias,\
            None, None, None, None, None, None
