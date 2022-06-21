#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

import DCN

class NormPreserveDeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias,
                stride, padding, dilation, group, deformable_groups, im2col_step, zero_padding=True):
        #assert input.shape[-3] % 2 == 0, "STANDING BUG: input must have an even number of channels"

        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.group = group
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.zero_padding = zero_padding
        output, deformed_columns = DCN.norm_preserve_deform_conv_forward(input, weight, bias,
                                         offset,
                                         ctx.kernel_size[0], ctx.kernel_size[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.group,
                                         ctx.deformable_groups,
                                         ctx.im2col_step,
                                         ctx.zero_padding)
        ctx.save_for_backward(input, offset, weight, bias, deformed_columns)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight, bias, deformed_columns = ctx.saved_tensors
        
        grad_input, grad_offset, grad_weight, grad_bias = \
            DCN.norm_preserve_deform_conv_backward(input, deformed_columns, weight,
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
        return grad_input, grad_offset, grad_weight, grad_bias,\
            None, None, None, None, None, None, None
