#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _pair

from functions.norm_preserve_deform_conv_func import NormPreserveDeformConvFunction

class NormPreserveDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                stride, padding, kernel_size=(3,3), init_weight=None, dilation=1, groups=1, deformable_groups=1,
                 im2col_step=64, bias=True):
        super(NormPreserveDeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = False
        self.weight = nn.Parameter(init_weight)
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
        if not self.use_bias:
            self.bias.requires_grad = False

    def forward(self, input, offset):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            offset.shape[1]
        return NormPreserveDeformConvFunction.apply(input, offset,
                                                   self.weight,
                                                   self.bias,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   self.groups,
                                                   self.deformable_groups,
                                                   self.im2col_step)

_NormPreserveDeformConv = NormPreserveDeformConvFunction.apply

class NormPreserveDeformConvPack(NormPreserveDeformConv):

    def __init__(self, in_channels, out_channels, stride, padding, kernel_size=(3,3), init_weight=None,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=1, bias=True, lr_mult=0.1):
        super(NormPreserveDeformConvPack, self).__init__(in_channels, out_channels, stride, padding, kernel_size, init_weight, dilation, groups, deformable_groups, im2col_step, bias)

        out_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(self.in_channels,
                                          out_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_weight = init_weight

        self.final_conv = nn.Conv2d(self.in_channels,
                                          self.out_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=False)

        self.init_offset()
        self.init_final_conv()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def init_final_conv(self):
        if self.init_weight != None:
            self.final_conv.weight.data = self.init_weight
            #self.conv_offset.bias.data.zero_()
        else:
            self.final_conv.weight.data.zero_()
            #self.conv_offset.bias.data.zero_()

    def forward(self, input, offset=None):
        if offset == None:
            offset = self.conv_offset(input)
        
        normalizing_factor = (self.weight.shape[-2] * self.weight.shape[-1])**0.5
        prototype_vector_length = torch.sqrt(torch.sum(torch.square(self.weight) + 1e-4, dim=-3))
        prototype_vector_length = prototype_vector_length.view(prototype_vector_length.size()[0], 
                                                                1,
                                                                prototype_vector_length.size()[1],
                                                                prototype_vector_length.size()[2])
        normalized_prototypes = self.weight / prototype_vector_length
        normalized_prototypes = normalized_prototypes / normalizing_factor
        result = NormPreserveDeformConvFunction.apply(input, offset, 
                                          normalized_prototypes, 
                                          self.bias, 
                                          self.stride, 
                                          self.padding, 
                                          self.dilation, 
                                          self.groups,
                                          self.deformable_groups,
                                          self.im2col_step)
        #result = self.final_conv(result)
        return result

