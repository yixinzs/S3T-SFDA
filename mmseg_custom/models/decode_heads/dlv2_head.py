# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.aspp_head import ASPPModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from torch import nn
import torch.nn.functional as F

@HEADS.register_module()
class DLV2Head(BaseDecodeHead):

    def __init__(self, dilations=(6, 12, 18, 24), **kwargs):
        assert 'channels' not in kwargs
        assert 'dropout_ratio' not in kwargs
        assert 'norm_cfg' not in kwargs
        kwargs['channels'] = 1
        kwargs['dropout_ratio'] = 0
        kwargs['norm_cfg'] = None
        super(DLV2Head, self).__init__(**kwargs)
        del self.conv_seg
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.num_classes,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)

    def forward(self, inputs):
        """Forward function."""
        # for f in inputs:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')
        x = self._transform_inputs(inputs)
        aspp_outs = self.aspp_modules(x)
        out = aspp_outs[0]
        for i in range(len(aspp_outs) - 1):
            out += aspp_outs[i + 1]
        return out

@HEADS.register_module()
class ASPP_Classifier_V2(BaseDecodeHead):
    def __init__(self, dilations=(6, 12, 18, 24), paddings=(6, 12, 18, 24), **kwargs):
        assert 'channels' not in kwargs
        assert 'dropout_ratio' not in kwargs
        assert 'norm_cfg' not in kwargs
        kwargs['channels'] = 512
        # kwargs['dropout_ratio'] = 0
        kwargs['norm_cfg'] = None
        super(ASPP_Classifier_V2, self).__init__(**kwargs)
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilations, paddings):
            self.conv2d_list.append(
                nn.Conv2d(
                    self.in_channels,
                    self.num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, inputs, size=None):
        x = self._transform_inputs(inputs)
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out
