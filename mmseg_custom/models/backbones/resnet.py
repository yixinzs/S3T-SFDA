import torch.nn as nn

from mmcv.runner import BaseModule, _load_checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
import math

__all__ = ['ResNetV1']


class BasicBlockV1b(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,
                               dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckV1b(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetV1(BaseModule):

    def __init__(self, block, layers, num_classes=1000, deep_stem=False,
                 zero_init_residual=False, output_stride=16, scale=1.0, norm_layer=nn.BatchNorm2d, pretrained=None, init_cfg=None):

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'

        self.pretrained = pretrained
        self.init_cfg = init_cfg

        if output_stride == 32:
            dilations = [1, 1]
            strides = [2, 2]
        elif output_stride == 16:
            dilations = [1, 2]
            strides = [2, 1]
        elif output_stride == 8:
            dilations = [2, 4]
            strides = [1, 1]
        else:
            raise NotImplementedError
        self.inplanes = int((128 if deep_stem else 64) * scale)
        super(ResNetV1, self).__init__()
        if deep_stem:
            # resnet vc
            mid_channel = int(64 * scale)
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, mid_channel, 3, 2, 1, bias=False),
                norm_layer(mid_channel),
                nn.ReLU(True),
                nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, bias=False),
                norm_layer(mid_channel),
                nn.ReLU(True),
                nn.Conv2d(mid_channel, self.inplanes, 3, 1, 1, bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, int(64 * scale), layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, int(128 * scale), layers[1], stride=2, norm_layer=norm_layer)

        self.layer3 = self._make_layer(block, int(256 * scale), layers[2], stride=strides[0], dilation=dilations[0],
                                       norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, int(512 * scale), layers[3], stride=strides[1], dilation=dilations[1],
                                       norm_layer=norm_layer)

        self.last_inp_channels = int(512 * block.expansion * scale)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * block.expansion * scale), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckV1b):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockV1b):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d,
                    multi_grid=False, multi_dilation=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if not multi_grid:
            if dilation in (1, 2):
                layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample,
                                    previous_dilation=dilation, norm_layer=norm_layer))
            elif dilation == 4:
                layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample,
                                    previous_dilation=dilation, norm_layer=norm_layer))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        else:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilation[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion

        if multi_grid:
            div = len(multi_dilation)
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=multi_dilation[i % div],
                                    previous_dilation=dilation, norm_layer=norm_layer))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation,
                                    previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        logger = get_root_logger()
        if self.pretrained is None:
            logger.info('Init mit from scratch.')
            for m in self.modules():
                self._init_weights(m)
        elif isinstance(self.pretrained, str):
            logger.info('Load mit checkpoint.')
            checkpoint = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            msg = self.load_state_dict(state_dict, False)
            logger.info(msg)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # for classification
        # x = self.avgpool(c4)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return [c1, c2, c3, c4]



# def resnet18(norm_layer=nn.BatchNorm2d):
#     num_block = [2, 2, 2, 2]
#     return ResNetV1(BasicBlockV1b, num_block, norm_layer=norm_layer)

@BACKBONES.register_module()
class resnet18(ResNetV1):

    def __init__(self, **kwargs):
        num_block = [2, 2, 2, 2]
        norm_layer = nn.BatchNorm2d
        super(resnet18, self).__init__(
            block=BasicBlockV1b,
            layers=num_block,
            output_stride=16,
            scale=1.0,
            norm_layer=norm_layer,
            **kwargs
        )

# def resnet34(norm_layer=nn.BatchNorm2d):
#     num_block = [3, 4, 6, 3]
#     return ResNetV1(BasicBlockV1b, num_block, norm_layer=norm_layer)


@BACKBONES.register_module()
class resnet34(ResNetV1):

    def __init__(self, **kwargs):
        num_block = [3, 4, 6, 3]
        norm_layer = nn.BatchNorm2d
        super(resnet34, self).__init__(
            block=BasicBlockV1b,
            layers=num_block,
            output_stride=16,
            scale=1.0,
            norm_layer=norm_layer,
            **kwargs
        )


# def resnet50(norm_layer=nn.BatchNorm2d):
#     num_block = [3, 4, 6, 3]
#     return ResNetV1(BottleneckV1b, num_block, norm_layer=norm_layer)


@BACKBONES.register_module()
class resnet50(ResNetV1):

    def __init__(self, **kwargs):
        num_block = [3, 4, 6, 3]
        norm_layer = nn.BatchNorm2d
        super(resnet50, self).__init__(
            block=BottleneckV1b,
            layers=num_block,
            output_stride=16,
            scale=1.0,
            norm_layer=norm_layer,
            **kwargs
        )



# def resnet101(norm_layer=nn.BatchNorm2d):
#     num_block = [3, 4, 23, 3]
#     return ResNetV1(BottleneckV1b, num_block, norm_layer=norm_layer)

@BACKBONES.register_module()
class resnet101(ResNetV1):

    def __init__(self, **kwargs):
        num_block = [3, 4, 23, 3]
        norm_layer = nn.BatchNorm2d
        super(resnet101, self).__init__(
            block=BottleneckV1b,
            layers=num_block,
            output_stride=16,
            scale=1.0,
            norm_layer=norm_layer,
            **kwargs
        )


# def resnet152(norm_layer=nn.BatchNorm2d):
#     num_block = [3, 8, 36, 3]
#     return ResNetV1(BottleneckV1b, num_block, norm_layer=norm_layer)


@BACKBONES.register_module()
class resnet152(ResNetV1):

    def __init__(self, **kwargs):
        num_block = [3, 8, 36, 3]
        norm_layer = nn.BatchNorm2d
        super(resnet152, self).__init__(
            block=BottleneckV1b,
            layers=num_block,
            output_stride=16,
            scale=1.0,
            norm_layer=norm_layer,
            **kwargs
        )


# @BACKBONES.register_module()
# def resnet50c(norm_layer=nn.BatchNorm2d):
#     num_block = [3, 4, 6, 3]
#     return ResNetV1(BottleneckV1b, num_block, norm_layer=norm_layer, deep_stem=True)


@BACKBONES.register_module()
class resnet50c(ResNetV1):

    def __init__(self, **kwargs):
        num_block = [3, 4, 6, 3]
        norm_layer = nn.BatchNorm2d
        super(resnet50c, self).__init__(
            block=BottleneckV1b,
            layers=num_block,
            output_stride=16,
            scale=1.0,
            norm_layer=norm_layer,
            deep_stem=True,
            **kwargs
        )


# def resnet101c(norm_layer=nn.BatchNorm2d):
#     num_block = [3, 4, 23, 3]
#     return ResNetV1(BottleneckV1b, num_block, norm_layer=norm_layer, deep_stem=True)


@BACKBONES.register_module()
class resnet101c(ResNetV1):

    def __init__(self, **kwargs):
        num_block = [3, 4, 23, 3]
        norm_layer = nn.BatchNorm2d
        super(resnet101c, self).__init__(
            block=BottleneckV1b,
            layers=num_block,
            output_stride=16,
            scale=1.0,
            norm_layer=norm_layer,
            deep_stem=True,
            **kwargs
        )




# def resnet152c(norm_layer=nn.BatchNorm2d):
#     num_block = [3, 8, 36, 3]
#     return ResNetV1(BottleneckV1b, num_block, norm_layer=norm_layer, deep_stem=True)

@BACKBONES.register_module()
class resnet152c(ResNetV1):

    def __init__(self, **kwargs):
        num_block = [3, 8, 36, 3]
        norm_layer = nn.BatchNorm2d
        super(resnet152c, self).__init__(
            block=BottleneckV1b,
            layers=num_block,
            output_stride=16,
            scale=1.0,
            norm_layer=norm_layer,
            deep_stem=True,
            **kwargs
        )