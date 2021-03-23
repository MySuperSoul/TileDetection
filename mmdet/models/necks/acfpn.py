import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

from ..builder import NECKS

# norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
# act_cfg = dict(type='ReLU', inplace=True),


class _DenseASPPConv(nn.Sequential):

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 atrous_rate,
                 drop_rate=0.1,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(_DenseASPPConv, self).__init__()
        self.add_module(
            'conv1',
            ConvModule(
                in_channels,
                inter_channels,
                1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
        ),
        self.add_module(
            'conv2',
            ConvModule(
                inter_channels,
                out_channels,
                3,
                stride=1,
                padding=atrous_rate,
                dilation=1 * atrous_rate,
                act_cfg=act_cfg),
        ),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(
                features, p=self.drop_rate, training=self.training)
        return features


class _DenseASPPBlock(nn.Module):

    def __init__(
            self,
            in_channels=2048,
            inter_channels1=512,
            inter_channels2=256,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    ):
        super(_DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1,
                                     inter_channels2, 3, 0.1)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1,
                                     inter_channels1, inter_channels2, 6, 0.1)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2,
                                      inter_channels1, inter_channels2, 12,
                                      0.1)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3,
                                      inter_channels1, inter_channels2, 18,
                                      0.1)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4,
                                      inter_channels1, inter_channels2, 24,
                                      0.1)

        d_feature1 = inter_channels2

        self.reduce = ConvModule(
            5 * d_feature1,
            256,
            1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            #conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            # norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
            # act_cfg=act_cfg,
            # inplace=False
        )

    def forward(self, x):
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)

        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)

        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)

        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)

        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)

        x = torch.cat([aspp3, aspp6, aspp12, aspp18, aspp24], dim=1)

        x = self.reduce(x)

        return x


@NECKS.register_module()
class ACFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(ACFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            if i < self.backbone_end_level - 1:
                self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self.dense_aspp = _DenseASPPBlock()
        # TODO:check weight init
        self.reduce_conv = ConvModule(2048, 256, 1, norm_cfg=norm_cfg)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        # xavier_init(self.reduce_conv, )

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            self.lateral_convs[i](inputs[i + self.start_level])
            for i in range(len(self.lateral_convs))
        ]

        aspp = self.dense_aspp(inputs[-1])
        reduced = self.reduce_conv(inputs[-1])
        laterals.append(reduced + aspp)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
