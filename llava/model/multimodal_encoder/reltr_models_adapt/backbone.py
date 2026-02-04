# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import clip
import math

from ..util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
import ipdb


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor, feat: torch.Tensor):
        x_h = 24
        x_w = 24
        feat = feat.to(dtype=torch.float32)
        feat = feat.permute(0, 2, 1)  # B, C, N

        feat = feat.reshape(feat.shape[0], feat.shape[1], x_h, x_w)
        xs = {}
        xs['0'] = feat
        

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]   # torch.Size([2, 28, 34])
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,                    
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 ):   # name: str,

        num_channels = 768
        super().__init__(train_backbone, num_channels, return_interm_layers)  # backbone, 


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor, feat: torch.Tensor):
        xs = self[0](tensor_list, feat)
        # # ipdb.set_trace()
        
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))    # [torch.Size([2, 256, 28, 34])]
            # # ipdb.set_trace()

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.return_interm_layers
    backbone = Backbone(train_backbone, return_interm_layers, args.dilation)  # args.backbone
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
