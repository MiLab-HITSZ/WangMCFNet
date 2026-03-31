
import sys
import os
sys.path.append("/home/wxp/25week/MCFNet/models")


from SEFFN import *
from IWP import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from MambaBranch.CNN import *
from MambaBranch.MBlock import *



class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 再 permute 回 (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
class Encoder(nn.Module):
    def __init__(self, base_channel=3, num_blocks=[4, 4, 4]):
        super(Encoder, self).__init__()
        self.num_blocks_stage2 = num_blocks[1]
        self.num_blocks_stage3 = num_blocks[2]
        self.encoder_sampling = nn.ModuleList([
            BasicConv(base_channel, base_channel * 2, kernel_size=3, act=True, stride=2, bias=False, norm=True),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, act=True, stride=2, bias=False, norm=True)
        ])
        self.cnn_stem = nn.Sequential(
            BasicConv(3, base_channel, kernel_size=3, act=True, stride=1, bias=False, norm=True),
            BasicConv(base_channel, base_channel, kernel_size=3, act=True, stride=1, bias=False, norm=True),
        )
        self.ecnn_stage = nn.Sequential(
            *[
                ResidualDepthBlock(base_channel, base_channel) for i in range(num_blocks[0])
            ]
        )

        self.emix_stage1 = nn.ModuleList([
            nn.Sequential(
                VSSBlock(base_channel * 2),
                SEFFN(base_channel * 2, patch_size=4)

            ) for i in range(num_blocks[1] // 2)
        ])

        self.emix_stage2 = nn.ModuleList([
            nn.Sequential(
                VSSBlock(base_channel * 4),
                SEFFN(base_channel * 4, patch_size=4)

            ) for i in range(num_blocks[2] // 2)
        ])

        self.SCM1 = SCM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 4)
        self.FAM1 = FAM(base_channel * 2)
        self.FAM2 = FAM(base_channel * 4)
        self.iwpm1 = IntelligentWaveletPoolingModule().cuda()
        self.iwpm2 = IntelligentWaveletPoolingModule().cuda()

        self.norm1 = LayerNorm2d(3)
        self.norm2 = LayerNorm2d(3)

    def forward(self, x):
        # x_2 = self.iwpm1(x)
        # x_4 = self.iwpm2(x_2)
        x_2 = self.norm1(self.iwpm1(x))
        x_4 = self.norm2(self.iwpm2(x_2))
        x_2 = self.SCM1(x_2)
        x_4 = self.SCM2(x_4)

        x = self.cnn_stem(x)
        feature_list = []
        # print
        for i in range(len(self.ecnn_stage)):
            x = self.ecnn_stage[i](x)
        feature_list.append(x)
        x = self.encoder_sampling[0](x)
        x = self.FAM1(x, x_2)
        for i in range(len(self.emix_stage1)):
            x = self.emix_stage1[i](x)
        feature_list.append(x)
        x = self.encoder_sampling[1](x)
        x = self.FAM2(x, x_4)
        for i in range(len(self.emix_stage2)):
            x = self.emix_stage2[i](x)
        feature_list.append(x)

        return feature_list