import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
try:
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d
except ImportError as e:
    ModulatedDeformConv2d = nn.Module
from mmcv.ops import ModulatedDeformConv2d  # 或 torchvision.ops.deform_conv2d


class FrequencySelection(nn.Module):
    def __init__(self, in_channels, k_list=[2], lp_type='avgpool', act='sigmoid', spatial_group=1):
        super().__init__()
        self.k_list = k_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.in_channels = in_channels
        self.spatial_group = spatial_group
        self.lp_type = lp_type

        if spatial_group > 64:
            spatial_group = in_channels
        self.spatial_group = spatial_group

        if self.lp_type == 'avgpool':
            for k in k_list:
                self.lp_list.append(nn.Sequential(
                    nn.ReplicationPad2d(padding=k // 2),
                    nn.AvgPool2d(kernel_size=k, padding=0, stride=1)
                ))

            for i in range(len(k_list)):
                freq_weight_conv = nn.Conv2d(in_channels=in_channels,
                                             out_channels=self.spatial_group,
                                             stride=1,
                                             kernel_size=3,
                                             groups=self.spatial_group,
                                             padding=3 // 2,
                                             bias=True)
                self.freq_weight_conv_list.append(freq_weight_conv)

        self.act = act

    def sp_act(self, freq_weight):
        if self.act == 'sigmoid':
            freq_weight = freq_weight.sigmoid() * 2
        elif self.act == 'softmax':
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        return freq_weight

    def forward(self, x):
        x_list = []

        # Ensure correct processing for the frequency selection
        if self.lp_type == 'avgpool':
            pre_x = x
            b, _, h, w = x.shape
            for idx, avg in enumerate(self.lp_list):
                low_part = avg(x)
                high_part = pre_x - low_part
                pre_x = low_part
                freq_weight = self.freq_weight_conv_list[idx](x)
                freq_weight = self.sp_act(freq_weight)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))

            x_list.append(pre_x)

        return x_list

class HPFD(ModulatedDeformConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, fs_cfg=None, **kwargs):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs)
        self.conv_type = kwargs.get('conv_type', 'conv')
        if fs_cfg is not None:
            self.FS = FrequencySelection(in_channels, **fs_cfg)

    def init_weights(self):
        super().init_weights()

    def freq_select(self, x):
        return x

    def forward(self, x):
        if hasattr(self, 'FS'):
            x_list = self.FS(x)
        else:
            x_list = [x]

        x = sum(x_list)
        return x
