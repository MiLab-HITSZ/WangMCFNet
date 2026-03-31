import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 再 permute 回 (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class SEFFN(nn.Module):
    def __init__(self, dim, patch_size, ffn_expansion_factor=4, bias=True):
        super(SEFFN, self).__init__()
        # 计算隐藏层的特征维度，通常是输入维度的若干倍
        hidden_features = int(dim * ffn_expansion_factor)
        # 保存patch大小，用于后续分块处理
        self.patch_size = patch_size
        self.dim = dim
        # 第一个1x1卷积层，用于提升特征维度，输出维度是隐藏层维度的两倍
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 深度可分离卷积，对每个通道单独处理，进一步提取特征
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        # 可学习的FFT参数，用于频域操作
        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        # 第二个1x1卷积层，用于将特征维度降回输入维度
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.norm  = LayerNorm2d(dim)

        # self.conv_blk = Mlp(hidden_dim, hidden_dim, hidden_dim)
        # self.ln_2 = nn.LayerNorm(hidden_dim)
        # self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
    def forward(self, x0):
        # x = x0 * self.skip_scale2 + self.conv_blk(self.ln_2(x0))
        # 通过第一个卷积层提升特征维度【提升维度】
        x = self.project_in(self.norm(x0))
        # 经过深度可分离卷积后，将输出分成两部分
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # 对第一部分应用GELU激活函数，然后与第二部分相乘
        x = F.gelu(x1) * x2
        # 通过第二个卷积层降低特征维度【降低维度】
        x = self.project_out(x)

        # 将特征图按指定patch大小进行分块重组
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,patch2=self.patch_size)
        # 对分块后的特征图进行二维快速傅里叶变换，转换到频域
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        # 在频域中应用可学习的参数，对频域特征进行调整
        x_patch_fft = x_patch_fft * self.fft
        # 进行二维逆快速傅里叶变换，将特征从频域转回空间域
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))

        # 将分块的特征图重新组合成完整的特征图
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,patch2=self.patch_size)

        return x0+ x
