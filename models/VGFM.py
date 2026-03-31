import torch
import torch.nn as nn
import torch.nn.functional as F

class VGFM(nn.Module):
    def __init__(self, dim=36):
        super(VGFM, self).__init__()
        # 定义1x1卷积层，将输入通道数从dim扩展到2*dim
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        # 定义1x1卷积层，保持通道数不变
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        # 定义1x1卷积层，保持通道数不变
        self.linear_2 = nn.Conv2d(dim, dim//2, 1, 1, 0)

        # 定义深度可分离卷积层（Depth-wise Convolution），保持通道数不变
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        # 定义GELU激活函数
        self.gelu = nn.GELU()
        # 设置下采样因子
        self.down_scale = 8

        # 定义可学习参数alpha，初始化为全1
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        # 定义可学习参数belt，初始化为全0
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, x1,x2):
        X = torch.cat([x1, x2], dim=1)
        _, _, h, w = X.shape  # 获取输入特征图的高度和宽度

        # 对X进行自适应最大池化操作，然后通过深度可分离卷积层
        x_s = self.dw_conv(F.adaptive_max_pool2d(X, (h // self.down_scale, w // self.down_scale)))

        # 计算X的方差 ，作为空间信息的统计差异
        x_v = torch.var(X, dim=(-2, -1), keepdim=True)

        # 计算局部细节估计
        Temp = x_s * self.alpha + x_v * self.belt

        x_l = X * F.interpolate(self.gelu(self.linear_1(Temp)), size=(h, w), mode='nearest')

        # 通过线性层2输出最终结果
        return self.linear_2(x_l)