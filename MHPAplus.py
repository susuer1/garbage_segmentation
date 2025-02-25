import torch
from torch import nn
import torch.nn.functional as F

class Multi_axis_Hadamard_Product_Attention_Plus(nn.Module):
    def __init__(self, input_dim, output_dim, p=8, q=8):
        super().__init__()

        channel_dim = input_dim // 4
        kernel_size = 3
        padding = (kernel_size - 1) // 2

        self.param_ij = nn.Parameter(torch.Tensor(1, channel_dim, p, q), requires_grad=True)
        nn.init.ones_(self.param_ij)
        self.conv_ij = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim, kernel_size=kernel_size, padding=padding, groups=channel_dim),
            nn.GELU(), nn.Conv2d(channel_dim, channel_dim, 1))

        self.param_mn = nn.Parameter(torch.Tensor(1, 1, channel_dim, p), requires_grad=True)
        nn.init.ones_(self.param_mn)
        self.conv_mn = nn.Sequential(
            nn.Conv1d(channel_dim, channel_dim, kernel_size=kernel_size, padding=padding, groups=channel_dim),
            nn.GELU(), nn.Conv1d(channel_dim, channel_dim, 1))

        self.param_op = nn.Parameter(torch.Tensor(1, 1, channel_dim, q), requires_grad=True)
        nn.init.ones_(self.param_op)
        self.conv_op = nn.Sequential(
            nn.Conv1d(channel_dim, channel_dim, kernel_size=kernel_size, padding=padding, groups=channel_dim),
            nn.GELU(), nn.Conv1d(channel_dim, channel_dim, 1))

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim, 1),
            nn.GELU(),
            nn.Conv2d(channel_dim, channel_dim, kernel_size=3, padding=1, groups=channel_dim)
        )

        self.layer_norm1 = LayerNorm(input_dim, eps=1e-6, data_format='channels_first')
        self.layer_norm2 = LayerNorm(input_dim, eps=1e-6, data_format='channels_first')

        self.final_dw = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, output_dim, 1),
        )

    def forward(self, input_tensor):
        # Layer normalization
        input_tensor = self.layer_norm1(input_tensor)

        # Split into four parts
        part1, part2, part3, part4 = torch.chunk(input_tensor, 4, dim=1)

        batch_size, channels, height, width = part1.size()

        # ----------ij----------#
        params_ij = self.param_ij
        residual_part1 = part1  # Save the original part1 for residual connection
        part1 = part1 * self.conv_ij(
            F.interpolate(params_ij, size=part1.shape[2:4], mode='bilinear', align_corners=True))
        part1 = part1 + residual_part1  # Residual connection

        # ----------mn----------#
        residual_part2 = part2  # Save the original part2 for residual connection
        part2 = part2.permute(0, 3, 1, 2)
        params_mn = self.param_mn
        part2 = part2 * self.conv_mn(
            F.interpolate(params_mn, size=part2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(
            0)
        part2 = part2.permute(0, 2, 3, 1)
        part2 = part2 + residual_part2  # Residual connection

        # ----------op----------#
        residual_part3 = part3  # Save the original part3 for residual connection
        part3 = part3.permute(0, 2, 1, 3)
        params_op = self.param_op
        part3 = part3 * self.conv_op(
            F.interpolate(params_op, size=part3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(
            0)
        part3 = part3.permute(0, 2, 1, 3)
        part3 = part3 + residual_part3  # Residual connection

        # ----------dw----------#
        residual_part4 = part4  # Save the original part4 for residual connection
        part4 = self.depthwise_conv(part4)
        part4 = part4 + residual_part4  # Residual connection

        # ----------concat----------#
        combined = torch.cat([part1, part2, part3, part4], dim=1)

        # ----------final dw----------#
        combined = self.layer_norm2(combined)
        combined = self.final_dw(combined)
        return combined


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_first'):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.data_format = data_format

        if self.data_format == 'channels_first':
            self.gamma = nn.Parameter(torch.ones(normalized_shape))  # 标准化后的尺度因子
            self.beta = nn.Parameter(torch.zeros(normalized_shape))  # 标准化后的偏移因子
        else:
            raise ValueError("Currently only 'channels_first' format is supported.")

    def forward(self, x):
        # 计算均值和方差
        if self.data_format == 'channels_first':
            mean = x.mean(dim=(2, 3), keepdim=True)  # 沿着 H 和 W 维度计算均值
            var = x.var(dim=(2, 3), keepdim=True, unbiased=False)  # 沿着 H 和 W 维度计算方差
        else:
            raise ValueError("Currently only 'channels_first' format is supported.")

        # 标准化操作
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 加上偏移和尺度
        out = self.gamma.view(1, -1, 1, 1) * x_norm + self.beta.view(1, -1, 1, 1)

        return out

