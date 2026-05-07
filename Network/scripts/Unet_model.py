# <<<<<<< HEAD
# import math
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class DoubleConv(nn.Module):
#     """使用GroupNorm替换BN，添加时间嵌入"""

#     def __init__(self, in_channels, out_channels, time_emb_dim=None):
#         super().__init__()
#         self.time_emb_dim = time_emb_dim

#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
#         self.act = nn.SiLU(inplace=True)  # SiLU通常比ReLU好
#         if in_channels != out_channels:
#             self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         else:
#             self.shortcut = nn.Identity()

#         if time_emb_dim is not None:
#             self.time_mlp = nn.Sequential(
#                 nn.SiLU(),
#                 nn.Linear(time_emb_dim, out_channels)
#             )

#     def forward(self, x, t=None):
#         # 第一层卷积
#         identity = self.shortcut(x)
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.act(x)

#         # 添加时间嵌入（在第一层卷积后添加效果更好）
#         if t is not None and self.time_emb_dim is not None:
#             time_emb = self.time_mlp(t)
#             time_emb = time_emb[:, :, None, None]
#             x = x + time_emb  # 简单相加

#         # 第二层卷积
#         x = self.conv2(x)
#         x = self.norm2(x)
#         # if identity.shape == x.shape:
#         x = x + identity
#         x = self.act(x)
#         return x


# class SelfAttention(nn.Module):
#     """自注意力机制，用于底层特征"""

#     def __init__(self, channels):
#         super().__init__()
#         self.norm = nn.GroupNorm(min(8, channels), channels)
#         self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
#         self.proj = nn.Conv2d(channels, channels, kernel_size=1)

#     # def forward(self, x):
#     #     B, C, H, W = x.shape
#     #     residual = x
#     #
#     #     x = self.norm(x)
#     #     qkv = self.qkv(x)
#     #     q, k, v = qkv.chunk(3, dim=1)
#     #
#     #     # 计算注意力
#     #     attn = torch.einsum('bchw,bcHW->bhwHW', q, k) * (C ** -0.5)
#     #     attn = attn.flatten(3).softmax(dim=-1).view(B, H, W, H, W)
#     #
#     #     out = torch.einsum('bhwHW,bcHW->bchw', attn, v)
#     #     out = self.proj(out)
#     #
#     #     return residual + out

#     def forward(self, x):
#         B, C, H, W = x.shape
#         residual = x
#         x = self.norm(x)
#         qkv = self.qkv(x).view(B, 3, C, H * W)
#         q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

#         # Standard Dot-Product Attention
#         attn = torch.matmul(q.transpose(-1, -2), k) * (C ** -0.5)
#         attn = attn.softmax(dim=-1)

#         out = torch.matmul(v, attn.transpose(-1, -2))
#         out = out.view(B, C, H, W)
#         return residual + self.proj(out)


# class Down(nn.Module):
#     def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
#         super().__init__()
#         # self.pool = nn.MaxPool2d(2)
#         self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
#         self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)
#         self.use_attention = use_attention
#         if use_attention:
#             self.attn = SelfAttention(out_channels)

#     def forward(self, x, t):
#         # x = self.pool(x)
#         x = self.downsample(x)
#         x = self.conv(x, t)
#         if self.use_attention:
#             x = self.attn(x)
#         return x


# class Up(nn.Module):
#     def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim, use_attention=False):
#         """
#         Args:
#             in_channels: 来自下层的输入通道数
#             skip_channels: 跳跃连接的通道数
#             out_channels: 输出通道数
#         """
#         super().__init__()
#         # 上采样层：将in_channels减半（ConvTranspose2d的out_channels = in_channels // 2）
#         self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

#         # 拼接后的通道数 = 上采样后的通道数 + 跳跃连接通道数
#         # 上采样后的通道数 = in_channels // 2
#         # 跳跃连接通道数 = skip_channels
#         concat_channels = in_channels // 2 + skip_channels

#         # DoubleConv接收拼接后的通道数，输出out_channels
#         self.conv = DoubleConv(concat_channels, out_channels, time_emb_dim)

#         self.use_attention = use_attention
#         if use_attention:
#             self.attn = SelfAttention(out_channels)

#     def forward(self, x1, x2, t):
#         """
#         x1: 来自下层的特征 (batch, in_channels, h, w)
#         x2: 来自跳跃连接的特征 (batch, skip_channels, H, W) 其中H=2h
#         """
#         x1 = self.up(x1)  # (batch, in_channels//2, 2h, 2w)

#         # 处理尺寸不匹配
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])

#         # 跳跃连接拼接
#         x = torch.cat([x2, x1], dim=1)  # (batch, in_channels//2 + skip_channels, H, W)

#         # DoubleConv包含时间嵌入
#         x = self.conv(x, t)

#         if self.use_attention:
#             x = self.attn(x)

#         return x


# class UNet(nn.Module):
#     def __init__(self, n_channels=3, n_classes=3, time_emb_dim=256):
#         super().__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.time_emb_dim = time_emb_dim

#         # 时间嵌入
#         self.time_embed = nn.Sequential(
#             TimestepEmbedder(time_emb_dim),
#             nn.Linear(time_emb_dim, time_emb_dim * 4),
#             nn.SiLU(),
#             nn.Linear(time_emb_dim * 4, time_emb_dim)
#         )

#         # 初始卷积
#         self.inc = DoubleConv(n_channels, 64, time_emb_dim)

#         # Downsampling
#         self.down1 = Down(64, 128, time_emb_dim)  # 64 -> 128
#         self.down2 = Down(128, 256, time_emb_dim)  # 128 -> 256
#         self.down3 = Down(256, 512, time_emb_dim)  # 256 -> 512
#         self.down4 = Down(512, 512, time_emb_dim, use_attention=True)  # 512 -> 512

#         # Bottleneck
#         self.bottleneck = DoubleConv(512, 512, time_emb_dim)
#         self.bottleneck_attn = SelfAttention(512)

#         # Upsampling - 注意每个Up的通道数设置
#         self.up1 = Up(in_channels=512, skip_channels=512, out_channels=256,
#                       time_emb_dim=time_emb_dim, use_attention=True)  # 512(bottleneck) + 512(skip) -> 256
#         self.up2 = Up(in_channels=256, skip_channels=256, out_channels=128,
#                       time_emb_dim=time_emb_dim)  # 256 + 256 -> 128
#         self.up3 = Up(in_channels=128, skip_channels=128, out_channels=64,
#                       time_emb_dim=time_emb_dim)  # 128 + 128 -> 64
#         self.up4 = Up(in_channels=64, skip_channels=64, out_channels=64,
#                       time_emb_dim=time_emb_dim)  # 64 + 64 -> 64

#         # 输出层
#         self.outc = nn.Sequential(
#             # nn.GroupNorm(min(8, 64), 64),
#             # nn.SiLU(),
#             nn.Conv2d(64, n_classes, kernel_size=1)
#         )

#     def forward(self, x, timestep):
#         # 时间嵌入
#         t = self.time_embed(timestep)

#         # Encoder with skip connections
#         x1 = self.inc(x, t)  # 64通道
#         x2 = self.down1(x1, t)  # 128通道
#         x3 = self.down2(x2, t)  # 256通道
#         x4 = self.down3(x3, t)  # 512通道
#         x5 = self.down4(x4, t)  # 512通道

#         # Bottleneck
#         x5 = self.bottleneck(x5, t)  # 512通道
#         x5 = self.bottleneck_attn(x5)

#         # Decoder with skip connections
#         x = self.up1(x5, x4, t)  # 输入: 512(bottleneck), 512(skip) -> 输出: 256
#         x = self.up2(x, x3, t)  # 输入: 256, 256(skip) -> 输出: 128
#         x = self.up3(x, x2, t)  # 输入: 128, 128(skip) -> 输出: 64
#         x = self.up4(x, x1, t)  # 输入: 64, 64(skip) -> 输出: 64

#         logits = self.outc(x)
#         return logits


# class TimestepEmbedder(nn.Module):
#     def __init__(self, embedding_dim):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.half_dim = embedding_dim // 2
#         emb = math.log(10000) / (self.half_dim - 1)
#         emb = torch.exp(torch.arange(self.half_dim, dtype=torch.float) * -emb)
#         self.register_buffer('emb', emb)

#     def forward(self, timestep):
#         emb = timestep.float()[:, None] * self.emb[None, :]
#         emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
#         return emb



import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .get_EMA import EMA

class DoubleConv(nn.Module):
    """使用GroupNorm替换BN，添加时间嵌入"""

    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)  # SiLU通常比ReLU好
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2)
            )

    def forward(self, x, t=None):
        # 第一层卷积
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        # 添加时间嵌入（在第一层卷积后添加效果更好）
        if t is not None and self.time_emb_dim is not None:
            time_emb = self.time_mlp(t)
            scale, shift = time_emb.chunk(2, dim=1)

            scale = scale[:, :, None, None]
            shift = shift[:, :, None, None]

            x = x * (1 + scale) + shift

        # 第二层卷积
        x = self.conv2(x)
        x = self.norm2(x)
        # if identity.shape == x.shape:
        x = x + identity
        x = self.act(x)
        return x


class SelfAttention(nn.Module):
    """自注意力机制，用于底层特征"""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    # def forward(self, x):
    #     B, C, H, W = x.shape
    #     residual = x
    #
    #     x = self.norm(x)
    #     qkv = self.qkv(x)
    #     q, k, v = qkv.chunk(3, dim=1)
    #
    #     # 计算注意力
    #     attn = torch.einsum('bchw,bcHW->bhwHW', q, k) * (C ** -0.5)
    #     attn = attn.flatten(3).softmax(dim=-1).view(B, H, W, H, W)
    #
    #     out = torch.einsum('bhwHW,bcHW->bchw', attn, v)
    #     out = self.proj(out)
    #
    #     return residual + out

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).view(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Standard Dot-Product Attention
        attn = torch.matmul(q.transpose(-1, -2), k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)

        out = torch.matmul(v, attn.transpose(-1, -2))
        out = out.view(B, C, H, W)
        return residual + self.proj(out)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
        super().__init__()
        # self.pool = nn.MaxPool2d(2)
        self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)
        self.use_attention = use_attention
        if use_attention:
            self.attn = SelfAttention(out_channels)

    def forward(self, x, t):
        # x = self.pool(x)
        x = self.downsample(x)
        
        x = self.conv(x, t)
        if self.use_attention:
            x = self.attn(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim, use_attention=False):
        """
        Args:
            in_channels: 来自下层的输入通道数
            skip_channels: 跳跃连接的通道数
            out_channels: 输出通道数
        """
        super().__init__()
        # 上采样层：将in_channels减半（ConvTranspose2d的out_channels = in_channels // 2）
        # self.up = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        # )
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # 拼接后的通道数 = 上采样后的通道数 + 跳跃连接通道数
        # 上采样后的通道数 = in_channels // 2
        # 跳跃连接通道数 = skip_channels
        concat_channels = in_channels // 2 + skip_channels

        # DoubleConv接收拼接后的通道数，输出out_channels
        self.conv = DoubleConv(concat_channels, out_channels, time_emb_dim)

        self.use_attention = use_attention
        if use_attention:
            self.attn = SelfAttention(out_channels)

    def forward(self, x1, x2, t):
        """
        x1: 来自下层的特征 (batch, in_channels, h, w)
        x2: 来自跳跃连接的特征 (batch, skip_channels, H, W) 其中H=2h
        """
        x1 = self.up(x1)  # (batch, in_channels//2, 2h, 2w)

        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 跳跃连接拼接
        x = torch.cat([x2, x1], dim=1)  # (batch, in_channels//2 + skip_channels, H, W)

        # DoubleConv包含时间嵌入
        x = self.conv(x, t)

        if self.use_attention:
            x = self.attn(x)

        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, time_emb_dim=256):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.time_emb_dim = time_emb_dim

        # 时间嵌入
        self.time_embed = nn.Sequential(
            TimestepEmbedder(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # 初始卷积
        self.inc = DoubleConv(n_channels, 64, time_emb_dim)

        # Downsampling
        self.down1 = Down(64, 128, time_emb_dim)  # 64 -> 128
        self.down2 = Down(128, 256, time_emb_dim)  # 128 -> 256
        self.down3 = Down(256, 512, time_emb_dim)  # 256 -> 512
        self.down4 = Down(512, 512, time_emb_dim, use_attention=True)  # 512 -> 512

        # Bottleneck
        self.bottleneck = DoubleConv(512, 512, time_emb_dim)
        self.bottleneck_attn = SelfAttention(512)

        # Upsampling - 注意每个Up的通道数设置
        self.up1 = Up(in_channels=512, skip_channels=512, out_channels=256,
                      time_emb_dim=time_emb_dim, use_attention=True)  # 512(bottleneck) + 512(skip) -> 256
        self.up2 = Up(in_channels=256, skip_channels=256, out_channels=128,
                      time_emb_dim=time_emb_dim)  # 256 + 256 -> 128
        self.up3 = Up(in_channels=128, skip_channels=128, out_channels=64,
                      time_emb_dim=time_emb_dim)  # 128 + 128 -> 64
        self.up4 = Up(in_channels=64, skip_channels=64, out_channels=64,
                      time_emb_dim=time_emb_dim)  # 64 + 64 -> 64

        # 输出层
        self.outc = nn.Sequential(
            nn.GroupNorm(min(8, 64), 64),
            nn.SiLU(),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )

    def forward(self, x, timestep):
        # 时间嵌入
        t = self.time_embed(timestep)

        # Encoder with skip connections
        x1 = self.inc(x, t)  # 64通道
        x2 = self.down1(x1, t)  # 128通道
        x3 = self.down2(x2, t)  # 256通道
        x4 = self.down3(x3, t)  # 512通道
        x5 = self.down4(x4, t)  # 512通道

        # Bottleneck
        x5 = self.bottleneck(x5, t)  # 512通道
        x5 = self.bottleneck_attn(x5)

        # Decoder with skip connections
        x = self.up1(x5, x4, t)  # 输入: 512(bottleneck), 512(skip) -> 输出: 256
        x = self.up2(x, x3, t)  # 输入: 256, 256(skip) -> 输出: 128
        x = self.up3(x, x2, t)  # 输入: 128, 128(skip) -> 输出: 64
        x = self.up4(x, x1, t)  # 输入: 64, 64(skip) -> 输出: 64

        logits = self.outc(x)
        return logits


class ResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.act(x + identity)
        return x


class TransformerBlock2d(nn.Module):
    def __init__(self, channels, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden_dim = channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attn_input = self.norm1(tokens)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)


class ConditionFusion(nn.Module):
    def __init__(self, x_channels, cond_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(x_channels + cond_channels, x_channels, kernel_size=1),
            nn.GroupNorm(min(8, x_channels), x_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, cond):
        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x + self.proj(torch.cat([x, cond], dim=1))


class IFFTConditionEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            ResBlock2d(32, 64),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            ResBlock2d(128, 128),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            ResBlock2d(256, 256),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            ResBlock2d(512, 512),
        )

    def forward(self, x):
        p1 = self.stem(x)
        p2 = self.down1(p1)
        p3 = self.down2(p2)
        p4 = self.down3(p3)
        return p1, p2, p3, p4


class VisibilityConditionEncoder(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            ResBlock2d(32, 64),
        )
        self.stage1 = nn.Sequential(
            ResBlock2d(64, 64),
        )
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.stage2 = nn.Sequential(
            ResBlock2d(128, 128),
            TransformerBlock2d(128, num_heads=4),
        )
        self.down2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.stage3 = nn.Sequential(
            ResBlock2d(256, 256),
            TransformerBlock2d(256, num_heads=8),
            TransformerBlock2d(256, num_heads=8),
        )
        self.down3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.stage4 = nn.Sequential(
            ResBlock2d(512, 512),
            TransformerBlock2d(512, num_heads=8),
        )

    def forward(self, x):
        v1 = self.stage1(self.stem(x))
        v2 = self.stage2(self.down1(v1))
        v3 = self.stage3(self.down2(v2))
        v4 = self.stage4(self.down3(v3))
        return v1, v2, v3, v4


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=1,
        time_emb_dim=256,
        ifft_channels=1,
        visibility_channels=2,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.time_emb_dim = time_emb_dim
        self.ifft_channels = ifft_channels
        self.visibility_channels = visibility_channels

        self.time_embed = nn.Sequential(
            TimestepEmbedder(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.ifft_encoder = IFFTConditionEncoder(in_channels=ifft_channels)
        self.visibility_encoder = VisibilityConditionEncoder(in_channels=visibility_channels + 3)

        self.cond_proj1 = nn.Conv2d(64 + 64, 64, kernel_size=1)
        self.cond_proj2 = nn.Conv2d(128 + 128, 128, kernel_size=1)
        self.cond_proj3 = nn.Conv2d(256 + 256, 256, kernel_size=1)
        self.cond_proj4 = nn.Conv2d(512 + 512, 512, kernel_size=1)

        self.inc = DoubleConv(n_channels, 64, time_emb_dim)
        self.down1 = Down(64, 128, time_emb_dim)
        self.down2 = Down(128, 256, time_emb_dim)
        self.down3 = Down(256, 512, time_emb_dim)
        self.down4 = Down(512, 512, time_emb_dim, use_attention=True)

        self.fuse_in = ConditionFusion(64, 64)
        self.fuse_d1 = ConditionFusion(128, 128)
        self.fuse_d2 = ConditionFusion(256, 256)
        self.fuse_d3 = ConditionFusion(512, 512)
        self.fuse_bot = ConditionFusion(512, 512)

        self.bottleneck = DoubleConv(512, 512, time_emb_dim)
        self.bottleneck_attn = SelfAttention(512)

        self.up1 = Up(in_channels=512, skip_channels=512, out_channels=256,
                      time_emb_dim=time_emb_dim, use_attention=True)
        self.up2 = Up(in_channels=256, skip_channels=256, out_channels=128,
                      time_emb_dim=time_emb_dim)
        self.up3 = Up(in_channels=128, skip_channels=128, out_channels=64,
                      time_emb_dim=time_emb_dim)
        self.up4 = Up(in_channels=64, skip_channels=64, out_channels=64,
                      time_emb_dim=time_emb_dim)

        self.outc = nn.Sequential(
            nn.GroupNorm(min(8, 64), 64),
            nn.SiLU(),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )

    def _build_geometry_maps(self, antenna_xy, dtype, device):
        if antenna_xy.dim() != 3:
            raise ValueError("antenna_xy must have shape [B, 2, N] or [B, N, 2]")
        if antenna_xy.shape[1] == 2:
            x = antenna_xy[:, 0, :]
            y = antenna_xy[:, 1, :]
        elif antenna_xy.shape[2] == 2:
            x = antenna_xy[:, :, 0]
            y = antenna_xy[:, :, 1]
        else:
            raise ValueError("antenna_xy must have a coordinate dimension of size 2")

        du = x.unsqueeze(2) - x.unsqueeze(1)
        dv = y.unsqueeze(2) - y.unsqueeze(1)
        r = torch.sqrt(du ** 2 + dv ** 2 + 1e-8)
        max_norm = torch.amax(r.flatten(1), dim=1).clamp_min(1e-6).view(-1, 1, 1)

        du = (du / max_norm).unsqueeze(1)
        dv = (dv / max_norm).unsqueeze(1)
        r = (r / max_norm).unsqueeze(1)
        return du.to(device=device, dtype=dtype), dv.to(device=device, dtype=dtype), r.to(device=device, dtype=dtype)

    def _encode_conditions(self, ifft_cond, visibility, antenna_xy, target_dtype, target_device):
        if ifft_cond is None or visibility is None or antenna_xy is None:
            raise ValueError("ifft_cond, visibility, and antenna_xy are required for ConditionalUNet")

        if ifft_cond.dim() == 3:
            ifft_cond = ifft_cond.unsqueeze(1)
        if visibility.dim() == 4 and visibility.shape[-1] == self.visibility_channels:
            visibility = visibility.permute(0, 3, 1, 2).contiguous()

        du, dv, r = self._build_geometry_maps(antenna_xy, target_dtype, target_device)
        vis_input = torch.cat([visibility.to(dtype=target_dtype, device=target_device), du, dv, r], dim=1)
        ifft_input = ifft_cond.to(dtype=target_dtype, device=target_device)

        p1, p2, p3, p4 = self.ifft_encoder(ifft_input)
        v1, v2, v3, v4 = self.visibility_encoder(vis_input)

        c1 = self.cond_proj1(torch.cat([p1, F.interpolate(v1, size=p1.shape[-2:], mode='bilinear', align_corners=False)], dim=1))
        c2 = self.cond_proj2(torch.cat([p2, F.interpolate(v2, size=p2.shape[-2:], mode='bilinear', align_corners=False)], dim=1))
        c3 = self.cond_proj3(torch.cat([p3, F.interpolate(v3, size=p3.shape[-2:], mode='bilinear', align_corners=False)], dim=1))
        c4 = self.cond_proj4(torch.cat([p4, F.interpolate(v4, size=p4.shape[-2:], mode='bilinear', align_corners=False)], dim=1))
        return c1, c2, c3, c4

    def forward(self, x, timestep, ifft_cond=None, visibility=None, antenna_xy=None):
        t = self.time_embed(timestep)
        c1, c2, c3, c4 = self._encode_conditions(ifft_cond, visibility, antenna_xy, x.dtype, x.device)

        x1 = self.fuse_in(self.inc(x, t), c1)
        x2 = self.fuse_d1(self.down1(x1, t), c2)
        x3 = self.fuse_d2(self.down2(x2, t), c3)
        x4 = self.fuse_d3(self.down3(x3, t), c4)
        x5 = self.down4(x4, t)

        x5 = self.bottleneck(x5, t)
        x5 = self.fuse_bot(x5, c4)
        x5 = self.bottleneck_attn(x5)

        x = self.up1(x5, x4, t)
        x = self.up2(x, x3, t)
        x = self.up3(x, x2, t)
        x = self.up4(x, x1, t)

        return self.outc(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 2
        emb = math.log(10000) / (self.half_dim - 1)
        emb = torch.exp(torch.arange(self.half_dim, dtype=torch.float) * -emb)
        self.register_buffer('emb', emb)

    def forward(self, timestep):
        emb = timestep.float()[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

def load_model(model_path, in_channels=3, out_channels=3, architecture='UNet', **kwargs):
    if architecture == 'ConditionalUNet':
        model = ConditionalUNet(
            n_channels=in_channels,
            n_classes=out_channels,
            time_emb_dim=kwargs.pop('time_emb_dim', 512),
            ifft_channels=kwargs.pop('ifft_channels', 1),
            visibility_channels=kwargs.pop('visibility_channels', 2),
        )
    else:
        model = UNet(n_channels=in_channels, n_classes=out_channels, time_emb_dim=kwargs.pop('time_emb_dim', 512))

    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return None, None

    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint.get('model_config', {})
    if model_config:
        architecture = model_config.get('architecture', architecture)
        if architecture == 'ConditionalUNet' and not isinstance(model, ConditionalUNet):
            model = ConditionalUNet(
                n_channels=model_config.get('n_channels', in_channels),
                n_classes=model_config.get('n_classes', out_channels),
                time_emb_dim=model_config.get('time_emb_dim', 512),
                ifft_channels=model_config.get('ifft_channels', 1),
                visibility_channels=model_config.get('visibility_channels', 2),
            )
        elif architecture == 'UNet' and not isinstance(model, UNet):
            model = UNet(
                n_channels=model_config.get('n_channels', in_channels),
                n_classes=model_config.get('n_classes', out_channels),
                time_emb_dim=model_config.get('time_emb_dim', 512),
            )
    elif architecture != model.__class__.__name__:
        raise ValueError(
            f"Checkpoint {model_path} has no model_config, but requested architecture "
            f"is {architecture} while the instantiated model is {model.__class__.__name__}. "
            "This usually means you are trying to load an old checkpoint into a new architecture."
        )

    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load checkpoint {model_path} into {model.__class__.__name__}. "
                "The checkpoint architecture is incompatible with the current model definition."
            ) from exc
        print(f"Loaded model weights from {model_path}")

        ema = EMA(model, decay=0.999)

        if 'ema_state_dict' in checkpoint:
            ema.shadow = checkpoint['ema_state_dict']
            print("Loaded EMA weights")
        else:
            print("No EMA weights found")

        return model, ema

    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded model weights (no EMA)")
        return model, None
    
    
# 保存模型（改进版）
def save_model(model, ema, model_path, epoch=None, optimizer=None, best_val_loss=None):
    """
    保存模型检查点

    Args:
        model: 模型
        ema: EMA对象
        model_path: 保存路径（完整文件路径）
        epoch: 当前epoch（可选）
        optimizer: 优化器（可选）
        best_val_loss: 最佳验证损失（可选）
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.shadow if hasattr(ema, 'shadow') else ema.state_dict(),
        'model_config': {
            'architecture': model.__class__.__name__,
            'n_channels': getattr(model, 'n_channels', None),
            'n_classes': getattr(model, 'n_classes', None),
            'time_emb_dim': getattr(model, 'time_emb_dim', None),
            'ifft_channels': getattr(model, 'ifft_channels', None),
            'visibility_channels': getattr(model, 'visibility_channels', None),
        }
    }

    # 可选信息（用于恢复训练）
    if epoch is not None:
        save_dict['epoch'] = epoch
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if best_val_loss is not None:
        save_dict['best_val_loss'] = best_val_loss

    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(save_dict, model_path)
    print(f'Model saved to {model_path}')


def load_pretrained_model(model, ema, model_path, device='cpu', load_optimizer=False, optimizer=None):
    """
    加载模型检查点

    Returns:
        model: 加载后的模型
        ema: 加载后的EMA
        epoch: 保存时的epoch（如果有）
        best_val_loss: 保存时的最佳验证损失（如果有）
    """
    checkpoint = torch.load(model_path, map_location=device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载EMA权重
    if 'ema_state_dict' in checkpoint:
        if hasattr(ema, 'shadow'):
            ema.shadow = checkpoint['ema_state_dict']
        else:
            ema.load_state_dict(checkpoint['ema_state_dict'])

    # 获取训练信息
    epoch = checkpoint.get('epoch', None)
    best_val_loss = checkpoint.get('best_val_loss', None)

    # 加载优化器（可选）
    if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")

    print(f'Model loaded from {model_path}')
    if epoch is not None:
        print(f'Saved at epoch: {epoch}')
    if best_val_loss is not None:
        print(f'Best val loss: {best_val_loss:.6f}')

    return model, ema, epoch, best_val_loss

