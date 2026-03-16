import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )

    def forward(self, x, t=None):
        # 第一层卷积
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        # 添加时间嵌入（在第一层卷积后添加效果更好）
        if t is not None and self.time_emb_dim is not None:
            time_emb = self.time_mlp(t)
            time_emb = time_emb[:, :, None, None]
            x = x + time_emb  # 简单相加

        # 第二层卷积
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        return x


class SelfAttention(nn.Module):
    """自注意力机制，用于底层特征"""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # 计算注意力
        attn = torch.einsum('bchw,bcHW->bhwHW', q, k) * (C ** -0.5)
        attn = attn.flatten(3).softmax(dim=-1).view(B, H, W, H, W)

        out = torch.einsum('bhwHW,bcHW->bchw', attn, v)
        out = self.proj(out)

        return residual + out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)
        self.use_attention = use_attention
        if use_attention:
            self.attn = SelfAttention(out_channels)

    def forward(self, x, t):
        x = self.pool(x)
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
    def __init__(self, n_channels=3, n_classes=3, time_emb_dim=128):
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

        # Upsampling - 注意每个Up的通道数设置
        self.up1 = Up(in_channels=512, skip_channels=512, out_channels=256,
                      time_emb_dim=time_emb_dim)  # 512(bottleneck) + 512(skip) -> 256
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

        # Decoder with skip connections
        x = self.up1(x5, x4, t)  # 输入: 512(bottleneck), 512(skip) -> 输出: 256
        x = self.up2(x, x3, t)  # 输入: 256, 256(skip) -> 输出: 128
        x = self.up3(x, x2, t)  # 输入: 128, 128(skip) -> 输出: 64
        x = self.up4(x, x1, t)  # 输入: 64, 64(skip) -> 输出: 64

        logits = self.outc(x)
        return logits


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

def load_model(model_path, in_channels=3, out_channels=3):
    # 创建模型并加载权重
    model = UNet(n_channels=in_channels, n_classes=out_channels, time_emb_dim=64)

    if os.path.exists(model_path):
        if model_path.endswith('.pth'):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded full model checkpoint from {model_path}")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded model weights from {model_path}")
        else:
            print(f"Model file {model_path} not found!")
            return None
    else:
        print(f"Model file {model_path} not found!")
        return None

    return model