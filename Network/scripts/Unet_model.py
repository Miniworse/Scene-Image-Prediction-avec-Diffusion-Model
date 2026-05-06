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
    model = UNet(n_channels=in_channels, n_classes=out_channels, time_emb_dim=512)

    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return None, None

    checkpoint = torch.load(model_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
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
        'ema_state_dict': ema.shadow if hasattr(ema, 'shadow') else ema.state_dict()
    }

    # 可选信息（用于恢复训练）
    if epoch is not None:
        save_dict['epoch'] = epoch
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if best_val_loss is not None:
        save_dict['best_val_loss'] = best_val_loss

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

