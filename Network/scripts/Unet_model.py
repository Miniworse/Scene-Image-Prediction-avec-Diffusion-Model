import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import math

class DoubleConv(nn.Module):
    """(卷积 => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

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


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

    def forward(self, x, t):
        x =self.maxpool_conv(x)
        # Add timestep embedding
        b, c, h, w = x.shape
        t = self.time_mlp(t)
        t = t[:, :, None, None].repeat(1, 1, h, w)
        x = x + t
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, in_channels)
        )

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        # Add timestep embedding
        b, c, h, w = x.shape
        t = self.time_mlp(t)
        t = t[:, :, None, None].repeat(1, 1, h, w)
        x = x + t
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, time_emb_dim=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.time_emb_dim = time_emb_dim

        # Timestep embedding
        self.time_embed = nn.Sequential(
            TimestepEmbedder(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, time_emb_dim)
        self.down2 = Down(128, 256, time_emb_dim)
        self.down3 = Down(256, 512, time_emb_dim)
        self.down4 = Down(512, 1024, time_emb_dim)

        self.up1 = Up(1024, 512, time_emb_dim)
        self.up2 = Up(512, 256, time_emb_dim)
        self.up3 = Up(256, 128, time_emb_dim)
        self.up4 = Up(128, 64, time_emb_dim)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, timestep):
        # Embed timestep
        t = self.time_embed(timestep)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x5 = self.down4(x4, t)

        x = self.up1(x5, x4, t)
        x = self.up2(x, x3, t)
        x = self.up3(x, x2, t)
        x = self.up4(x, x1, t)
        logits = self.outc(x)
        return logits

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