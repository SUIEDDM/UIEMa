""" Full assembly of the parts to form the complete network """

from .unet_part import *
from .UIEMa import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import SLUT
class FC(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.fc(x)



class SLUT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        _, C, H, W = x.shape
        y = F.interpolate(x, size=[C, C], mode='bilinear', align_corners=True)
        # b c w h -> b c h w
        y = self.act1(self.conv1(y)).permute(0, 1, 3, 2)
        # b c h w -> b w h c
        y = self.act2(self.conv2(y)).permute(0, 3, 2, 1)
        # b w h c -> b c w h
        y = self.act3(self.conv3(y)).permute(0, 3, 1, 2)
        y = F.interpolate(y, size=[H, W], mode='bilinear', align_corners=True)
        return x + y

#
class CFFM(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = nn.ReLU()
        self.norm2 = nn.ReLU()

        self.gobal = Gobal(dim)
        self.conv = nn.Conv3d(2, 1, 3, 1, 1)
        self.fc = FC(dim, ffn_scale)

    def forward(self, x):
        y = self.norm1(x)
        y_g = self.gobal(y)
        y = self.conv(torch.cat([y.unsqueeze(1), y_g.unsqueeze(1)], dim=1)).squeeze(1) + x
        y = self.fc(self.norm2(y)) + y
        return y



class GLIMamba(nn.Module):
    def __init__(self):
        super(GLIMamba, self).__init__()
        # 初始化模型参数

    def forward(self, x):
        # 模型的前向传播
        # 返回一个张量或张量组成的元组
        return x  # 示例：直接返回输入

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        model = cls()
        model.load_state_dict(checkpoint['state_dict'])
        return model
    def __init__(self, dim=48, n_blocks=16, ffn_scale=2.0):
        super().__init__()
        self.to_feat = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1),
        )
        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x


class DAMixNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = GLIMamba(dim=48, n_blocks=8, ffn_scale=2.0)
        self.model2 = GLIMamba(dim=48, n_blocks=8, ffn_scale=2.0)
        self.model3 = GLIMamba(dim=48, n_blocks=8, ffn_scale=2.0)
        self.model4 = GLIMamba(dim=48, n_blocks=8, ffn_scale=2.0)

        self.fconv = nn.Conv2d(48 * 4, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, x):

        x1 = F.interpolate(x, size=[32, 32], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x, size=[64, 64], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x, size=[128, 128], mode='bilinear', align_corners=True)

        x1 = self.model1(x1)
        x2 = self.model1(x2)
        x3 = self.model1(x3)
        x4 = self.model1(x)

        x1 = F.interpolate(x1, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=True)

        output = self.fconv(torch.cat([x1, x2, x3, x4], dim=1))
        return output

class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 3)

    def forward(self, inp):
        x = inp
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x) + inp
        return x

data = torch.randn(2, 3, 128, 128).to("cuda")

model = UNet(n_channels=3).to("cuda")


# print(model(data).shape)


