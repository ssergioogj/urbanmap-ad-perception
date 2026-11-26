import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, BasicBlock

class BevEncode(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 1):
        super().__init__()
        trunk = resnet18(weights=None)

        # Input: 200 x 200, 64 channels
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        # 3 First Meta-layers
        self.layer1 = trunk.layer1  # x1: (B, 64, 100, 100)
        self.layer2 = trunk.layer2  # x2: (B, 120, 50, 50)
        self.layer3 = trunk.layer3  # x3: (B, 256, 25, 25)

        # Upsample of x3 by a factor of 4 and concatenate with x1
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels=256 + 64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            BasicBlock(128, 128)
        )

        # Unsample x2 to restore 200x200 grid and project output channels
        self.out_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.conv1(x)))  # From paper, make: 1. Convolution 2. Batchnorm 3. ReLu

        # Meta layers
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3= self.layer3(x2)

        # Upsample de x3
        x3_up = F.interpolate(input=x3, size=x1.shape[-2:], mode="bilinear", align_corners=True)
        x_cat = torch.cat([x3_up, x1], dim=1)

        feat = self.up_conv(x_cat)
        return self.out_head(feat)