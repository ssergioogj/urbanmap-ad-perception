import timm
import torch
import torch.nn as nn

from lss_bev.models.cam_encode import Up


class BevEncode(nn.Module):
    """Bird's-Eye-View CNN

    Procesa el pseudo-mapa BEV (B, C, 200, 200) que sale de Splat para
    interpolar huecos radiales, fusionar vistas y predecir las clases finales.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 1):
        super().__init__()
        # Extrae las 3 primeras etapas de ResNet-18 (strides 2, 4 y 8 -> tamaños 100, 50 y 25)
        self.trunk = timm.create_model(
            "resnet18",
            pretrained=False,
            in_chans=in_channels,
            features_only=True,
            out_indices=(0, 1, 2),
        )

        # En ResNet-18 estándar: x1 (stride 2) tiene 64 canales, x3 (stride 8) tiene 128 canales
        c1_dim, _, c3_dim = self.trunk.feature_info.channels()

        # Reutilizamos tu módulo Up (interpolación bilineal + 2 convs 3x3) para fusionar x3 con x1
        self.up1 = Up(in_channels=c3_dim, skip_channels=c1_dim, out_channels=256)

        # Upsampling final x2 para recuperar la resolución original (100x100 -> 200x200)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64, 200, 200)
        x1, _, x3 = self.trunk(x)   # x1: (B, 64, 100, 100), x3: (B, 128, 25, 25)
        x = self.up1(x3, x1)         # (B, 256, 100, 100)
        return self.up2(x)           # (B, out_channels, 200, 200)
