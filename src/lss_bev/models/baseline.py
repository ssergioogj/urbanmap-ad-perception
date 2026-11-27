"""
CNN baseline of the paper: the same pieces as LSS but without
3D structure (no depth, no Lift geometry, no Splat).
  1. Each camera goes through the same image backbone + neck as CamEncode and a
     1x1 conv gives C features per cell: (B, N, C, 8, 22). No depth is predicted.
  2. The 6 cameras are concatenated along the channels: (B, N*C, 8, 22).
  3. A 1x1 conv mixes them back to C channels and a bilinear interpolation
     stretches the 8x22 map to 200x200. Calibration (K, R, t) is NOT used: the
     network has to learn by itself where each image region goes on the map.
  4. The same BEV CNN as LSS predicts the output.
"""
import torch.nn as nn
import torch.nn.functional as F

from lss_bev.models.bev_encode import BevEncode
from lss_bev.models.cam_encode import CamEncode


class CNNBaseline(nn.Module):
    def __init__(self, n_cams=6, c_channels=64, out_channels=1, bev_size=(200, 200), pretrained=True):
        super().__init__()
        self.bev_size = bev_size
        encoder = CamEncode(c_chanels=c_channels, pretrained=pretrained)
        self.image_backbone = encoder.image_backbone          # same EfficientNet-B0 layers
        self.neck = encoder.neck                              # same Up module (128 channels)
        self.features = nn.Conv2d(128, c_channels, kernel_size=1)
        self.mix_cams = nn.Sequential(
            nn.Conv2d(n_cams * c_channels, c_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_channels),
            nn.ReLU(inplace=True),
        )
        self.bev_encode = BevEncode(in_channels=c_channels, out_channels=out_channels)

    def forward(self, imgs, intrins=None, rots=None, trans=None):
        """Same signature as LiftSplat; the calibration arguments are ignored on purpose."""
        B, N, C_in, H, W = imgs.shape
        c3, c4 = self.image_backbone(imgs.reshape(B * N, C_in, H, W))
        feats = self.features(self.neck(c4, c3))                         # (B*N, C, 8, 22)
        _, C, Hf, Wf = feats.shape
        feats = feats.view(B, N * C, Hf, Wf)                              # concatenate the cameras
        bev = F.interpolate(self.mix_cams(feats), size=self.bev_size,
                            mode="bilinear", align_corners=True)          # (B, C, 200, 200)
        return self.bev_encode(bev)
