"""Lift + Splat: from 6 images to a bird's-eye-view feature map."""
import torch.nn as nn

from lss_bev.models.cam_encode import CamEncode
from lss_bev.models.geometry import create_frustum, frustum_to_ego
from lss_bev.models.splat import Splat


class LiftSplat(nn.Module):
    def __init__(self, image_size=(128, 352), downsample=16, dbound=(4.0, 45.0, 1.0),
                 xbound=(-50.0, 50.0, 0.5), ybound=(-50.0, 50.0, 0.5),
                 zbound=(-10.0, 10.0, 20.0), c_channels=64, pretrained=True):
        super().__init__()
        frustum = create_frustum(image_size, downsample, dbound)
        d_bins = frustum.shape[0]
        # buffer: saved with the model and moved to the GPU with .to(), but not trained
        self.register_buffer("frustum", frustum, persistent=False)
        self.cam_encode = CamEncode(d_bins=d_bins, c_chanels=c_channels, pretrained=pretrained)
        self.splat = Splat(xbound, ybound, zbound)

    def forward(self, imgs, intrins, rots, trans):
        """
        Args:
            imgs:    (B, N, 3, H, W) images already resized and normalized
            intrins: (B, N, 3, 3) intrinsics ALREADY adjusted to the H x W size
            rots:    (B, N, 3, 3) camera -> ego rotation
            trans:   (B, N, 3)    camera -> ego translation
        Returns:
            bev: (B, C, 200, 200)
        """
        features, _ = self.cam_encode(imgs)                                 # what is there
        geom = frustum_to_ego(self.frustum, intrins, rots, trans)           # where it is
        return self.splat(features, geom)                                   # onto the map
