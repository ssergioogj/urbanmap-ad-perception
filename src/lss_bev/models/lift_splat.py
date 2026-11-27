"""Lift + Splat + BEV CNN: from 6 images to a bird's-eye-view segmentation."""
import torch.nn as nn

from lss_bev.models.bev_encode import BevEncode
from lss_bev.models.cam_encode import CamEncode
from lss_bev.models.geometry import create_frustum, frustum_to_ego
from lss_bev.models.splat import Splat


class LiftSplat(nn.Module):
    def __init__(self, image_size=(128, 352), downsample=16, dbound=(4.0, 45.0, 1.0),
                 xbound=(-50.0, 50.0, 0.5), ybound=(-50.0, 50.0, 0.5),
                 zbound=(-10.0, 10.0, 20.0), c_channels=64, out_channels=1, pretrained=True,
                 freeze_backbone=False):
        super().__init__()
        frustum = create_frustum(image_size, downsample, dbound)
        d_bins = frustum.shape[0]
        # buffer: saved with the model and moved to the GPU with .to(), but not trained
        self.register_buffer("frustum", frustum, persistent=False)
        self.cam_encode = CamEncode(d_bins=d_bins, c_chanels=c_channels, pretrained=pretrained)
        self.splat = Splat(xbound, ybound, zbound)
        self.bev_encode = BevEncode(in_channels=c_channels, out_channels=out_channels)

        # "Frozen Encoder" baseline (paper section 5.1): the pretrained image backbone
        # is not trained. Only the neck, the depth/context head and the BEV CNN learn.
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.cam_encode.image_backbone.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            # keep BatchNorm statistics of the frozen backbone fixed too
            self.cam_encode.image_backbone.eval()
        return self

    def forward(self, imgs, intrins, rots, trans):
        """
        Args:
            imgs:    (B, N, 3, H, W) images already resized and normalized
            intrins: (B, N, 3, 3) intrinsics ALREADY adjusted to the H x W size
            rots:    (B, N, 3, 3) camera -> ego rotation
            trans:   (B, N, 3)    camera -> ego translation
        Returns:
            logits: (B, out_channels, 200, 200), one logit per cell and class
        """
        features, _ = self.cam_encode(imgs)                                 # what is there
        geom = frustum_to_ego(self.frustum, intrins, rots, trans)           # where it is
        bev = self.splat(features, geom)                                    # onto the map
        return self.bev_encode(bev)                                         # what is in each cell
