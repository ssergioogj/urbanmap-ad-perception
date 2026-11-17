"""
El resultado se alinea posición a posición con `frustum_features` de
CamEncode: el punto [b, n, d, h, w] tiene sus características en
frustum_features[b, n, d, :, h, w] y su posición en geom[b, n, d, h, w, :].
Here is developed the part to asociate |D| points {(h, w, d) | d in D} to each pixel
There are no learnable parameters in this transformation
"""
import torch


def create_frustum(image_size=(128, 352), downsample=16, dbound=(4.0, 45.0, 1.0)):
    """Rack (u, v, d) of all the points of the frustrum of one camera"""
    H, W = image_size
    Hf, Wf = H // downsample, W // downsample

    depths = torch.arange(*dbound, dtype=torch.float32)       # (D)  el array of 41 depths
    D = depths.shape[0]

    # Center of each cell in pixels
    # Cell j covers the pixels [16j, 16j+15] -> center 16j + 7.5
    us = (torch.arange(Wf, dtype=torch.float32) + 0.5) * downsample - 0.5   # (Wf)
    vs = (torch.arange(Hf, dtype=torch.float32) + 0.5) * downsample - 0.5   # (Hf)

    # Repeat with each axis to form the full rack(D, Hf, Wf)
    d = depths.view(D, 1, 1).expand(D, Hf, Wf)
    v = vs.view(1, Hf, 1).expand(D, Hf, Wf)
    u = us.view(1, 1, Wf).expand(D, Hf, Wf)

    return torch.stack([u, v, d], dim=-1)                       # (D, Hf, Wf, 3)


def adjust_intrinsics(intrins, scale, crop_top=0.0, crop_left=0.0):
    """Adapts the intrinsec matrix to the cropped and resized image 128 x 352
    fx, fy, cx, cy  ->  scale
    cx -= crop_left,  cy -= crop_top
    """
    K = intrins.clone()
    K[..., 0, :] *= scale    # rows of u: fx, 0, cx
    K[..., 1, :] *= scale    # rows of v: 0, fy, cy
    K[..., 0, 2] -= crop_left
    K[..., 1, 2] -= crop_top
    return K


def frustum_to_ego(frustum, intrins, rots, trans):
    """Pass all the points (u, v, d) of the frustrum to coordinates of the vehicle"""
    B, N = intrins.shape[:2]

    # Pixel + depth -> (u*d, v*d, d).
    # Pass from pixel to camera using intrinsic matrix K
    u, v, d = frustum.unbind(dim=-1)
    points = torch.stack([u * d, v * d, d], dim=-1)            # (D, Hf, Wf, 3)

    # Pass from camera to vehicle
    combine = rots @ torch.inverse(intrins)                    # (B, N, 3, 3)

    # Matrix-vector production for all the points at the same time
    points = points.to(combine).view(1, 1, *points.shape, 1)
    geom = combine.view(B, N, 1, 1, 1, 3, 3) @ points          # (B, N, D, Hf, Wf, 3, 1)
    geom = geom.squeeze(-1) + trans.view(B, N, 1, 1, 1, 3)     # (B, N, D, Hf, Wf, 3)
    return geom
