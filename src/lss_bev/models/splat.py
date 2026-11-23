"""
"Splat" step of LSS (paper sections 3.2 and 4.2).

    "We assign every point to its nearest pillar and perform sum pooling to
     create a C x H x W tensor that can be processed by a standard CNN."

Input: the point cloud produced by Lift
    - features: what is at each point   (B, N, D, C, Hf, Wf)   <- CamEncode
    - geom:     where each point is     (B, N, D, Hf, Wf, 3)   <- frustum_to_ego
Output: a bird's-eye-view map (B, C, X, Y), by default (B, 64, 200, 200).

Output grid convention:
    bev[b, :, i, j] is the cell with
        x (forward) in [x_min + i*dx, x_min + (i+1)*dx)
        y (left)    in [y_min + j*dy, y_min + (j+1)*dy)
    With the paper configuration: i = 0 is 50 m behind, i = 199 is 50 m ahead;
    j = 0 is 50 m to the right, j = 199 is 50 m to the left.
    The BEV labels MUST be generated with this SAME convention.

"Pillars": there is a single cell along z (from -10 to 10 m), so all points
above the same ground cell are summed together, whatever their height.
"""
import torch
import torch.nn as nn


def grid_size(xbound, ybound, zbound):
    """Number of cells along each axis. Each bound is (min, max, cell_size)."""
    return tuple(int(round((b[1] - b[0]) / b[2])) for b in (xbound, ybound, zbound))


def points_to_cells(geom, xbound, ybound, zbound):
    """Converts positions in meters into integer cell indices.

    We use floor instead of calling .long() directly: .long() rounds towards
    zero, so a point at x = -50.2 (outside the map) would end up in cell 0
    instead of being discarded.
    """
    lower = geom.new_tensor([xbound[0], ybound[0], zbound[0]])
    size = geom.new_tensor([xbound[2], ybound[2], zbound[2]])
    return torch.floor((geom - lower) / size).long()


class QuickCumsum(torch.autograd.Function):
    """Per-cell sum using the "cumsum trick" (section 4.2) with an analytic gradient.

    Points must come SORTED by cell (ranks).
    Example with 5 points and cells [7, 7, 7, 9, 9]:
        cumsum:            [a, a+b, a+b+c, a+b+c+d, a+b+c+d+e]
        last of each cell: [      a+b+c,            a+b+c+d+e]
        minus previous:    [a+b+c, d+e]   -> the sum of each cell
    """

    @staticmethod
    def forward(ctx, feats, ranks):
        x = feats.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]          # True at the last point of each cell
        x, ranks_out = x[kept], ranks[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # For backward: which group (cell) each point belongs to
        group = torch.cumsum(kept, 0)
        group[kept] -= 1
        ctx.save_for_backward(group)
        ctx.mark_non_differentiable(ranks_out)
        return x, ranks_out

    @staticmethod
    def backward(ctx, grad_x, grad_ranks):
        # The sum of a cell depends with weight 1 on each of its points,
        # so every point receives the gradient of its cell.
        (group,) = ctx.saved_tensors
        return grad_x[group], None


class Splat(nn.Module):
    """Sum pooling of the Lift point cloud onto the BEV grid."""

    def __init__(self, xbound=(-50.0, 50.0, 0.5), ybound=(-50.0, 50.0, 0.5),
                 zbound=(-10.0, 10.0, 20.0), use_cumsum=True):
        super().__init__()
        self.xbound, self.ybound, self.zbound = xbound, ybound, zbound
        self.nx, self.ny, self.nz = grid_size(xbound, ybound, zbound)
        self.use_cumsum = use_cumsum

    def forward(self, features, geom):
        """
        Args:
            features: (B, N, D, C, Hf, Wf)
            geom:     (B, N, D, Hf, Wf, 3)
        Returns:
            bev:      (B, C * nz, nx, ny)   -> (B, 64, 200, 200) with the default config
        """
        B, N, D, C, Hf, Wf = features.shape
        nx, ny, nz = self.nx, self.ny, self.nz

        # 1) Flatten: one row per point. Features get C last so that they are
        #    aligned point by point with geom.
        feats = features.permute(0, 1, 2, 4, 5, 3).reshape(-1, C)          # (P, C)
        cells = points_to_cells(geom, self.xbound, self.ybound, self.zbound).reshape(-1, 3)
        batch_ix = torch.arange(B, device=feats.device).repeat_interleave(N * D * Hf * Wf)

        # 2) Drop the points that fall outside the map
        ix, iy, iz = cells.unbind(1)
        keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        feats, ix, iy, iz, batch_ix = feats[keep], ix[keep], iy[keep], iz[keep], batch_ix[keep]

        # 3) A unique cell number for each point (as if the output tensor were flattened)
        ranks = ((batch_ix * nz + iz) * nx + ix) * ny + iy                    # (P_inside,)

        # 4) Sum pooling: add up the points that fall into the same cell
        out = feats.new_zeros(B * nz * nx * ny, C)
        if self.use_cumsum:
            order = ranks.argsort()
            sums, cells_used = QuickCumsum.apply(feats[order], ranks[order])
            out = out.index_copy(0, cells_used, sums)
        else:
            out = out.index_add(0, ranks, feats)   # direct version, easier to read

        # 5) Back to image shape: (B, nz, nx, ny, C) -> (B, nz*C, nx, ny)
        out = out.view(B, nz, nx, ny, C).permute(0, 1, 4, 2, 3)
        return out.reshape(B, nz * C, nx, ny)
