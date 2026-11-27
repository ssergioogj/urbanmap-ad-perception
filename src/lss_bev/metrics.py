"""Evaluation metric: IoU on the BEV grid, accumulated over the whole set (as in the paper)."""
import torch


class IoUMeter:
    """Accumulates intersection and union over many batches, then divides once.

    This is how LSS reports IoU: sum all intersections and all unions of the
    validation set and divide at the end (not the mean of per-sample IoUs).
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.intersection = 0
        self.union = 0

    @torch.no_grad()
    def update(self, logits, label):
        """logits, label: (B, 1, X, Y). A cell is predicted as vehicle if sigmoid > threshold."""
        pred = logits.sigmoid() > self.threshold
        gt = label > 0.5
        self.intersection += (pred & gt).sum().item()
        self.union += (pred | gt).sum().item()

    def compute(self):
        return self.intersection / self.union if self.union > 0 else 0.0


class SegmentationMeter:
    """IoU, precision and recall, overall and by distance to the ego car.

    Everything is accumulated over the whole set and divided at the end.
    Distance bands are measured from the car center to the center of each cell.
    """

    def __init__(self, threshold=0.5, bands=((0, 10), (10, 20), (20, 30), (30, 50)),
                 xbound=(-50.0, 50.0, 0.5), ybound=(-50.0, 50.0, 0.5)):
        self.threshold = threshold
        self.bands = bands
        xs = torch.arange(xbound[0] + xbound[2] / 2, xbound[1], xbound[2])
        ys = torch.arange(ybound[0] + ybound[2] / 2, ybound[1], ybound[2])
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        self.dist = torch.sqrt(gx ** 2 + gy ** 2)                     # (nx, ny) meters
        self.tp = self.fp = self.fn = 0
        self.band_inter = [0] * len(bands)
        self.band_union = [0] * len(bands)

    @torch.no_grad()
    def update(self, logits, label):
        pred = (logits.sigmoid() > self.threshold).cpu()
        gt = (label > 0.5).cpu()
        self.tp += (pred & gt).sum().item()
        self.fp += (pred & ~gt).sum().item()
        self.fn += (~pred & gt).sum().item()
        for k, (lo, hi) in enumerate(self.bands):
            ring = (self.dist >= lo) & (self.dist < hi)
            self.band_inter[k] += (pred & gt & ring).sum().item()
            self.band_union[k] += ((pred | gt) & ring).sum().item()

    def compute(self):
        div = lambda a, b: a / b if b else 0.0  # noqa: E731
        out = dict(iou=div(self.tp, self.tp + self.fp + self.fn),
                   precision=div(self.tp, self.tp + self.fp),
                   recall=div(self.tp, self.tp + self.fn))
        for (lo, hi), i, u in zip(self.bands, self.band_inter, self.band_union):
            out[f"iou_{lo}-{hi}m"] = div(i, u)
        return out
