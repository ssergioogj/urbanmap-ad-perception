"""
Draw what the trained models predict next to the ground truth, for a few
validation samples.

    uv run python scripts/visualize_predictions.py runs/mini-lss/best.pt runs/mini-cnn/best.pt
    uv run python scripts/visualize_predictions.py runs/mini-lss/best.pt --samples 0 20 40 60

Each row is one sample: the 6 cameras, the ground truth, and the probability
map of each model (white outline = ground truth). Maps are drawn like the other
figures: forward up, left of the car on the left. -> outputs/predictions.png
"""
import argparse
from pathlib import Path

import matplotlib
import torch

from lss_bev.dataset import IMAGENET_MEAN, IMAGENET_STD, NuScenesBEV
from lss_bev.models.build import load_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"cnn": "CNN (no 3D)", "frozen": "Frozen Encoder", "lss": "Lift-Splat"}
EXTENT = [-50, 50, -50, 50]


def camera_mosaic(imgs):
    """(6, 3, H, W) normalized -> (2H, 3W, 3) image: front cameras on top, back below."""
    raw = (imgs * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(0, 2, 3, 1)
    top = torch.cat(list(raw[:3]), dim=1)
    bottom = torch.cat(list(raw[3:]), dim=1)
    return torch.cat([top, bottom], dim=0)


def show_map(ax, data, title, label=None, **kw):
    ax.imshow(data, origin="lower", extent=EXTENT, interpolation="nearest", **kw)
    if label is not None:
        ax.contour(label, levels=[0.5], colors="white", linewidths=0.6, origin="lower", extent=EXTENT)
    ax.add_patch(plt.Rectangle((-0.95, -2.3), 1.9, 4.6, color="red"))
    ax.invert_xaxis()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--samples", type=int, nargs="+", default=[0, 25, 50, 75])
    p.add_argument("--dataroot", default=str(ROOT / "data"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--out", default=str(ROOT / "outputs" / "predictions.png"))
    args = p.parse_args()

    val = NuScenesBEV(args.dataroot, args.version, "val", augment=False)
    models = [load_checkpoint(c, "cpu") for c in args.checkpoints]
    titles = [NAMES.get(ck.get("args", {}).get("model", "lss"), "model") for _, ck in models]

    rows, cols = len(args.samples), 2 + len(models)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols + 3, 3.2 * rows),
                             gridspec_kw=dict(width_ratios=[2.2] + [1] * (cols - 1)), squeeze=False)
    for r, idx in enumerate(args.samples):
        imgs, intrins, rots, trans, label = val[idx]
        axes[r, 0].imshow(camera_mosaic(imgs)); axes[r, 0].set_axis_off()
        axes[r, 0].set_title(f"val sample {idx}", fontsize=9)
        show_map(axes[r, 1], label[0], "ground truth", cmap="Greys", vmin=0, vmax=1)
        for c, (model, _) in enumerate(models):
            with torch.no_grad():
                prob = model(imgs[None], intrins[None], rots[None], trans[None]).sigmoid()[0, 0]
            show_map(axes[r, 2 + c], prob, titles[c], label=label[0], cmap="magma", vmin=0, vmax=1)

    fig.tight_layout()
    Path(args.out).parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
