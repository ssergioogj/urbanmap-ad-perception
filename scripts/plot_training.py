"""
Training curves of several runs, from their log.csv files.
    uv run python scripts/plot_training.py runs/mini-lss runs/mini-frozen runs/mini-cnn
Left: validation IoU per epoch (dots) and its 3-epoch moving average (line).
Right: train and validation loss.  -> outputs/training_curves.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"lss": "Lift-Splat", "frozen": "Frozen Encoder", "cnn": "CNN (no 3D)"}
COLORS = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]


def moving_average(values, k=3):
    return [sum(values[max(0, i - k + 1):i + 1]) / len(values[max(0, i - k + 1):i + 1])
            for i in range(len(values))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default=str(ROOT / "outputs" / "training_curves.png"))
    args = p.parse_args()

    fig, (ax_iou, ax_loss) = plt.subplots(1, 2, figsize=(13, 4.8))
    for color, run in zip(COLORS, args.runs):
        rows = list(csv.DictReader(open(Path(run) / "log.csv")))
        rows = [r for r in rows if r["val_iou"]]
        epoch = [int(r["epoch"]) for r in rows]
        iou = [100 * float(r["val_iou"]) for r in rows]
        name = Path(run).name
        label = next((v for k, v in LABELS.items() if name.endswith(k)), name)

        ax_iou.scatter(epoch, iou, s=12, color=color, alpha=0.45)
        ax_iou.plot(epoch, moving_average(iou), color=color, lw=2, label=label)
        ax_loss.plot(epoch, [float(r["train_loss"]) for r in rows], color=color, lw=1.5, ls="--")
        ax_loss.plot(epoch, [float(r["val_loss"]) for r in rows], color=color, lw=2, label=label)

    ax_iou.set_xlabel("epoch"); ax_iou.set_ylabel("validation IoU (%)")
    ax_iou.set_title("Vehicle IoU on validation (dots) and 3-epoch moving average")
    ax_iou.legend(); ax_iou.grid(alpha=0.3)
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("BCE loss")
    ax_loss.set_title("Loss: train (dashed) and validation (solid)")
    ax_loss.legend(); ax_loss.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.out).parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
