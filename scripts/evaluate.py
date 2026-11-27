"""
Evaluate trained checkpoints on the validation split and print the results table.

    uv run python scripts/evaluate.py runs/mini-lss/best.pt runs/mini-cnn/best.pt runs/mini-frozen/best.pt

Prints IoU (the paper metric), precision, recall, accuracy and IoU by distance to the car,
next to the paper numbers (Table 1, nuScenes "Vehicles"), and saves
outputs/results.csv.
"""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lss_bev.dataset import NuScenesBEV
from lss_bev.metrics import SegmentationMeter
from lss_bev.models.build import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
PAPER = {"cnn": 24.25, "frozen": 26.83, "lss": 32.07}       # Table 1, nuScenes, Vehicles
NAMES = {"cnn": "CNN (no 3D)", "frozen": "Frozen Encoder", "lss": "Lift-Splat"}


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--dataroot", default=str(ROOT / "data"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(ROOT / "outputs" / "results.csv"))
    args = p.parse_args()

    device = pick_device(args.device)
    val = NuScenesBEV(args.dataroot, args.version, "val", augment=False)
    loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers)

    rows = []
    for path in args.checkpoints:
        model, ckpt = load_checkpoint(path, device)
        kind = ckpt.get("args", {}).get("model", "lss")
        meter = SegmentationMeter()
        with torch.no_grad():
            for imgs, intrins, rots, trans, label in loader:
                logits = model(imgs.to(device), intrins.to(device), rots.to(device), trans.to(device))
                meter.update(logits, label)
        r = meter.compute()
        r.update(model=NAMES.get(kind, kind), checkpoint=path, epoch=ckpt.get("epoch"),
                 paper_iou=PAPER.get(kind))
        rows.append(r)
        print(f"evaluated {path}: IoU {100 * r['iou']:.2f}")

    bands = [k for k in rows[0] if k.startswith("iou_")]
    header = ["Model", "IoU", "Paper IoU", "Precision", "Recall", "Accuracy"] + [b[4:] for b in bands]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        paper = f"{r['paper_iou']:.2f}" if r["paper_iou"] else "-"
        cells = [r["model"], f"{100 * r['iou']:.2f}", paper,
                 f"{100 * r['precision']:.1f}", f"{100 * r['recall']:.1f}", f"{100 * r['accuracy']:.2f}"]
        cells += [f"{100 * r[b]:.1f}" for b in bands]
        print("| " + " | ".join(cells) + " |")
    print(f"\n{len(val)} validation samples ({args.version}). Values in %.")
    print(f"Accuracy of a model that never predicts a vehicle: {100 * rows[0]['empty_model_accuracy']:.2f}% "
          "(why accuracy is not a useful metric here).")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
