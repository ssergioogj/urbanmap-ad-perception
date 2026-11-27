"""
Train Lift-Splat (+ BEV CNN) for vehicle segmentation on nuScenes.
Examples (from the project root):
    # 1) Sanity check: memorize 4 samples. IoU on them must go up a lot.
    uv run python scripts/train.py --overfit 4 --epochs 150 --val-every 25

    # 2) Real training on nuScenes mini
    uv run python scripts/train.py --epochs 30

    # 3) Baselines of the paper, same settings
    uv run python scripts/train.py --epochs 30 --model cnn --name mini-cnn
    uv run python scripts/train.py --epochs 30 --model frozen --name mini-frozen
Hyperparameters follow the paper (section 5) and the official code: Adam, lr 1e-3,
weight decay 1e-7, batch size 4, BCE with positive weight 1.0 (paper; the official
code uses 2.13, available with --pos-weight 2.13), gradient clipping at 5.
Outputs go to runs/<name>/: log.csv, last.pt and best.pt (best validation IoU).
"""
import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from lss_bev.dataset import NuScenesBEV
from lss_bev.metrics import IoUMeter
from lss_bev.models.build import MODELS, build_model

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", default=str(ROOT / "data"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--model", default="lss", choices=MODELS,
                   help="lss: Lift-Splat | cnn: CNN baseline without 3D | frozen: LSS with frozen backbone")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-7)
    p.add_argument("--pos-weight", type=float, default=1.0,
                   help="BCE positive weight. Paper (sec. 5): 1.0 for object segmentation; "
                        "the official code uses 2.13")
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    p.add_argument("--no-pretrained", action="store_true", help="do not load ImageNet weights")
    p.add_argument("--overfit", type=int, default=0,
                   help="train AND evaluate on the first N training samples (sanity check)")
    p.add_argument("--val-every", type=int, default=1, help="evaluate every N epochs")
    p.add_argument("--name", default=None, help="run name (folder inside runs/)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")          # Case Apple GPU (M1-M4)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(batch, device):
    return [x.to(device, non_blocking=True) for x in batch]


@torch.no_grad()
def evaluate(model, loader, device, pos_weight):
    model.eval()
    meter, total_loss, n = IoUMeter(), 0.0, 0
    for batch in loader:
        imgs, intrins, rots, trans, label = to_device(batch, device)
        logits = model(imgs, intrins, rots, trans)
        total_loss += F.binary_cross_entropy_with_logits(logits, label, pos_weight=pos_weight).item() * len(label)
        n += len(label)
        meter.update(logits, label)
    model.train()
    return total_loss / n, meter.compute()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    run = ROOT / "runs" / (args.name or time.strftime("%Y%m%d-%H%M%S"))
    run.mkdir(parents=True, exist_ok=True)

    # Data
    if args.overfit:
        base = NuScenesBEV(args.dataroot, args.version, "train", augment=False)
        train_set = val_set = Subset(base, range(args.overfit))
    else:
        train_set = NuScenesBEV(args.dataroot, args.version, "train", augment=True)
        val_set = NuScenesBEV(args.dataroot, args.version, "val", augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=not args.overfit,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, persistent_workers=args.workers > 0)

    # Model, loss, optimizer
    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pos_weight = torch.tensor([args.pos_weight], device=device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr, weight_decay=args.weight_decay)

    print(f"model={args.model}  params={n_params / 1e6:.2f}M (trainable {n_train / 1e6:.2f}M)  "
          f"device={device}  train={len(train_set)}  val={len(val_set)}  run={run}")
    log = open(run / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "step", "train_loss", "val_loss", "val_iou", "seconds"])

    best_iou, step, start = -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        running, count = 0.0, 0
        for batch in train_loader:
            imgs, intrins, rots, trans, label = to_device(batch, device)
            logits = model(imgs, intrins, rots, trans)
            loss = F.binary_cross_entropy_with_logits(logits, label, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            running += loss.item()
            count += 1
            step += 1
        train_loss = running / max(count, 1)

        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_loss, val_iou = evaluate(model, val_loader, device, pos_weight)
            elapsed = time.time() - start
            print(f"epoch {epoch:3d}  step {step:5d}  train_loss {train_loss:.4f}  "
                  f"val_loss {val_loss:.4f}  val_IoU {val_iou:.3f}  ({elapsed / 60:.1f} min)")
            writer.writerow([epoch, step, f"{train_loss:.5f}", f"{val_loss:.5f}", f"{val_iou:.5f}", int(elapsed)])
            log.flush()
            ckpt = dict(model=model.state_dict(), epoch=epoch, val_iou=val_iou, args=vars(args))
            torch.save(ckpt, run / "last.pt")
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(ckpt, run / "best.pt")
        else:
            print(f"epoch {epoch:3d}  step {step:5d}  train_loss {train_loss:.4f}")

    log.close()
    print(f"done. best val IoU {best_iou:.3f}  -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
