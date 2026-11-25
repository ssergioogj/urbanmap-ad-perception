from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lss_bev.dataset import NuScenesBevDataset, get_scene_splits
from lss_bev.models.lift_splat import LiftSplat

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Split por escenas (8 escenas train ~323 samples, 2 escenas val ~81 samples)
train_idx, val_idx = get_scene_splits(ROOT / "data", val_scenes=2)
train_ds = NuScenesBevDataset(ROOT / "data", indices=train_idx)
val_ds = NuScenesBevDataset(ROOT / "data", indices=val_idx)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

# 2. Modelo, optimizador y pérdida (Sec. 5 del paper)
model = LiftSplat(pretrained=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-7)
criterion = nn.BCEWithLogitsLoss()

best_val_iou = 0.0
epochs = 20

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for imgs, intrins, rots, trans, target in train_loader:
        imgs, intrins, rots, trans, target = [x.to(device) for x in (imgs, intrins, rots, trans, target)]
        optimizer.zero_grad()
        logits = model(imgs, intrins, rots, trans)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_ds)

    # Evaluación en validación
    model.eval()
    val_loss, total_inter, total_union = 0.0, 0.0, 0.0
    with torch.no_grad():
        for imgs, intrins, rots, trans, target in val_loader:
            imgs, intrins, rots, trans, target = [x.to(device) for x in (imgs, intrins, rots, trans, target)]
            logits = model(imgs, intrins, rots, trans)
            val_loss += criterion(logits, target).item() * imgs.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_inter += (preds * target).sum().item()
            total_union += ((preds + target) > 0).float().sum().item()

    val_loss /= len(val_ds)
    val_iou = total_inter / max(total_union, 1.0)
    print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), OUT_DIR / "best_lss.pt")

print(f"Entrenamiento finalizado. Mejor Val IoU: {best_val_iou:.4f}")
