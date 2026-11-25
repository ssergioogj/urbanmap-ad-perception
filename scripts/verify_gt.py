from pathlib import Path
import matplotlib
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from lss_bev.models.geometry import adjust_intrinsics, create_frustum, frustum_to_ego
from lss_bev.models.splat import Splat
from lss_bev.nusc_calib import CAMS, load_bev_vehicle_mask, load_sample_cameras

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
H, W, STRIDE = 128, 352, 16
SCALE, CROP_TOP = 0.22, 48
COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

# 1. Cargar calibración y máscara GT del mismo sample
sample_idx = 0
cal = load_sample_cameras(ROOT / "data", sample_index=sample_idx)
mask = load_bev_vehicle_mask(ROOT / "data", sample_index=sample_idx)[0].numpy()  # (200, 200)

# 2. Proyectar conos de cámara (igual que en visualize_splat.py)
K = adjust_intrinsics(cal["intrins"], SCALE, crop_top=CROP_TOP)
frustum = create_frustum((H, W), STRIDE, (4.0, 45.0, 1.0))
geom = frustum_to_ego(frustum, K[None], cal["rots"][None], cal["trans"][None])
D, Hf, Wf = frustum.shape[:3]

feats = torch.zeros(1, 6, D, 6, Hf, Wf)
for n in range(6):
    feats[:, n, :, n] = 1.0
bev = Splat()(feats, geom)[0].numpy()
owner = np.where(bev.sum(0) > 0, bev.argmax(0), np.nan)

# 3. Dibujar conos + máscara de vehículos superpuesta
fig, ax = plt.subplots(figsize=(8, 8))
extent = [-50, 50, -50, 50]
kw = dict(origin="lower", extent=extent, interpolation="nearest")

# Conos de cámara semitransparentes de fondo
ax.imshow(owner, cmap=ListedColormap(COLORS), vmin=-0.5, vmax=5.5, alpha=0.35, **kw)

# Máscara de vehículos en rojo sólido (solo donde mask > 0)
ax.imshow(np.where(mask > 0, 1.0, np.nan), cmap=ListedColormap(["#d62728"]), **kw)

# Coche ego en el centro
ax.add_patch(plt.Rectangle((-0.95, -2.3), 1.9, 4.6, color="black", label="Ego vehicle"))
ax.invert_xaxis()  # izquierda del coche a la izquierda de la imagen (igual que visualize_splat.py)
ax.set_xlabel("y (m) — izquierda ←   → derecha")
ax.set_ylabel("x (m) — hacia delante")
ax.set_title(f"Verificación GT Vehículos (sample {sample_idx}) — Celdas activas: {int(mask.sum())}")
ax.grid(alpha=0.25)

out = ROOT / "outputs" / "verify_gt.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"Guardado en {out} (celdas con vehículo: {int(mask.sum())})")
