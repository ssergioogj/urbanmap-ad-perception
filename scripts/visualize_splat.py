"""
Muestra qué hace Splat con la calibración real de un sample de nuScenes mini.

  Izquierda: a qué cámara pertenece cada casilla del mapa 200x200.
  Derecha:   cuántos puntos del frustum caen en cada casilla (lo que Splat suma).

    python scripts/visualize_splat.py            -> outputs/splat_bev.png
"""
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib.colors import ListedColormap, LogNorm

from lss_bev.models.geometry import adjust_intrinsics, create_frustum, frustum_to_ego
from lss_bev.models.splat import Splat
from lss_bev.nusc_calib import CAMS, load_sample_cameras

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
H, W, STRIDE = 128, 352, 16
SCALE, CROP_TOP = 0.22, 48
COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

cal = load_sample_cameras(ROOT / "data")
K = adjust_intrinsics(cal["intrins"], SCALE, crop_top=CROP_TOP)
frustum = create_frustum((H, W), STRIDE, (4.0, 45.0, 1.0))
geom = frustum_to_ego(frustum, K[None], cal["rots"][None], cal["trans"][None])
D, Hf, Wf = frustum.shape[:3]

# Característica = "de qué cámara viene": canal n vale 1 si el punto es de la cámara n
feats = torch.zeros(1, 6, D, 6, Hf, Wf)
for n in range(6):
    feats[:, n, :, n] = 1.0
bev = Splat()(feats, geom)[0].numpy()               # (6, 200, 200): puntos por cámara y casilla

count = bev.sum(0)
owner = np.where(count > 0, bev.argmax(0), np.nan)  # cámara con más puntos en la casilla

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
extent = [-50, 50, -50, 50]                         # en metros
kw = dict(origin="lower", extent=extent, interpolation="nearest")

axes[0].imshow(owner, cmap=ListedColormap(COLORS), vmin=-0.5, vmax=5.5, **kw)
for n, cam in enumerate(CAMS):
    axes[0].scatter([], [], color=COLORS[n], label=cam, s=30)
axes[0].legend(loc="lower left", fontsize=8)
axes[0].set_title("Casillas del mapa 200×200 según la cámara que las llena")

im = axes[1].imshow(np.where(count > 0, count, np.nan), cmap="viridis", norm=LogNorm(), **kw)
fig.colorbar(im, ax=axes[1], fraction=0.046, label="puntos sumados en la casilla")
axes[1].set_title("Cuántos puntos caen en cada casilla (escala log.)")

for ax in axes:
    ax.invert_xaxis()                               # izquierda del coche a la izquierda de la imagen
    ax.add_patch(plt.Rectangle((-0.95, -2.3), 1.9, 4.6, color="black"))
    ax.set_xlabel("y (m) — izquierda ←   → derecha")
    ax.set_ylabel("x (m) — hacia delante")
    ax.set_aspect("equal")

# Nota: imshow pinta bev[i, j] con i (x) en vertical y j (y) en horizontal
fig.suptitle(f"Splat con la calibración real: {int(count.sum())} de {6 * D * Hf * Wf} puntos caen dentro del mapa")
fig.tight_layout()
out = ROOT / "outputs" / "splat_bev.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130)
print(f"guardado en {out}")
