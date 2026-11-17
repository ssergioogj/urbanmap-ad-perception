"""
It draws, seen from the top, the frustrum points of the 6 camera of a sample of nuScenes mini
Used to check that the geometry of Lift put each camera where it belongs
"""
from pathlib import Path
from PIL import Image
from lss_bev.models.geometry import adjust_intrinsics, create_frustum, frustum_to_ego
from lss_bev.nusc_calib import CAMS, load_sample_cameras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
H, W, STRIDE = 128, 352, 8
SCALE, CROP_TOP = 0.22, 48
COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

cal = load_sample_cameras(ROOT / "data")
K = adjust_intrinsics(cal["intrins"], SCALE, crop_top=CROP_TOP)
frustum = create_frustum((H, W), STRIDE, (4.0, 45.0, 1.0))
geom = frustum_to_ego(frustum, K[None], cal["rots"][None], cal["trans"][None])[0]

fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0.03, 0.06, 0.5, 0.88])
for n, cam in enumerate(CAMS):
    g = geom[n]                                  # (D, Hf, Wf, 3)
    ax.scatter(g[::2, ::3, :, 0].flatten(), g[::2, ::3, :, 1].flatten(),
               s=0.4, color=COLORS[n], alpha=0.5)
    ax.scatter([], [], s=30, color=COLORS[n], label=cam)
    tx, ty, _ = cal["trans"][n]
    ax.plot(tx, ty, "k^", ms=5)

# Rack BEV of the paper: among -50 and 50 m with cell of size 0.5 meters -> 50+50/0.5 = 200
# 
ax.add_patch(plt.Rectangle((-50, -50), 100, 100, fill=False, ls="--", color="gray"))
ax.add_patch(plt.Rectangle((-2.3, -0.95), 4.6, 1.9, color="black"))  # a car (aprox)
ax.annotate("", xy=(6, 0), xytext=(2.5, 0), arrowprops=dict(arrowstyle="->", lw=2))
ax.set_xlim(-55, 55); ax.set_ylim(-55, 55); ax.set_aspect("equal")
ax.set_xlabel("x (m) — hacia delante"); ax.set_ylabel("y (m) — hacia la izquierda")
ax.set_title("Puntos del frustum vistos desde arriba (4–44 m)\nrecuadro: rejilla BEV de 200×200 (±50 m)")
ax.legend(loc="lower left", fontsize=8, markerscale=1)
ax.grid(alpha=0.3)

# The 6 images in the same disposition as the car
order = [0, 1, 2, 3, 4, 5]
for i, n in enumerate(order):
    r, c = divmod(i, 3)
    a = fig.add_axes([0.56 + c * 0.145, 0.55 - r * 0.3, 0.14, 0.3])
    im = Image.open(cal["paths"][n]).convert("RGB")
    im = im.resize((int(im.width * SCALE), int(im.height * SCALE)))
    im = im.crop((0, CROP_TOP, W, CROP_TOP + H))
    a.imshow(im); a.set_axis_off()
    a.set_title(CAMS[n], fontsize=8, color=COLORS[n])

out = ROOT / "outputs" / "frustum_bev.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"guardado en {out}")
