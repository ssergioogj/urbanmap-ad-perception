"""
Visual check of the ground-truth labels for one nuScenes mini sample.

  Left:  the 6 camera images with every annotated vehicle drawn on them
         (3D box), to see which vehicles are labelled.
  Right: the 200x200 label map, drawn like splat_bev.png (forward up, left on the left).

    python scripts/visualize_labels.py [sample_index]   -> outputs/labels_bev.png
"""
import sys
from pathlib import Path

import matplotlib
import torch
from PIL import Image

from lss_bev.labels import VehicleLabels
from lss_bev.models.geometry import adjust_intrinsics
from lss_bev.nusc_calib import CAMS, load_sample_cameras

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
H, W = 128, 352
SCALE, CROP_TOP = 0.22, 48
index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

cal = load_sample_cameras(ROOT / "data", sample_index=index)
labels = VehicleLabels(ROOT / "data")
token = cal["sample_token"]
label = labels(token)[0]
boxes = labels.vehicle_boxes(token)
K = adjust_intrinsics(cal["intrins"], SCALE, crop_top=CROP_TOP)


def box_edges(b):
    """The 12 edges of the 3D box in the ego frame, as a (24, 3) list of segment ends."""
    c, s_ = torch.cos(b["yaw"]), torch.sin(b["yaw"])
    l, w, h = b["length"] / 2, b["width"] / 2, b["height"] / 2
    local = torch.tensor([[x, y, z] for x, y, z in
                          [(l, w, -h), (l, -w, -h), (-l, -w, -h), (-l, w, -h),
                           (l, w, h), (l, -w, h), (-l, -w, h), (-l, w, h)]])
    rot = torch.tensor([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    corners = local @ rot.T + b["center"]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    return torch.stack([corners[k] for pair in pairs for k in pair])


fig = plt.figure(figsize=(16, 7))
for n, cam in enumerate(CAMS):
    r, col = divmod(n, 3)
    ax = fig.add_axes([0.01 + col * 0.19, 0.52 - r * 0.42, 0.185, 0.4])
    im = Image.open(cal["paths"][n]).convert("RGB")
    im = im.resize((int(im.width * SCALE), int(im.height * SCALE))).crop((0, CROP_TOP, W, CROP_TOP + H))
    ax.imshow(im)
    for b in boxes:
        p_cam = (cal["rots"][n].T @ (box_edges(b) - cal["trans"][n]).T).T
        if (p_cam[:, 2] < 1).any():
            continue                                  # (partly) behind this camera
        uv = (K[n] @ p_cam.T).T
        uv = uv[:, :2] / uv[:, 2:]
        if uv[:, 0].max() < 0 or uv[:, 0].min() > W or uv[:, 1].max() < 0 or uv[:, 1].min() > H:
            continue
        for k in range(0, 24, 2):
            ax.plot(uv[k:k + 2, 0], uv[k:k + 2, 1], color="red", lw=0.8)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
    ax.set_title(cam, fontsize=8)

ax = fig.add_axes([0.6, 0.06, 0.38, 0.86])
ax.imshow(label, origin="lower", extent=[-50, 50, -50, 50], cmap="Reds", interpolation="nearest")
ax.invert_xaxis()                                     # left of the car on the left of the image
ax.add_patch(plt.Rectangle((-0.95, -2.3), 1.9, 4.6, color="black"))
ax.set_xlabel("y (m) — left ←   → right"); ax.set_ylabel("x (m) — forward")
ax.set_title(f"Label 200×200: {len(boxes)} vehicles, {int(label.sum())} cells")
ax.grid(alpha=0.3); ax.set_aspect("equal")
fig.suptitle(f"Ground-truth vehicles, sample {index}")

out = ROOT / "outputs" / "labels_bev.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130)
print(f"saved to {out}")
