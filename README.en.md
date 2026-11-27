# BEV-UrbanMap: Lift-Splat-Shoot from scratch

🇪🇸 [Leer en español](README.es.md)

Bird's-eye-view (BEV) perception for autonomous driving from the 6 cameras of a car.
We implement the **Lift-Splat-Shoot** architecture (Philion & Fidler, ECCV 2020) in
PyTorch and train it on **nuScenes** to segment vehicles on a 100 m × 100 m map around the car.

Final project of *Inteligencia Artificial en las Organizaciones* (UC3M, 2026/27), group 14:
Sergio González Jiménez, Alfonso Gabriel Pineda Fernández, Daniel López-Antona Pesquera,
Edgar Antero Sánchez Elizondo.

## Status (27 November 2026)

| Part | Status |
|---|---|
| **Lift**: per-pixel depth distribution × context (outer product) + frustum geometry | ✅ done and tested |
| **Splat**: pillar sum pooling with the cumsum trick and its analytic gradient | ✅ done and tested |
| **BEV CNN** (ResNet-18 blocks, paper section 4.1) | ✅ done and tested |
| BEV vehicle labels from the nuScenes 3D boxes | ✅ done and tested |
| Dataset with the official splits and data augmentation | ✅ done and tested |
| Training, evaluation (IoU, precision, recall, accuracy, IoU by distance) | ✅ done |
| Paper baselines: CNN without 3D and Frozen Encoder | ✅ done and trained |
| **Shoot**: motion planning with template trajectories | ⏳ pending (planned for the final delivery) |
| Drivable-area segmentation | ⏳ pending |
| Training on (part of) nuScenes trainval | ⏳ pending, only nuScenes mini so far |

## How it works

```
6 images ──► CamEncode (Lift) ─────────► what is at each 3D point   (B, 6, 41, 64, 8, 22)
             EfficientNet-B0 + depth/context head
calibration ──► frustum geometry (Lift) ► where each point is       (B, 6, 41, 8, 22, 3)
                   K⁻¹, R, t
                         │
                         ▼
                 Splat (sum pooling) ───► BEV feature map            (B, 64, 200, 200)
                         ▼
                 BEV CNN ───────────────► vehicle logit per cell     (B, 1, 200, 200)
```

- **Lift** (`models/cam_encode.py`, `models/geometry.py`). For every cell of the image
  feature map the network predicts a probability over 41 depths (4–44 m) and a 64-d context
  vector; their outer product places the context along the camera ray. The camera calibration
  gives the (x, y, z) position of each of these points around the car.
- **Splat** (`models/splat.py`). Every point falls into a 0.5 m × 0.5 m pillar of a 200 × 200
  grid; points in the same pillar are summed (the paper's "cumsum trick").
- **BEV CNN** (`models/bev_encode.py`). Fills the gaps between camera rays and predicts the map.

**Fine-tuning vs. developed from scratch.** The image backbone (EfficientNet-B0) starts from
ImageNet weights and is fine-tuned. Everything else (neck, depth/context head, frustum geometry,
Splat, BEV CNN, labels, dataset, training) is our own code and is trained from scratch.
The official LSS repository was only used as a reference.

## Project structure

```
src/lss_bev/
  models/cam_encode.py    Lift: image backbone + depth/context head + outer product
  models/geometry.py      Lift: frustum grid, intrinsics adjustment, camera -> ego
  models/splat.py         Splat: pillar sum pooling (cumsum trick)
  models/bev_encode.py    BEV CNN
  models/lift_splat.py    Full model (Lift + Splat + BEV CNN), optional frozen backbone
  models/baseline.py      CNN baseline without 3D structure
  models/build.py         Build a model by name / load a checkpoint
  labels.py               Ground-truth BEV vehicle maps from nuScenes annotations
  dataset.py              PyTorch Dataset (images, calibration, label) + augmentation
  metrics.py              IoU, precision, recall, accuracy, IoU by distance
  nusc_calib.py           Read camera calibration from the nuScenes JSON files
scripts/                  train, evaluate, and visualizations
tests/                    89 tests (pytest)
```

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it if missing).

```bash
uv sync --extra dev
```

Main dependencies (exact versions in `uv.lock`): PyTorch and torchvision, timm (EfficientNet-B0),
nuscenes-devkit (official splits), pyquaternion, Pillow, matplotlib; pytest for the tests.

On **Windows with an NVIDIA GPU**, PyPI only has CPU builds of PyTorch: install the CUDA build
(see pytorch.org) or training will silently run on the CPU.

### Data

Download **nuScenes mini** (v1.0) from <https://www.nuscenes.org/nuscenes> and extract it into
`data/` so that it looks like:

```
data/
  maps/  samples/  sweeps/  v1.0-mini/
```

Only the camera images, the calibration and the 3D annotations are used (no LiDAR or radar data).
Other versions (e.g. `v1.0-trainval`, even a partial download) work with `--version`; samples
with missing images are skipped.

**License.** nuScenes is distributed under CC BY-NC-SA 4.0 and requires accepting its terms of use
on the nuScenes website, so it is not included in this repository or in the delivery ZIP.

### Trained models

The trained models are not included: generate them with the three training commands in Usage
(`runs/mini-lss`, `runs/mini-cnn`, `runs/mini-frozen`). Each takes about 25 minutes on an Apple M4
(GPU via MPS); on a CPU it is much slower. The quick sanity check (`--overfit 4`) takes about 2 minutes.
The figures and the results table are generated the same way, with the scripts in Usage.

## Usage

All commands from the project root.

```bash
# Tests (~45 s)
uv run pytest tests -q

# Visual checks
uv run python scripts/visualize_frustum.py       # frustum points of the 6 cameras, seen from above
uv run python scripts/visualize_splat.py         # which camera fills each BEV cell
uv run python scripts/visualize_labels.py 0      # ground-truth vehicles of a sample (0-403)

# Sanity check: memorize 4 samples (IoU should go above ~0.8)
uv run python scripts/train.py --overfit 4 --epochs 150 --val-every 25 --name overfit

# Training on nuScenes mini (~25 min each on an Apple M4)
uv run python scripts/train.py --epochs 30 --name mini-lss
uv run python scripts/train.py --epochs 30 --model cnn --name mini-cnn
uv run python scripts/train.py --epochs 30 --model frozen --name mini-frozen

# Results table, predictions and training curves
uv run python scripts/evaluate.py runs/mini-lss/best.pt runs/mini-cnn/best.pt runs/mini-frozen/best.pt
uv run python scripts/visualize_predictions.py runs/mini-lss/best.pt runs/mini-cnn/best.pt runs/mini-frozen/best.pt
uv run python scripts/plot_training.py runs/mini-lss runs/mini-frozen runs/mini-cnn
```

The training script uses the Apple GPU (MPS), an NVIDIA GPU (CUDA) or the CPU automatically.
Each run writes `runs/<name>/log.csv`, `last.pt` and `best.pt` (best validation IoU).
Figures and tables go to `outputs/`. Neither folder is in git.

## Results (nuScenes mini)

Vehicle segmentation on the official `mini_val` split (81 samples, 2 scenes), best epoch of a
30-epoch run (one seed), batch size 4, Adam 1e-3, BCE with positive weight 1.0 (paper, sec. 5).

| Model | IoU | Paper IoU | Precision | Recall | Accuracy | IoU 0–10 m | 10–20 m | 20–30 m | 30–50 m |
|---|---|---|---|---|---|---|---|---|---|
| CNN (no 3D) | 0.16 | 24.25 | 16.2 | 0.2 | 97.34 | 1.4 | 0.3 | 0.0 | 0.0 |
| Frozen Encoder | 9.25 | 26.83 | 37.0 | 11.0 | 97.15 | 37.9 | 17.4 | 2.7 | 0.7 |
| **Lift-Splat (ours)** | **9.74** | 32.07 | 42.5 | 11.2 | 97.25 | 38.8 | 18.9 | 2.0 | 0.7 |

Values in %. Paper values: Table 1, nuScenes, "Vehicles".

- **3D structure is what matters.** Without Lift and Splat the CNN learns almost nothing with
  this little data, while both 3D models place vehicles correctly near the car.
- **Accuracy is misleading**: 97.36 % of the cells are empty, so a model that never predicts a
  vehicle reaches 97.36 % accuracy, more than any trained model. IoU is the paper metric.
- **Lift-Splat vs. Frozen Encoder** is too close to call with one seed (the validation IoU moves
  2–3 points between epochs).
- **Distance**: good up to 20 m, poor beyond 20 m (monocular depth gets harder with distance).
- **Gap with the paper**: we use ~1 % of its data (323 vs. ~28k training samples) and
  2.4k vs. 300k training steps; the models overfit after ~8 epochs.

## Implementation notes

- Image features at stride 16 (8 × 22 cells per image) as in the official code.
- Grid cells are computed with `floor`, so points just outside the map are dropped
  (the official code uses `.long()`, which rounds towards zero and keeps them in border cells).
- Augmentation: random resize, crop and horizontal flip; the intrinsics are changed with the image
  so the labels never change. The official ±5.4° rotation is not implemented.
- Labels include every annotated vehicle, whatever its visibility, as in the paper.

## Pending work (until 13 December)

1. **Shoot**: template trajectories from the ego poses (K-Means), cost map head, the loss of
   eq. 2 and top-k accuracy.
2. **Drivable area** segmentation from the nuScenes maps.
3. More data (part of `v1.0-trainval`) and several seeds per model.

## References

1. J. Philion, S. Fidler. *Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by
   Implicitly Unprojecting to 3D.* ECCV 2020.
2. H. Caesar et al. *nuScenes: A multimodal dataset for autonomous driving.* CVPR 2020.
3. K. He et al. *Deep residual learning for image recognition.* CVPR 2016.
4. M. Tan, Q. Le. *EfficientNet: Rethinking model scaling for convolutional neural networks.* ICML 2019.
