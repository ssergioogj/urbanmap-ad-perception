"""
Tests for the ground-truth BEV labels (labels.py).
Run from the project root:
    pytest tests/test_labels.py -v
"""
import json
import math
from pathlib import Path

import pytest
import torch
from pyquaternion import Quaternion

from lss_bev.labels import VehicleLabels, box_footprint_mask
from lss_bev.models.geometry import adjust_intrinsics
from lss_bev.models.splat import Splat
from lss_bev.nusc_calib import CAMS, load_sample_cameras

ROOT = Path(__file__).resolve().parents[1]
DATAROOT = ROOT / "data"
HAS_MINI = (DATAROOT / "v1.0-mini").exists()
H, W = 128, 352
SCALE, CROP_TOP = 0.22, 48


def cell_of(x, y):
    return int((x + 50) // 0.5), int((y + 50) // 0.5)

# 1. One box, computable by hand
def test_axis_aligned_car_area():
    """A 4 m x 2 m car facing forward covers 4*2 / (0.5*0.5) = 32 cells."""
    m = box_footprint_mask((10.0, 0.0), length=4.0, width=2.0, yaw=0.0)
    assert m.shape == (200, 200)
    assert m.sum() == 32
    ii, jj = torch.nonzero(m, as_tuple=True)
    assert ii.max() - ii.min() + 1 == 8          # 4 m along x -> 8 cells
    assert jj.max() - jj.min() + 1 == 4          # 2 m along y -> 4 cells


def test_rotated_car_swaps_extent():
    """Turned 90 degrees, the same car is long along y and narrow along x."""
    m = box_footprint_mask((10.0, 0.0), length=4.0, width=2.0, yaw=math.pi / 2)
    ii, jj = torch.nonzero(m, as_tuple=True)
    assert ii.max() - ii.min() + 1 == 4
    assert jj.max() - jj.min() + 1 == 8


def test_diagonal_car_keeps_its_area():
    m = box_footprint_mask((-15.0, 12.0), length=4.6, width=1.9, yaw=math.radians(35))
    assert abs(m.sum().item() - 4.6 * 1.9 / 0.25) < 6       # ~35 cells, borders vary


def test_orientation_matches_splat():
    """A car ahead-left must land in the same cell where Splat puts a point there."""
    m = box_footprint_mask((20.0, 10.0), length=1.0, width=1.0, yaw=0.0)
    feats = torch.ones(1, 1, 1, 1, 1, 1)
    geom = torch.tensor([20.0, 10.0, 0.0]).view(1, 1, 1, 1, 1, 3)
    bev = Splat()(feats, geom)[0, 0]
    assert m[bev > 0].all()
    i, j = cell_of(20.0, 10.0)
    assert i > 100 and j > 100 and m[i, j]


def test_box_outside_map_is_empty():
    assert box_footprint_mask((80.0, 0.0), 4.0, 2.0, 0.0).sum() == 0


def test_box_on_the_edge_is_clipped():
    m = box_footprint_mask((49.0, 0.0), length=4.0, width=2.0, yaw=0.0)
    assert 0 < m.sum() < 32


# 2. Real nuScenes mini samples
@pytest.fixture(scope="module")
def labels():
    return VehicleLabels(DATAROOT)


@pytest.mark.skipif(not HAS_MINI, reason="no nuScenes mini")
def test_label_shape_and_values(labels):
    token = labels.samples[0]["token"]
    lab = labels(token)
    assert lab.shape == (1, 200, 200)
    assert set(lab.unique().tolist()) <= {0.0, 1.0}
    assert lab.sum() > 0


@pytest.mark.skipif(not HAS_MINI, reason="no nuScenes mini")
def test_only_vehicles(labels):
    for s in labels.samples[:20]:
        for b in labels.vehicle_boxes(s["token"]):
            assert b["category"].startswith("vehicle.")


@pytest.mark.skipif(not HAS_MINI, reason="no nuScenes mini")
def test_own_car_is_not_labelled(labels):
    """Our own car is not an annotation: the cell at the origin must be empty
    in every sample."""
    for s in labels.samples[:50]:
        assert labels(s["token"])[0, 99:101, 99:101].sum() == 0


@pytest.mark.skipif(not HAS_MINI, reason="no nuScenes mini")
def test_label_agrees_with_camera_view(labels):
    """Independent path: take each annotated vehicle, move it to the car frame
    with the CAM_FRONT ego pose (not the LIDAR one used by the labels), keep the
    ones CAM_FRONT can see, and check their center lands on a labelled cell."""
    cal = load_sample_cameras(DATAROOT)
    token = cal["sample_token"]
    lab = labels(token)[0]
    meta = DATAROOT / "v1.0-mini"
    sd = next(x for x in json.loads((meta / "sample_data.json").read_text())
              if x["sample_token"] == token and x["is_key_frame"] and "/CAM_FRONT/" in x["filename"])
    pose = next(p for p in json.loads((meta / "ego_pose.json").read_text()) if p["token"] == sd["ego_pose_token"])
    R_ego = torch.tensor(Quaternion(pose["rotation"]).rotation_matrix, dtype=torch.float32)
    t_ego = torch.tensor(pose["translation"], dtype=torch.float32)
    n = CAMS.index("CAM_FRONT")
    K = adjust_intrinsics(cal["intrins"][n], SCALE, crop_top=CROP_TOP)
    R, t = cal["rots"][n], cal["trans"][n]

    checked = 0
    for a in labels.anns_by_sample[token]:
        if not labels.category_of[a["instance_token"]].startswith("vehicle."):
            continue
        p_ego = R_ego.T @ (torch.tensor(a["translation"], dtype=torch.float32) - t_ego)
        p_cam = R.T @ (p_ego - t)
        if p_cam[2] < 4:
            continue
        uvw = K @ p_cam
        u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
        if not (0 <= u < W and 0 <= v < H):
            continue
        i, j = cell_of(p_ego[0].item(), p_ego[1].item())
        if 0 <= i < 200 and 0 <= j < 200:
            assert lab[i, j] == 1, labels.category_of[a["instance_token"]]
            checked += 1
    assert checked >= 1
