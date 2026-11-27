"""
Tests for the nuScenes BEV dataset (dataset.py).

Run from the project root:
    pytest tests/test_dataset.py -v
"""
import random
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from lss_bev.dataset import IMAGENET_MEAN, IMAGENET_STD, NuScenesBEV
from lss_bev.models.geometry import adjust_intrinsics
from lss_bev.models.lift_splat import LiftSplat

ROOT = Path(__file__).resolve().parents[1]
DATAROOT = ROOT / "data"
pytestmark = pytest.mark.skipif(not (DATAROOT / "v1.0-mini").exists(), reason="no nuScenes mini")


@pytest.fixture(scope="module")
def train():
    return NuScenesBEV(DATAROOT, "v1.0-mini", "train")


@pytest.fixture(scope="module")
def val():
    return NuScenesBEV(DATAROOT, "v1.0-mini", "val")


# --------------------------------------------------------------------------
# 1. Splits
# --------------------------------------------------------------------------
def test_official_mini_splits(train, val):
    """mini_train = 8 scenes, mini_val = 2 scenes, with no sample in both."""
    assert len(train) > 300 and len(val) > 70
    tokens_train = {s["token"] for s in train.samples}
    tokens_val = {s["token"] for s in val.samples}
    assert not tokens_train & tokens_val


# --------------------------------------------------------------------------
# 2. One item
# --------------------------------------------------------------------------
def test_item_shapes(val):
    imgs, intrins, rots, trans, label = val[0]
    assert imgs.shape == (6, 3, 128, 352)
    assert intrins.shape == (6, 3, 3) and rots.shape == (6, 3, 3) and trans.shape == (6, 3)
    assert label.shape == (1, 200, 200)
    assert imgs.dtype == intrins.dtype == label.dtype == torch.float32


def test_images_are_normalized(val):
    imgs = val[0][0]
    raw = imgs * IMAGENET_STD + IMAGENET_MEAN          # undo the normalization
    assert raw.min() >= -1e-5 and raw.max() <= 1 + 1e-5
    assert -1.5 < imgs.mean() < 1.5


def test_eval_preprocessing_is_resize_022_crop_48(val):
    """Without augmentation: same image and K as the scripts and tests use."""
    imgs, intrins, *_ = val[3]
    info = val.samples[3]["cams"]["CAM_FRONT"]
    im = Image.open(info["path"]).convert("RGB")
    im = im.resize((352, 198), Image.BILINEAR).crop((0, 48, 352, 176))
    x = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).view(128, 352, 3).permute(2, 0, 1).float() / 255
    assert torch.allclose(imgs[1], (x - IMAGENET_MEAN) / IMAGENET_STD, atol=1e-5)
    assert torch.allclose(intrins[1], adjust_intrinsics(info["K"], 0.22, crop_top=48))


def test_eval_is_deterministic(val):
    a, b = val[5], val[5]
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_label_matches_labels_module(val):
    assert torch.equal(val[7][4], val.labels(val.samples[7]["token"]))


# --------------------------------------------------------------------------
# 3. Augmentation keeps image and intrinsics consistent
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(8))
def test_augmentation_moves_pixels_like_intrinsics(train, seed):
    """Draw a dot where a 3D point projects in a fake 1600x900 image, augment
    image and K together, and check the dot is where the new K projects it."""
    random.seed(seed)
    ds = NuScenesBEV.__new__(NuScenesBEV)
    ds.augment, ds.final_dim = True, (128, 352)
    ds.resize_lim, ds.bot_pct_lim, ds.rand_flip = train.resize_lim, train.bot_pct_lim, True

    K = torch.tensor([[1266.4, 0, 816.3], [0, 1266.4, 491.5], [0, 0, 1.0]])
    point = torch.tensor([1.5 - seed * 0.4, 0.4, 12.0])          # in camera coordinates
    uvw = K @ point
    u, v = (uvw[0] / uvw[2]).item(), (uvw[1] / uvw[2]).item()
    img = Image.new("RGB", (1600, 900))
    ImageDraw.Draw(img).ellipse((u - 12, v - 12, u + 12, v + 12), fill=(255, 255, 255))

    aug = ds.sample_augmentation(1600, 900)
    out, K2 = ds.transform(img, K, *aug)
    uvw2 = K2 @ point
    u2, v2 = (uvw2[0] / uvw2[2]).item(), (uvw2[1] / uvw2[2]).item()
    if not (0 <= u2 < 352 and 0 <= v2 < 128):
        pytest.skip("dot cropped out with this augmentation")

    mask = torch.frombuffer(bytearray(out.convert("L").tobytes()), dtype=torch.uint8).view(128, 352) > 128
    vs, us = torch.nonzero(mask, as_tuple=True)
    # Up to ~1 px of difference is expected: PIL resizes around pixel centers
    # (half-pixel offset) and rounds the new size to integers. At stride 16 this is
    # far below one feature cell. The official LSS code has the same approximation.
    assert abs(us.float().mean() - u2) < 1.5 and abs(vs.float().mean() - v2) < 1.5, aug


def test_augmented_items_differ_but_label_does_not(train):
    ds = NuScenesBEV(DATAROOT, "v1.0-mini", "train", augment=True)
    random.seed(0)
    a = ds[0]
    random.seed(1)
    b = ds[0]
    assert not torch.equal(a[0], b[0]) and not torch.equal(a[1], b[1])
    assert torch.equal(a[4], b[4])                    # augmentation never changes the label


# --------------------------------------------------------------------------
# 4. Ready for training
# --------------------------------------------------------------------------
def test_dataloader_and_one_training_step(val):
    loader = DataLoader(val, batch_size=2, shuffle=True)
    imgs, intrins, rots, trans, label = next(iter(loader))
    assert imgs.shape == (2, 6, 3, 128, 352) and label.shape == (2, 1, 200, 200)

    torch.manual_seed(0)
    model = LiftSplat(pretrained=False).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    logits = model(imgs, intrins, rots, trans)
    loss = F.binary_cross_entropy_with_logits(logits, label)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
