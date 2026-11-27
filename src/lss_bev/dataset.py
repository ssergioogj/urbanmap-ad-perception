"""
PyTorch Dataset for LSS on nuScenes.

Each item is one sample (one instant) with the 6 cameras:
    imgs:    (6, 3, 128, 352)  resized, cropped and normalized images
    intrins: (6, 3, 3)         intrinsics ADJUSTED to the resized/cropped image
    rots:    (6, 3, 3)         camera -> ego rotation
    trans:   (6, 3)            camera -> ego translation
    label:   (1, 200, 200)     1 where there is a vehicle (see labels.py)

Preprocessing follows the official LSS code:
  - Evaluation (augment=False): resize x0.22 (1600x900 -> 352x198) and crop the
    rows [48, 176): removes most of the sky and a bit of the car hood.
  - Training (augment=True): random resize in [0.193, 0.225], random crop
    (vertical position from bot_pct_lim, random horizontal position) and random
    horizontal flip. Every change to the image is applied to its intrinsics too,
    so the geometry stays correct and the labels do not change.
    (The official code also rotates the image up to +-5.4 degrees; not done here.)
"""
import json
import random
from pathlib import Path

import torch
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from lss_bev.labels import VehicleLabels
from lss_bev.models.geometry import adjust_intrinsics, flip_intrinsics
from lss_bev.nusc_calib import CAMS

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def split_scenes(version, split):
    """Scene names of the official nuScenes split ('train' or 'val')."""
    from nuscenes.utils.splits import create_splits_scenes
    name = f"mini_{split}" if version == "v1.0-mini" else split
    return set(create_splits_scenes(verbose=False)[name])


class NuScenesBEV(Dataset):
    def __init__(self, dataroot, version="v1.0-mini", split="train", augment=False,
                 final_dim=(128, 352), resize_lim=(0.193, 0.225), bot_pct_lim=(0.0, 0.22),
                 rand_flip=True, cams=CAMS):
        self.root = Path(dataroot)
        self.augment = augment
        self.final_dim = final_dim
        self.resize_lim, self.bot_pct_lim, self.rand_flip = resize_lim, bot_pct_lim, rand_flip
        self.cams = cams
        self.labels = VehicleLabels(dataroot, version)

        meta = self.root / version
        load = lambda name: json.loads((meta / f"{name}.json").read_text())  # noqa: E731

        # 1. Samples of the scenes in this split
        wanted = split_scenes(version, split)
        scene_tokens = {s["token"] for s in load("scene") if s["name"] in wanted}
        samples = [s for s in load("sample") if s["scene_token"] in scene_tokens]
        samples.sort(key=lambda s: (s["scene_token"], s["timestamp"]))

        # 2. For each sample, the keyframe image and calibration of each camera
        calib = {c["token"]: c for c in load("calibrated_sensor")}
        cams_of = {}
        for sd in load("sample_data"):
            if not sd["is_key_frame"]:
                continue
            cam = sd["filename"].split("/")[1]
            if cam in cams:
                c = calib[sd["calibrated_sensor_token"]]
                cams_of.setdefault(sd["sample_token"], {})[cam] = dict(
                    path=self.root / sd["filename"],
                    K=torch.tensor(c["camera_intrinsic"], dtype=torch.float32),
                    R=torch.tensor(Quaternion(c["rotation"]).rotation_matrix, dtype=torch.float32),
                    t=torch.tensor(c["translation"], dtype=torch.float32),
                )

        # 3. Keep only samples whose 6 images are on disk (useful with partial downloads)
        self.samples = []
        for s in samples:
            info = cams_of.get(s["token"], {})
            if len(info) == len(cams) and all(info[c]["path"].exists() for c in cams):
                self.samples.append(dict(token=s["token"], cams=info))

    def __len__(self):
        return len(self.samples)

    def sample_augmentation(self, W, H):
        """Resize factor, crop box (left, top) and flip for one image."""
        fH, fW = self.final_dim
        if self.augment:
            resize = random.uniform(*self.resize_lim)
            newW, newH = int(W * resize), int(H * resize)
            crop_top = int((1 - random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_left = int(random.uniform(0, max(0, newW - fW)))
            flip = self.rand_flip and random.random() < 0.5
        else:
            resize = max(fH / H, fW / W)                              # 0.22
            newW, newH = int(W * resize), int(H * resize)
            crop_top = int((1 - sum(self.bot_pct_lim) / 2) * newH) - fH    # 48
            crop_left = int(max(0, newW - fW) / 2)                        # 0
            flip = False
        return resize, crop_left, crop_top, flip

    def transform(self, img, K, resize, crop_left, crop_top, flip):
        """Applies resize -> crop -> flip to the image and the same to K."""
        fH, fW = self.final_dim
        W, H = img.size
        img = img.resize((int(W * resize), int(H * resize)), Image.BILINEAR)
        img = img.crop((crop_left, crop_top, crop_left + fW, crop_top + fH))
        K = adjust_intrinsics(K, resize, crop_top=crop_top, crop_left=crop_left)
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            K = flip_intrinsics(K, fW)
        return img, K

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs, intrins, rots, trans = [], [], [], []
        for cam in self.cams:
            info = sample["cams"][cam]
            img = Image.open(info["path"]).convert("RGB")
            aug = self.sample_augmentation(*img.size)
            img, K = self.transform(img, info["K"], *aug)

            x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
            x = x.view(img.height, img.width, 3).permute(2, 0, 1).float() / 255
            imgs.append((x - IMAGENET_MEAN) / IMAGENET_STD)
            intrins.append(K)
            rots.append(info["R"])
            trans.append(info["t"])

        label = self.labels(sample["token"])
        return torch.stack(imgs), torch.stack(intrins), torch.stack(rots), torch.stack(trans), label
