from collections import defaultdict
from pathlib import Path
import json

from PIL import Image, ImageDraw
from pyquaternion import Quaternion
import numpy as np
import torch
from torch.utils.data import Dataset

from lss_bev.models.geometry import adjust_intrinsics
from lss_bev.nusc_calib import CAMS

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def get_scene_splits(dataroot, version="v1.0-mini", val_scenes=2):
    """Divide los indices de sample.json por escena completa (evita data leakage)."""
    meta = Path(dataroot) / version
    scenes = json.loads((meta / "scene.json").read_text())
    samples = json.loads((meta / "sample.json").read_text())

    val_scene_tokens = {s["token"] for s in scenes[-val_scenes:]}
    train_idx, val_idx = [], []
    for idx, s in enumerate(samples):
        if s["scene_token"] in val_scene_tokens:
            val_idx.append(idx)
        else:
            train_idx.append(idx)
    return train_idx, val_idx


class NuScenesBevDataset(Dataset):
    def __init__(
        self,
        dataroot,
        version="v1.0-mini",
        indices=None,
        image_size=(128, 352),
        scale=0.22,
        crop_top=48,
        xbound=(-50.0, 50.0, 0.5),
        ybound=(-50.0, 50.0, 0.5),
        category_prefix="vehicle.",
    ):
        self.dataroot = Path(dataroot)
        meta = self.dataroot / version
        self.image_size = image_size
        self.scale = scale
        self.crop_top = crop_top
        self.xbound = xbound
        self.ybound = ybound
        self.nx = int(round((xbound[1] - xbound[0]) / xbound[2]))
        self.ny = int(round((ybound[1] - ybound[0]) / ybound[2]))
        self.category_prefix = category_prefix

        # Cachear metadatos en memoria una sola vez
        self.samples = json.loads((meta / "sample.json").read_text())
        self.calib = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}
        self.ego_poses = {ep["token"]: ep for ep in json.loads((meta / "ego_pose.json").read_text())}
        categories = {c["token"]: c["name"] for c in json.loads((meta / "category.json").read_text())}
        self.instances = {i["token"]: categories[i["category_token"]] for i in json.loads((meta / "instance.json").read_text())}

        self.sd_by_sample = defaultdict(dict)
        for sd in json.loads((meta / "sample_data.json").read_text()):
            if sd["is_key_frame"]:
                cam = sd["filename"].split("/")[1]
                if cam in CAMS:
                    self.sd_by_sample[sd["sample_token"]][cam] = sd

        self.anns_by_sample = defaultdict(list)
        for ann in json.loads((meta / "sample_annotation.json").read_text()):
            if self.instances[ann["instance_token"]].startswith(category_prefix):
                self.anns_by_sample[ann["sample_token"]].append(ann)

        self.indices = list(range(len(self.samples))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.samples[self.indices[idx]]
        token = sample["token"]
        by_cam = self.sd_by_sample[token]

        H, W = self.image_size
        imgs, intrins, rots, trans = [], [], [], []
        for cam in CAMS:
            sd = by_cam[cam]
            cs = self.calib[sd["calibrated_sensor_token"]]

            im = Image.open(self.dataroot / sd["filename"]).convert("RGB")
            im = im.resize((int(im.width * self.scale), int(im.height * self.scale)))
            im = im.crop((0, self.crop_top, W, self.crop_top + H))
            t = torch.from_numpy(np.array(im, dtype=np.float32)).permute(2, 0, 1) / 255.0
            imgs.append((t - IMAGENET_MEAN) / IMAGENET_STD)

            intrins.append(torch.tensor(cs["camera_intrinsic"], dtype=torch.float32))
            rots.append(torch.tensor(Quaternion(cs["rotation"]).rotation_matrix, dtype=torch.float32))
            trans.append(torch.tensor(cs["translation"], dtype=torch.float32))

        intrins = adjust_intrinsics(torch.stack(intrins), self.scale, crop_top=self.crop_top)

        # Generar máscara GT
        ep = self.ego_poses[by_cam["CAM_FRONT"]["ego_pose_token"]]
        ego_trans = np.array(ep["translation"], dtype=np.float64)
        ego_rot_inv = Quaternion(ep["rotation"]).rotation_matrix.T

        canvas = Image.new("L", (self.ny, self.nx), 0)
        draw = ImageDraw.Draw(canvas)
        for ann in self.anns_by_sample[token]:
            w, l, _ = ann["size"]
            corners_local = np.array([
                [ l / 2,  w / 2, 0.0],
                [ l / 2, -w / 2, 0.0],
                [-l / 2, -w / 2, 0.0],
                [-l / 2,  w / 2, 0.0],
            ], dtype=np.float64)
            box_rot = Quaternion(ann["rotation"]).rotation_matrix
            box_trans = np.array(ann["translation"], dtype=np.float64)
            corners_ego = (ego_rot_inv @ ((box_rot @ corners_local.T).T + box_trans - ego_trans).T).T

            ix = np.floor((corners_ego[:, 0] - self.xbound[0]) / self.xbound[2]).astype(int)
            iy = np.floor((corners_ego[:, 1] - self.ybound[0]) / self.ybound[2]).astype(int)
            if np.all(ix < 0) or np.all(ix >= self.nx) or np.all(iy < 0) or np.all(iy >= self.ny):
                continue
            draw.polygon([(int(j), int(i)) for i, j in zip(ix, iy)], outline=1, fill=1)

        mask = torch.from_numpy(np.array(canvas, dtype=np.float32)).unsqueeze(0)
        return torch.stack(imgs), intrins, torch.stack(rots), torch.stack(trans), mask
