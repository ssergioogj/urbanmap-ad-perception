from pathlib import Path
import json

from PIL import Image, ImageDraw
from pyquaternion import Quaternion
import numpy as np
import torch

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]


def load_sample_cameras(dataroot, version="v1.0-mini", sample_index=0, cams=CAMS):
    """Returns, for a sample, image's route and calibration

    Returns:
        dict con:
            paths:   6 image's routes
            intrins: (N, 3, 3) instrisec
            rots:    (N, 3, 3) camera rotation R -> vehicle
            trans:   (N, 3)    camera translation -> vehicle
            sample_token
    """
    root = Path(dataroot)
    meta = root / version
    sample = json.loads((meta / "sample.json").read_text())[sample_index]
    calib = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}

    by_cam = {}
    for sd in json.loads((meta / "sample_data.json").read_text()):
        if sd["sample_token"] == sample["token"] and sd["is_key_frame"]:
            cam = sd["filename"].split("/")[1]
            if cam in cams:
                by_cam[cam] = sd

    paths, intrins, rots, trans = [], [], [], []
    for cam in cams:
        sd = by_cam[cam]
        cs = calib[sd["calibrated_sensor_token"]]
        paths.append(root / sd["filename"])
        intrins.append(torch.tensor(cs["camera_intrinsic"], dtype=torch.float32))
        rots.append(torch.tensor(Quaternion(cs["rotation"]).rotation_matrix, dtype=torch.float32))
        trans.append(torch.tensor(cs["translation"], dtype=torch.float32))

    return {
        "paths": paths,
        "intrins": torch.stack(intrins),
        "rots": torch.stack(rots),
        "trans": torch.stack(trans),
        "sample_token": sample["token"],
    }


def load_bev_vehicle_mask(
    dataroot,
    version="v1.0-mini",
    sample_index=0,
    xbound=(-50.0, 50.0, 0.5),
    ybound=(-50.0, 50.0, 0.5),
    category_prefix="vehicle.",
):
    """Generates a binary BEV vehicle mask (1, nx, ny) matching Splat's grid convention."""
    root = Path(dataroot)
    meta = root / version
    sample = json.loads((meta / "sample.json").read_text())[sample_index]
    sample_token = sample["token"]

    ego_pose_token = next(
        sd["ego_pose_token"]
        for sd in json.loads((meta / "sample_data.json").read_text())
        if sd["sample_token"] == sample_token and sd["is_key_frame"] and "CAM_FRONT/" in sd["filename"]
    )
    ep = next(ep for ep in json.loads((meta / "ego_pose.json").read_text()) if ep["token"] == ego_pose_token)
    ego_trans = np.array(ep["translation"], dtype=np.float64)
    ego_rot_inv = Quaternion(ep["rotation"]).rotation_matrix.T

    categories = {c["token"]: c["name"] for c in json.loads((meta / "category.json").read_text())}
    instances = {i["token"]: categories[i["category_token"]] for i in json.loads((meta / "instance.json").read_text())}

    nx = int(round((xbound[1] - xbound[0]) / xbound[2]))
    ny = int(round((ybound[1] - ybound[0]) / ybound[2]))
    canvas = Image.new("L", (ny, nx), 0)
    draw = ImageDraw.Draw(canvas)

    for ann in json.loads((meta / "sample_annotation.json").read_text()):
        if ann["sample_token"] != sample_token or not instances[ann["instance_token"]].startswith(category_prefix):
            continue

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

        ix = np.floor((corners_ego[:, 0] - xbound[0]) / xbound[2]).astype(int)
        iy = np.floor((corners_ego[:, 1] - ybound[0]) / ybound[2]).astype(int)

        if np.all(ix < 0) or np.all(ix >= nx) or np.all(iy < 0) or np.all(iy >= ny):
            continue

        draw.polygon([(int(j), int(i)) for i, j in zip(ix, iy)], outline=1, fill=1)

    return torch.from_numpy(np.array(canvas, dtype=np.float32)).unsqueeze(0)