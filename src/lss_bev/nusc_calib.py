from pyquaternion import Quaternion
from pathlib import Path
import json
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
