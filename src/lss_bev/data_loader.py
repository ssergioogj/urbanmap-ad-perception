import os

from nuscenes.nuscenes import NuScenes

DATAROOT = "./data"

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=True)

sample = nusc.sample[0]

cam_keys = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
]

for cam in cam_keys:
    cam_token = sample['data'][cam]
    
    img_path = nusc.get_sample_data_path(cam_token)
    print(f"[{cam}] -> {os.path.basename(img_path)} (Existe: {os.path.exists(img_path)})")


