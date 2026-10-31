import numpy as np
import torch

from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from PIL import Image

class NuscenesDataset(Dataset):
    def __init__(self, nusc: NuScenes):
        self.__nusc = nusc
        self.__samples = self.__nusc.sample
        self.__lss_img_dim = (128, 352) # Img dim for LSS
        self.__CAMS = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
        ]

    def __len__(self):
        return len(self.__samples)

    def __getitem__(self, idx):
        sample = self.__samples[idx]

        imgs = []
        rots = []
        trans = []
        intrinsics = []

        for cam in self.__CAMS:
            cam_token = sample['data'][cam]
            cam_record = self.__nusc.get('sample_data', cam_token)
            calib = self.__nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])

            # Image and adjust K
            img_path = self.__nusc.get_sample_data_path(cam_token)
            # Intrinsic matrix
            K_orig = np.array(calib['camera_intrinsic'], dtype=np.float32)
            # K_adj is K (intrinsic) after resizing the image
            img_tensor, K_adj = self.__get_img_intrinsics(img_path, K_orig)

            # Extrinsic matrix (rotation and translation from the cam POV to the position of the car)
            r = Quaternion(calib['rotation']).rotation_matrix.astype(np.float32) # From quaternion to 3x3 matrix
            t = np.array(calib['translation'], dtype=np.float32)

            imgs.append(img_tensor)
            intrinsics.append(K_adj)
            rots.append(torch.from_numpy(r))
            trans.append(torch.from_numpy(t))

        return (
            torch.stack(imgs),
            torch.stack(rots),
            torch.stack(trans),
            torch.stack(intrinsics)
        )

    def __get_img_intrinsics(self, img_path, K_orig):
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        target_w, target_h = self.__lss_img_dim

        # Scaling factor
        scale = target_w / orig_w
        resize_dim = (int(orig_w * scale), int(orig_h * scale))
        img.resize(resize_dim, Image.BILINEAR) # Computes the new pixel color as the weighted mean of the 4 nearest neighbours

        # Crop height
        crop_y = (resize_dim[1] - target_h) // 2
        crop_x = 0
        img = img.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))

        # Tensor
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 # OpenCV/PIl img format = (H, W, C), Conv2D format = (C, H, W)

        # Stats from ImageNet (for pretrained Resnet-50 or EfficientNet as backbone)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Adjust intrinsic matrix K
        # u' = u * scale - crop_x
        # v' = v * scale - crop_y
        K = np.copy(K_orig)
        K[0, 0] *= scale        # fx
        K[1, 1] *= scale        # fy
        K[0, 2] = K[0, 2] * scale - crop_x  # cx
        K[1, 2] = K[1, 2] * scale - crop_y  # cy

        return img_tensor, torch.from_numpy(K).float()


        


