"""
Ground-truth BEV labels: where the vehicles really are.
For one nuScenes sample we take the annotated 3D boxes of every vehicle, move
them to the ego (car) frame and paint their footprint (the box seen from above)
on the same 200x200 grid that Splat produces.
Grid convention:
    label[0, i, j] is the cell with
        x (forward) in [x_min + i*dx, x_min + (i+1)*dx)
        y (left)    in [y_min + j*dy, y_min + (j+1)*dy)
"""
import json
from pathlib import Path

import torch
from pyquaternion import Quaternion

from lss_bev.models.splat import grid_size

XBOUND = (-50.0, 50.0, 0.5)
YBOUND = (-50.0, 50.0, 0.5)


def yaw_of(rotation_matrix):
    """Heading angle (radians) of a 3x3 rotation, measured from the x axis."""
    return torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])


def box_footprint_mask(center_xy, length, width, yaw, xbound=XBOUND, ybound=YBOUND):
    """Cells of the grid covered by one box seen from above.
    A cell is marked if its CENTER is inside the rectangle.
    Args:
        center_xy: (x, y) of the box center in ego coordinates (meters).
        length:    size along the box heading (meters).
        width:     size across the box heading (meters).
        yaw:       box heading in the ego frame (radians, 0 = pointing forward).
    Returns:
        (nx, ny) bool tensor.
    """
    nx, ny, _ = grid_size(xbound, ybound, (0.0, 1.0, 1.0))
    xs = xbound[0] + (torch.arange(nx) + 0.5) * xbound[2]     # cell centers along x
    ys = ybound[0] + (torch.arange(ny) + 0.5) * ybound[2]     # cell centers along y
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")              # (nx, ny)

    # Express every cell center in the box frame: rotate by -yaw around the center
    dx, dy = gx - center_xy[0], gy - center_xy[1]
    c, s = torch.cos(torch.as_tensor(yaw)), torch.sin(torch.as_tensor(yaw))
    along = c * dx + s * dy           # coordinate along the box length
    across = -s * dx + c * dy         # coordinate along the box width
    return (along.abs() <= length / 2) & (across.abs() <= width / 2)


class VehicleLabels:
    """Builds vehicle labels for nuScenes samples. JSON files are read only once."""

    def __init__(self, dataroot, version="v1.0-mini", xbound=XBOUND, ybound=YBOUND):
        meta = Path(dataroot) / version
        load = lambda name: json.loads((meta / f"{name}.json").read_text())
        self.xbound, self.ybound = xbound, ybound

        categories = {c["token"]: c["name"] for c in load("category")}
        self.category_of = {i["token"]: categories[i["category_token"]] for i in load("instance")}
        self.samples = load("sample")
        self.ego_poses = {p["token"]: p for p in load("ego_pose")}

        # Annotations grouped by sample
        self.anns_by_sample = {}
        for a in load("sample_annotation"):
            self.anns_by_sample.setdefault(a["sample_token"], []).append(a)

        # Reference ego pose of each sample: the one of the LIDAR_TOP keyframe
        # As in the official LSS code, cameras are within a few ms of it.
        self.ego_pose_of_sample = {}
        for sd in load("sample_data"):
            if sd["is_key_frame"] and "/LIDAR_TOP/" in sd["filename"]:
                self.ego_pose_of_sample[sd["sample_token"]] = self.ego_poses[sd["ego_pose_token"]]

    def vehicle_boxes(self, sample_token):
        """Vehicle boxes of a sample in ego coordinates.
        Returns a list of dicts with center (x, y, z), length, width, height, yaw, category.
        """
        pose = self.ego_pose_of_sample[sample_token]
        R_ego = torch.tensor(Quaternion(pose["rotation"]).rotation_matrix, dtype=torch.float32)
        t_ego = torch.tensor(pose["translation"], dtype=torch.float32)

        boxes = []
        for a in self.anns_by_sample.get(sample_token, []):
            category = self.category_of[a["instance_token"]]
            if not category.startswith("vehicle."):
                continue
            # global -> ego: p_ego = R_ego^T (p_global - t_ego); same for the box rotation
            center = R_ego.T @ (torch.tensor(a["translation"], dtype=torch.float32) - t_ego)
            R_box = R_ego.T @ torch.tensor(Quaternion(a["rotation"]).rotation_matrix, dtype=torch.float32)
            width, length, height = a["size"]     # nuScenes stores size as (width, length, height)
            boxes.append(dict(center=center, length=length, width=width, height=height,
                              yaw=yaw_of(R_box), category=category))
        return boxes

    def __call__(self, sample_token):
        """Label map (1, nx, ny) float: 1 where there is a vehicle, 0 elsewhere."""
        nx, ny, _ = grid_size(self.xbound, self.ybound, (0.0, 1.0, 1.0))
        label = torch.zeros(nx, ny, dtype=torch.bool)
        for b in self.vehicle_boxes(sample_token):
            label |= box_footprint_mask(b["center"][:2], b["length"], b["width"], b["yaw"],
                                        self.xbound, self.ybound)
        return label.float().unsqueeze(0)
