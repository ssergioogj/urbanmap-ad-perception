"""Create a model by name and load trained checkpoints."""
import torch

from lss_bev.models.baseline import CNNBaseline
from lss_bev.models.lift_splat import LiftSplat

MODELS = ("lss", "cnn", "frozen")


def build_model(name, pretrained=True):
    """lss: Lift-Splat | cnn: CNN baseline without 3D | frozen: LSS with frozen backbone."""
    if name == "cnn":
        return CNNBaseline(pretrained=pretrained)
    if name in ("lss", "frozen"):
        return LiftSplat(pretrained=pretrained, freeze_backbone=name == "frozen")
    raise ValueError(f"unknown model {name!r}, expected one of {MODELS}")


def load_checkpoint(path, device="cpu"):
    """Rebuild the model saved by scripts/train.py. Returns (model, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    name = ckpt.get("args", {}).get("model", "lss")       # runs made before --model existed are LSS
    model = build_model(name, pretrained=False)            # weights come from the checkpoint
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
