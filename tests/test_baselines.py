"""
Tests for the paper baselines: CNN without 3D structure and
Lift-Splat with a frozen image backbone.
Run from the project root:
    pytest tests/test_baselines.py -v
"""
import torch
import torch.nn.functional as F

from lss_bev.models.baseline import CNNBaseline
from lss_bev.models.lift_splat import LiftSplat

H, W = 128, 352


def fake_batch(B=1, seed=0):
    torch.manual_seed(seed)
    K = torch.tensor([[275.6, 0, 176.0], [0, 275.6, 55.0], [0, 0, 1.0]]).expand(B, 6, 3, 3).clone()
    R = torch.tensor([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]).expand(B, 6, 3, 3).clone()
    t = torch.tensor([1.7, 0.0, 1.5]).expand(B, 6, 3).clone()
    return torch.randn(B, 6, 3, H, W), K, R, t


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# CNN baseline
def test_cnn_output_shape():
    m = CNNBaseline(pretrained=False).eval()
    with torch.no_grad():
        assert m(*fake_batch(2)).shape == (2, 1, 200, 200)


def test_cnn_has_roughly_same_params_as_lss():
    """Paper: 'roughly the same number of parameters as our model'."""
    a, b = n_params(CNNBaseline(pretrained=False)), n_params(LiftSplat(pretrained=False))
    assert abs(a - b) / b < 0.05


def test_cnn_ignores_calibration_but_lss_does_not():
    """The baseline has no 3D structure: changing K, R, t changes nothing.
    LSS uses them, so its output must change."""
    imgs, K, R, t = fake_batch()
    t2 = t + torch.tensor([5.0, 3.0, 0.0])               # move every camera
    cnn = CNNBaseline(pretrained=False).eval()
    lss = LiftSplat(pretrained=False).eval()
    with torch.no_grad():
        assert torch.equal(cnn(imgs, K, R, t), cnn(imgs, K, R, t2))
        assert not torch.allclose(lss(imgs, K, R, t), lss(imgs, K, R, t2))


def test_cnn_trains():
    m = CNNBaseline(pretrained=False).train()
    imgs, K, R, t = fake_batch(2)
    F.binary_cross_entropy_with_logits(m(imgs, K, R, t), torch.zeros(2, 1, 200, 200)).backward()
    assert m.image_backbone.conv_stem.weight.grad is not None
    assert m.bev_encode.conv1.weight.grad is not None


# Frozen encoder
def test_frozen_backbone_does_not_change():
    torch.manual_seed(1)
    m = LiftSplat(pretrained=False, freeze_backbone=True).train()
    backbone_before = {k: v.clone() for k, v in m.cam_encode.image_backbone.state_dict().items()}
    head_before = m.cam_encode.depth_context_conv.weight.clone()

    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    for seed in range(2):
        imgs, K, R, t = fake_batch(2, seed)
        loss = F.binary_cross_entropy_with_logits(m(imgs, K, R, t), torch.ones(2, 1, 200, 200))
        opt.zero_grad()
        loss.backward()
        opt.step()

    # weights and BatchNorm running statistics of the backbone are untouched
    for k, v in m.cam_encode.image_backbone.state_dict().items():
        assert torch.equal(v, backbone_before[k]), k
    # the rest of the network does learn
    assert not torch.equal(m.cam_encode.depth_context_conv.weight, head_before)


def test_frozen_only_backbone_is_frozen():
    m = LiftSplat(pretrained=False, freeze_backbone=True)
    assert not any(p.requires_grad for p in m.cam_encode.image_backbone.parameters())
    assert all(p.requires_grad for p in m.cam_encode.neck.parameters())
    assert all(p.requires_grad for p in m.bev_encode.parameters())
    m.train()
    assert not m.cam_encode.image_backbone.training and m.bev_encode.training
