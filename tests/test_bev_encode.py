"""
Tests for the bird's-eye-view CNN (bev_encode.py).
Run from the project root:
    pytest tests/test_bev_encode.py -v
"""
import torch
import torch.nn.functional as F

from lss_bev.models.bev_encode import BevEncode


def intermediate_shapes(m, x):
    h = m.relu(m.bn1(m.conv1(x)))
    x1 = m.layer1(h)
    x2 = m.layer2(x1)
    x3 = m.layer3(x2)
    return x1.shape, x2.shape, x3.shape


def test_output_shape():
    m = BevEncode().eval()
    with torch.no_grad():
        assert m(torch.randn(2, 64, 200, 200)).shape == (2, 1, 200, 200)


def test_output_shape_several_classes():
    m = BevEncode(out_channels=2).eval()
    with torch.no_grad():
        assert m(torch.randn(1, 64, 200, 200)).shape == (1, 2, 200, 200)


def test_matches_paper_architecture():
    """Paper 4.1: conv 7x7 stride 2 + BN + ReLU, then the first 3 ResNet-18
    meta-layers -> x1, x2, x3 at three resolutions."""
    m = BevEncode().eval()
    with torch.no_grad():
        s1, s2, s3 = intermediate_shapes(m, torch.randn(1, 64, 200, 200))
    assert s1 == (1, 64, 100, 100)
    assert s2 == (1, 128, 50, 50)
    assert s3 == (1, 256, 25, 25)


def test_gradients_reach_every_layer():
    m = BevEncode().train()
    m(torch.randn(2, 64, 200, 200)).pow(2).mean().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, missing


def test_spreads_information_to_neighbours():
    """Splat leaves gaps between the camera rays: the BEV CNN must be able to
    move information from a filled cell to nearby empty cells."""
    torch.manual_seed(0)
    m = BevEncode().eval()
    x = torch.zeros(1, 64, 200, 200)
    with torch.no_grad():
        base = m(x)
        x[0, :, 100, 100] = 5.0                       # one single filled cell
        out = m(x)
    changed = (out - base).abs()[0, 0] > 1e-6
    assert changed[100, 108] and changed[92, 100]     # 4 m away still affected
    assert not changed[100, 190]                      # far away: untouched


def test_can_learn_to_fill_gaps():
    """Overfit test: the input only has sparse dots inside two "vehicles" (as
    after Splat); the target is the full vehicle. The CNN must learn to fill them."""
    torch.manual_seed(1)
    S = 96                                            # smaller map to keep the test fast
    target = torch.zeros(1, 1, S, S)
    target[..., 20:30, 40:50] = 1.0
    target[..., 60:68, 15:31] = 1.0
    x = torch.zeros(1, 64, S, S)
    x[:, :8, ::3, ::3] = target[:, :, ::3, ::3] * torch.randn(8, 1, 1).abs()

    m = BevEncode().train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(80):
        loss = F.binary_cross_entropy_with_logits(m(x), target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    m.eval()
    with torch.no_grad():
        pred = m(x).sigmoid() > 0.5
    gt = target.bool()
    iou = (pred & gt).sum() / (pred | gt).sum()
    assert iou > 0.7, f"IoU {iou:.2f}"
