"""
Tests de la fase Lift (CamEncode).

Ejecutar desde la raíz del proyecto:
    pytest tests/test_lift.py -v

Por defecto se usa el backbone SIN pesos preentrenados para no depender de
internet: la arquitectura y las dimensiones son idénticas. El test que carga
los pesos de ImageNet se salta si no hay conexión.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lss_bev.models.cam_encode import CamEncode

ROOT = Path(__file__).resolve().parents[1]
DATAROOT = ROOT / "data"

# Configuración del paper: imagen 128x352, profundidades de 4 a 45 m cada 1 m
H, W = 128, 352
D_BINS = 41
C = 64
STRIDE = 8  # out_indices=(2, 4) -> el frustum sale a resolución 1/8, cambiar a (3, 4) si queremos que salga como en el paper 1/16
CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = CamEncode(d_bins=D_BINS, c_chanels=C, pretrained=False)
    m.eval()
    return m


@pytest.fixture(scope="module")
def imgs():
    torch.manual_seed(1)
    return torch.randn(2, 6, 3, H, W)


@pytest.fixture(scope="module")
def outputs(model, imgs):
    with torch.no_grad():
        return model(imgs)


# --------------------------------------------------------------------------
# 1. Dimensiones
# --------------------------------------------------------------------------
def test_output_shapes(outputs):
    frustum, depth = outputs
    hf, wf = H // STRIDE, W // STRIDE
    assert depth.shape == (2, 6, D_BINS, hf, wf)
    assert frustum.shape == (2, 6, D_BINS, C, hf, wf)


def test_feature_stride_is_8(model):
    # Splat tendrá que construir el frustum con este mismo stride.
    assert model.image_backbone.feature_info.reduction() == [8, 32]


# --------------------------------------------------------------------------
# 2. Distribución de profundidad
# --------------------------------------------------------------------------
def test_depth_is_probability_distribution(outputs):
    _, depth = outputs
    assert torch.all(depth >= 0) and torch.all(depth <= 1)
    sums = depth.sum(dim=2)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_no_nans(outputs):
    frustum, depth = outputs
    assert torch.isfinite(frustum).all() and torch.isfinite(depth).all()


# --------------------------------------------------------------------------
# 3. Producto exterior (el núcleo de Lift)
# --------------------------------------------------------------------------
def test_outer_product(outputs):
    """frustum[d, c] debe ser exactamente depth[d] * context[c].

    Como las probabilidades suman 1, sumar el frustum sobre D recupera el
    vector de contexto; con él reconstruimos el producto y lo comparamos.
    """
    frustum, depth = outputs
    context = frustum.sum(dim=2)                          # (B, N, C, h, w)
    rebuilt = depth.unsqueeze(3) * context.unsqueeze(2)   # (B, N, D, C, h, w)
    assert torch.allclose(frustum, rebuilt, atol=1e-5)


def test_one_hot_depth_puts_all_features_in_one_bin(model):
    """Si la red está segura de la profundidad (one-hot), todo el contexto
    debe caer en ese bin y el resto debe ser cero (caso límite del paper)."""
    torch.manual_seed(2)
    x = torch.randn(1, 1, 3, H, W)
    conv = model.depth_context_conv
    old_b = conv.bias.data.clone()
    try:
        conv.bias.data[:D_BINS] = -1e4
        conv.bias.data[10] = 1e4      # fuerza todo al bin 10 (14 m)
        with torch.no_grad():
            frustum, depth = model(x)
    finally:
        conv.bias.data.copy_(old_b)
    assert torch.allclose(depth[:, :, 10], torch.ones_like(depth[:, :, 10]))
    others = torch.cat([frustum[:, :, :10], frustum[:, :, 11:]], dim=2)
    assert others.abs().max() < 1e-6
    assert frustum[:, :, 10].abs().sum() > 0


# --------------------------------------------------------------------------
# 4. Orden batch / cámaras (errores típicos de view/reshape)
# --------------------------------------------------------------------------
def test_cameras_are_independent(model, imgs, outputs):
    """Cambiar la imagen de una cámara no debe afectar a las demás."""
    frustum, _ = outputs
    x2 = imgs.clone()
    x2[0, 0] = torch.randn(3, H, W)
    with torch.no_grad():
        frustum2, _ = model(x2)
    assert not torch.allclose(frustum[0, 0], frustum2[0, 0])
    assert torch.allclose(frustum[0, 1:], frustum2[0, 1:], atol=1e-5)
    assert torch.allclose(frustum[1], frustum2[1], atol=1e-5)


def test_batch_matches_single_camera(model, imgs, outputs):
    """Procesar cada cámara por separado da lo mismo que procesarlas juntas."""
    frustum, depth = outputs
    with torch.no_grad():
        f, d = model(imgs[1:2, 4:5])
    assert torch.allclose(frustum[1:2, 4:5], f, atol=1e-5)
    assert torch.allclose(depth[1:2, 4:5], d, atol=1e-5)


# --------------------------------------------------------------------------
# 5. Entrenabilidad
# --------------------------------------------------------------------------
def test_gradients_reach_backbone_and_heads():
    """Todo el modelo debe recibir gradiente (el backbone se ajusta por
    fine-tuning y la neck y la cabeza se entrenan desde cero)."""
    torch.manual_seed(3)
    m = CamEncode(d_bins=D_BINS, c_chanels=C, pretrained=False).train()
    frustum, _ = m(torch.randn(1, 2, 3, 64, 176))
    frustum.pow(2).mean().backward()
    for name, module in [("backbone", m.image_backbone), ("neck", m.neck),
                         ("depth_context_conv", m.depth_context_conv)]:
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(g is not None for g in grads), name
        assert sum(g.abs().sum() for g in grads) > 0, name


def test_can_overfit_a_depth_target():
    """Prueba de sobreajuste: con una sola imagen, la red debe aprender a
    predecir una profundidad que depende de la fila del píxel (como en una
    carretera real: abajo cerca, arriba lejos)."""
    torch.manual_seed(4)
    m = CamEncode(d_bins=D_BINS, c_chanels=C, pretrained=False).train()
    x = torch.randn(1, 1, 3, 64, 176)
    hf, wf = 64 // STRIDE, 176 // STRIDE
    rows = torch.arange(hf).flip(0) * (D_BINS - 1) // (hf - 1)
    target = rows.view(1, hf, 1).expand(1, hf, wf)        # (1, h, w)

    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(60):
        _, depth = m(x)
        loss = F.nll_loss(torch.log(depth[:, 0] + 1e-8), target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    m.eval()
    with torch.no_grad():
        _, depth = m(x)
    acc = (depth[:, 0].argmax(1) == target).float().mean().item()
    assert acc > 0.9, f"precisión {acc:.2f}, loss {loss.item():.3f}"


# --------------------------------------------------------------------------
# 6. Datos reales (nuScenes mini) y pesos preentrenados
# --------------------------------------------------------------------------
def _load_first_sample_cams():
    meta = DATAROOT / "v1.0-mini"
    sample = json.loads((meta / "sample.json").read_text())[0]
    sds = json.loads((meta / "sample_data.json").read_text())
    files = {}
    for sd in sds:
        if sd["sample_token"] == sample["token"] and sd["is_key_frame"]:
            cam = sd["filename"].split("/")[1]
            if cam in CAMS:
                files[cam] = DATAROOT / sd["filename"]
    return [files[c] for c in CAMS]


@pytest.mark.skipif(not (DATAROOT / "v1.0-mini").exists(), reason="sin nuScenes mini")
def test_real_nuscenes_sample(model):
    from PIL import Image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    ims = []
    for f in _load_first_sample_cams():
        im = Image.open(f).convert("RGB").resize((W, H))
        t = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255
        ims.append((t - mean) / std)
    x = torch.stack(ims).unsqueeze(0)
    with torch.no_grad():
        frustum, depth = model(x)
    assert frustum.shape == (1, 6, D_BINS, C, H // STRIDE, W // STRIDE)
    assert torch.isfinite(frustum).all()


def test_pretrained_weights_load():
    try:
        m = CamEncode(pretrained=True)
    except Exception as e:  # sin internet / Hugging Face bloqueado
        pytest.skip(f"no se pudieron descargar los pesos: {type(e).__name__}")
    w = m.image_backbone.conv_stem.weight
    m2 = CamEncode(pretrained=False)
    assert not torch.allclose(w, m2.image_backbone.conv_stem.weight)
