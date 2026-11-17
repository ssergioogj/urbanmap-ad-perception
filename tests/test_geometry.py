"""
Tests de la parte geométrica de Lift (geometry.py).

Ejecutar desde la raíz del proyecto:
    pytest tests/test_geometry.py -v
"""
import json
import math
from pathlib import Path

import pytest
import torch
from pyquaternion import Quaternion

from lss_bev.models.cam_encode import CamEncode
from lss_bev.models.geometry import adjust_intrinsics, create_frustum, frustum_to_ego
from lss_bev.nusc_calib import CAMS, load_sample_cameras

ROOT = Path(__file__).resolve().parents[1]
DATAROOT = ROOT / "data"
HAS_MINI = (DATAROOT / "v1.0-mini").exists()

H, W, STRIDE = 128, 352, 16
# Preprocesado de ejemplo: 1600x900 -> x0.22 -> 352x198 -> recorte de 48 filas arriba -> 352x128
SCALE, CROP_TOP = 0.22, 48

# Rotación de una cámara frontal "ideal": z de la cámara -> x del vehículo,
# x de la cámara (derecha) -> -y del vehículo, y de la cámara (abajo) -> -z del vehículo
R_FRONT = torch.tensor([[0., 0., 1.],
                        [-1., 0., 0.],
                        [0., -1., 0.]])


def simple_K(f=100.0, cx=176.0, cy=64.0):
    return torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def one_point_frustum(u, v, d):
    """Frustum con un único punto, para probar posiciones concretas."""
    return torch.tensor([u, v, d], dtype=torch.float32).view(1, 1, 1, 3)


def to_ego(frustum, K, R=torch.eye(3), t=torch.zeros(3)):
    return frustum_to_ego(frustum, K.view(1, 1, 3, 3), R.view(1, 1, 3, 3), t.view(1, 1, 3))


# --------------------------------------------------------------------------
# 1. La rejilla del frustum
# --------------------------------------------------------------------------
def test_frustum_shape_and_depths():
    fr = create_frustum((H, W), STRIDE, (4.0, 45.0, 1.0))
    assert fr.shape == (41, 8, 22, 3)
    depths = fr[:, 0, 0, 2]
    assert torch.equal(depths, torch.arange(4.0, 45.0, 1.0))   # 4, 5, ..., 44


def test_frustum_pixel_centers():
    fr = create_frustum((H, W), STRIDE)
    us, vs = fr[0, 0, :, 0], fr[0, :, 0, 1]
    assert us[0] == 7.5 and us[-1] == W - 8.5          # centro de la primera y última celda
    assert vs[0] == 7.5 and vs[-1] == H - 8.5
    assert torch.allclose(us.diff(), torch.full_like(us.diff(), STRIDE))


def test_frustum_matches_camencode_output():
    """La rejilla tiene que tener el mismo tamaño que lo que sale de Lift."""
    model = CamEncode(pretrained=False).eval()
    with torch.no_grad():
        _, depth = model(torch.randn(1, 1, 3, H, W))
    fr = create_frustum((H, W), STRIDE)
    assert depth.shape[2:] == fr.shape[:3]                # (D, Hf, Wf)


# --------------------------------------------------------------------------
# 2. Cámara simple (sin rotación ni traslación): de píxel a cámara
# --------------------------------------------------------------------------
def test_principal_point_goes_straight_ahead():
    """El centro óptico a profundidad 10 está justo delante: (0, 0, 10)."""
    p = to_ego(one_point_frustum(176.0, 64.0, 10.0), simple_K())
    assert torch.allclose(p.view(3), torch.tensor([0.0, 0.0, 10.0]), atol=1e-5)


def test_pixel_offsets():
    """100 px a la derecha con f=100 y d=10 -> 10 m a la derecha (x de la cámara)."""
    p = to_ego(one_point_frustum(276.0, 64.0, 10.0), simple_K()).view(3)
    assert torch.allclose(p, torch.tensor([10.0, 0.0, 10.0]), atol=1e-5)
    p = to_ego(one_point_frustum(176.0, 114.0, 10.0), simple_K()).view(3)
    assert torch.allclose(p, torch.tensor([0.0, 5.0, 10.0]), atol=1e-5)     # abajo = y positiva


def test_depth_is_z_not_euclidean_distance():
    """d es la distancia a lo largo del eje óptico (z), no la distancia en línea
    recta: todos los puntos del bin d=10 tienen z_cam = 10 exacto."""
    fr = create_frustum((H, W), STRIDE)
    geom = to_ego(fr, simple_K())[0, 0]                   # (D, Hf, Wf, 3)
    assert torch.allclose(geom[..., 2], fr[..., 2])


def test_roundtrip_project_and_lift():
    """Proyectar puntos 3D a la imagen y volver a subirlos da los mismos puntos."""
    torch.manual_seed(0)
    K = simple_K(f=150.0)
    pts = torch.rand(50, 3) * torch.tensor([20, 10, 40]) + torch.tensor([-10, -5, 4])
    uvw = (K @ pts.T).T
    u, v, d = uvw[:, 0] / uvw[:, 2], uvw[:, 1] / uvw[:, 2], pts[:, 2]
    fr = torch.stack([u, v, d], -1).view(50, 1, 1, 3)
    back = to_ego(fr, K).view(50, 3)
    assert torch.allclose(back, pts, atol=1e-4)


# --------------------------------------------------------------------------
# 3. Extrínsecos: de cámara a vehículo
# --------------------------------------------------------------------------
def test_front_camera_axes():
    """Cámara frontal montada en (1.7, 0, 1.5) mirando hacia delante."""
    t = torch.tensor([1.7, 0.0, 1.5])
    K = simple_K()
    ahead = to_ego(one_point_frustum(176.0, 64.0, 10.0), K, R_FRONT, t).view(3)
    assert torch.allclose(ahead, torch.tensor([11.7, 0.0, 1.5]), atol=1e-5)
    right = to_ego(one_point_frustum(276.0, 64.0, 10.0), K, R_FRONT, t).view(3)
    assert right[1] < 0                       # derecha en la imagen = y negativa del coche
    down = to_ego(one_point_frustum(176.0, 114.0, 10.0), K, R_FRONT, t).view(3)
    assert down[2] < 1.5                      # abajo en la imagen = más cerca del suelo


def test_batch_and_cameras_shapes():
    fr = create_frustum((H, W), STRIDE)
    B, N = 2, 6
    K = simple_K().expand(B, N, 3, 3)
    R = torch.eye(3).expand(B, N, 3, 3)
    t = torch.randn(B, N, 3)
    geom = frustum_to_ego(fr, K, R, t)
    assert geom.shape == (B, N, 41, 8, 22, 3)
    # cada cámara usa su propia traslación
    assert torch.allclose(geom[1, 3] - geom[0, 0], t[1, 3] - t[0, 0], atol=1e-4)


# --------------------------------------------------------------------------
# 4. Ajuste de intrínsecos por redimensionado y recorte
# --------------------------------------------------------------------------
def test_adjust_intrinsics_matches_resizing_the_image():
    """Proyectar con K ajustada = proyectar con K original y luego escalar/recortar."""
    K = torch.tensor([[1266.4, 0, 816.3], [0, 1266.4, 491.5], [0, 0, 1.0]])
    Kn = adjust_intrinsics(K, SCALE, crop_top=CROP_TOP)
    p = torch.tensor([2.0, 0.5, 15.0])
    a = K @ p
    u, v = a[0] / a[2], a[1] / a[2]
    b = Kn @ p
    assert torch.allclose(b[0] / b[2], SCALE * u, atol=1e-3)
    assert torch.allclose(b[1] / b[2], SCALE * v - CROP_TOP, atol=1e-3)


# --------------------------------------------------------------------------
# 5. Calibración real de nuScenes mini
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real():
    cal = load_sample_cameras(DATAROOT)
    K = adjust_intrinsics(cal["intrins"], SCALE, crop_top=CROP_TOP)
    fr = create_frustum((H, W), STRIDE)
    geom = frustum_to_ego(fr, K[None], cal["rots"][None], cal["trans"][None])[0]
    return cal, K, fr, geom


@pytest.mark.skipif(not HAS_MINI, reason="sin nuScenes mini")
def test_each_camera_looks_where_it_should(real):
    """El centro de cada imagen a 20 m debe apuntar en la dirección de su cámara."""
    _, _, _, geom = real
    expected_deg = {"CAM_FRONT": 0, "CAM_FRONT_LEFT": 55, "CAM_FRONT_RIGHT": -55,
                    "CAM_BACK": 180, "CAM_BACK_LEFT": 110, "CAM_BACK_RIGHT": -110}
    d20 = 16                                            # índice de 20 m (20 - 4)
    for n, cam in enumerate(CAMS):
        x, y, _ = geom[n, d20, 4, 11]                 # celda central de la cuadrícula 8x22
        ang = math.degrees(math.atan2(y, x))
        diff = (ang - expected_deg[cam] + 180) % 360 - 180
        assert abs(diff) < 15, f"{cam}: {ang:.0f}º, esperado ~{expected_deg[cam]}º"


@pytest.mark.skipif(not HAS_MINI, reason="sin nuScenes mini")
def test_horizon_and_ground(real):
    """Con las cámaras a ~1.5 m, la fila de arriba apunta por encima del suelo y
    la de abajo hacia el suelo: a 20 m la z de abajo debe ser menor."""
    _, _, _, geom = real
    top, bottom = geom[1, 16, 0, 11, 2], geom[1, 16, -1, 11, 2]
    assert top > bottom
    assert bottom < 0.5


@pytest.mark.skipif(not HAS_MINI, reason="sin nuScenes mini")
def test_real_annotations_roundtrip(real):
    """Prueba independiente con datos reales: proyectamos los centros de las cajas
    anotadas a la cámara frontal (con la fórmula directa de nuScenes) y
    comprobamos que frustum_to_ego los devuelve a su posición."""
    cal, K, _, _ = real
    meta = DATAROOT / "v1.0-mini"
    anns = [a for a in json.loads((meta / "sample_annotation.json").read_text())
            if a["sample_token"] == cal["sample_token"]]
    sds = json.loads((meta / "sample_data.json").read_text())
    sd_front = next(s for s in sds if s["sample_token"] == cal["sample_token"]
                    and s["is_key_frame"] and "/CAM_FRONT/" in s["filename"])
    pose = next(p for p in json.loads((meta / "ego_pose.json").read_text())
                if p["token"] == sd_front["ego_pose_token"])

    q_ego, t_ego = Quaternion(pose["rotation"]), torch.tensor(pose["translation"])
    R_ego = torch.tensor(q_ego.rotation_matrix, dtype=torch.float32)
    n = CAMS.index("CAM_FRONT")
    R, t, Kf = cal["rots"][n], cal["trans"][n], K[n]

    checked = 0
    for a in anns:
        p_glob = torch.tensor(a["translation"], dtype=torch.float32)
        p_ego = R_ego.T @ (p_glob - t_ego)                 # global -> vehículo
        p_cam = R.T @ (p_ego - t)                          # vehículo -> cámara
        if p_cam[2] < 4:                                   # detrás o demasiado cerca
            continue
        uvw = Kf @ p_cam
        u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
        if not (0 <= u < W and 0 <= v < H):
            continue
        back = frustum_to_ego(one_point_frustum(u, v, p_cam[2]),
                              Kf.view(1, 1, 3, 3), R.view(1, 1, 3, 3), t.view(1, 1, 3)).view(3)
        assert torch.allclose(back, p_ego, atol=1e-2), a["token"]
        checked += 1
    assert checked >= 3, "pocas cajas visibles para la prueba"
