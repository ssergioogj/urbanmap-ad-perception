"""
Tests de Splat (splat.py) y de Lift + Splat juntos (lift_splat.py).

Ejecutar desde la raíz del proyecto:
    pytest tests/test_splat.py -v
"""
from pathlib import Path

import pytest
import torch

from lss_bev.models.geometry import adjust_intrinsics, create_frustum, frustum_to_ego
from lss_bev.models.lift_splat import LiftSplat
from lss_bev.models.splat import QuickCumsum, Splat, points_to_cells
from lss_bev.nusc_calib import CAMS, load_sample_cameras

ROOT = Path(__file__).resolve().parents[1]
DATAROOT = ROOT / "data"
HAS_MINI = (DATAROOT / "v1.0-mini").exists()

H, W, STRIDE = 128, 352, 16
SCALE, CROP_TOP = 0.22, 48
GRID = dict(xbound=(-50.0, 50.0, 0.5), ybound=(-50.0, 50.0, 0.5), zbound=(-10.0, 10.0, 20.0))


def make_points(xyz, feats):
    """Construye entradas de Splat a partir de una lista de puntos sueltos.

    xyz:   (P, 3) posiciones en metros
    feats: (P, C) características
    Devuelve features (1, 1, P, C, 1, 1) y geom (1, 1, P, 1, 1, 3): cada punto
    se trata como una "profundidad" distinta de un único píxel.
    """
    xyz, feats = torch.as_tensor(xyz, dtype=torch.float32), torch.as_tensor(feats, dtype=torch.float32)
    P, C = feats.shape
    return feats.view(1, 1, P, C, 1, 1), xyz.view(1, 1, P, 1, 1, 3)


def cell_of(x, y):
    return int((x + 50) // 0.5), int((y + 50) // 0.5)


@pytest.fixture(params=[True, False], ids=["cumsum", "index_add"])
def splat(request):
    return Splat(**GRID, use_cumsum=request.param)


# --------------------------------------------------------------------------
# 1. La rejilla
# --------------------------------------------------------------------------
def test_grid_is_200x200x1():
    s = Splat(**GRID)
    assert (s.nx, s.ny, s.nz) == (200, 200, 1)


def test_points_to_cells_rounds_down_negatives():
    """-50.2 m está fuera del mapa: debe dar índice -1, no 0."""
    cells = points_to_cells(torch.tensor([[-50.2, -49.9, 0.0]]), **GRID)
    assert cells.tolist() == [[-1, 0, 0]]


# --------------------------------------------------------------------------
# 2. Casos pequeños calculables a mano
# --------------------------------------------------------------------------
def test_single_point_lands_in_its_cell(splat):
    f, g = make_points([[10.2, -3.7, 1.0]], [[1.0, 2.0, 3.0]])
    bev = splat(f, g)
    assert bev.shape == (1, 3, 200, 200)
    i, j = cell_of(10.2, -3.7)                       # (120, 92)
    assert (i, j) == (120, 92)
    assert bev[0, :, i, j].tolist() == [1.0, 2.0, 3.0]
    assert bev.abs().sum() == 6.0                    # en ningún otro sitio hay nada


def test_points_in_same_cell_are_summed(splat):
    f, g = make_points([[10.1, 5.1, 0.0], [10.4, 5.4, 2.0], [-20.0, 0.0, 0.0]],
                       [[1.0], [2.0], [5.0]])
    bev = splat(f, g)
    assert bev[(0, 0, *cell_of(10.1, 5.1))] == 3.0     # 1 + 2: misma casilla de 0.5 m
    assert bev[(0, 0, *cell_of(-20.0, 0.0))] == 5.0


def test_height_does_not_matter_pillars(splat):
    """Pilares: dos puntos a distinta altura sobre la misma casilla se suman."""
    f, g = make_points([[3.0, 3.0, -2.0], [3.0, 3.0, 4.0]], [[1.0], [1.0]])
    assert splat(f, g)[(0, 0, *cell_of(3.0, 3.0))] == 2.0


def test_points_outside_are_dropped(splat):
    f, g = make_points([[60.0, 0.0, 0.0], [0.0, -50.2, 0.0], [0.0, 0.0, 15.0], [49.9, 49.9, 0.0]],
                       [[1.0], [1.0], [1.0], [7.0]])
    bev = splat(f, g)
    assert bev.sum() == 7.0                          # solo sobrevive el último
    assert bev[0, 0, 199, 199] == 7.0


def test_orientation():
    """i crece hacia delante y j hacia la izquierda (convención documentada)."""
    s = Splat(**GRID)
    f, g = make_points([[40.0, 0.0, 0.0], [0.0, 30.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])
    bev = s(f, g)
    i_front, _ = torch.nonzero(bev[0, 0], as_tuple=True)
    _, j_left = torch.nonzero(bev[0, 1], as_tuple=True)
    assert i_front.item() > 100 and j_left.item() > 100


def test_batches_do_not_mix(splat):
    f1, g1 = make_points([[1.0, 1.0, 0.0]], [[1.0]])
    f2, g2 = make_points([[-30.0, 20.0, 0.0]], [[2.0]])
    bev = splat(torch.cat([f1, f2]), torch.cat([g1, g2]))
    assert bev[0].sum() == 1.0 and bev[(0, 0, *cell_of(1.0, 1.0))] == 1.0
    assert bev[1].sum() == 2.0 and bev[(1, 0, *cell_of(-30.0, 20.0))] == 2.0


# --------------------------------------------------------------------------
# 3. El truco cumsum del paper = la suma directa (valores y gradientes)
# --------------------------------------------------------------------------
def random_cloud(seed=0, B=2, N=6, D=41, C=8):
    torch.manual_seed(seed)
    feats = torch.randn(B, N, D, C, H // STRIDE, W // STRIDE)
    geom = torch.rand(B, N, D, H // STRIDE, W // STRIDE, 3) * 140 - 70   # parte cae fuera
    geom[..., 0:2] = geom[..., 0:2].round(decimals=0)                     # fuerza casillas repetidas
    return feats, geom


def test_cumsum_equals_index_add():
    feats, geom = random_cloud()
    a = Splat(**GRID, use_cumsum=True)(feats, geom)
    b = Splat(**GRID, use_cumsum=False)(feats, geom)
    assert torch.allclose(a, b, atol=1e-4)


def test_cumsum_gradient_equals_index_add():
    feats, geom = random_cloud(1)
    weights = torch.randn(2, 8, 200, 200)
    grads = []
    for use_cumsum in (True, False):
        f = feats.clone().requires_grad_(True)
        (Splat(**GRID, use_cumsum=use_cumsum)(f, geom) * weights).sum().backward()
        grads.append(f.grad)
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_quick_cumsum_gradcheck():
    """Comprueba numéricamente el gradiente analítico del truco cumsum."""
    feats = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    ranks = torch.tensor([2, 2, 5, 5, 5, 8, 9])
    assert torch.autograd.gradcheck(lambda f: QuickCumsum.apply(f, ranks)[0], (feats,))


def test_sum_is_conserved():
    """Splat no crea ni pierde información: suma de salida = suma de lo que cae dentro."""
    feats, geom = random_cloud(2)
    bev = Splat(**GRID)(feats, geom)
    cells = points_to_cells(geom, **GRID)
    inside = ((cells[..., 0] >= 0) & (cells[..., 0] < 200) & (cells[..., 1] >= 0)
              & (cells[..., 1] < 200) & (cells[..., 2] == 0))
    expected = (feats.permute(0, 1, 2, 4, 5, 3)[inside]).sum()
    assert torch.allclose(bev.sum(), expected, rtol=1e-4)


# --------------------------------------------------------------------------
# 4. Con la calibración real de nuScenes mini
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_geom():
    cal = load_sample_cameras(DATAROOT)
    K = adjust_intrinsics(cal["intrins"], SCALE, crop_top=CROP_TOP)
    fr = create_frustum((H, W), STRIDE)
    return cal, K, frustum_to_ego(fr, K[None], cal["rots"][None], cal["trans"][None])


@pytest.mark.skipif(not HAS_MINI, reason="sin nuScenes mini")
def test_each_camera_fills_its_side_of_the_map(real_geom):
    """Damos a cada punto una característica que dice de qué cámara viene
    (canal n = 1 si viene de la cámara n) y miramos dónde acaba cada canal."""
    _, _, geom = real_geom
    D, Hf, Wf = geom.shape[2:5]
    feats = torch.zeros(1, 6, D, 6, Hf, Wf)
    for n in range(6):
        feats[:, n, :, n] = 1.0
    bev = Splat(**GRID)(feats, geom)[0]              # (6, 200, 200)

    ii, jj = torch.meshgrid(torch.arange(200.), torch.arange(200.), indexing="ij")
    x, y = ii * 0.5 - 50 + 0.25, jj * 0.5 - 50 + 0.25  # centro de cada casilla en metros
    cx = {cam: (bev[n] * x).sum() / bev[n].sum() for n, cam in enumerate(CAMS)}
    cy = {cam: (bev[n] * y).sum() / bev[n].sum() for n, cam in enumerate(CAMS)}
    assert cx["CAM_FRONT"] > 10 and abs(cy["CAM_FRONT"]) < 5
    assert cx["CAM_BACK"] < -10 and abs(cy["CAM_BACK"]) < 5
    assert cy["CAM_FRONT_LEFT"] > 5 and cy["CAM_BACK_LEFT"] > 5
    assert cy["CAM_FRONT_RIGHT"] < -5 and cy["CAM_BACK_RIGHT"] < -5


@pytest.mark.skipif(not HAS_MINI, reason="sin nuScenes mini")
def test_lift_splat_end_to_end(real_geom):
    """Modelo completo Lift + Splat: forma de salida y gradiente hasta el backbone."""
    cal, K, _ = real_geom
    torch.manual_seed(0)
    model = LiftSplat(pretrained=False)
    imgs = torch.randn(1, 6, 3, H, W)
    bev = model(imgs, K[None], cal["rots"][None], cal["trans"][None])
    assert bev.shape == (1, 1, 200, 200)
    assert torch.isfinite(bev).all()
    bev.pow(2).mean().backward()
    g = model.cam_encode.image_backbone.conv_stem.weight.grad
    assert g is not None and g.abs().sum() > 0


@pytest.mark.skipif(not HAS_MINI, reason="sin nuScenes mini")
def test_one_hot_depth_puts_feature_at_that_distance(real_geom):
    """Si toda la probabilidad está en 20 m, la información de CAM_FRONT debe caer
    en un arco a ~20 m delante del coche (como en el test one-hot de Lift)."""
    _, _, geom = real_geom
    D, Hf, Wf = geom.shape[2:5]
    feats = torch.zeros(1, 6, D, 1, Hf, Wf)
    d20 = 16                                           # índice de 20 m
    feats[0, CAMS.index("CAM_FRONT"), d20] = 1.0       # solo CAM_FRONT, solo 20 m
    bev = Splat(**GRID)(feats, geom)[0, 0]
    ii, jj = torch.nonzero(bev, as_tuple=True)
    x = ii * 0.5 - 50 + 0.25
    assert x.min() > 18 and x.max() < 23               # 20 m + 1.7 m de la cámara, ± bordes
