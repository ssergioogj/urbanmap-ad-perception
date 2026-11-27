"""Tests for the IoU metric (metrics.py)."""
import torch

from lss_bev.metrics import IoUMeter
from lss_bev.metrics import SegmentationMeter

BIG = 20.0   # logit whose sigmoid is ~1


def test_perfect_prediction():
    label = torch.zeros(1, 1, 4, 4)
    label[..., :2, :2] = 1
    m = IoUMeter()
    m.update(label * 2 * BIG - BIG, label)
    assert m.compute() == 1.0


def test_hand_computed_iou():
    """Prediction 2 cells, label 3 cells, 1 in common -> IoU = 1 / 4."""
    label = torch.zeros(1, 1, 1, 5)
    label[..., [0, 1, 2]] = 1
    logits = torch.full((1, 1, 1, 5), -BIG)
    logits[..., [2, 3]] = BIG
    m = IoUMeter()
    m.update(logits, label)
    assert m.compute() == 0.25


def test_accumulates_over_batches_not_mean():
    """Sum of intersections / sum of unions, not the mean of per-batch IoUs."""
    m = IoUMeter()
    l1 = torch.ones(1, 1, 1, 4)                       # 4 cells, all predicted -> 4/4
    m.update(torch.full_like(l1, BIG), l1)
    l2 = torch.zeros(1, 1, 1, 4); l2[..., 0] = 1      # 1 cell, predicted 4 -> 1/4
    m.update(torch.full_like(l2, BIG), l2)
    assert m.compute() == (4 + 1) / (4 + 4)           # 0.625, not (1 + 0.25) / 2


def test_empty_is_zero():
    m = IoUMeter()
    m.update(torch.full((1, 1, 2, 2), -BIG), torch.zeros(1, 1, 2, 2))
    assert m.compute() == 0.0


# SegmentationMeter: precision, recall and IoU by distance
def test_precision_recall_hand_computed():
    """Same example as above: TP 1, FP 1, FN 2."""
    m = SegmentationMeter(bands=())
    label = torch.zeros(1, 1, 200, 200)
    label[..., 0, [0, 1, 2]] = 1
    logits = torch.full((1, 1, 200, 200), -BIG)
    logits[..., 0, [2, 3]] = BIG
    m.update(logits, label)
    r = m.compute()
    assert r["iou"] == 0.25 and r["precision"] == 0.5 and r["recall"] == 1 / 3


def test_iou_by_distance():
    """A perfect prediction near the car and a missed vehicle far away."""
    m = SegmentationMeter(bands=((0, 10), (30, 50)))
    label = torch.zeros(1, 1, 200, 200)
    label[..., 110:114, 100:104] = 1        # ~6 m ahead
    label[..., 180:184, 100:104] = 1        # ~41 m ahead
    logits = torch.full_like(label, -BIG)
    logits[..., 110:114, 100:104] = BIG     # only the near one is found
    m.update(logits, label)
    r = m.compute()
    assert r["iou_0-10m"] == 1.0 and r["iou_30-50m"] == 0.0
    assert r["iou"] == 0.5


def test_accuracy_is_misleading():
    """A model that predicts nothing gets a high accuracy but IoU 0."""
    m = SegmentationMeter(bands=())
    label = torch.zeros(1, 1, 200, 200)
    label[..., 100:110, 100:110] = 1                   # 100 of 40000 cells
    m.update(torch.full_like(label, -BIG), label)
    r = m.compute()
    assert r["accuracy"] == 1 - 100 / 40000 and r["iou"] == 0.0
    assert r["empty_model_accuracy"] == r["accuracy"]
