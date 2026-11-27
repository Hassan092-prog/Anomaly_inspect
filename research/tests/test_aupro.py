"""Unit tests and Anomalib cross-checks for custom AUPRO metric."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from anomalib.metrics.aupro import _AUPRO as AnomalibAUPRO

from research.data.datasets import AnomalyImageDataset, load_dataset_samples
from research.metrics import compute_aupro, compute_pro_curve


def test_aupro_hand_computed_two_region_toy_case() -> None:
    """Verify AUPRO matches exact hand-computed values on a 2-region toy mask."""
    # Shape (1, 3, 5): 15 pixels total -> 5 defect pixels across 2 regions, 10 normal pixels.
    # Region 1 (row 0, cols 0..1): size 2, scores 0.9, 0.7
    # Region 2 (row 2, cols 2..4): size 3, scores 0.8, 0.6, 0.4
    # Normal row 1: one pixel 0.75, two tied pixels 0.5, remaining normal pixels 0.1.
    masks = np.zeros((1, 3, 5), dtype=np.uint8)
    masks[0, 0, 0:2] = 1
    masks[0, 2, 2:5] = 1

    anomaly_maps = np.full((1, 3, 5), 0.1, dtype=np.float64)
    anomaly_maps[0, 0, 0] = 0.9
    anomaly_maps[0, 0, 1] = 0.7
    anomaly_maps[0, 2, 2] = 0.8
    anomaly_maps[0, 2, 3] = 0.6
    anomaly_maps[0, 2, 4] = 0.4
    anomaly_maps[0, 1, 0] = 0.75
    anomaly_maps[0, 1, 1] = 0.5
    anomaly_maps[0, 1, 2] = 0.5

    # Exact curve:
    # (0.0, 0.0) -> (0.0, 0.25) -> (0.0, 5/12) -> (0.1, 5/12)
    # -> (0.1, 2/3) -> (0.1, 5/6) -> (0.3, 5/6) -> (0.3, 1.0)
    # At fpr_limit=0.3: area = 0.1*(5/12) + 0.2*(5/6) = 5/24 -> AUPRO = (5/24)/0.3 = 25/36.
    assert compute_aupro(masks, anomaly_maps, fpr_limit=0.3) == pytest.approx(25.0 / 36.0)

    # At fpr_limit=0.2 (interpolates inside [0.1, 0.3]):
    # area = 0.1*(5/12) + 0.1*(5/6) = 1/8 -> AUPRO = 0.125 / 0.2 = 0.625.
    assert compute_aupro(masks, anomaly_maps, fpr_limit=0.2) == pytest.approx(0.625)

    fpr, pro = compute_pro_curve(masks, anomaly_maps, fpr_limit=0.3)
    assert fpr[0] == pytest.approx(0.0)
    assert pro[0] == pytest.approx(0.0)
    assert fpr[-1] == pytest.approx(0.3)
    assert np.all(np.diff(fpr) >= 0.0)
    assert np.all(np.diff(pro) >= 0.0)


def test_aupro_normalization_monotonicity_and_predictors() -> None:
    """Verify perfect/inverted/constant predictors and non-decreasing behavior across fpr_limit."""
    # Include an all-normal image, an image with 2 regions, and diagonal connectivity.
    masks = np.zeros((3, 6, 6), dtype=np.uint8)
    masks[1, 1:3, 1:3] = 1
    masks[2, 0, 0] = 1
    masks[2, 1, 1] = 1  # diagonally touching (0, 0) -> single 8-connected region

    perfect_maps = np.full((3, 6, 6), 0.05, dtype=np.float64)
    perfect_maps[masks == 1] = 0.95
    assert compute_aupro(masks, perfect_maps, fpr_limit=0.3) == pytest.approx(1.0)
    assert compute_aupro(masks, perfect_maps, fpr_limit=0.1) == pytest.approx(1.0)

    # Inverted predictor places all normal pixels strictly ahead of all defect pixels.
    inverted_maps = 1.0 - perfect_maps
    assert compute_aupro(masks, inverted_maps, fpr_limit=0.3) == pytest.approx(0.0)

    # Constant predictor gives PRO(f) = f -> integral_0^L f df / L = L / 2.
    constant_maps = np.full((3, 6, 6), 0.42, dtype=np.float64)
    for limit in (0.1, 0.3, 0.5, 1.0):
        assert compute_aupro(masks, constant_maps, fpr_limit=limit) == pytest.approx(limit / 2.0)

    # Monotonic non-decreasing raw area and expected normalized behavior on random scores.
    rng = np.random.default_rng(42)
    rand_maps = rng.uniform(0.0, 1.0, size=masks.shape)
    limits = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    raw_areas: list[float] = []
    aupros: list[float] = []
    for limit in limits:
        val = compute_aupro(masks, rand_maps, fpr_limit=limit)
        aupros.append(val)
        raw_areas.append(val * limit)
    assert all(b >= a - 1e-12 for a, b in zip(raw_areas[:-1], raw_areas[1:], strict=True))
    assert all(b >= a - 1e-12 for a, b in zip(aupros[:-1], aupros[1:], strict=True))


def test_aupro_connectivity_mode_distinguishes_diagonal_touch() -> None:
    """8-connected merges diagonal defect pixels; 4-connected keeps them separate."""
    masks = np.zeros((1, 4, 4), dtype=np.uint8)
    # 1x2 horizontal bar + 1 pixel at (1, 2) touching (0, 1) diagonally.
    masks[0, 0, 0] = 1
    masks[0, 0, 1] = 1
    masks[0, 1, 2] = 1

    # Background pixels score 0.5 so (1, 2) is covered at FPR=0, (0, 0)/(0, 1) after bg.
    scores = np.full((1, 4, 4), 0.5, dtype=np.float64)
    scores[0, 1, 2] = 0.9
    scores[0, 0, 0] = 0.1
    scores[0, 0, 1] = 0.1

    # 8-connected: 1 region of size 3 -> covering (1, 2) first gives PRO = 1/3 across [0, 0.3].
    # 4-connected: 2 regions (sizes 2, 1) -> covering (1, 2) first gives PRO = 0.5 across [0, 0.3].
    score_8 = compute_aupro(masks, scores, fpr_limit=0.3, connectivity=2)
    score_4 = compute_aupro(masks, scores, fpr_limit=0.3, connectivity=1)
    assert score_8 == pytest.approx(1.0 / 3.0)
    assert score_4 == pytest.approx(0.5)
    assert score_8 < score_4


def test_aupro_rejects_invalid_inputs() -> None:
    """Malformed shapes, NaNs, degenerate masks, and bad params must raise ValueError."""
    masks = np.zeros((2, 4, 4), dtype=np.uint8)
    masks[0, 1:3, 1:3] = 1
    scores = np.zeros((2, 4, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_aupro(masks, np.zeros((2, 4, 5)))

    with pytest.raises(ValueError, match="3D"):
        compute_aupro(np.array([0, 1]), np.array([0.1, 0.9]))

    with pytest.raises(ValueError, match="non-empty"):
        compute_aupro(np.empty((0, 4, 4), dtype=np.uint8), np.empty((0, 4, 4)))

    bad_scores = scores.copy()
    bad_scores[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_aupro(masks, bad_scores)

    bad_mask = masks.copy()
    bad_mask[0, 0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        compute_aupro(bad_mask, scores)

    with pytest.raises(ValueError, match="dtype"):
        compute_aupro(masks.astype(str), scores)

    with pytest.raises(ValueError, match="at least one anomalous region"):
        compute_aupro(np.zeros((2, 4, 4), dtype=np.uint8), scores)

    with pytest.raises(ValueError, match="at least one background pixel"):
        compute_aupro(np.ones((2, 4, 4), dtype=np.uint8), scores)

    for bad_limit in (0.0, 1.2, float("nan"), True):
        with pytest.raises(ValueError, match="fpr_limit"):
            compute_aupro(masks, scores, fpr_limit=bad_limit)  # type: ignore[arg-type]

    for bad_conn in (8, True, 1.0):
        with pytest.raises(ValueError, match="connectivity"):
            compute_aupro(masks, scores, connectivity=bad_conn)  # type: ignore[arg-type]


def test_aupro_accepts_bool_masks_torch_tensors_and_all_defective_image_in_batch() -> None:
    """Verify bool masks, CPU PyTorch tensors, and an all-foreground image in a mixed batch."""
    masks_bool = np.zeros((2, 4, 4), dtype=bool)
    masks_bool[1, :, :] = True  # image 1 is 100% anomalous; image 0 provides normal pixels

    scores_np = np.full((2, 4, 4), 0.2, dtype=np.float32)
    scores_np[1, :, :] = 0.8

    assert compute_aupro(masks_bool, scores_np, fpr_limit=0.3) == pytest.approx(1.0)
    torch_score = compute_aupro(torch.from_numpy(masks_bool), torch.from_numpy(scores_np))
    assert torch_score == pytest.approx(1.0)


def test_aupro_is_invariant_to_batch_image_order() -> None:
    """Reordering images in a batch must leave AUPRO identical despite ties."""
    masks = np.zeros((4, 8, 8), dtype=np.uint8)
    masks[0, 1:3, 1:3] = 1
    masks[2, 4:7, 2:5] = 1
    masks[3, 0, 0] = 1

    rng = np.random.default_rng(99)
    scores = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], size=masks.shape)
    baseline = compute_aupro(masks, scores, fpr_limit=0.3)

    perm = [3, 1, 0, 2]
    reordered = compute_aupro(masks[perm], scores[perm], fpr_limit=0.3)
    assert reordered == pytest.approx(baseline, abs=1e-12)


def test_aupro_matches_anomalib_reference_on_synthetic_batch() -> None:
    """Cross-check custom compute_aupro against Anomalib 2.7.1 AUPRO on multi-image maps."""
    rng = np.random.default_rng(123)
    masks = np.zeros((6, 16, 16), dtype=np.uint8)
    # Image 0: all normal
    # Image 1: two separate regions
    masks[1, 2:5, 2:5] = 1
    masks[1, 10:12, 11:14] = 1
    # Image 2: diagonal crack (tests 8-connectivity agreement)
    for k in range(5):
        masks[2, 3 + k, 4 + k] = 1
    # Image 4: tiny 1-pixel defect
    masks[4, 8, 8] = 1

    # Semi-informative anomaly maps + tied plateaus
    scores = rng.uniform(0.0, 0.4, size=masks.shape).astype(np.float32)
    scores[masks == 1] += np.float32(0.35)
    scores[3, :4, :4] = np.float32(0.25)

    for limit in (0.1, 0.3, 0.5):
        ours = compute_aupro(masks, scores, fpr_limit=limit)
        ref_metric = AnomalibAUPRO(fpr_limit=limit)
        ref = float(
            ref_metric(
                torch.from_numpy(scores),
                torch.from_numpy(masks.astype(np.float32)),
            ).item()
        )
        assert ours == pytest.approx(ref, abs=1e-5)


@pytest.mark.slow
def test_aupro_matches_anomalib_reference_on_mvtec_bottle() -> None:
    """Real-category cross-check on MVTec AD bottle test masks (docs/05 section 3.3)."""
    bottle_root = Path("data/mvtec_ad")
    if not (bottle_root / "bottle").is_dir():
        pytest.skip("MVTec AD bottle dataset not present under data/mvtec_ad/bottle")

    all_samples = load_dataset_samples("mvtec", bottle_root, "bottle")
    test_samples = [s for s in all_samples if s.split == "test"]
    ds = AnomalyImageDataset(test_samples)

    masks = np.stack([ds[i]["mask"].numpy() for i in range(len(ds))], axis=0)
    rng = np.random.default_rng(7)
    # Synthetic localized predictor scores anchored to real MVTec masks (+ false-alarm spots).
    scores = rng.normal(loc=0.2, scale=0.1, size=masks.shape).astype(np.float32)
    scores[masks == 1] += np.float32(0.45)

    ours = compute_aupro(masks, scores, fpr_limit=0.3)
    ref_metric = AnomalibAUPRO(fpr_limit=0.3)
    ref = float(
        ref_metric(
            torch.from_numpy(scores),
            torch.from_numpy(masks.astype(np.float32)),
        ).item()
    )
    assert ours == pytest.approx(ref, abs=1e-5)
