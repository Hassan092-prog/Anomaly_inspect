"""Unit tests for image AUROC, pixel AUROC, and F1-max metrics."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from research.metrics import (
    compute_f1_max,
    compute_f1_max_with_threshold,
    compute_image_auroc,
    compute_pixel_auroc,
)


def test_image_auroc_perfect_inverted_constant_and_random() -> None:
    """Verify image AUROC on perfect, inverted, constant, and random predictors."""
    labels = np.array([0, 0, 0, 1, 1, 1])

    assert compute_image_auroc(labels, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == pytest.approx(
        1.0
    )
    assert compute_image_auroc(labels, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])) == pytest.approx(
        0.0
    )
    assert compute_image_auroc(labels, np.full(6, 0.5)) == pytest.approx(0.5)

    rng = np.random.default_rng(0)
    many_labels = np.concatenate([np.zeros(500, dtype=np.int8), np.ones(500, dtype=np.int8)])
    random_scores = rng.uniform(0.0, 1.0, size=1000)
    assert 0.45 <= compute_image_auroc(many_labels, random_scores) <= 0.55


def test_pixel_auroc_with_all_normal_image_mixed_in() -> None:
    """Verify pixel AUROC handles all-normal images alongside defective masks."""
    # Image 0: all-normal (all-zero mask); Image 1: 2x2 anomalous patch.
    masks = np.zeros((2, 4, 4), dtype=np.uint8)
    masks[1, 1:3, 1:3] = 1

    perfect_maps = np.full((2, 4, 4), 0.1, dtype=np.float32)
    perfect_maps[1, 1:3, 1:3] = 0.9
    assert compute_pixel_auroc(masks, perfect_maps) == pytest.approx(1.0)

    inverted_maps = 1.0 - perfect_maps
    assert compute_pixel_auroc(masks, inverted_maps) == pytest.approx(0.0)

    constant_maps = np.full((2, 4, 4), 0.42, dtype=np.float32)
    assert compute_pixel_auroc(masks, constant_maps) == pytest.approx(0.5)


def test_f1_max_known_small_examples() -> None:
    """Verify oracle F1-max and threshold against hand-calculated examples."""
    # At threshold 0.35: preds = [0, 1, 1, 1] -> TP=2, FP=1, FN=0 -> F1 = 4/5 = 0.8.
    targets = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    f1_max, threshold = compute_f1_max_with_threshold(targets, scores)
    assert f1_max == pytest.approx(0.8)
    assert threshold == pytest.approx(0.35)
    assert compute_f1_max(targets, scores) == pytest.approx(0.8)

    # Pixel-level 3D input (1 normal image + 1 defective image).
    masks = np.zeros((2, 2, 2), dtype=np.uint8)
    masks[1, 0, :] = 1
    maps = np.zeros((2, 2, 2), dtype=np.float32)
    maps[1, 0, :] = 0.9
    assert compute_f1_max(masks, maps) == pytest.approx(1.0)

    # Inverted predictor hits precision=0, recall=0 at high thresholds; must not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        inv_f1 = compute_f1_max(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1]))
    assert inv_f1 == pytest.approx(2.0 / 3.0)


def test_metrics_reject_invalid_inputs() -> None:
    """Malformed shapes, NaN/Inf scores, non-binary labels, or single-class inputs must raise."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_image_auroc([0, 1], [0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="finite"):
        compute_image_auroc([0, 1], [0.1, np.nan])

    with pytest.raises(ValueError, match="binary"):
        compute_image_auroc([0, 2], [0.1, 0.9])

    with pytest.raises(ValueError, match="both normal"):
        compute_image_auroc([0, 0, 0], [0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="3D"):
        compute_pixel_auroc(np.array([0, 1]), np.array([0.1, 0.9]))

    with pytest.raises(ValueError, match="at least one positive"):
        compute_f1_max([0, 0], [0.1, 0.2])
