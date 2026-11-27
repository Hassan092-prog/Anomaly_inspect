"""Image-level and pixel-level classification metrics (AUROC and F1-max)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import precision_recall_curve, roc_auc_score


def _validate_binary_targets_and_scores(
    targets: ArrayLike,
    scores: ArrayLike,
    *,
    expected_ndim: int | None = None,
    require_both_classes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate shapes, finiteness, and binary {0, 1} target values."""
    y_true = np.asarray(targets)
    y_score = np.asarray(scores, dtype=np.float64)

    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: targets {y_true.shape} != scores {y_score.shape}")
    if y_true.size == 0:
        raise ValueError("Inputs must be non-empty")
    if expected_ndim is not None and y_true.ndim != expected_ndim:
        raise ValueError(f"Expected {expected_ndim}D arrays, got {y_true.ndim}D")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("Scores must be finite (no NaN or Inf)")

    if y_true.dtype == np.bool_:
        y_true_int = y_true.astype(np.int8)
    else:
        y_true_float = y_true.astype(np.float64)
        if not np.all(np.isin(y_true_float, (0.0, 1.0))):
            raise ValueError("Targets must contain only binary values {0, 1}")
        y_true_int = y_true_float.astype(np.int8)

    n_pos = int(np.sum(y_true_int == 1))
    n_neg = int(y_true_int.size - n_pos)
    if require_both_classes and (n_pos == 0 or n_neg == 0):
        raise ValueError("Targets must contain both normal (0) and anomalous (1) entries")
    if not require_both_classes and n_pos == 0:
        raise ValueError("Targets must contain at least one positive (1) entry")

    return y_true_int.ravel(), y_score.ravel()


def compute_image_auroc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Compute image-level ROC AUC.

    Args:
        labels: Binary image labels of shape ``(N,)`` where ``0`` = normal and
            ``1`` = anomalous.
        scores: Continuous image-level anomaly scores of shape ``(N,)``
            (unitless; higher means more anomalous).

    Returns:
        Scalar AUROC in ``[0.0, 1.0]``.
    """
    y_true, y_score = _validate_binary_targets_and_scores(
        labels, scores, expected_ndim=1, require_both_classes=True
    )
    return float(roc_auc_score(y_true, y_score))


def compute_pixel_auroc(masks: ArrayLike, anomaly_maps: ArrayLike) -> float:
    """Compute pixel-level ROC AUC across all pixels in a test batch.

    Normal images contribute all-zero masks of shape ``(H, W)`` per
    ``docs/05_EVALUATION_PROTOCOL.md`` section 3.2.

    Args:
        masks: Binary ground-truth masks of shape ``(N, H, W)`` with values in ``{0, 1}``.
        anomaly_maps: Continuous anomaly maps of shape ``(N, H, W)``
            (unitless; higher means more anomalous).

    Returns:
        Scalar pixel AUROC in ``[0.0, 1.0]``.
    """
    y_true, y_score = _validate_binary_targets_and_scores(
        masks, anomaly_maps, expected_ndim=3, require_both_classes=True
    )
    return float(roc_auc_score(y_true, y_score))


def compute_f1_max_with_threshold(targets: ArrayLike, scores: ArrayLike) -> tuple[float, float]:
    """Compute oracle maximum F1 and its corresponding threshold.

    Works on 1D image-level arrays ``(N,)`` or 3D pixel-level arrays ``(N, H, W)``.
    Note: this uses ground-truth test labels to select the threshold and must be
    labeled as an **oracle** metric in all benchmark tables.

    Args:
        targets: Binary ground-truth array of shape ``(N,)`` or ``(N, H, W)``.
        scores: Continuous anomaly scores matching ``targets.shape``.

    Returns:
        Tuple ``(f1_max, oracle_threshold)`` where ``f1_max`` is in ``[0.0, 1.0]``
        and ``oracle_threshold`` is in raw score units.
    """
    y_true, y_score = _validate_binary_targets_and_scores(
        targets, scores, expected_ndim=None, require_both_classes=False
    )
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    f1_scores = np.divide(
        2.0 * p * r,
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=denom > 0.0,
    )
    best_idx = int(np.argmax(f1_scores))
    return float(f1_scores[best_idx]), float(thresholds[best_idx])


def compute_f1_max(targets: ArrayLike, scores: ArrayLike) -> float:
    """Compute oracle maximum F1 over all thresholds (``docs/05`` section 3.4).

    Args:
        targets: Binary ground-truth array of shape ``(N,)`` or ``(N, H, W)``.
        scores: Continuous anomaly scores matching ``targets.shape``.

    Returns:
        Maximum F1 score in ``[0.0, 1.0]``.
    """
    f1_max, _ = compute_f1_max_with_threshold(targets, scores)
    return f1_max
