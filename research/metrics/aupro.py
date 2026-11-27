"""Area Under Per-Region Overlap (AUPRO) localization metric."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from skimage.measure import label as label_connected_components


def _validate_aupro_inputs(
    masks: ArrayLike,
    anomaly_maps: ArrayLike,
    fpr_limit: float,
    connectivity: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate inputs for AUPRO computation and return normalized arrays."""
    if not isinstance(fpr_limit, (int, float)) or isinstance(fpr_limit, bool):
        raise ValueError(f"fpr_limit must be a float in (0, 1], got {fpr_limit!r}")
    if not math.isfinite(fpr_limit) or fpr_limit <= 0.0 or fpr_limit > 1.0:
        raise ValueError(f"fpr_limit must be in (0, 1], got {fpr_limit}")
    if (
        isinstance(connectivity, bool)
        or not isinstance(connectivity, int)
        or connectivity not in (1, 2)
    ):
        raise ValueError(
            f"connectivity must be 1 (4-connected) or 2 (8-connected), got {connectivity!r}"
        )

    mask_arr = np.asarray(masks)
    score_arr = np.asarray(anomaly_maps, dtype=np.float64)

    if mask_arr.shape != score_arr.shape:
        raise ValueError(
            f"Shape mismatch: masks {mask_arr.shape} != anomaly_maps {score_arr.shape}"
        )
    if mask_arr.ndim != 3:
        raise ValueError(f"Expected 3D arrays (N, H, W), got {mask_arr.ndim}D")
    if mask_arr.size == 0:
        raise ValueError("Inputs must be non-empty")
    if not np.all(np.isfinite(score_arr)):
        raise ValueError("Anomaly maps must be finite (no NaN or Inf)")

    if mask_arr.dtype == np.bool_:
        mask_bin = mask_arr.astype(np.uint8)
    elif np.issubdtype(mask_arr.dtype, np.integer) or np.issubdtype(mask_arr.dtype, np.floating):
        if not np.all((mask_arr == 0) | (mask_arr == 1)):
            raise ValueError("Masks must contain only binary values {0, 1}")
        mask_bin = mask_arr.astype(np.uint8, copy=False)
    else:
        raise ValueError(f"Masks must have numeric or boolean dtype, got {mask_arr.dtype}")

    n_fg = int(np.count_nonzero(mask_bin))
    n_bg = mask_bin.size - n_fg
    if n_fg == 0:
        raise ValueError("Masks must contain at least one anomalous region (no foreground pixels)")
    if n_bg == 0:
        raise ValueError("Masks must contain at least one background pixel (no normal pixels)")

    return mask_bin, score_arr


def _label_batch_regions(masks: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    """Label connected anomalous regions independently per image with batch-unique IDs."""
    region_labels = np.zeros(masks.shape, dtype=np.int64)
    region_offset = 0

    for idx in range(masks.shape[0]):
        img_mask = masks[idx]
        if not np.any(img_mask):
            continue
        img_labels, num_found = label_connected_components(
            img_mask,
            background=0,
            return_num=True,
            connectivity=connectivity,
        )
        fg = img_labels > 0
        region_labels[idx, fg] = img_labels[fg] + region_offset
        region_offset += int(num_found)

    if region_offset == 0:
        raise ValueError("Masks must contain at least one anomalous region")
    return region_labels, region_offset


def _clip_curve_at_fpr_limit(
    fpr: np.ndarray,
    pro: np.ndarray,
    fpr_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip a monotonic (fpr, pro) curve at fpr_limit using linear interpolation."""
    within_limit = np.flatnonzero(fpr <= fpr_limit)
    if within_limit.size == 0:
        return np.array([0.0, fpr_limit], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)

    last = int(within_limit[-1])
    if fpr[last] < fpr_limit and last + 1 < fpr.size:
        f1 = float(fpr[last])
        f2 = float(fpr[last + 1])
        p1 = float(pro[last])
        p2 = float(pro[last + 1])
        p_lim = p1 + (p2 - p1) * (fpr_limit - f1) / (f2 - f1) if f2 > f1 else p2
        return (
            np.concatenate([fpr[: last + 1], np.array([fpr_limit], dtype=np.float64)]),
            np.concatenate([pro[: last + 1], np.array([p_lim], dtype=np.float64)]),
        )

    return fpr[: last + 1], pro[: last + 1]


def compute_pro_curve(
    masks: ArrayLike,
    anomaly_maps: ArrayLike,
    *,
    fpr_limit: float = 0.3,
    connectivity: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the normalized Per-Region Overlap (PRO) vs FPR curve clipped at ``fpr_limit``.

    Each connected anomalous component in ``masks`` (per image) is weighted equally,
    following ``docs/05_EVALUATION_PROTOCOL.md`` section 3.3. Default ``connectivity=2``
    uses 8-connectivity, matching ``skimage.measure.label`` and Anomalib 2.7.1.

    Args:
        masks: Binary ground-truth masks of shape ``(N, H, W)`` with values in ``{0, 1}``.
            All-normal images must be represented with all-zero ``(H, W)`` masks.
        anomaly_maps: Continuous anomaly maps of shape ``(N, H, W)``
            (unitless; higher means more anomalous).
        fpr_limit: Upper pixel-level false positive rate bound in ``(0, 1]``.
            Defaults to ``0.3``.
        connectivity: Pixel neighborhood connectivity passed to ``skimage.measure.label``:
            ``1`` for 4-connected or ``2`` for 8-connected (default).

    Returns:
        Tuple ``(fpr, pro)`` of 1D ``float64`` arrays sorted by non-decreasing ``fpr``,
        starting at ``(0.0, 0.0)`` and ending at ``fpr == fpr_limit``.
    """
    mask_bin, score_arr = _validate_aupro_inputs(masks, anomaly_maps, fpr_limit, connectivity)
    region_labels, num_regions = _label_batch_regions(mask_bin, connectivity=connectivity)

    flat_labels = region_labels.ravel()
    flat_scores = score_arr.ravel()

    bg_mask = flat_labels == 0
    num_bg = int(np.count_nonzero(bg_mask))
    region_sizes = np.bincount(flat_labels, minlength=num_regions + 1).astype(np.float64)

    pro_change = np.zeros_like(flat_scores, dtype=np.float64)
    fg_mask = flat_labels > 0
    pro_change[fg_mask] = 1.0 / region_sizes[flat_labels[fg_mask]]

    # Exact global descending sort; tied scores collapse to the last element after cumsum.
    order = np.argsort(flat_scores)[::-1]
    sorted_scores = flat_scores[order]
    fpr = np.cumsum(bg_mask[order], dtype=np.float64) / float(num_bg)
    pro = np.cumsum(pro_change[order], dtype=np.float64) / float(num_regions)
    np.clip(fpr, 0.0, 1.0, out=fpr)
    np.clip(pro, 0.0, 1.0, out=pro)

    keep = np.ones(sorted_scores.shape, dtype=bool)
    keep[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    fpr = np.concatenate([np.array([0.0], dtype=np.float64), fpr[keep]])
    pro = np.concatenate([np.array([0.0], dtype=np.float64), pro[keep]])

    return _clip_curve_at_fpr_limit(fpr, pro, float(fpr_limit))


def compute_aupro(
    masks: ArrayLike,
    anomaly_maps: ArrayLike,
    *,
    fpr_limit: float = 0.3,
    connectivity: int = 2,
) -> float:
    """Compute normalized Area Under Per-Region Overlap (AUPRO) up to ``fpr_limit``.

    Computes ``(1 / fpr_limit) * integral_0^fpr_limit PRO(fpr) dfpr`` using
    exact threshold sweeps and trapezoidal integration.

    Args:
        masks: Binary ground-truth masks of shape ``(N, H, W)`` with values in ``{0, 1}``.
        anomaly_maps: Continuous anomaly maps of shape ``(N, H, W)``
            (unitless; higher means more anomalous).
        fpr_limit: Integration upper bound on pixel FPR in ``(0, 1]``.
            Defaults to ``0.3``.
        connectivity: Pixel neighborhood connectivity: ``1`` (4-connected) or
            ``2`` (8-connected, default).

    Returns:
        Scalar AUPRO score in ``[0.0, 1.0]``.
    """
    fpr, pro = compute_pro_curve(
        masks,
        anomaly_maps,
        fpr_limit=fpr_limit,
        connectivity=connectivity,
    )
    dx = fpr[1:] - fpr[:-1]
    area = float(np.sum(0.5 * dx * (pro[1:] + pro[:-1])))
    return float(np.clip(area / float(fpr_limit), 0.0, 1.0))
