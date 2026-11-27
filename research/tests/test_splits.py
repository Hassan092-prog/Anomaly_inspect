"""Unit tests for deterministic val-normal and k-shot split generation and leakage guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research.data import (
    LeakageError,
    SampleRecord,
    SplitError,
    assert_no_test_leakage,
    create_category_splits,
    load_mvtec_samples,
    load_split_records,
)


def _build_mvtec_for_splits(root: Path, n_train: int = 20, category: str = "bottle") -> Path:
    cat_dir = root / category
    for i in range(n_train):
        p = cat_dir / "train" / "good" / f"{i:03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((64, 64, 3), i % 255, dtype=np.uint8)).save(p)

    test_ok = cat_dir / "test" / "good" / "000.png"
    test_ok.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(test_ok)

    test_bad = cat_dir / "test" / "broken" / "000.png"
    gt_bad = cat_dir / "ground_truth" / "broken" / "000_mask.png"
    test_bad.parent.mkdir(parents=True, exist_ok=True)
    gt_bad.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8)).save(test_bad)
    Image.fromarray(np.full((64, 64), 255, dtype=np.uint8)).save(gt_bad)
    return root


def test_splits_are_disjoint_subsets_and_byte_identical_on_rerun(tmp_path: Path) -> None:
    """Val and train-fit must be disjoint, k-shot subsets of train-fit, and byte-identical."""
    data_root = _build_mvtec_for_splits(tmp_path / "mvtec", n_train=20, category="bottle")
    splits_dir = tmp_path / "splits"

    files_run1 = create_category_splits("mvtec", data_root, "bottle", splits_dir=splits_dir)
    bytes_run1 = {p.name: p.read_bytes() for p in files_run1}

    files_run2 = create_category_splits("mvtec", data_root, "bottle", splits_dir=splits_dir)
    bytes_run2 = {p.name: p.read_bytes() for p in files_run2}
    assert bytes_run1 == bytes_run2

    full_train, val_full, test_recs = load_split_records(
        "mvtec", data_root, "bottle", splits_dir=splits_dir, k_shot="full"
    )
    assert len(val_full) == 3  # 15% of 20
    assert len(full_train) == 17
    assert len(test_recs) == 2

    full_train_ids = {r.sample_id for r in full_train}
    val_ids = {r.sample_id for r in val_full}
    assert full_train_ids.isdisjoint(val_ids)
    assert all(r.split == "train" for r in full_train)
    assert all(r.split == "val" for r in val_full)
    assert all(r.split == "test" for r in test_recs)

    # Every k-shot subset must be strictly inside full train-fit and share identical val-normal.
    for k in (1, 5, 10):
        for seed in (0, 1, 2):
            k_train, k_val, _ = load_split_records(
                "mvtec", data_root, "bottle", splits_dir=splits_dir, k_shot=k, seed=seed
            )
            assert len(k_train) == k
            assert {r.sample_id for r in k_train}.issubset(full_train_ids)
            assert {r.sample_id for r in k_val} == val_ids

    # Zero-shot (k=0, e.g. WinCLIP) yields 0 train-fit images while keeping identical val-normal.
    zero_train, zero_val, _ = load_split_records(
        "mvtec", data_root, "bottle", splits_dir=splits_dir, k_shot=0
    )
    assert zero_train == []
    assert {r.sample_id for r in zero_val} == val_ids


def test_leakage_guard_blocks_test_samples_in_fitting_pool(tmp_path: Path) -> None:
    """Fitting/normalization helpers must reject any sample carrying split='test'."""
    data_root = _build_mvtec_for_splits(tmp_path / "mvtec", n_train=6, category="bottle")
    test_samples = load_mvtec_samples(data_root, "bottle", split="test")

    with pytest.raises(LeakageError, match="leaked"):
        assert_no_test_leakage(test_samples, context="normalization")

    anomalous_train = SampleRecord(
        dataset="mvtec",
        category="bottle",
        split="train",
        label=1,
        defect_type="broken",
        image_path=tmp_path / "x.png",
        mask_path=None,
        sample_id="bad",
    )
    with pytest.raises(LeakageError, match="Anomalous"):
        assert_no_test_leakage([anomalous_train], context="fitting")


def test_split_rejects_k_larger_than_train_fit(tmp_path: Path) -> None:
    """Requesting more few-shot images than available train-fit samples must raise."""
    data_root = _build_mvtec_for_splits(tmp_path / "mvtec", n_train=5, category="bottle")
    with pytest.raises(SplitError, match="exceeds available"):
        create_category_splits(
            "mvtec",
            data_root,
            "bottle",
            splits_dir=tmp_path / "splits",
            k_shots=(10,),
            seeds=(0,),
        )
