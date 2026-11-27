"""Unit tests for MVTec AD and VisA dataset wrappers and mask alignment."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research.common.config import PreprocessConfig
from research.data import (
    MVTEC_CATEGORIES,
    VISA_CATEGORIES,
    AnomalyImageDataset,
    DatasetError,
    category_counts,
    load_mvtec_samples,
    load_visa_samples,
    preprocess_image_and_mask,
)
from research.data.datasets import load_mask_array


def _save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _build_synthetic_mvtec(root: Path, category: str = "bottle") -> Path:
    cat_dir = root / category
    # 3 normal train images
    for i in range(3):
        img = np.full((300, 300, 3), 128, dtype=np.uint8)
        _save_png(cat_dir / "train" / "good" / f"{i:03d}.png", img)

    # 2 normal test images
    for i in range(2):
        img = np.full((300, 300, 3), 120, dtype=np.uint8)
        _save_png(cat_dir / "test" / "good" / f"{i:03d}.png", img)

    # 2 anomalous test images with a central white square + matching mask
    for i in range(2):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[100:200, 100:200] = 255
        mask = np.zeros((300, 300), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        _save_png(cat_dir / "test" / "broken" / f"{i:03d}.png", img)
        _save_png(cat_dir / "ground_truth" / "broken" / f"{i:03d}_mask.png", mask)
    return root


def _build_synthetic_visa_csv(root: Path, category: str = "candle") -> Path:
    rows = [("object", "split", "label", "image", "mask")]
    for i in range(2):
        rel = f"{category}/Data/Images/Normal/{i:04d}.jpg"
        Image.fromarray(np.full((256, 256, 3), 100, dtype=np.uint8)).save(
            _ensure_parent(root / rel)
        )
        rows.append((category, "train", "normal", rel, ""))

    rel_test_ok = f"{category}/Data/Images/Normal/0099.jpg"
    Image.fromarray(np.full((256, 256, 3), 110, dtype=np.uint8)).save(
        _ensure_parent(root / rel_test_ok)
    )
    rows.append((category, "test", "normal", rel_test_ok, ""))

    rel_anom = f"{category}/Data/Images/Anomaly/0000.jpg"
    rel_mask = f"{category}/Data/Masks/Anomaly/0000.png"
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[64:192, 64:192] = 200
    mask = np.zeros((256, 256), dtype=np.uint8)
    # VisA masks can store instance IDs like 1, 2 rather than 255.
    mask[64:192, 64:192] = 1
    Image.fromarray(img).save(_ensure_parent(root / rel_anom))
    Image.fromarray(mask).save(_ensure_parent(root / rel_mask))
    rows.append((category, "test", "anomaly", rel_anom, rel_mask))

    csv_path = _ensure_parent(root / "split_csv" / "1cls.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    return root


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_category_lists_match_official_counts() -> None:
    """MVTec AD has 15 categories and VisA has 12 categories."""
    assert len(MVTEC_CATEGORIES) == 15
    assert len(VISA_CATEGORIES) == 12


def test_mvtec_loader_counts_and_mask_alignment(tmp_path: Path) -> None:
    """MVTec loader should return expected counts and spatially aligned masks."""
    root = _build_synthetic_mvtec(tmp_path, "bottle")
    samples = load_mvtec_samples(root, "bottle")
    assert category_counts(samples) == {
        "total": 7,
        "train": 3,
        "val": 0,
        "test": 4,
        "test_normal": 2,
        "test_anomalous": 2,
    }

    ds = AnomalyImageDataset(samples, PreprocessConfig(resize=256, crop_size=224))
    assert len(ds) == 7

    train_item = ds[0]
    assert train_item["split"] == "train"
    assert train_item["label"] == 0
    assert train_item["image"].shape == (3, 224, 224)
    assert train_item["mask"].shape == (224, 224)
    assert int(train_item["mask"].sum()) == 0

    anom_item = next(ds[i] for i in range(len(ds)) if ds[i]["label"] == 1)
    assert anom_item["split"] == "test"
    # Central bright region in the synthetic image must coincide with mask == 1.
    mask_np = anom_item["mask"].numpy()
    assert set(np.unique(mask_np).tolist()) == {0, 1}
    raw_center_mean = float(anom_item["image"][0, mask_np == 1].mean())
    raw_bg_mean = float(anom_item["image"][0, mask_np == 0].mean())
    assert raw_center_mean > raw_bg_mean


def test_visa_csv_loader_counts_and_non255_mask_binarization(tmp_path: Path) -> None:
    """VisA loader should parse 1cls.csv and binarize instance-ID masks cleanly."""
    root = _build_synthetic_visa_csv(tmp_path, "candle")
    samples = load_visa_samples(root, "candle")
    assert category_counts(samples) == {
        "total": 4,
        "train": 2,
        "val": 0,
        "test": 2,
        "test_normal": 1,
        "test_anomalous": 1,
    }

    ds = AnomalyImageDataset(samples)
    anom_item = next(ds[i] for i in range(len(ds)) if ds[i]["label"] == 1)
    mask_vals = set(np.unique(anom_item["mask"].numpy()).tolist())
    assert mask_vals == {0, 1}


def test_loader_rejects_missing_mask_and_non_normal_train(tmp_path: Path) -> None:
    """Missing test masks, non-normal train folders, or mismatched sizes must fail loud."""
    root = _build_synthetic_mvtec(tmp_path, "bottle")
    (root / "bottle" / "ground_truth" / "broken" / "000_mask.png").unlink()
    with pytest.raises(DatasetError, match="mask"):
        load_mvtec_samples(root, "bottle")

    bad_train = tmp_path / "bad_train"
    _build_synthetic_mvtec(bad_train, "bottle")
    _save_png(
        bad_train / "bottle" / "train" / "defect" / "000.png",
        np.zeros((256, 256, 3), dtype=np.uint8),
    )
    with pytest.raises(DatasetError, match="normal"):
        load_mvtec_samples(bad_train, "bottle")

    img = np.zeros((256, 256, 3), dtype=np.uint8)
    wrong_mask = np.zeros((128, 128), dtype=np.uint8)
    with pytest.raises(DatasetError, match="shape"):
        preprocess_image_and_mask(img, wrong_mask)


def test_visa_rejects_csv_path_traversal(tmp_path: Path) -> None:
    """VisA split CSV paths escaping dataset root must raise DatasetError."""
    root = _build_synthetic_visa_csv(tmp_path, "candle")
    csv_path = root / "split_csv" / "1cls.csv"
    csv_path.write_text(
        "object,split,label,image,mask\ncandle,train,normal,../secret.png,\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="escapes dataset root"):
        load_visa_samples(root, "candle")


def test_rgba_and_bool_masks_and_exact_mask_stem_matching(tmp_path: Path) -> None:
    """RGBA alpha must not corrupt masks, bool masks must resize cleanly, and stems stay exact."""
    rgba_mask = np.zeros((256, 256, 4), dtype=np.uint8)
    rgba_mask[..., 3] = 255  # opaque alpha everywhere
    rgba_mask[64:128, 64:128, :3] = 255
    mask_path = _ensure_parent(tmp_path / "rgba_mask.png")
    Image.fromarray(rgba_mask, mode="RGBA").save(mask_path)

    loaded_mask = load_mask_array(mask_path)
    assert loaded_mask.shape == (256, 256)
    assert int(loaded_mask[0, 0]) == 0
    assert int(loaded_mask[80, 80]) == 255

    img = np.zeros((256, 256, 3), dtype=np.uint8)
    bool_mask = np.zeros((256, 256), dtype=bool)
    bool_mask[64:128, 64:128] = True
    _, mask_tensor = preprocess_image_and_mask(img, bool_mask)
    assert set(np.unique(mask_tensor.numpy()).tolist()) == {0, 1}

    # Exact stem match: 1.jpg must match 1_mask.tif without colliding with 10_mask.tif.
    mvtec_root = _build_synthetic_mvtec(tmp_path / "mvtec", "bottle")
    cat_dir = mvtec_root / "bottle"
    _save_png(cat_dir / "test" / "broken" / "1.jpg", np.zeros((300, 300, 3), dtype=np.uint8))
    _save_png(cat_dir / "test" / "broken" / "10.jpg", np.zeros((300, 300, 3), dtype=np.uint8))
    _save_png(
        cat_dir / "ground_truth" / "broken" / "1_mask.tif",
        np.zeros((300, 300), dtype=np.uint8),
    )
    _save_png(
        cat_dir / "ground_truth" / "broken" / "10_mask.tif",
        np.zeros((300, 300), dtype=np.uint8),
    )
    samples = load_mvtec_samples(mvtec_root, "bottle", split="test")
    assert len(samples) == 6
