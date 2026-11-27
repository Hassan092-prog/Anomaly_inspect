"""Dataset wrappers and aligned preprocessing for MVTec AD and VisA."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from research.common.config import PreprocessConfig

MVTEC_CATEGORIES: tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

VISA_CATEGORIES: tuple[str, ...] = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)

DATASET_CATEGORIES: dict[str, tuple[str, ...]] = {
    "mvtec": MVTEC_CATEGORIES,
    "visa": VISA_CATEGORIES,
}

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})

SplitTag = Literal["train", "val", "test"]
RawSplit = Literal["train", "test"]
DatasetName = Literal["mvtec", "visa"]


class DatasetError(ValueError):
    """Raised when dataset directories, CSV splits, or masks are invalid."""


@dataclass(frozen=True)
class SampleRecord:
    """Metadata for a single image sample (carrying explicit split provenance)."""

    dataset: DatasetName
    category: str
    split: SplitTag
    label: int
    defect_type: str
    image_path: Path
    mask_path: Path | None
    sample_id: str


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _resolve_under_root(root: Path, rel_path: str) -> Path:
    root_resolved = root.resolve()
    normalized = rel_path.replace("\\", "/")
    candidate = (root_resolved / normalized).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise DatasetError(f"Path escapes dataset root {root}: {rel_path!r}")
    return candidate


def _find_mvtec_mask(gt_defect_dir: Path, image_path: Path) -> Path:
    if not gt_defect_dir.is_dir():
        raise DatasetError(f"Missing ground-truth mask directory: {gt_defect_dir}")
    stem = image_path.stem
    for candidate_name in (f"{stem}_mask.png", f"{stem}_mask.PNG", f"{stem}.png", f"{stem}.PNG"):
        candidate = gt_defect_dir / candidate_name
        if candidate.is_file():
            return candidate
    valid_stems = {stem, f"{stem}_mask"}
    matches = sorted(
        p for p in gt_defect_dir.iterdir() if _is_image_file(p) and p.stem in valid_stems
    )
    if len(matches) == 1:
        return matches[0]
    raise DatasetError(f"Missing or ambiguous mask for {image_path} in {gt_defect_dir}")


def _scan_mvtec_style_category(
    category_dir: Path,
    *,
    dataset: DatasetName,
    category: str,
    split: RawSplit | None,
) -> list[SampleRecord]:
    if not category_dir.is_dir():
        raise DatasetError(f"Category directory not found: {category_dir}")

    splits_to_scan: tuple[RawSplit, ...] = ("train", "test") if split is None else (split,)
    records: list[SampleRecord] = []

    for split_name in splits_to_scan:
        split_dir = category_dir / split_name
        if not split_dir.is_dir():
            raise DatasetError(f"Missing split directory: {split_dir}")

        defect_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        if not defect_dirs:
            raise DatasetError(f"No class subdirectories found under {split_dir}")

        for defect_dir in defect_dirs:
            defect_type = defect_dir.name
            if split_name == "train" and defect_type != "good":
                raise DatasetError(
                    f"Training split must contain only normal ('good') images, found {defect_dir}"
                )

            is_normal = defect_type == "good"
            label = 0 if is_normal else 1
            gt_defect_dir = category_dir / "ground_truth" / defect_type

            img_paths = sorted(p for p in defect_dir.iterdir() if _is_image_file(p))
            if not img_paths:
                raise DatasetError(f"Empty class directory with 0 images: {defect_dir}")
            for img_path in img_paths:
                mask_path: Path | None = None
                if not is_normal:
                    mask_path = _find_mvtec_mask(gt_defect_dir, img_path)
                rel_id = img_path.relative_to(category_dir).as_posix()
                records.append(
                    SampleRecord(
                        dataset=dataset,
                        category=category,
                        split=split_name,
                        label=label,
                        defect_type=defect_type,
                        image_path=img_path,
                        mask_path=mask_path,
                        sample_id=rel_id,
                    )
                )

    if not records:
        raise DatasetError(f"Found 0 images for {dataset}/{category} under {category_dir}")
    return records


def load_mvtec_samples(
    root: str | Path,
    category: str,
    split: RawSplit | None = None,
) -> list[SampleRecord]:
    """Load MVTec AD sample records for one category.

    Args:
        root: Path to MVTec AD root directory containing category subdirectories.
        category: One of ``MVTEC_CATEGORIES``.
        split: Optional ``"train"`` or ``"test"`` filter; ``None`` loads both.
    """
    if category not in MVTEC_CATEGORIES:
        raise DatasetError(f"Unknown MVTec AD category {category!r}")
    root_path = Path(root)
    return _scan_mvtec_style_category(
        root_path / category,
        dataset="mvtec",
        category=category,
        split=split,
    )


def _load_visa_from_split_csv(
    root_path: Path,
    split_csv: Path,
    category: str,
    split: RawSplit | None,
) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with split_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected_cols = {"object", "split", "label", "image", "mask"}
        if reader.fieldnames is None or not expected_cols.issubset(set(reader.fieldnames)):
            raise DatasetError(f"VisA split CSV {split_csv} missing expected columns")

        for row in reader:
            if row["object"] != category:
                continue
            row_split = row["split"]
            if row_split not in ("train", "test"):
                raise DatasetError(f"Unexpected VisA split {row_split!r} in {split_csv}")
            if split is not None and row_split != split:
                continue

            raw_label = row["label"]
            if raw_label not in ("normal", "anomaly"):
                raise DatasetError(f"Unexpected VisA label {raw_label!r} in {split_csv}")
            if row_split == "train" and raw_label != "normal":
                raise DatasetError("VisA train split must contain only normal samples")

            img_rel = row["image"]
            img_path = _resolve_under_root(root_path, img_rel)
            if not _is_image_file(img_path):
                raise DatasetError(f"Missing VisA image: {img_path}")

            is_normal = raw_label == "normal"
            mask_rel = (row["mask"] or "").strip()
            mask_path: Path | None = None
            if not is_normal:
                if not mask_rel:
                    raise DatasetError(f"Missing mask entry for anomalous VisA sample: {img_rel}")
                mask_path = _resolve_under_root(root_path, mask_rel)
                if not _is_image_file(mask_path):
                    raise DatasetError(f"Missing VisA mask file: {mask_path}")

            records.append(
                SampleRecord(
                    dataset="visa",
                    category=category,
                    split=row_split,  # type: ignore[arg-type]
                    label=0 if is_normal else 1,
                    defect_type="good" if is_normal else "anomaly",
                    image_path=img_path,
                    mask_path=mask_path,
                    sample_id=img_rel.replace("\\", "/"),
                )
            )

    if not records:
        raise DatasetError(f"Found 0 VisA samples for {category!r} in {split_csv}")
    return sorted(records, key=lambda r: (r.split, r.sample_id))


def load_visa_samples(
    root: str | Path,
    category: str,
    split: RawSplit | None = None,
) -> list[SampleRecord]:
    """Load VisA sample records from official ``split_csv/1cls.csv`` or MVTec-style folders."""
    if category not in VISA_CATEGORIES:
        raise DatasetError(f"Unknown VisA category {category!r}")
    root_path = Path(root)

    split_csv = root_path / "split_csv" / "1cls.csv"
    if split_csv.is_file():
        return _load_visa_from_split_csv(root_path, split_csv, category, split)

    visa_pytorch_dir = root_path / "visa_pytorch" / category
    if visa_pytorch_dir.is_dir():
        return _scan_mvtec_style_category(
            visa_pytorch_dir, dataset="visa", category=category, split=split
        )

    direct_dir = root_path / category
    if (direct_dir / "train").is_dir() or (direct_dir / "test").is_dir():
        return _scan_mvtec_style_category(
            direct_dir, dataset="visa", category=category, split=split
        )

    raise DatasetError(
        f"Could not find VisA split_csv/1cls.csv or prepared category folder under {root_path}"
    )


def load_dataset_samples(
    dataset: DatasetName,
    root: str | Path,
    category: str,
    split: RawSplit | None = None,
) -> list[SampleRecord]:
    """Load sample records for either ``'mvtec'`` or ``'visa'``."""
    if dataset == "mvtec":
        return load_mvtec_samples(root, category, split=split)
    if dataset == "visa":
        return load_visa_samples(root, category, split=split)
    raise DatasetError(f"Unsupported dataset {dataset!r}")


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as uint8 RGB array of shape ``(H, W, 3)``."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def load_mask_array(path: str | Path) -> np.ndarray:
    """Load a ground-truth mask as a 2D uint8 array of shape ``(H, W)``."""
    with Image.open(path) as img:
        if img.mode in {"L", "1", "P", "I", "F"}:
            arr = np.asarray(img)
        else:
            # Drop alpha channel so RGBA/LA masks do not become all-foreground.
            arr = np.max(np.asarray(img.convert("RGB")), axis=2)
    return arr.astype(np.uint8, copy=False)


def _center_crop(array: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = array.shape[:2]
    if h < crop_size or w < crop_size:
        raise DatasetError(f"Cannot center-crop {crop_size} from shape {(h, w)}")
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return array[top : top + crop_size, left : left + crop_size]


def preprocess_image_and_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray | None,
    preprocess: PreprocessConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize, center-crop, and normalize image while keeping mask aligned.

    Args:
        image_rgb: RGB uint8 image of shape ``(H, W, 3)``.
        mask: Optional 2D mask of shape ``(H, W)``; ``None`` produces an all-zero mask.
        preprocess: Preprocessing config (defaults to ``256 -> 224``, ImageNet stats).

    Returns:
        Tuple ``(image_tensor, mask_tensor)`` where ``image_tensor`` has shape
        ``(3, crop_size, crop_size)`` (float32) and ``mask_tensor`` has shape
        ``(crop_size, crop_size)`` (uint8 with values in ``{0, 1}``).
    """
    cfg = preprocess or PreprocessConfig()
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise DatasetError(f"Expected RGB image (H, W, 3), got {image_rgb.shape}")

    h, w = image_rgb.shape[:2]
    resized_img = cv2.resize(image_rgb, (cfg.resize, cfg.resize), interpolation=cv2.INTER_LINEAR)
    cropped_img = _center_crop(resized_img, cfg.crop_size).astype(np.float32) / 255.0

    mean = np.asarray(cfg.normalize_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(cfg.normalize_std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (cropped_img - mean) / std
    image_tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))

    if mask is None:
        mask_tensor = torch.zeros((cfg.crop_size, cfg.crop_size), dtype=torch.uint8)
        return image_tensor, mask_tensor

    if mask.ndim != 2:
        raise DatasetError(f"Expected 2D mask (H, W), got {mask.shape}")
    if mask.shape != (h, w):
        raise DatasetError(f"Image shape {(h, w)} != mask shape {mask.shape}")

    mask_for_resize = mask.astype(np.uint8, copy=False) if mask.dtype == np.bool_ else mask
    resized_mask = cv2.resize(
        mask_for_resize, (cfg.resize, cfg.resize), interpolation=cv2.INTER_NEAREST
    )
    cropped_mask = _center_crop(resized_mask, cfg.crop_size)
    binary_mask = (cropped_mask > 0.5).astype(np.uint8)
    mask_tensor = torch.from_numpy(np.ascontiguousarray(binary_mask))
    return image_tensor, mask_tensor


class AnomalyImageDataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset returning aligned preprocessed image/mask tensors and split tags."""

    def __init__(
        self,
        samples: Sequence[SampleRecord],
        preprocess: PreprocessConfig | None = None,
    ) -> None:
        self.samples = list(samples)
        self.preprocess = preprocess or PreprocessConfig()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.samples[index]
        image_rgb = load_image_rgb(rec.image_path)
        raw_mask = load_mask_array(rec.mask_path) if rec.mask_path is not None else None
        image_tensor, mask_tensor = preprocess_image_and_mask(
            image_rgb, raw_mask, preprocess=self.preprocess
        )
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": rec.label,
            "split": rec.split,
            "dataset": rec.dataset,
            "category": rec.category,
            "defect_type": rec.defect_type,
            "sample_id": rec.sample_id,
            "image_path": str(rec.image_path),
            "mask_path": str(rec.mask_path) if rec.mask_path is not None else None,
        }


def category_counts(samples: Sequence[SampleRecord]) -> dict[str, int]:
    """Summarize sample counts across train/val/test and normal/anomalous subsets."""
    return {
        "total": len(samples),
        "train": sum(1 for s in samples if s.split == "train"),
        "val": sum(1 for s in samples if s.split == "val"),
        "test": sum(1 for s in samples if s.split == "test"),
        "test_normal": sum(1 for s in samples if s.split == "test" and s.label == 0),
        "test_anomalous": sum(1 for s in samples if s.split == "test" and s.label == 1),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Verify MVTec AD or VisA dataset discovery.")
    parser.add_argument("--dataset", choices=("mvtec", "visa"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()

    categories = (args.category,) if args.category else DATASET_CATEGORIES[args.dataset]
    for cat in categories:
        samples = load_dataset_samples(args.dataset, args.root, cat)
        counts = category_counts(samples)
        print(f"{args.dataset}/{cat}: {counts}")


if __name__ == "__main__":
    _main()
