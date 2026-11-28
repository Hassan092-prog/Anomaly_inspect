"""Bridge deterministic splits and ``AnomalyImageDataset`` to Anomalib ``ImageBatch`` loaders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from anomalib.data import ImageBatch
from torch.utils.data import DataLoader

from research.common.config import ExperimentConfig
from research.data.datasets import AnomalyImageDataset, DatasetName, SampleRecord
from research.data.make_splits import (
    assert_no_test_leakage,
    create_category_splits,
    load_split_records,
)


@dataclass(frozen=True)
class SplitDataLoaders:
    """Isolated train-fit, validation-normal, and test dataloaders for one category run."""

    train_fit: DataLoader[Any] | None
    val_normal: DataLoader[Any]
    test: DataLoader[Any]
    train_records: tuple[SampleRecord, ...]
    val_records: tuple[SampleRecord, ...]
    test_records: tuple[SampleRecord, ...]


def resolve_dataset_root(data_dir: str | Path, dataset: DatasetName) -> Path:
    """Resolve dataset directory under ``data_dir`` per ``scripts/download_data.md``."""
    base = Path(data_dir)
    return base / "mvtec_ad" if dataset == "mvtec" else base / "visa"


def collate_image_batch(items: Sequence[dict[str, Any]]) -> ImageBatch:
    """Collate items from ``AnomalyImageDataset`` into an Anomalib ``ImageBatch``."""
    if not items:
        raise ValueError("Cannot collate an empty batch")

    images = torch.stack([item["image"] for item in items], dim=0)
    gt_label = torch.tensor([int(item["label"]) for item in items], dtype=torch.bool)
    gt_mask = torch.stack([item["mask"] for item in items], dim=0)
    image_path = [str(item["image_path"]) for item in items]
    raw_mask_paths = [item.get("mask_path") for item in items]
    mask_path = (
        [str(p) if p is not None else "" for p in raw_mask_paths]
        if any(p is not None for p in raw_mask_paths)
        else None
    )
    return ImageBatch(
        image=images,
        gt_label=gt_label,
        gt_mask=gt_mask,
        image_path=image_path,
        mask_path=mask_path,
    )


def _split_file_for_config(cfg: ExperimentConfig, splits_dir: Path) -> Path:
    category_dir = splits_dir / cfg.data.dataset / cfg.data.category
    if cfg.data.k_shot in ("full", 0):
        return category_dir / "val_split.json"
    return category_dir / f"k{cfg.data.k_shot}_seed{cfg.seed}.json"


def load_split_dataloaders(
    cfg: ExperimentConfig,
    *,
    batch_size: int = 16,
    num_workers: int = 0,
    create_splits_if_missing: bool = True,
) -> SplitDataLoaders:
    """Load deterministic ``train_fit``, ``val_normal``, and ``test`` ``ImageBatch`` dataloaders.

    Explicitly bypasses Anomalib's built-in ``MVTecAD``/``Visa`` datamodule split logic
    (which defaults to ``ValSplitMode.SAME_AS_TEST`` in Anomalib 2.7.1) and verifies zero
    test-set leakage on both ``train_fit`` and ``val_normal``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")

    data_root = resolve_dataset_root(cfg.data.data_dir, cfg.data.dataset)
    splits_dir = Path(cfg.data.data_dir) / "splits"
    split_file = _split_file_for_config(cfg, splits_dir)

    if not split_file.is_file() and create_splits_if_missing:
        if isinstance(cfg.data.k_shot, int) and cfg.data.k_shot > 0:
            k_shots: tuple[int, ...] = (cfg.data.k_shot,)
            seeds: tuple[int, ...] = (cfg.seed,)
        else:
            k_shots = ()
            seeds = ()
        create_category_splits(
            cfg.data.dataset,
            data_root,
            cfg.data.category,
            splits_dir=splits_dir,
            val_fraction=cfg.data.val_fraction,
            val_seed=cfg.data.val_seed,
            k_shots=k_shots,
            seeds=seeds,
        )

    train_records, val_records, test_records = load_split_records(
        cfg.data.dataset,
        data_root,
        cfg.data.category,
        splits_dir=splits_dir,
        k_shot=cfg.data.k_shot,
        seed=cfg.seed,
    )

    assert_no_test_leakage(train_records, context="anomalib_data.train_fit")
    assert_no_test_leakage(val_records, context="anomalib_data.val_normal")
    if any(rec.label != 0 for rec in train_records):
        raise ValueError("train_fit must contain only normal (label == 0) images")
    if any(rec.label != 0 for rec in val_records):
        raise ValueError("val_normal must contain only normal (label == 0) images")
    if not val_records:
        raise ValueError("val_normal split must be non-empty")
    if not test_records:
        raise ValueError("test split must be non-empty")

    train_loader: DataLoader[Any] | None = None
    if train_records:
        train_ds = AnomalyImageDataset(train_records, preprocess=cfg.preprocess)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_image_batch,
        )

    val_ds = AnomalyImageDataset(val_records, preprocess=cfg.preprocess)
    test_ds = AnomalyImageDataset(test_records, preprocess=cfg.preprocess)
    val_loader: DataLoader[Any] = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_image_batch,
    )
    test_loader: DataLoader[Any] = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_image_batch,
    )

    return SplitDataLoaders(
        train_fit=train_loader,
        val_normal=val_loader,
        test=test_loader,
        train_records=tuple(train_records),
        val_records=tuple(val_records),
        test_records=tuple(test_records),
    )
