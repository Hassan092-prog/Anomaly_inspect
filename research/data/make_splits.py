"""Deterministic validation and few-shot split creation (``docs/05`` section 1)."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from research.common.results import PROTOCOL_VERSION
from research.data.datasets import (
    DATASET_CATEGORIES,
    DatasetName,
    SampleRecord,
    load_dataset_samples,
)

DEFAULT_VAL_FRACTION = 0.15
DEFAULT_VAL_SEED = 0
DEFAULT_K_SHOTS: tuple[int, ...] = (1, 5, 10)
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
DEFAULT_SPLITS_DIR = Path("data/splits")


class SplitError(ValueError):
    """Raised when split parameters or split integrity checks fail."""


class LeakageError(SplitError):
    """Raised when test-split samples appear in a fitting or validation context."""


@dataclass(frozen=True)
class CategorySplit:
    """Resolved deterministic normal train-fit and val-normal sample IDs."""

    dataset: DatasetName
    category: str
    val_fraction: float
    val_seed: int
    train_fit: tuple[str, ...]
    val_normal: tuple[str, ...]


def assert_no_test_leakage(records: Sequence[SampleRecord], *, context: str = "fitting") -> None:
    """Raise ``LeakageError`` if any record belongs to the ``test`` split or is anomalous."""
    for rec in records:
        if rec.split == "test":
            raise LeakageError(f"Test sample {rec.sample_id!r} leaked into {context} split!")
        if rec.label != 0:
            raise LeakageError(
                f"Anomalous sample {rec.sample_id!r} leaked into normal-only {context} split!"
            )


def split_train_val(
    train_samples: Sequence[SampleRecord],
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    val_seed: int = DEFAULT_VAL_SEED,
) -> CategorySplit:
    """Carve a deterministic ``val-normal`` subset from normal training samples."""
    if not (0.0 < val_fraction < 0.5):
        raise SplitError(f"val_fraction must be in (0, 0.5), got {val_fraction}")
    if isinstance(val_seed, bool) or not isinstance(val_seed, int) or val_seed < 0:
        raise SplitError(f"val_seed must be a non-negative int, got {val_seed!r}")
    if not train_samples:
        raise SplitError("Cannot split empty train_samples")

    assert_no_test_leakage(train_samples, context="train/val splitting")

    dataset = train_samples[0].dataset
    category = train_samples[0].category
    for rec in train_samples:
        if rec.dataset != dataset or rec.category != category:
            raise SplitError("All train_samples must share the same dataset and category")

    ordered_ids = sorted({rec.sample_id for rec in train_samples})
    if len(ordered_ids) != len(train_samples):
        raise SplitError("Duplicate sample_id values found in train_samples")
    if len(ordered_ids) < 2:
        raise SplitError(f"Need at least 2 normal train images to split, got {len(ordered_ids)}")

    n_val = max(1, round(len(ordered_ids) * val_fraction))
    if n_val >= len(ordered_ids):
        n_val = len(ordered_ids) - 1

    rng = random.Random(val_seed)
    val_ids = tuple(sorted(rng.sample(ordered_ids, n_val)))
    val_set = set(val_ids)
    train_fit_ids = tuple(sid for sid in ordered_ids if sid not in val_set)

    return CategorySplit(
        dataset=dataset,
        category=category,
        val_fraction=val_fraction,
        val_seed=val_seed,
        train_fit=train_fit_ids,
        val_normal=val_ids,
    )


def sample_k_shot(train_fit_ids: Sequence[str], k: int, seed: int) -> tuple[str, ...]:
    """Sample ``k`` sample IDs deterministically from ``train_fit_ids`` only."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise SplitError(f"k must be a positive int, got {k!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SplitError(f"seed must be a non-negative int, got {seed!r}")
    unique_ids = sorted(set(train_fit_ids))
    if k > len(unique_ids):
        raise SplitError(f"Requested k={k} exceeds available train-fit samples ({len(unique_ids)})")
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(unique_ids, k)))


def _write_canonical_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def create_category_splits(
    dataset: DatasetName,
    data_root: str | Path,
    category: str,
    *,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    val_seed: int = DEFAULT_VAL_SEED,
    k_shots: Sequence[int] = DEFAULT_K_SHOTS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> list[Path]:
    """Write deterministic ``val_split.json`` and ``k{k}_seed{s}.json`` files."""
    train_samples = load_dataset_samples(dataset, data_root, category, split="train")
    base_split = split_train_val(train_samples, val_fraction=val_fraction, val_seed=val_seed)

    out_dir = Path(splits_dir) / dataset / category
    written: list[Path] = []

    val_payload: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "category": category,
        "val_fraction": val_fraction,
        "val_seed": val_seed,
        "train_fit": list(base_split.train_fit),
        "val_normal": list(base_split.val_normal),
    }
    written.append(_write_canonical_json(out_dir / "val_split.json", val_payload))

    for k in k_shots:
        for seed in seeds:
            k_ids = sample_k_shot(base_split.train_fit, k, seed)
            k_payload: dict[str, object] = {
                "protocol_version": PROTOCOL_VERSION,
                "dataset": dataset,
                "category": category,
                "k_shot": k,
                "seed": seed,
                "val_seed": val_seed,
                "train_fit": list(k_ids),
                "val_normal": list(base_split.val_normal),
            }
            written.append(_write_canonical_json(out_dir / f"k{k}_seed{seed}.json", k_payload))

    return written


def load_split_records(
    dataset: DatasetName,
    data_root: str | Path,
    category: str,
    *,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    k_shot: int | str = "full",
    seed: int = 0,
) -> tuple[list[SampleRecord], list[SampleRecord], list[SampleRecord]]:
    """Load ``(train_fit, val_normal, test)`` ``SampleRecord`` lists from saved split JSON."""
    category_split_dir = Path(splits_dir) / dataset / category
    if k_shot in ("full", 0):
        split_file = category_split_dir / "val_split.json"
    elif isinstance(k_shot, int) and not isinstance(k_shot, bool) and k_shot > 0:
        split_file = category_split_dir / f"k{k_shot}_seed{seed}.json"
    else:
        raise SplitError(f"Unsupported k_shot={k_shot!r}; expected 'full' or non-negative int")

    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    payload = json.loads(split_file.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset or payload.get("category") != category:
        raise SplitError(f"Split metadata mismatch in {split_file}")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise SplitError(f"Protocol version mismatch in {split_file}")

    train_ids = set() if k_shot == 0 else set(payload["train_fit"])
    val_ids = set(payload["val_normal"])
    if not train_ids.isdisjoint(val_ids):
        raise SplitError(f"Overlap between train_fit and val_normal in {split_file}")

    all_train = load_dataset_samples(dataset, data_root, category, split="train")
    by_id = {rec.sample_id: rec for rec in all_train}

    missing_ids = (train_ids | val_ids) - set(by_id.keys())
    if missing_ids:
        raise SplitError(f"Split {split_file} references missing train IDs: {sorted(missing_ids)}")

    train_records = [replace(by_id[sid], split="train") for sid in sorted(train_ids)]
    val_records = [replace(by_id[sid], split="val") for sid in sorted(val_ids)]
    test_records = load_dataset_samples(dataset, data_root, category, split="test")

    assert_no_test_leakage(train_records, context="train_fit")
    assert_no_test_leakage(val_records, context="val_normal")
    return train_records, val_records, test_records


def _main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic val and k-shot splits.")
    parser.add_argument("--dataset", choices=("mvtec", "visa"), required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--val-seed", type=int, default=DEFAULT_VAL_SEED)
    parser.add_argument("--k-shots", type=int, nargs="+", default=list(DEFAULT_K_SHOTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    default_root = Path("data/mvtec_ad") if args.dataset == "mvtec" else Path("data/visa")
    data_root = args.root or default_root
    categories = (args.category,) if args.category else DATASET_CATEGORIES[args.dataset]

    total_files = 0
    for cat in categories:
        written = create_category_splits(
            args.dataset,
            data_root,
            cat,
            splits_dir=args.splits_dir,
            val_fraction=args.val_fraction,
            val_seed=args.val_seed,
            k_shots=args.k_shots,
            seeds=args.seeds,
        )
        total_files += len(written)
    print(f"Wrote {total_files} split files for {args.dataset} under {args.splits_dir}")


if __name__ == "__main__":
    _main()
