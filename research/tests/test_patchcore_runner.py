"""Unit and integration tests for the Anomalib PatchCore runner (``T-010``)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research.benchmarks.anomalib_data import collate_image_batch, load_split_dataloaders
from research.benchmarks.patchcore_runner import (
    compute_normal_threshold,
    run_patchcore_experiment,
    summarize_val_stats,
)
from research.benchmarks.run import _apply_cli_overrides
from research.common.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PreprocessConfig,
    load_config,
)
from research.common.results import read_results
from research.data.datasets import AnomalyImageDataset, SampleRecord
from research.data.make_splits import LeakageError, create_category_splits


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _create_synthetic_mvtec_category(
    data_dir: Path,
    category: str = "bottle",
    *,
    size: int = 64,
    n_train: int = 12,
) -> Path:
    cat_dir = data_dir / "mvtec_ad" / category
    rng = np.random.default_rng(123)

    # Normal training images: clean mid-gray with slight texture.
    for idx in range(n_train):
        img = np.full((size, size, 3), 120, dtype=np.uint8)
        noise = rng.integers(-6, 7, size=(size, size, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        _write_png(cat_dir / "train" / "good" / f"{idx:03d}.png", img)

    # Normal test images.
    for idx in range(3):
        img = np.full((size, size, 3), 120, dtype=np.uint8)
        _write_png(cat_dir / "test" / "good" / f"{idx:03d}.png", img)

    # Defective test images with a bright square anomaly + matching GT mask.
    for idx in range(3):
        img = np.full((size, size, 3), 120, dtype=np.uint8)
        img[20:44, 20:44, :] = 245
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[20:44, 20:44] = 255
        _write_png(cat_dir / "test" / "crack" / f"{idx:03d}.png", img)
        _write_png(cat_dir / "ground_truth" / "crack" / f"{idx:03d}_mask.png", mask)

    return data_dir


def test_patchcore_bottle_config_is_valid() -> None:
    """Checked-in bottle YAML config must load and validate cleanly."""
    cfg = load_config("research/configs/patchcore_bottle.yaml")
    assert cfg.model.name == "patchcore"
    assert cfg.data.dataset == "mvtec"
    assert cfg.data.category == "bottle"
    assert cfg.preprocess.resize == 256
    assert cfg.preprocess.crop_size == 224


def test_compute_normal_threshold_and_stats_reject_invalid() -> None:
    """Threshold derivation uses val-normal scores only and fails loud on malformed inputs."""
    scores = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert compute_normal_threshold(scores, "val_p99") == pytest.approx(np.percentile(scores, 99.0))
    expected_mean3 = float(np.mean(scores) + 3.0 * np.std(scores))
    assert compute_normal_threshold(scores, "mean_3std") == pytest.approx(expected_mean3)

    stats = summarize_val_stats(scores, np.ones((4, 8, 8), dtype=np.float32))
    assert stats["n_val"] == 4
    assert stats["map_mean"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="non-empty"):
        compute_normal_threshold(np.array([]), "val_p99")
    with pytest.raises(ValueError, match="finite"):
        compute_normal_threshold(np.array([1.0, np.nan]), "val_p99")
    with pytest.raises(ValueError, match="Unsupported threshold method"):
        compute_normal_threshold(scores, "oracle")


def test_anomalib_data_loader_enforces_split_isolation_and_collates(tmp_path: Path) -> None:
    """DataLoader bridge must isolate train_fit/val_normal from test and build ImageBatch."""
    data_dir = _create_synthetic_mvtec_category(tmp_path / "data")
    cfg = ExperimentConfig(
        experiment_name="split_check",
        data=DataConfig(dataset="mvtec", category="bottle", data_dir=str(data_dir)),
        preprocess=PreprocessConfig(resize=64, crop_size=48),
    )
    loaders = load_split_dataloaders(cfg, batch_size=4)
    assert loaders.train_fit is not None
    assert all(r.split == "train" and r.label == 0 for r in loaders.train_records)
    assert all(r.split == "val" and r.label == 0 for r in loaders.val_records)
    assert any(r.split == "test" for r in loaders.test_records)

    batch = next(iter(loaders.test))
    assert batch.image.shape[1:] == (3, 48, 48)
    assert batch.gt_mask is not None
    assert batch.gt_mask.shape[1:] == (48, 48)

    # Passing a leaked test sample to collate still works structurally; load_split_records guards.
    leaked_rec = replace(loaders.test_records[0], split="test")
    assert isinstance(leaked_rec, SampleRecord)
    ds = AnomalyImageDataset([leaked_rec], preprocess=cfg.preprocess)
    collated = collate_image_batch([ds[0]])
    assert collated.image.shape == (1, 3, 48, 48)


def test_load_split_dataloaders_blocks_leaked_test_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any test sample leaking into train_fit or val_normal must raise LeakageError."""
    data_dir = _create_synthetic_mvtec_category(tmp_path / "data")
    create_category_splits("mvtec", data_dir / "mvtec_ad", "bottle", splits_dir=data_dir / "splits")
    cfg = ExperimentConfig(
        data=DataConfig(dataset="mvtec", category="bottle", data_dir=str(data_dir)),
        preprocess=PreprocessConfig(resize=64, crop_size=48),
    )

    leaked = SampleRecord(
        dataset="mvtec",
        category="bottle",
        split="test",
        image_path=data_dir / "mvtec_ad" / "bottle" / "test" / "good" / "000.png",
        mask_path=None,
        label=0,
        defect_type="good",
        sample_id="test/good/000.png",
    )
    monkeypatch.setattr(
        "research.benchmarks.anomalib_data.load_split_records",
        lambda *args, **kwargs: ([leaked], [replace(leaked, split="val")], [leaked]),
    )
    with pytest.raises(LeakageError):
        load_split_dataloaders(cfg)


def test_patchcore_runner_end_to_end_synthetic_offline(tmp_path: Path) -> None:
    """Full fit -> val_normal threshold -> test eval -> artifact & CSV export without downloads."""
    data_dir = _create_synthetic_mvtec_category(tmp_path / "data", size=64, n_train=12)
    results_csv = tmp_path / "results.csv"
    runs_dir = tmp_path / "runs"
    artifact_root = tmp_path / "artifacts"

    cfg = ExperimentConfig(
        experiment_name="synthetic_patchcore",
        seed=0,
        output_dir=str(runs_dir),
        results_csv=str(results_csv),
        data=DataConfig(dataset="mvtec", category="bottle", data_dir=str(data_dir)),
        preprocess=PreprocessConfig(resize=64, crop_size=48),
        model=ModelConfig(
            name="patchcore",
            backbone="resnet18",
            coreset_sampling_ratio=0.5,
            layers=("layer2",),
            params={"num_neighbors": 3, "batch_size": 4, "num_workers": 0, "pre_trained": False},
        ),
    )

    row = run_patchcore_experiment(
        cfg,
        artifact_root=artifact_root,
        device="cpu",
        write_results=True,
    )
    assert row.image_auroc is not None and 0.0 <= row.image_auroc <= 1.0
    assert row.pixel_auroc is not None and 0.0 <= row.pixel_auroc <= 1.0
    assert row.aupro is not None and 0.0 <= row.aupro <= 1.0
    assert row.f1_max_image is not None and 0.0 <= row.f1_max_image <= 1.0
    assert row.f1_max_pixel is not None and 0.0 <= row.f1_max_pixel <= 1.0
    assert row.bank_size_mb is not None and row.bank_size_mb > 0.0

    # Artifacts and threshold provenance exist.
    art_dir = artifact_root / "patchcore" / "mvtec" / "bottle"
    for fname in ("model.pt", "stats.json", "threshold.json", "meta.json"):
        assert (art_dir / fname).is_file()
    thr_payload = json.loads((art_dir / "threshold.json").read_text(encoding="utf-8"))
    assert thr_payload["source"] == "val_normal"

    # Exactly one valid row written to CSV.
    rows = read_results(results_csv)
    assert len(rows) == 1
    assert rows[0].model == "patchcore"
    assert rows[0].category == "bottle"

    # Re-running with the same seed reproduces metrics and does not overwrite run_dir.
    row_repeat = run_patchcore_experiment(
        cfg,
        artifact_root=artifact_root,
        device="cpu",
        write_results=False,
    )
    assert row_repeat.image_auroc == pytest.approx(row.image_auroc)
    assert row_repeat.pixel_auroc == pytest.approx(row.pixel_auroc)
    assert row_repeat.aupro == pytest.approx(row.aupro)
    assert row_repeat.run_id != row.run_id
    assert len(list(runs_dir.iterdir())) == 2


def test_split_mismatch_and_few_shot_artifact_isolation(tmp_path: Path) -> None:
    """Reject stale val_split metadata mismatch and protect canonical artifacts on k-shot."""
    data_dir = _create_synthetic_mvtec_category(tmp_path / "data", size=64, n_train=12)
    create_category_splits("mvtec", data_dir / "mvtec_ad", "bottle", splits_dir=data_dir / "splits")

    bad_val_cfg = ExperimentConfig(
        data=DataConfig(dataset="mvtec", category="bottle", data_dir=str(data_dir), val_seed=9),
        preprocess=PreprocessConfig(resize=64, crop_size=48),
    )
    with pytest.raises(ValueError, match="differs from config"):
        load_split_dataloaders(bad_val_cfg)

    runs_dir = tmp_path / "runs"
    artifact_root = tmp_path / "artifacts"
    few_cfg = ExperimentConfig(
        experiment_name="k5_run",
        seed=0,
        output_dir=str(runs_dir),
        data=DataConfig(dataset="mvtec", category="bottle", data_dir=str(data_dir), k_shot=5),
        preprocess=PreprocessConfig(resize=64, crop_size=48),
        model=ModelConfig(
            name="patchcore",
            backbone="resnet18",
            coreset_sampling_ratio=0.5,
            layers=("layer2",),
            params={"num_neighbors": 3, "batch_size": 4, "num_workers": 0, "pre_trained": False},
        ),
    )
    row = run_patchcore_experiment(
        few_cfg, artifact_root=artifact_root, device="cpu", write_results=False
    )
    assert row.k_shot == 5
    assert not (artifact_root / "patchcore" / "mvtec" / "bottle").exists()
    assert any((runs_dir / d.name / "artifacts" / "model.pt").is_file() for d in runs_dir.iterdir())


def test_cli_overrides_update_experiment_name_and_reject_bad_kshot() -> None:
    """CLI override helper must keep experiment_name accurate and fail loud on bad k-shot."""
    base = load_config("research/configs/patchcore_bottle.yaml")
    ns = argparse.Namespace(
        model=None,
        dataset="visa",
        category="candle",
        k_shot="5",
        seed=2,
        results_csv=None,
    )
    updated = _apply_cli_overrides(base, ns)
    assert updated.experiment_name == "patchcore_visa_candle_k5_seed2"
    assert updated.data.k_shot == 5
    assert updated.seed == 2

    bad_ns = argparse.Namespace(
        model=None,
        dataset=None,
        category=None,
        k_shot="bad",
        seed=None,
        results_csv=None,
    )
    with pytest.raises(ValueError, match="--k-shot"):
        _apply_cli_overrides(base, bad_ns)


@pytest.mark.slow
def test_patchcore_runner_real_mvtec_bottle(tmp_path: Path) -> None:
    """Real end-to-end PatchCore integration on mvtec/bottle if staged locally."""
    if not Path("data/mvtec_ad/bottle").is_dir():
        pytest.skip("MVTec AD bottle dataset not present under data/mvtec_ad/bottle")

    base_cfg = load_config("research/configs/patchcore_bottle.yaml")
    cfg = replace(
        base_cfg,
        output_dir=str(tmp_path / "runs"),
        results_csv=str(tmp_path / "results.csv"),
    )
    row = run_patchcore_experiment(
        cfg,
        artifact_root=tmp_path / "artifacts",
        device="cpu",
        write_results=True,
    )
    assert row.image_auroc is not None and row.image_auroc >= 0.90
    assert row.pixel_auroc is not None and row.pixel_auroc >= 0.90
    assert row.aupro is not None and row.aupro >= 0.80
