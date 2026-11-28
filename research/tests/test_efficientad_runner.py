"""Unit tests for the Anomalib EfficientAD-S runner (``T-011``)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research.benchmarks.efficientad_runner import (
    build_efficientad_model,
    run_efficientad_experiment,
    validate_efficientad_config,
)
from research.common.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PreprocessConfig,
    load_config,
)
from research.common.results import read_results


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _create_synthetic_mvtec_and_imagenette(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    cat_dir = data_dir / "mvtec_ad" / "bottle"
    size = 256
    rng = np.random.default_rng(17)

    for idx in range(4):
        img = np.full((size, size, 3), 128, dtype=np.uint8)
        noise = rng.integers(-4, 5, size=(size, size, 3), dtype=np.int16)
        arr = np.clip(img + noise, 0, 255).astype(np.uint8)
        _write_png(cat_dir / "train" / "good" / f"{idx:03d}.png", arr)

    for idx in range(2):
        normal = np.full((size, size, 3), 128, dtype=np.uint8)
        _write_png(cat_dir / "test" / "good" / f"{idx:03d}.png", normal)

    for idx in range(2):
        anom = np.full((size, size, 3), 128, dtype=np.uint8)
        anom[80:140, 80:140, :] = 240
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[80:140, 80:140] = 255
        _write_png(cat_dir / "test" / "crack" / f"{idx:03d}.png", anom)
        _write_png(cat_dir / "ground_truth" / "crack" / f"{idx:03d}_mask.png", mask)

    imagenet_dir = tmp_path / "imagenette"
    for idx in range(2):
        dummy = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        _write_png(imagenet_dir / "class0" / f"img_{idx}.png", dummy)

    return data_dir, imagenet_dir


def test_efficientad_bottle_config_is_valid() -> None:
    """Checked-in EfficientAD-S config loads and passes architectural constraints."""
    cfg = load_config("research/configs/efficientad_bottle.yaml")
    assert cfg.model.name == "efficientad_s"
    assert cfg.preprocess.crop_size == 256
    assert cfg.preprocess.normalize_mean == (0.0, 0.0, 0.0)
    assert cfg.preprocess.normalize_std == (1.0, 1.0, 1.0)
    validate_efficientad_config(cfg)


def test_efficientad_rejects_invalid_crop_or_double_normalization() -> None:
    """Must fail loud if crop_size < 256 or ImageNet normalization is left enabled."""
    base = load_config("research/configs/efficientad_bottle.yaml")

    bad_crop = replace(base, preprocess=PreprocessConfig(resize=256, crop_size=224))
    with pytest.raises(ValueError, match="crop_size >= 256"):
        validate_efficientad_config(bad_crop)

    bad_norm = replace(
        base,
        preprocess=PreprocessConfig(
            resize=256,
            crop_size=256,
            normalize_mean=(0.485, 0.456, 0.406),
            normalize_std=(0.229, 0.224, 0.225),
        ),
    )
    with pytest.raises(ValueError, match=r"\[0, 1\] inputs"):
        validate_efficientad_config(bad_norm)

    bad_size = replace(
        base,
        model=replace(base.model, params={**base.model.params, "model_size": "xl"}),
    )
    with pytest.raises(ValueError, match="model_size"):
        validate_efficientad_config(bad_size)


def test_efficientad_runner_end_to_end_synthetic_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline 2-step EfficientAD-S train -> val_normal quantile/threshold -> test -> CSV."""
    data_dir, imagenet_dir = _create_synthetic_mvtec_and_imagenette(tmp_path)
    results_csv = tmp_path / "results.csv"
    runs_dir = tmp_path / "runs"
    artifact_root = tmp_path / "artifacts"

    cfg = ExperimentConfig(
        experiment_name="synthetic_efficientad",
        seed=0,
        output_dir=str(runs_dir),
        results_csv=str(results_csv),
        data=DataConfig(dataset="mvtec", category="bottle", data_dir=str(data_dir)),
        preprocess=PreprocessConfig(
            resize=256,
            crop_size=256,
            normalize_mean=(0.0, 0.0, 0.0),
            normalize_std=(1.0, 1.0, 1.0),
        ),
        model=ModelConfig(
            name="efficientad_s",
            backbone="pdn_small",
            coreset_sampling_ratio=1.0,
            layers=(),
            params={
                "model_size": "small",
                "teacher_out_channels": 384,
                "imagenet_dir": str(imagenet_dir),
                "max_epochs": 1,
                "max_steps": 2,
                "eval_batch_size": 2,
                "num_workers": 0,
                "padding": False,
                "pad_maps": True,
            },
        ),
    )

    model = build_efficientad_model(cfg)
    # Skip downloading pretrained teacher zip in unit tests; keep random init weights.
    monkeypatch.setattr(model, "prepare_pretrained_model", lambda: None)

    row = run_efficientad_experiment(
        cfg,
        artifact_root=artifact_root,
        device="cpu",
        write_results=True,
        profile=True,
        warmup_runs=1,
        timed_runs=2,
        model_override=model,
    )

    assert row.model == "efficientad_s"
    assert row.image_auroc is not None and 0.0 <= row.image_auroc <= 1.0
    assert row.pixel_auroc is not None and 0.0 <= row.pixel_auroc <= 1.0
    assert row.aupro is not None and 0.0 <= row.aupro <= 1.0
    assert row.latency_ms_p50 is not None and row.latency_ms_p50 > 0.0
    assert row.latency_ms_p95 is not None and row.latency_ms_p95 >= row.latency_ms_p50
    assert "train_time_s=" in row.notes

    art_dir = artifact_root / "efficientad_s" / "mvtec" / "bottle"
    for fname in ("model.pt", "stats.json", "threshold.json", "meta.json"):
        assert (art_dir / fname).is_file()
    meta_payload = json.loads((art_dir / "meta.json").read_text(encoding="utf-8"))
    assert "train_time_s" in meta_payload and meta_payload["train_time_s"] >= 0.0

    rows = read_results(results_csv)
    assert len(rows) == 1
    assert rows[0].model == "efficientad_s"

    # Step-controlled training (max_epochs=-1, max_steps=2) must work cleanly.
    step_only_cfg = replace(
        cfg,
        model=replace(cfg.model, params={**cfg.model.params, "max_epochs": -1, "max_steps": 2}),
    )
    step_model = build_efficientad_model(step_only_cfg)
    monkeypatch.setattr(step_model, "prepare_pretrained_model", lambda: None)
    step_row = run_efficientad_experiment(
        step_only_cfg,
        artifact_root=artifact_root,
        device="cpu",
        write_results=False,
        model_override=step_model,
    )
    assert step_row.image_auroc is not None

    # Reject unbounded or single-step training that crashes EfficientAd's StepLR(step_size=0).
    for bad_limits in ({"max_epochs": -1, "max_steps": -1}, {"max_steps": 1}):
        bad_cfg = replace(
            cfg,
            model=replace(cfg.model, params={**cfg.model.params, **bad_limits}),
        )
        with pytest.raises(ValueError, match="EfficientAd"):
            run_efficientad_experiment(
                bad_cfg,
                artifact_root=artifact_root,
                device="cpu",
                write_results=False,
                model_override=model,
            )
