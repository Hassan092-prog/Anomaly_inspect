"""Anomalib EfficientAD-S runner (``T-011``): train student/AE, calibrate quantiles, eval."""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from anomalib.engine import Engine
from anomalib.models import EfficientAd

from research.benchmarks.anomalib_data import SplitLightningDataModule, load_split_dataloaders
from research.benchmarks.patchcore_runner import (
    compute_normal_threshold,
    predict_on_dataloader,
    summarize_val_stats,
)
from research.benchmarks.profiler import ProfileResult, profile_inference, resolve_device_string
from research.common.config import (
    ExperimentConfig,
    collect_env_versions,
    config_hash,
    config_to_dict,
    get_git_sha,
    write_run_metadata,
)
from research.common.results import (
    DEFAULT_RESULTS_PATH,
    ResultRow,
    append_result_row,
    utc_timestamp_iso,
)
from research.common.seed import set_seed
from research.metrics import (
    compute_aupro,
    compute_f1_max,
    compute_image_auroc,
    compute_pixel_auroc,
)


def validate_efficientad_config(cfg: ExperimentConfig) -> None:
    """Validate ``EfficientAd``-specific architectural and preprocessing constraints."""
    if cfg.model.name not in {"efficientad_s", "efficientad"}:
        raise ValueError(
            f"run_efficientad_experiment requires model.name in "
            f"{{'efficientad_s', 'efficientad'}}, got {cfg.model.name!r}"
        )
    if cfg.preprocess.crop_size < 256:
        raise ValueError(
            "EfficientAd Encoder requires crop_size >= 256 (8x8 conv after 5x stride-2 "
            f"downsampling), got {cfg.preprocess.crop_size}"
        )
    if cfg.preprocess.normalize_mean != (0.0, 0.0, 0.0) or cfg.preprocess.normalize_std != (
        1.0,
        1.0,
        1.0,
    ):
        raise ValueError(
            "EfficientAd normalizes inside PDN forward() and expects [0, 1] inputs "
            "(normalize_mean=(0,0,0), normalize_std=(1,1,1))"
        )
    model_size = str(cfg.model.params.get("model_size", "small"))
    if model_size not in {"small", "medium"}:
        raise ValueError(f"model.params.model_size must be 'small' or 'medium', got {model_size!r}")
    out_ch = int(cfg.model.params.get("teacher_out_channels", 384))
    if out_ch <= 0:
        raise ValueError(f"model.params.teacher_out_channels must be > 0, got {out_ch}")


def build_efficientad_model(cfg: ExperimentConfig) -> EfficientAd:
    """Instantiate Anomalib ``EfficientAd`` with external aligned [0, 1] preprocessing."""
    validate_efficientad_config(cfg)
    params = cfg.model.params
    return EfficientAd(
        imagenet_dir=str(params.get("imagenet_dir", "data/imagenette")),
        teacher_out_channels=int(params.get("teacher_out_channels", 384)),
        model_size=str(params.get("model_size", "small")),
        lr=float(params.get("lr", 1e-4)),
        weight_decay=float(params.get("weight_decay", 1e-5)),
        padding=bool(params.get("padding", False)),
        pad_maps=bool(params.get("pad_maps", True)),
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )


def save_efficientad_artifacts(
    artifact_dir: Path,
    model: EfficientAd,
    cfg: ExperimentConfig,
    val_stats: dict[str, Any],
    val_scores: np.ndarray,
    *,
    git_sha: str,
    train_time_s: float,
) -> Path:
    """Save EfficientAD checkpoint and normal-only stats/thresholds per ``docs/02`` §5."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    params = cfg.model.params

    model_path = artifact_dir / "model.pt"
    torch.save(
        {
            "model_name": cfg.model.name,
            "model_size": str(params.get("model_size", "small")),
            "teacher_out_channels": int(params.get("teacher_out_channels", 384)),
            "padding": bool(params.get("padding", False)),
            "pad_maps": bool(params.get("pad_maps", True)),
            "preprocess": config_to_dict(cfg)["preprocess"],
            "state_dict": model.model.state_dict(),
        },
        model_path,
    )

    val_p99 = compute_normal_threshold(val_scores, method="val_p99")
    mean_3std = compute_normal_threshold(val_scores, method="mean_3std")
    chosen = val_p99 if cfg.threshold_method == "val_p99" else mean_3std
    if not math.isfinite(chosen):
        raise ValueError(f"Non-finite derived threshold: {chosen}")

    (artifact_dir / "stats.json").write_text(
        json.dumps(val_stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "threshold.json").write_text(
        json.dumps(
            {
                "method": cfg.threshold_method,
                "threshold": chosen,
                "val_p99": val_p99,
                "mean_3std": mean_3std,
                "source": "val_normal",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "meta.json").write_text(
        json.dumps(
            {
                "config_hash": config_hash(cfg),
                "git_sha": git_sha,
                "dataset": cfg.data.dataset,
                "category": cfg.data.category,
                "k_shot": cfg.data.k_shot,
                "seed": cfg.seed,
                "model": cfg.model.name,
                "backbone": cfg.model.backbone,
                "train_time_s": train_time_s,
                "created_at": utc_timestamp_iso(),
                "env": collect_env_versions(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return model_path


def run_efficientad_experiment(
    cfg: ExperimentConfig,
    *,
    artifact_root: Path = Path("artifacts"),
    device: str | None = None,
    write_results: bool = True,
    results_path: str | Path | None = None,
    profile: bool = False,
    warmup_runs: int = 10,
    timed_runs: int = 100,
    model_override: EfficientAd | None = None,
) -> ResultRow:
    """Train EfficientAD-S on normal images, calibrate on val_normal, and evaluate on test."""
    validate_efficientad_config(cfg)
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    exec_device, hw_desc = resolve_device_string(device)

    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_run_id = f"{cfg.experiment_name}_{run_stamp}_{config_hash(cfg)[:6]}"
    run_id = base_run_id
    run_dir = Path(cfg.output_dir) / run_id
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_id = f"{base_run_id}_{suffix}"
        run_dir = Path(cfg.output_dir) / run_id
    write_run_metadata(cfg, run_dir)
    git_sha = get_git_sha()

    params = cfg.model.params
    eval_batch_size = int(params.get("eval_batch_size", 16))
    num_workers = int(params.get("num_workers", 0))
    splits = load_split_dataloaders(
        cfg,
        batch_size=1,  # EfficientAd enforces train_batch_size == 1
        eval_batch_size=eval_batch_size,
        train_shuffle=True,
        num_workers=num_workers,
    )
    if splits.train_fit is None:
        raise ValueError("EfficientAD requires non-empty train_fit samples (k_shot cannot be 0)")

    max_epochs = int(params.get("max_epochs", 10))
    max_steps = int(params.get("max_steps", -1))
    if max_epochs < 0 and max_steps < 0:
        raise ValueError(
            "EfficientAd requires either max_epochs > 0 or max_steps >= 2 (both are < 0)"
        )
    if max_epochs < 0:
        effective_steps = max_steps
    elif max_steps < 0:
        effective_steps = max_epochs * len(splits.train_records)
    else:
        effective_steps = min(max_steps, max_epochs * len(splits.train_records))
    if effective_steps < 2:
        raise ValueError(
            "EfficientAd StepLR scheduler requires at least 2 total training steps "
            f"(got {effective_steps}; increase max_epochs or max_steps)"
        )

    datamodule = SplitLightningDataModule(
        name=cfg.data.dataset,
        category=cfg.data.category,
        splits=splits,
        train_batch_size=1,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )

    model = model_override if model_override is not None else build_efficientad_model(cfg)

    engine = Engine(
        default_root_dir=run_dir / "anomalib",
        accelerator=exec_device,
        devices=1,
        logger=False,
        max_epochs=max_epochs,
        max_steps=max_steps,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    t_train_start = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    train_time_s = time.perf_counter() - t_train_start

    torch_device = torch.device(exec_device)
    val_scores, val_maps, val_labels, _ = predict_on_dataloader(
        model, splits.val_normal, device=torch_device
    )
    if np.any(val_labels != 0):
        raise ValueError("Validation split contained non-normal labels")
    val_stats = summarize_val_stats(val_scores, val_maps)

    if cfg.data.k_shot == "full" and cfg.shift.corruption == "none":
        artifact_dir = Path(artifact_root) / cfg.model.name / cfg.data.dataset / cfg.data.category
    else:
        artifact_dir = run_dir / "artifacts"

    model_path = save_efficientad_artifacts(
        artifact_dir,
        model,
        cfg,
        val_stats,
        val_scores,
        git_sha=git_sha,
        train_time_s=train_time_s,
    )
    model_size_mb = float(model_path.stat().st_size) / (1024.0 * 1024.0)

    test_scores, test_maps, test_labels, test_masks = predict_on_dataloader(
        model, splits.test, device=torch_device
    )

    prof: ProfileResult | None = None
    if profile:
        crop = cfg.preprocess.crop_size
        sample = torch.rand((1, 3, crop, crop), dtype=torch.float32)
        prof = profile_inference(
            model.model,
            sample,
            device=exec_device,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
            model=model.model,
            bank_path=model_path,
            fit_time_s=train_time_s,
        )

    notes = (
        f"n_train={len(splits.train_records)},n_val={len(splits.val_records)},"
        f"n_test={len(splits.test_records)},train_time_s={train_time_s:.2f}"
    )
    if prof is not None:
        notes = f"{notes},params={prof.param_count}"

    row = ResultRow(
        run_id=run_id,
        git_sha=git_sha,
        timestamp=utc_timestamp_iso(),
        model=cfg.model.name,
        backbone=cfg.model.backbone,
        dataset=cfg.data.dataset,
        category=cfg.data.category,
        k_shot=cfg.data.k_shot,
        seed=cfg.seed,
        shift=cfg.shift.corruption,
        severity=cfg.shift.severity,
        fusion=cfg.fusion.method,
        fusion_signals="+".join(cfg.fusion.signals),
        threshold_type=cfg.threshold_method,
        device=hw_desc,
        image_auroc=compute_image_auroc(test_labels, test_scores),
        pixel_auroc=compute_pixel_auroc(test_masks, test_maps),
        aupro=compute_aupro(test_masks, test_maps),
        f1_max_image=compute_f1_max(test_labels, test_scores),
        f1_max_pixel=compute_f1_max(test_masks, test_maps),
        latency_ms_p50=prof.latency_ms_p50 if prof is not None else None,
        latency_ms_p95=prof.latency_ms_p95 if prof is not None else None,
        peak_mem_mb=prof.peak_mem_mb if prof is not None else None,
        bank_size_mb=prof.bank_size_mb if prof is not None else model_size_mb,
        notes=notes,
    )

    if write_results:
        target_csv = Path(results_path) if results_path is not None else Path(cfg.results_csv)
        if results_path is None and not cfg.results_csv:
            target_csv = DEFAULT_RESULTS_PATH
        append_result_row(row, target_csv)

    return row
