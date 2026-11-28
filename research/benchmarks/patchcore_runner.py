"""Anomalib PatchCore runner (``T-010``): fit on normals, calibrate on val, evaluate on test."""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from anomalib.data import ImageBatch
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.models.components import AnomalibModule
from torch.utils.data import DataLoader

from research.benchmarks.anomalib_data import load_split_dataloaders
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


def compute_normal_threshold(val_scores: np.ndarray, method: str = "val_p99") -> float:
    """Derive deployment anomaly threshold from normal validation scores only (``docs/05`` §3.5)."""
    scores = np.asarray(val_scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError("val_scores must be non-empty")
    if not np.all(np.isfinite(scores)):
        raise ValueError("val_scores must be finite (no NaN or Inf)")

    if method == "val_p99":
        return float(np.percentile(scores, 99.0))
    if method == "mean_3std":
        return float(np.mean(scores) + 3.0 * np.std(scores))
    raise ValueError(f"Unsupported threshold method {method!r}; expected 'val_p99' or 'mean_3std'")


def summarize_val_stats(
    val_scores: np.ndarray,
    val_maps: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute validation-normal summary statistics for calibration and thresholding."""
    scores = np.asarray(val_scores, dtype=np.float64).ravel()
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("val_scores must be non-empty and finite")

    stats: dict[str, Any] = {
        "n_val": int(scores.size),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_p50": float(np.percentile(scores, 50.0)),
        "score_p95": float(np.percentile(scores, 95.0)),
        "score_p99": float(np.percentile(scores, 99.0)),
        "score_max": float(np.max(scores)),
    }
    if val_maps is not None:
        maps = np.asarray(val_maps, dtype=np.float64)
        if maps.size == 0 or not np.all(np.isfinite(maps)):
            raise ValueError("val_maps must be non-empty and finite")
        stats["map_mean"] = float(np.mean(maps))
        stats["map_std"] = float(np.std(maps))
        stats["map_p99"] = float(np.percentile(maps, 99.0))
    return stats


def build_patchcore_model(cfg: ExperimentConfig) -> Patchcore:
    """Instantiate Anomalib ``Patchcore`` with external aligned preprocessing disabled."""
    params = cfg.model.params
    num_neighbors = int(params.get("num_neighbors", 9))
    pre_trained = bool(params.get("pre_trained", True))
    return Patchcore(
        backbone=cfg.model.backbone,
        layers=list(cfg.model.layers),
        pre_trained=pre_trained,
        coreset_sampling_ratio=cfg.model.coreset_sampling_ratio,
        num_neighbors=num_neighbors,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )


def predict_on_dataloader(
    model: AnomalibModule,
    dataloader: DataLoader[ImageBatch],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run ``AnomalibModule`` inference on all batches and return scores, maps, labels, masks."""
    model.eval()
    model.to(device)

    all_scores: list[np.ndarray] = []
    all_maps: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch.image.to(device)
            outputs = model.model(images)
            pred_score = outputs.pred_score.detach().cpu().numpy().astype(np.float64)
            anomaly_map = outputs.anomaly_map.detach().cpu().numpy().astype(np.float32)
            if anomaly_map.ndim == 4 and anomaly_map.shape[1] == 1:
                anomaly_map = anomaly_map[:, 0, :, :]

            if batch.gt_label is None or batch.gt_mask is None:
                raise ValueError("Expected gt_label and gt_mask on evaluation ImageBatch")

            all_scores.append(pred_score.reshape(-1))
            all_maps.append(anomaly_map)
            all_labels.append(batch.gt_label.detach().cpu().numpy().astype(np.int8).reshape(-1))
            all_masks.append(batch.gt_mask.detach().cpu().numpy().astype(np.uint8))

    if not all_scores:
        raise ValueError("Evaluation dataloader produced zero batches")

    return (
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_maps, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_masks, axis=0),
    )


def save_patchcore_artifacts(
    artifact_dir: Path,
    model: Patchcore,
    cfg: ExperimentConfig,
    val_stats: dict[str, Any],
    val_scores: np.ndarray,
    *,
    git_sha: str,
) -> Path:
    """Save PatchCore checkpoint/memory-bank and normal-only stats/thresholds per ``docs/02`` §5."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    num_neighbors = int(cfg.model.params.get("num_neighbors", 9))

    model_path = artifact_dir / "model.pt"
    torch.save(
        {
            "model_name": cfg.model.name,
            "backbone": cfg.model.backbone,
            "layers": list(cfg.model.layers),
            "coreset_sampling_ratio": cfg.model.coreset_sampling_ratio,
            "num_neighbors": num_neighbors,
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
                "layers": list(cfg.model.layers),
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


def run_patchcore_experiment(
    cfg: ExperimentConfig,
    *,
    artifact_root: Path = Path("artifacts"),
    device: str | None = None,
    write_results: bool = True,
    results_path: str | Path | None = None,
    profile: bool = False,
    warmup_runs: int = 10,
    timed_runs: int = 100,
) -> ResultRow:
    """Run one reproducible PatchCore fit/val/test experiment and optionally append CSV row."""
    if cfg.model.name != "patchcore":
        raise ValueError(
            f"run_patchcore_experiment requires model.name=='patchcore', got {cfg.model.name!r}"
        )

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

    batch_size = int(cfg.model.params.get("batch_size", 16))
    num_workers = int(cfg.model.params.get("num_workers", 0))
    splits = load_split_dataloaders(cfg, batch_size=batch_size, num_workers=num_workers)
    if splits.train_fit is None:
        raise ValueError("PatchCore requires non-empty train_fit samples (k_shot cannot be 0)")

    model = build_patchcore_model(cfg)
    engine = Engine(
        default_root_dir=run_dir / "anomalib",
        accelerator=exec_device,
        devices=1,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    t_fit_start = time.perf_counter()
    engine.fit(model=model, train_dataloaders=splits.train_fit)
    fit_time_s = time.perf_counter() - t_fit_start

    torch_device = torch.device(exec_device)
    val_scores, val_maps, val_labels, _ = predict_on_dataloader(
        model, splits.val_normal, device=torch_device
    )
    if np.any(val_labels != 0):
        raise ValueError("Validation split contained non-normal labels")
    val_stats = summarize_val_stats(val_scores, val_maps)

    # Keep canonical artifacts/{model}/{dataset}/{category}/ strictly for clean full-shot runs.
    if cfg.data.k_shot == "full" and cfg.shift.corruption == "none":
        artifact_dir = Path(artifact_root) / cfg.model.name / cfg.data.dataset / cfg.data.category
    else:
        artifact_dir = run_dir / "artifacts"
    model_path = save_patchcore_artifacts(
        artifact_dir,
        model,
        cfg,
        val_stats,
        val_scores,
        git_sha=git_sha,
    )
    bank_size_mb = float(model_path.stat().st_size) / (1024.0 * 1024.0)

    test_scores, test_maps, test_labels, test_masks = predict_on_dataloader(
        model, splits.test, device=torch_device
    )

    prof: ProfileResult | None = None
    if profile:
        crop = cfg.preprocess.crop_size
        sample = torch.randn((1, 3, crop, crop), dtype=torch.float32)
        bank_tensor = model.model.memory_bank
        tensor_mb = float(bank_tensor.numel() * bank_tensor.element_size()) / (1024.0 * 1024.0)
        prof = profile_inference(
            model.model,
            sample,
            device=exec_device,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
            model=model.model,
            bank_path=model_path,
            fit_time_s=fit_time_s,
            memory_bank_tensor_mb=tensor_mb,
        )

    notes = (
        f"n_train={len(splits.train_records)},n_val={len(splits.val_records)},"
        f"n_test={len(splits.test_records)},fit_time_s={fit_time_s:.2f}"
    )
    if prof is not None:
        notes = f"{notes},params={prof.param_count}"

    measured_bank_mb = (
        prof.bank_size_mb if prof is not None and prof.bank_size_mb is not None else bank_size_mb
    )
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
        bank_size_mb=measured_bank_mb,
        notes=notes,
    )

    if write_results:
        target_csv = Path(results_path) if results_path is not None else Path(cfg.results_csv)
        if results_path is None and not cfg.results_csv:
            target_csv = DEFAULT_RESULTS_PATH
        append_result_row(row, target_csv)

    return row
