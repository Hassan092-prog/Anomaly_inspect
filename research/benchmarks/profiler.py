"""Reusable inference latency and memory profiler (``docs/05`` section 3.9, ``T-015``)."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from research.common.results import ResultRow, append_result_row, utc_timestamp_iso


@dataclass(frozen=True)
class ProfileResult:
    """Efficiency measurements matching ``docs/05_EVALUATION_PROTOCOL.md`` section 3.9."""

    latency_ms_p50: float
    latency_ms_p95: float
    peak_mem_mb: float
    bank_size_mb: float | None
    param_count: int
    fit_time_s: float | None
    device: str
    warmup_runs: int
    timed_runs: int
    memory_bank_tensor_mb: float | None = None


def resolve_device_string(device: str | None = None) -> tuple[str, str]:
    """Resolve execution device (``cpu`` / ``cuda``) and human-readable hardware string."""
    if device is None:
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = device.lower().strip()
        if resolved not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported device {device!r}; expected 'cpu' or 'cuda'")
        if resolved == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if resolved == "cuda":
        hw = f"cuda:{torch.cuda.get_device_name(0)}"
    else:
        proc = platform.processor().strip() or platform.machine()
        hw = f"cpu ({proc})"
    return resolved, hw


def count_parameters(module: torch.nn.Module) -> int:
    """Count total scalar parameters in a PyTorch module."""
    return int(sum(param.numel() for param in module.parameters()))


def measure_disk_size_mb(path: str | Path | None) -> float | None:
    """Return file size on disk in MiB, or ``None`` if ``path`` is ``None``."""
    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Artifact file not found for size measurement: {target}")
    return float(target.stat().st_size) / (1024.0 * 1024.0)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_peak_memory_mb(device: torch.device) -> float:
    """Return peak allocated VRAM (MiB) on CUDA or process high-water RSS (MiB) on CPU.

    Note: on Linux/macOS CPU, ``resource.getrusage`` reports the process lifetime
    maximum RSS, so running profiling in a standalone process isolates inference memory.
    """
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
    if sys.platform.startswith("linux"):
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    if sys.platform == "darwin":
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)
    return 0.0


def profile_inference(
    forward_fn: Callable[[torch.Tensor], Any],
    sample_input: torch.Tensor,
    *,
    device: str | None = None,
    warmup_runs: int = 10,
    timed_runs: int = 100,
    model: torch.nn.Module | None = None,
    param_count: int | None = None,
    bank_path: str | Path | None = None,
    fit_time_s: float | None = None,
    memory_bank_tensor_mb: float | None = None,
) -> ProfileResult:
    """Profile batch-size-1 inference latency (p50/p95 ms) and memory per ``docs/05`` §3.9.

    Args:
        forward_fn: Callable accepting a single ``(1, C, H, W)`` input tensor.
        sample_input: Batch-size-1 sample input tensor with shape ``(1, ...)``.
        device: Target device (``"cpu"``, ``"cuda"``, or ``None`` for auto-select).
        warmup_runs: Number of un-timed warm-up passes (default ``10``).
        timed_runs: Number of timed passes (default ``100``).
        model: Optional ``nn.Module`` to set to ``eval()``, move to device, and count params.
        param_count: Optional explicit parameter count (overrides ``model`` count if set).
        bank_path: Optional saved checkpoint/memory-bank file path to measure size on disk.
        fit_time_s: Optional recorded fit/training duration in seconds.
        memory_bank_tensor_mb: Optional raw memory-bank tensor footprint in MiB.

    Returns:
        Populated ``ProfileResult`` instance.
    """
    if sample_input.ndim < 1 or sample_input.shape[0] != 1:
        raise ValueError(
            f"Profiling requires batch size 1 per docs/05 §3.9, got {tuple(sample_input.shape)}"
        )
    if isinstance(warmup_runs, bool) or not isinstance(warmup_runs, int) or warmup_runs < 0:
        raise ValueError(f"warmup_runs must be a non-negative int, got {warmup_runs!r}")
    if isinstance(timed_runs, bool) or not isinstance(timed_runs, int) or timed_runs < 1:
        raise ValueError(f"timed_runs must be a positive int, got {timed_runs!r}")
    if fit_time_s is not None and (isinstance(fit_time_s, bool) or fit_time_s < 0.0):
        raise ValueError(f"fit_time_s must be >= 0, got {fit_time_s!r}")
    if param_count is not None and (
        isinstance(param_count, bool) or not isinstance(param_count, int) or param_count < 0
    ):
        raise ValueError(f"param_count must be a non-negative int, got {param_count!r}")

    exec_device, hw_desc = resolve_device_string(device)
    torch_device = torch.device(exec_device)

    resolved_params = param_count
    if model is not None:
        model.eval()
        model.to(torch_device)
        if resolved_params is None:
            resolved_params = count_parameters(model)
    if resolved_params is None:
        resolved_params = 0

    disk_size_mb = measure_disk_size_mb(bank_path)
    x = sample_input.to(torch_device)

    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    _sync_if_cuda(torch_device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = forward_fn(x)
        _sync_if_cuda(torch_device)

        latencies_ms: list[float] = []
        for _ in range(timed_runs):
            _sync_if_cuda(torch_device)
            t0 = time.perf_counter()
            _ = forward_fn(x)
            _sync_if_cuda(torch_device)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_mem_mb = _measure_peak_memory_mb(torch_device)
    lat_arr = np.asarray(latencies_ms, dtype=np.float64)

    return ProfileResult(
        latency_ms_p50=float(np.percentile(lat_arr, 50.0)),
        latency_ms_p95=float(np.percentile(lat_arr, 95.0)),
        peak_mem_mb=peak_mem_mb,
        bank_size_mb=disk_size_mb,
        param_count=resolved_params,
        fit_time_s=fit_time_s,
        device=hw_desc,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        memory_bank_tensor_mb=memory_bank_tensor_mb,
    )


def load_patchcore_model_from_artifact(
    artifact_dir: str | Path,
    *,
    device: str | None = None,
) -> tuple[PatchcoreModel, dict[str, Any], Path]:
    """Load a saved PatchCore ``model.pt`` artifact into eval mode offline."""
    target_dir = Path(artifact_dir)
    model_path = target_dir / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved PatchCore artifact not found: {model_path}")

    exec_device, _ = resolve_device_string(device)
    payload = torch.load(model_path, map_location=exec_device, weights_only=True)
    required_keys = {"state_dict", "backbone", "layers", "num_neighbors"}
    if not isinstance(payload, dict) or not required_keys.issubset(payload.keys()):
        raise ValueError(f"Malformed PatchCore checkpoint payload in {model_path}")

    patchcore_model = PatchcoreModel(
        backbone=str(payload["backbone"]),
        layers=list(payload["layers"]),
        pre_trained=False,
        num_neighbors=int(payload["num_neighbors"]),
    )
    patchcore_model.load_state_dict(payload["state_dict"])
    if patchcore_model.memory_bank.numel() == 0:
        raise ValueError(f"Loaded PatchCore memory_bank is empty in {model_path}")
    patchcore_model.eval()
    patchcore_model.to(torch.device(exec_device))
    return patchcore_model, payload, model_path


def profile_patchcore_artifact(
    artifact_dir: str | Path,
    *,
    device: str | None = None,
    warmup_runs: int = 10,
    timed_runs: int = 100,
) -> ProfileResult:
    """Profile a saved PatchCore artifact directory at batch size 1."""
    model, payload, model_path = load_patchcore_model_from_artifact(artifact_dir, device=device)
    preprocess = payload.get("preprocess", {})
    crop_size = int(preprocess.get("crop_size", 224))
    sample_input = torch.randn((1, 3, crop_size, crop_size), dtype=torch.float32)

    bank_tensor = model.memory_bank
    tensor_mb = float(bank_tensor.numel() * bank_tensor.element_size()) / (1024.0 * 1024.0)

    return profile_inference(
        model,
        sample_input,
        device=device,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        model=model,
        bank_path=model_path,
        memory_bank_tensor_mb=tensor_mb,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Profile a saved PatchCore model artifact.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/patchcore/mvtec/bottle"),
        help="Path to artifact directory containing model.pt.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="Target device.")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations.")
    parser.add_argument("--runs", type=int, default=100, help="Timed iterations.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output only.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save profile JSON.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Optional CSV path to append a profiling ResultRow (requires meta.json).",
    )
    args = parser.parse_args()

    result = profile_patchcore_artifact(
        args.artifact_dir,
        device=args.device,
        warmup_runs=args.warmup,
        timed_runs=args.runs,
    )
    payload = {
        "artifact_dir": str(args.artifact_dir),
        "device": result.device,
        "warmup_runs": result.warmup_runs,
        "timed_runs": result.timed_runs,
        "latency_ms_p50": result.latency_ms_p50,
        "latency_ms_p95": result.latency_ms_p95,
        "peak_mem_mb": result.peak_mem_mb,
        "bank_size_mb": result.bank_size_mb,
        "memory_bank_tensor_mb": result.memory_bank_tensor_mb,
        "param_count": result.param_count,
    }
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.results_csv is not None:
        meta_path = args.artifact_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Artifact metadata required for CSV append: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        threshold_path = args.artifact_dir / "threshold.json"
        threshold_method = "val_p99"
        if threshold_path.is_file():
            threshold_method = str(
                json.loads(threshold_path.read_text(encoding="utf-8")).get("method", "val_p99")
            )
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        row = ResultRow(
            run_id=f"profile_{meta['model']}_{meta['dataset']}_{meta['category']}_{stamp}",
            git_sha=str(meta.get("git_sha", "unknown")),
            timestamp=utc_timestamp_iso(),
            model=str(meta["model"]),
            backbone=str(meta["backbone"]),
            dataset=str(meta["dataset"]),
            category=str(meta["category"]),
            k_shot=meta.get("k_shot", "full"),
            seed=int(meta.get("seed", 0)),
            shift="none",
            severity=0,
            fusion="none",
            fusion_signals="",
            threshold_type=threshold_method,
            device=result.device,
            latency_ms_p50=result.latency_ms_p50,
            latency_ms_p95=result.latency_ms_p95,
            peak_mem_mb=result.peak_mem_mb,
            bank_size_mb=result.bank_size_mb,
            notes=(
                f"profile_only=true,warmup={result.warmup_runs},runs={result.timed_runs},"
                f"params={result.param_count}"
                + (
                    f",bank_tensor_mb={result.memory_bank_tensor_mb:.2f}"
                    if result.memory_bank_tensor_mb is not None
                    else ""
                )
            ),
        )
        append_result_row(row, args.results_csv)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        disk_str = f"{result.bank_size_mb:.2f}MB" if result.bank_size_mb is not None else "n/a"
        tensor_str = (
            f"{result.memory_bank_tensor_mb:.2f}MB"
            if result.memory_bank_tensor_mb is not None
            else "n/a"
        )
        print(
            f"[profile] {args.artifact_dir} on {result.device}: "
            f"p50={result.latency_ms_p50:.2f}ms, p95={result.latency_ms_p95:.2f}ms, "
            f"peak_mem={result.peak_mem_mb:.1f}MB, disk={disk_str}, "
            f"bank_tensor={tensor_str}, params={result.param_count}"
        )


if __name__ == "__main__":
    _main()
