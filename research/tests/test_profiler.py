"""Unit tests for the reusable inference profiler (``T-015``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from research.benchmarks.profiler import (
    _main as profiler_cli_main,
)
from research.benchmarks.profiler import (
    count_parameters,
    load_patchcore_model_from_artifact,
    measure_disk_size_mb,
    profile_inference,
    profile_patchcore_artifact,
    resolve_device_string,
)
from research.common.results import read_results


def test_profile_inference_counts_warmup_and_timed_runs(tmp_path: Path) -> None:
    """Verify exact 10+100 run count, percentile ordering, param count, and disk size."""
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
    )
    bank_file = tmp_path / "model.pt"
    bank_file.write_bytes(b"0" * (1024 * 512))  # 0.5 MiB

    call_count = 0

    def tracked_forward(x: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return model(x)

    sample = torch.randn((1, 3, 16, 16))
    result = profile_inference(
        tracked_forward,
        sample,
        device="cpu",
        warmup_runs=10,
        timed_runs=100,
        model=model,
        bank_path=bank_file,
        fit_time_s=1.25,
    )

    assert call_count == 110
    assert result.warmup_runs == 10
    assert result.timed_runs == 100
    assert result.latency_ms_p50 > 0.0
    assert result.latency_ms_p95 >= result.latency_ms_p50
    assert result.peak_mem_mb >= 0.0
    assert result.bank_size_mb == pytest.approx(0.5)
    assert result.param_count == count_parameters(model)
    assert result.fit_time_s == pytest.approx(1.25)
    assert result.device.startswith("cpu")


def test_profile_inference_rejects_invalid_inputs(tmp_path: Path) -> None:
    """Enforce batch-size-1, valid run counts, and existing artifact files."""
    model = torch.nn.Identity()

    with pytest.raises(ValueError, match="batch size 1"):
        profile_inference(model, torch.randn((2, 3, 8, 8)), device="cpu")

    with pytest.raises(ValueError, match="warmup_runs"):
        profile_inference(model, torch.randn((1, 3, 8, 8)), device="cpu", warmup_runs=-1)

    with pytest.raises(ValueError, match="warmup_runs"):
        profile_inference(
            model,
            torch.randn((1, 3, 8, 8)),
            device="cpu",
            warmup_runs=True,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="timed_runs"):
        profile_inference(model, torch.randn((1, 3, 8, 8)), device="cpu", timed_runs=0)

    with pytest.raises(FileNotFoundError, match="not found"):
        measure_disk_size_mb(tmp_path / "missing.pt")

    with pytest.raises(FileNotFoundError, match="not found"):
        load_patchcore_model_from_artifact(tmp_path / "missing_dir", device="cpu")

    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device_string("tpu")


def test_profile_patchcore_artifact_round_trip(tmp_path: Path) -> None:
    """Reload a tiny saved PatchcoreModel checkpoint offline and profile at batch size 1."""
    tiny = PatchcoreModel(
        backbone="resnet18", layers=["layer2"], pre_trained=False, num_neighbors=2
    )
    tiny.memory_bank = torch.randn((12, 128), dtype=torch.float32)

    artifact_dir = tmp_path / "patchcore" / "mvtec" / "bottle"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "patchcore",
            "backbone": "resnet18",
            "layers": ["layer2"],
            "coreset_sampling_ratio": 0.1,
            "num_neighbors": 2,
            "preprocess": {"resize": 64, "crop_size": 32},
            "state_dict": tiny.state_dict(),
        },
        artifact_dir / "model.pt",
    )

    loaded, _, _ = load_patchcore_model_from_artifact(artifact_dir, device="cpu")
    assert loaded.memory_bank.shape == (12, 128)

    result = profile_patchcore_artifact(
        artifact_dir,
        device="cpu",
        warmup_runs=2,
        timed_runs=5,
    )
    assert result.warmup_runs == 2
    assert result.timed_runs == 5
    assert result.latency_ms_p50 > 0.0
    assert result.latency_ms_p95 >= result.latency_ms_p50
    assert result.bank_size_mb is not None and result.bank_size_mb > 0.0
    assert result.memory_bank_tensor_mb is not None and result.memory_bank_tensor_mb > 0.0
    assert result.param_count == count_parameters(loaded)


def test_profiler_rejects_malformed_or_empty_checkpoint_and_cli_reads_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject empty/incomplete checkpoints and preserve threshold.json method via CLI."""
    empty_dir = tmp_path / "empty_bank"
    empty_dir.mkdir(parents=True)
    tiny = PatchcoreModel(
        backbone="resnet18", layers=["layer2"], pre_trained=False, num_neighbors=2
    )
    torch.save(
        {
            "backbone": "resnet18",
            "layers": ["layer2"],
            "num_neighbors": 2,
            "state_dict": tiny.state_dict(),
        },
        empty_dir / "model.pt",
    )
    with pytest.raises(ValueError, match="empty"):
        load_patchcore_model_from_artifact(empty_dir, device="cpu")

    bad_dir = tmp_path / "bad_payload"
    bad_dir.mkdir(parents=True)
    torch.save({"state_dict": {}}, bad_dir / "model.pt")
    with pytest.raises(ValueError, match="Malformed"):
        load_patchcore_model_from_artifact(bad_dir, device="cpu")

    art_dir = tmp_path / "valid_art"
    art_dir.mkdir(parents=True)
    tiny.memory_bank = torch.randn((8, 128), dtype=torch.float32)
    torch.save(
        {
            "backbone": "resnet18",
            "layers": ["layer2"],
            "num_neighbors": 2,
            "preprocess": {"crop_size": 32},
            "state_dict": tiny.state_dict(),
        },
        art_dir / "model.pt",
    )
    (art_dir / "meta.json").write_text(
        json.dumps(
            {
                "model": "patchcore",
                "backbone": "resnet18",
                "dataset": "mvtec",
                "category": "bottle",
                "seed": 0,
                "k_shot": "full",
                "git_sha": "abc123",
            }
        ),
        encoding="utf-8",
    )
    (art_dir / "threshold.json").write_text(json.dumps({"method": "mean_3std"}), encoding="utf-8")

    csv_path = tmp_path / "results.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profiler",
            "--artifact-dir",
            str(art_dir),
            "--device",
            "cpu",
            "--warmup",
            "1",
            "--runs",
            "2",
            "--results-csv",
            str(csv_path),
        ],
    )
    profiler_cli_main()
    rows = read_results(csv_path)
    assert len(rows) == 1
    assert rows[0].threshold_type == "mean_3std"
    assert rows[0].run_id.startswith("profile_patchcore_mvtec_bottle_20")
