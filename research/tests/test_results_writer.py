"""Unit tests for the append-only results CSV writer and schema validator."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from research.common.results import (
    RESULT_COLUMNS,
    ResultRow,
    ResultValidationError,
    append_result_row,
    read_results,
    utc_timestamp_iso,
)


def _make_valid_row(**overrides: object) -> ResultRow:
    defaults: dict[str, object] = {
        "run_id": "run-001",
        "git_sha": "abc1234",
        "timestamp": utc_timestamp_iso(),
        "model": "patchcore",
        "backbone": "wide_resnet50_2",
        "dataset": "mvtec",
        "category": "bottle",
        "k_shot": "full",
        "seed": 0,
        "shift": "none",
        "severity": 0,
        "fusion": "none",
        "fusion_signals": "",
        "threshold_type": "val_p99",
        "image_auroc": 0.99,
        "pixel_auroc": 0.98,
        "aupro": 0.94,
        "f1_max_image": 0.97,
        "f1_max_pixel": 0.75,
        "ece": 0.0,
        "risk_cov_area": None,
        "device": "cpu",
        "notes": "baseline",
    }
    defaults.update(overrides)
    return ResultRow(**defaults)  # type: ignore[arg-type]


def test_append_result_row_matches_schema_and_preserves_empty_vs_zero(tmp_path: Path) -> None:
    """Missing metrics serialize as empty string while 0.0 serializes as 0.0."""
    csv_path = tmp_path / "results.csv"
    row1 = _make_valid_row()
    row2 = _make_valid_row(run_id="run-002", k_shot=5, seed=1, shift="gaussian_noise", severity=3)

    append_result_row(row1, csv_path)
    append_result_row(row2, csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == list(RESULT_COLUMNS)
    assert len(rows) == 3

    col_idx = {col: i for i, col in enumerate(RESULT_COLUMNS)}
    assert rows[1][col_idx["ece"]] == "0.0"
    assert rows[1][col_idx["risk_cov_area"]] == ""

    parsed = read_results(csv_path)
    assert parsed == [row1, row2]


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"run_id": ""},
        {"protocol_version": "2.0"},
        {"timestamp": "not-a-timestamp"},
        {"dataset": "imagenet"},
        {"k_shot": -1},
        {"k_shot": "10"},
        {"seed": -1},
        {"shift": "none", "severity": 1},
        {"shift": "gaussian_blur", "severity": 0},
        {"image_auroc": 1.2},
        {"aupro": float("nan")},
        {"latency_ms_p50": -0.5},
    ],
)
def test_invalid_result_row_raises(tmp_path: Path, bad_kwargs: dict[str, object]) -> None:
    """Schema violations must raise ResultValidationError without writing to CSV."""
    csv_path = tmp_path / "results.csv"
    row = _make_valid_row(**bad_kwargs)

    with pytest.raises(ResultValidationError):
        append_result_row(row, csv_path)

    assert not csv_path.exists()


def test_append_rejects_mismatched_existing_header(tmp_path: Path) -> None:
    """Appending to a CSV with non-conforming columns must raise."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("col_a,col_b\n1,2\n", encoding="utf-8")

    with pytest.raises(ResultValidationError, match="header"):
        append_result_row(_make_valid_row(), csv_path)


def test_read_results_missing_file_returns_empty(tmp_path: Path) -> None:
    """Missing results file returns empty list."""
    assert read_results(tmp_path / "missing.csv") == []


def test_read_results_rejects_truncated_and_extra_column_rows(tmp_path: Path) -> None:
    """Malformed CSV rows with missing or extra cells must raise ResultValidationError."""
    csv_path = tmp_path / "malformed.csv"
    header = ",".join(RESULT_COLUMNS)
    csv_path.write_text(f"{header}\nonly_one_cell\n", encoding="utf-8")
    with pytest.raises(ResultValidationError, match="Malformed CSV row"):
        read_results(csv_path)

    row_ok = tmp_path / "lf.csv"
    append_result_row(_make_valid_row(), row_ok)
    assert b"\r\n" not in row_ok.read_bytes()
