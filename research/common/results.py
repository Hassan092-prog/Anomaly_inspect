"""Append-only CSV results writer and schema validator (``docs/05`` section 5)."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research.common.config import VALID_DATASETS, VALID_SHIFTS

PROTOCOL_VERSION = "1.0"
DEFAULT_RESULTS_PATH = Path("results/results.csv")

RESULT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "protocol_version",
    "git_sha",
    "timestamp",
    "model",
    "backbone",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "shift",
    "severity",
    "fusion",
    "fusion_signals",
    "threshold_type",
    "image_auroc",
    "pixel_auroc",
    "aupro",
    "f1_max_image",
    "f1_max_pixel",
    "ece",
    "risk_cov_area",
    "latency_ms_p50",
    "latency_ms_p95",
    "peak_mem_mb",
    "bank_size_mb",
    "device",
    "notes",
)

_UNIT_INTERVAL_METRICS: tuple[str, ...] = (
    "image_auroc",
    "pixel_auroc",
    "aupro",
    "f1_max_image",
    "f1_max_pixel",
    "ece",
)

_NON_NEGATIVE_METRICS: tuple[str, ...] = (
    "risk_cov_area",
    "latency_ms_p50",
    "latency_ms_p95",
    "peak_mem_mb",
    "bank_size_mb",
)


class ResultValidationError(ValueError):
    """Raised when a result row or CSV header violates ``docs/05`` schema."""


@dataclass(frozen=True)
class ResultRow:
    """One evaluated condition row matching ``docs/05_EVALUATION_PROTOCOL.md`` section 5."""

    run_id: str
    git_sha: str
    timestamp: str
    model: str
    backbone: str
    dataset: str
    category: str
    k_shot: int | str
    seed: int
    shift: str
    severity: int
    fusion: str
    fusion_signals: str
    threshold_type: str
    device: str
    protocol_version: str = PROTOCOL_VERSION
    image_auroc: float | None = None
    pixel_auroc: float | None = None
    aupro: float | None = None
    f1_max_image: float | None = None
    f1_max_pixel: float | None = None
    ece: float | None = None
    risk_cov_area: float | None = None
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None
    peak_mem_mb: float | None = None
    bank_size_mb: float | None = None
    notes: str = ""


def utc_timestamp_iso() -> str:
    """Return current UTC timestamp formatted as ISO 8601."""
    return datetime.now(UTC).isoformat()


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_optional_float(
    name: str,
    value: Any,
    *,
    min_value: float = 0.0,
    max_value: float | None = None,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultValidationError(f"{name} must be float or None, got {value!r}")
    fval = float(value)
    if not math.isfinite(fval):
        raise ResultValidationError(f"{name} must be finite, got {value!r}")
    if fval < min_value:
        raise ResultValidationError(f"{name} must be >= {min_value}, got {fval}")
    if max_value is not None and fval > max_value:
        raise ResultValidationError(f"{name} must be <= {max_value}, got {fval}")


def _validate_identifiers_and_axis(row: ResultRow) -> None:
    required_non_empty = (
        "run_id",
        "protocol_version",
        "git_sha",
        "timestamp",
        "model",
        "backbone",
        "dataset",
        "category",
        "shift",
        "fusion",
        "threshold_type",
        "device",
    )
    for field_name in required_non_empty:
        val = getattr(row, field_name)
        if not isinstance(val, str) or not val.strip():
            raise ResultValidationError(f"{field_name} must be a non-empty string, got {val!r}")

    if row.protocol_version != PROTOCOL_VERSION:
        raise ResultValidationError(
            f"protocol_version must be {PROTOCOL_VERSION!r}, got {row.protocol_version!r}"
        )
    try:
        datetime.fromisoformat(row.timestamp)
    except ValueError as exc:
        raise ResultValidationError(f"timestamp must be ISO 8601, got {row.timestamp!r}") from exc

    if row.dataset not in VALID_DATASETS:
        raise ResultValidationError(f"dataset must be one of {VALID_DATASETS}, got {row.dataset!r}")
    if not isinstance(row.fusion_signals, str) or not isinstance(row.notes, str):
        raise ResultValidationError("fusion_signals and notes must be strings")


def validate_result_row(row: ResultRow) -> None:
    """Validate every field of ``row`` against the protocol schema."""
    _validate_identifiers_and_axis(row)

    if isinstance(row.k_shot, bool) or not isinstance(row.k_shot, (int, str)):
        raise ResultValidationError(f"k_shot must be 'full' or int >= 0, got {row.k_shot!r}")
    if isinstance(row.k_shot, str) and row.k_shot != "full":
        raise ResultValidationError(f"k_shot string must be 'full', got {row.k_shot!r}")
    if isinstance(row.k_shot, int) and row.k_shot < 0:
        raise ResultValidationError(f"k_shot must be >= 0, got {row.k_shot}")

    if not _is_strict_int(row.seed) or row.seed < 0:
        raise ResultValidationError(f"seed must be an int >= 0, got {row.seed!r}")
    if row.shift not in VALID_SHIFTS:
        raise ResultValidationError(f"shift must be one of {VALID_SHIFTS}, got {row.shift!r}")
    if not _is_strict_int(row.severity):
        raise ResultValidationError(f"severity must be an int, got {row.severity!r}")
    if row.shift == "none" and row.severity != 0:
        raise ResultValidationError("severity must be 0 when shift is 'none'")
    if row.shift != "none" and not (1 <= row.severity <= 5):
        raise ResultValidationError(f"severity must be in 1..5, got {row.severity}")

    for metric in _UNIT_INTERVAL_METRICS:
        _validate_optional_float(metric, getattr(row, metric), min_value=0.0, max_value=1.0)
    for metric in _NON_NEGATIVE_METRICS:
        _validate_optional_float(metric, getattr(row, metric), min_value=0.0, max_value=None)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def row_to_csv_dict(row: ResultRow) -> dict[str, str]:
    """Validate ``row`` and serialize it to ordered CSV string fields."""
    validate_result_row(row)
    return {col: _format_cell(getattr(row, col)) for col in RESULT_COLUMNS}


def _parse_optional_float(raw: str) -> float | None:
    return None if raw == "" else float(raw)


def _parse_k_shot(raw: str) -> int | str:
    if raw == "full":
        return "full"
    try:
        return int(raw)
    except ValueError as exc:
        raise ResultValidationError(f"Invalid k_shot cell in CSV: {raw!r}") from exc


def csv_dict_to_row(raw: dict[str | None, Any]) -> ResultRow:
    """Parse a CSV row dictionary back into a validated ``ResultRow``."""
    if None in raw:
        raise ResultValidationError("Malformed CSV row: unexpected extra columns")
    try:
        row = ResultRow(
            run_id=raw["run_id"],
            protocol_version=raw["protocol_version"],
            git_sha=raw["git_sha"],
            timestamp=raw["timestamp"],
            model=raw["model"],
            backbone=raw["backbone"],
            dataset=raw["dataset"],
            category=raw["category"],
            k_shot=_parse_k_shot(raw["k_shot"]),
            seed=int(raw["seed"]),
            shift=raw["shift"],
            severity=int(raw["severity"]),
            fusion=raw["fusion"],
            fusion_signals=raw["fusion_signals"],
            threshold_type=raw["threshold_type"],
            image_auroc=_parse_optional_float(raw["image_auroc"]),
            pixel_auroc=_parse_optional_float(raw["pixel_auroc"]),
            aupro=_parse_optional_float(raw["aupro"]),
            f1_max_image=_parse_optional_float(raw["f1_max_image"]),
            f1_max_pixel=_parse_optional_float(raw["f1_max_pixel"]),
            ece=_parse_optional_float(raw["ece"]),
            risk_cov_area=_parse_optional_float(raw["risk_cov_area"]),
            latency_ms_p50=_parse_optional_float(raw["latency_ms_p50"]),
            latency_ms_p95=_parse_optional_float(raw["latency_ms_p95"]),
            peak_mem_mb=_parse_optional_float(raw["peak_mem_mb"]),
            bank_size_mb=_parse_optional_float(raw["bank_size_mb"]),
            device=raw["device"],
            notes=raw["notes"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultValidationError(f"Malformed CSV row: {exc}") from exc

    validate_result_row(row)
    return row


def append_result_row(row: ResultRow, path: str | Path = DEFAULT_RESULTS_PATH) -> Path:
    """Validate and append one ``ResultRow`` to ``path`` (writing header if new/empty)."""
    serialized = row_to_csv_dict(row)
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_has_content = csv_path.is_file() and csv_path.stat().st_size > 0
    if file_has_content:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header != list(RESULT_COLUMNS):
            raise ResultValidationError(
                f"Existing CSV header in {csv_path} does not match protocol schema"
            )
        mode = "a"
    else:
        mode = "w"

    with csv_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RESULT_COLUMNS), lineterminator="\n")
        if not file_has_content:
            writer.writeheader()
        writer.writerow(serialized)

    return csv_path


def read_results(path: str | Path = DEFAULT_RESULTS_PATH) -> list[ResultRow]:
    """Read and validate all rows from ``path`` (returning ``[]`` if missing/empty)."""
    csv_path = Path(path)
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(RESULT_COLUMNS):
            raise ResultValidationError(f"CSV header in {csv_path} does not match protocol schema")
        return [csv_dict_to_row(raw) for raw in reader]
