"""YAML-to-dataclass experiment configuration loader and run metadata recorder."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml  # type: ignore[import-untyped]

T = TypeVar("T")

VALID_DATASETS = ("mvtec", "visa")
VALID_SHIFTS = ("none", "brightness", "contrast", "gaussian_noise", "gaussian_blur", "jpeg")
VALID_THRESHOLDS = ("val_p99", "mean_3std")


class ConfigError(ValueError):
    """Raised when an experiment YAML config is malformed or violates constraints."""


@dataclass(frozen=True)
class PreprocessConfig:
    """Image and mask preprocessing configuration."""

    resize: int = 256
    crop_size: int = 224
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataConfig:
    """Dataset and deterministic split configuration."""

    dataset: Literal["mvtec", "visa"] = "mvtec"
    category: str = "bottle"
    data_dir: str = "data"
    val_fraction: float = 0.15
    val_seed: int = 0
    k_shot: int | str = "full"


@dataclass(frozen=True)
class ModelConfig:
    """Model selection and architecture hyperparameters."""

    name: str = "patchcore"
    backbone: str = "wide_resnet50_2"
    coreset_sampling_ratio: float = 0.1
    layers: tuple[str, ...] = ("layer2", "layer3")
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusionConfig:
    """Multi-scale fusion configuration."""

    method: str = "none"
    signals: tuple[str, ...] = ()
    temperature: float = 1.0
    weights: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ShiftConfig:
    """Photometric distribution-shift configuration (test images only)."""

    corruption: str = "none"
    severity: int = 0
    recipes_path: str = "research/shift/recipes.yaml"


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration for a single reproducible experiment run."""

    experiment_name: str = "default"
    seed: int = 0
    deterministic: bool = True
    threshold_method: str = "val_p99"
    output_dir: str = "runs"
    results_csv: str = "results/results.csv"
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    shift: ShiftConfig = field(default_factory=ShiftConfig)


def _coerce_value(field_name: str, value: Any, default_val: Any) -> Any:
    """Coerce YAML-loaded values to expected dataclass field container types."""
    if value is None:
        return None
    if field_name == "weights":
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"fusion.weights must be a list, got {type(value).__name__}")
        return tuple(float(x) for x in value)
    if isinstance(default_val, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{field_name} must be a list, got {type(value).__name__}")
        if field_name in {"normalize_mean", "normalize_std"}:
            if len(value) != 3:
                raise ConfigError(f"{field_name} must have 3 elements, got {len(value)}")
            return tuple(float(x) for x in value)
        return tuple(str(x) for x in value)
    return value


def _build_section(cls: type[T], raw: Any, section_name: str) -> T:
    """Instantiate a dataclass section from a raw mapping, rejecting unknown keys."""
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"Section '{section_name}' must be a mapping, got {type(raw).__name__}")

    valid_fields = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw.keys()) - set(valid_fields.keys())
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{section_name}': {sorted(unknown)}")

    defaults = cls()
    kwargs: dict[str, Any] = {}
    for name in valid_fields:
        if name not in raw:
            continue
        default_val = getattr(defaults, name)
        kwargs[name] = _coerce_value(name, raw[name], default_val)
    return cls(**kwargs)


def _is_strict_int(value: Any) -> bool:
    """Return True if value is an int and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def validate_config(cfg: ExperimentConfig) -> None:
    """Validate cross-field invariants on an ``ExperimentConfig``."""
    if not _is_strict_int(cfg.seed) or not (0 <= cfg.seed < 2**32):
        raise ConfigError(f"seed must be an int in [0, 2**32 - 1], got {cfg.seed!r}")
    if cfg.data.dataset not in VALID_DATASETS:
        raise ConfigError(f"data.dataset must be one of {VALID_DATASETS}, got {cfg.data.dataset!r}")
    if not cfg.data.category.strip():
        raise ConfigError("data.category must be non-empty")
    if not (0.0 < cfg.data.val_fraction < 0.5):
        raise ConfigError(f"data.val_fraction must be in (0, 0.5), got {cfg.data.val_fraction}")
    if not _is_strict_int(cfg.data.val_seed) or cfg.data.val_seed < 0:
        raise ConfigError(f"data.val_seed must be a non-negative int, got {cfg.data.val_seed!r}")
    if isinstance(cfg.data.k_shot, bool) or not isinstance(cfg.data.k_shot, (int, str)):
        raise ConfigError(
            f"data.k_shot must be 'full' or non-negative int, got {cfg.data.k_shot!r}"
        )
    if isinstance(cfg.data.k_shot, str) and cfg.data.k_shot != "full":
        raise ConfigError(f"data.k_shot string must be 'full', got {cfg.data.k_shot!r}")
    if isinstance(cfg.data.k_shot, int) and cfg.data.k_shot < 0:
        raise ConfigError(f"data.k_shot must be >= 0, got {cfg.data.k_shot}")
    if not _is_strict_int(cfg.preprocess.resize) or not _is_strict_int(cfg.preprocess.crop_size):
        raise ConfigError("preprocess.resize and preprocess.crop_size must be integers")
    if cfg.preprocess.resize <= 0 or cfg.preprocess.crop_size <= 0:
        raise ConfigError("preprocess.resize and preprocess.crop_size must be positive")
    if cfg.preprocess.crop_size > cfg.preprocess.resize:
        raise ConfigError("preprocess.crop_size cannot exceed preprocess.resize")
    if not (0.0 < cfg.model.coreset_sampling_ratio <= 1.0):
        raise ConfigError("model.coreset_sampling_ratio must be in (0, 1]")
    if not isinstance(cfg.model.params, dict):
        raise ConfigError("model.params must be a mapping")
    if cfg.fusion.temperature <= 0.0:
        raise ConfigError("fusion.temperature must be > 0")
    if cfg.fusion.weights is not None and (
        len(cfg.fusion.weights) == 0
        or any(not math.isfinite(w) or w < 0.0 for w in cfg.fusion.weights)
        or sum(cfg.fusion.weights) <= 0.0
    ):
        raise ConfigError("fusion.weights must be non-empty, non-negative, with positive sum")
    if cfg.shift.corruption not in VALID_SHIFTS:
        raise ConfigError(
            f"shift.corruption must be one of {VALID_SHIFTS}, got {cfg.shift.corruption!r}"
        )
    if not _is_strict_int(cfg.shift.severity):
        raise ConfigError(f"shift.severity must be an int, got {cfg.shift.severity!r}")
    if cfg.shift.corruption == "none" and cfg.shift.severity != 0:
        raise ConfigError("shift.severity must be 0 when shift.corruption is 'none'")
    if cfg.shift.corruption != "none" and not (1 <= cfg.shift.severity <= 5):
        raise ConfigError(f"shift.severity must be in 1..5, got {cfg.shift.severity}")
    if cfg.threshold_method not in VALID_THRESHOLDS:
        raise ConfigError(
            f"threshold_method must be one of {VALID_THRESHOLDS}, got {cfg.threshold_method!r}"
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an ``ExperimentConfig`` from a YAML file."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML syntax in {cfg_path}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")

    top_fields = {f.name for f in fields(ExperimentConfig)}
    unknown = set(raw.keys()) - top_fields
    if unknown:
        raise ConfigError(f"Unknown top-level config key(s): {sorted(unknown)}")

    section_types: dict[str, type[Any]] = {
        "data": DataConfig,
        "preprocess": PreprocessConfig,
        "model": ModelConfig,
        "fusion": FusionConfig,
        "shift": ShiftConfig,
    }
    kwargs: dict[str, Any] = {}
    for key, val in raw.items():
        if key in section_types:
            kwargs[key] = _build_section(section_types[key], val, key)
        else:
            kwargs[key] = val

    cfg = ExperimentConfig(**kwargs)
    validate_config(cfg)
    return cfg


def _normalize_for_yaml(obj: Any) -> Any:
    """Convert dataclasses and tuples recursively to YAML-safe primitives."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _normalize_for_yaml(asdict(obj))
    if isinstance(obj, dict):
        return {k: _normalize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_normalize_for_yaml(v) for v in obj]
    if isinstance(obj, list):
        return [_normalize_for_yaml(v) for v in obj]
    return obj


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Serialize ``ExperimentConfig`` to a plain dictionary suitable for YAML/JSON."""
    data = _normalize_for_yaml(cfg)
    if not isinstance(data, dict):
        raise ConfigError("Expected serialized config to be a dictionary")
    return data


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Write ``ExperimentConfig`` to a YAML file, creating parent directories."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)


def config_hash(cfg: ExperimentConfig) -> str:
    """Compute a deterministic 12-character SHA-256 hash of the config."""
    canonical = json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def get_git_sha() -> str:
    """Return the current git commit SHA, or ``'unknown'`` if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_env_versions() -> list[str]:
    """Return sorted key runtime and package versions for run reproducibility."""
    lines = [f"python=={platform.python_version()}"]
    packages = (
        "anomalib",
        "torch",
        "torchvision",
        "numpy",
        "scikit-learn",
        "torchmetrics",
        "albumentations",
        "faiss-cpu",
        "opencv-python-headless",
        "pyyaml",
    )
    for pkg in packages:
        try:
            ver = importlib.metadata.version(pkg)
            lines.append(f"{pkg}=={ver}")
        except importlib.metadata.PackageNotFoundError:
            continue
    return lines


def write_run_metadata(cfg: ExperimentConfig, run_dir: str | Path) -> Path:
    """Write ``config.yaml``, ``git_sha.txt``, and ``env.txt`` into ``run_dir``."""
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    save_config(cfg, target / "config.yaml")
    (target / "git_sha.txt").write_text(f"{get_git_sha()}\n", encoding="utf-8")
    (target / "env.txt").write_text("\n".join(collect_env_versions()) + "\n", encoding="utf-8")
    return target
