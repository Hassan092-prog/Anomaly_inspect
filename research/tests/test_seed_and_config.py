"""Unit tests for deterministic seeding and YAML config loading."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from research.common.config import (
    ConfigError,
    ExperimentConfig,
    config_hash,
    load_config,
    save_config,
    write_run_metadata,
)
from research.common.seed import set_seed


def test_set_seed_produces_identical_sequences() -> None:
    """Same seed must yield identical Python, NumPy, and PyTorch random sequences."""
    set_seed(42)
    py_a = [random.random() for _ in range(5)]
    np_a = np.random.rand(4, 4)
    torch_a = torch.randn(3, 5)

    set_seed(42)
    py_b = [random.random() for _ in range(5)]
    np_b = np.random.rand(4, 4)
    torch_b = torch.randn(3, 5)

    assert py_a == py_b
    np.testing.assert_array_equal(np_a, np_b)
    assert torch.equal(torch_a, torch_b)


def test_different_seeds_produce_different_tensors() -> None:
    """Distinct seeds must not produce identical random tensors."""
    set_seed(0)
    t0 = torch.randn(16)
    set_seed(1)
    t1 = torch.randn(16)
    assert not torch.equal(t0, t1)


@pytest.mark.parametrize("bad_seed", [-1, 2**32, True])
def test_set_seed_rejects_invalid_values(bad_seed: int | bool) -> None:
    """Invalid seed values should raise ValueError."""
    with pytest.raises(ValueError):
        set_seed(bad_seed)


def test_config_yaml_round_trip(tmp_path: Path) -> None:
    """Saving and reloading an ExperimentConfig should preserve all fields and hash."""
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(
        """
experiment_name: patchcore_bottle
seed: 7
data:
  dataset: mvtec
  category: bottle
  val_fraction: 0.15
  k_shot: 5
preprocess:
  resize: 256
  crop_size: 224
model:
  name: pc_layerwise
  layers: [layer2, layer3]
  coreset_sampling_ratio: 0.1
fusion:
  method: confidence
  signals: [sep, agree, focus]
  temperature: 0.5
shift:
  corruption: gaussian_noise
  severity: 3
""",
        encoding="utf-8",
    )

    loaded = load_config(cfg_path)
    assert loaded.experiment_name == "patchcore_bottle"
    assert loaded.seed == 7
    assert loaded.data.k_shot == 5
    assert loaded.model.layers == ("layer2", "layer3")
    assert loaded.fusion.signals == ("sep", "agree", "focus")
    assert loaded.shift.severity == 3

    saved_path = tmp_path / "out.yaml"
    save_config(loaded, saved_path)
    reloaded = load_config(saved_path)
    assert reloaded == loaded
    assert config_hash(reloaded) == config_hash(loaded)


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    """Typos or unknown YAML fields should fail loudly with ConfigError."""
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("model:\n  unknown_param: 123\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(cfg_path)


def test_config_rejects_invalid_shift_severity(tmp_path: Path) -> None:
    """Severity must be 0 for 'none' and 1..5 for active corruptions."""
    cfg_path = tmp_path / "bad_shift.yaml"
    cfg_path.write_text("shift:\n  corruption: none\n  severity: 2\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="severity"):
        load_config(cfg_path)


def test_config_missing_file_raises(tmp_path: Path) -> None:
    """Missing YAML file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_config_validates_container_types_and_weights(tmp_path: Path) -> None:
    """Scalar passed for tuple field or non-positive fusion weights should raise ConfigError."""
    bad_layers = tmp_path / "bad_layers.yaml"
    bad_layers.write_text("model:\n  layers: layer2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a list"):
        load_config(bad_layers)

    bad_weights = tmp_path / "bad_weights.yaml"
    bad_weights.write_text("fusion:\n  weights: [0.0, 0.0]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="positive sum"):
        load_config(bad_weights)

    good_weights = tmp_path / "good_weights.yaml"
    good_weights.write_text("fusion:\n  method: fixed\n  weights: [0.3, 0.7]\n", encoding="utf-8")
    cfg = load_config(good_weights)
    assert cfg.fusion.weights == (0.3, 0.7)

    neg_weights = tmp_path / "neg_weights.yaml"
    neg_weights.write_text("fusion:\n  weights: [-0.5, 1.5]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="non-negative"):
        load_config(neg_weights)

    float_seed = tmp_path / "float_seed.yaml"
    float_seed.write_text("seed: 1.5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="seed"):
        load_config(float_seed)

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("model: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(malformed)


def test_write_run_metadata_creates_expected_files(tmp_path: Path) -> None:
    """write_run_metadata should record config.yaml, git_sha.txt, and env.txt."""
    cfg = ExperimentConfig(experiment_name="smoke")
    run_dir = write_run_metadata(cfg, tmp_path / "run_001")

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "git_sha.txt").is_file()
    assert (run_dir / "env.txt").is_file()
    assert load_config(run_dir / "config.yaml") == cfg
    env_text = (run_dir / "env.txt").read_text(encoding="utf-8")
    assert "python==" in env_text
    assert "torch==" in env_text
