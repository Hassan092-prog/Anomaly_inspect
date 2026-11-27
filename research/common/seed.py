"""Deterministic random seed management across Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible experiments.

    Args:
        seed: Non-negative integer seed in ``[0, 2**32 - 1]``.
        deterministic: If True, enable deterministic cuDNN and PyTorch ops
            (with ``warn_only=True`` because some CUDA kernels lack deterministic
            implementations unless ``CUBLAS_WORKSPACE_CONFIG`` is configured).

    Raises:
        ValueError: If ``seed`` is negative, boolean, or exceeds 32-bit range.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed < 2**32):
        raise ValueError(f"seed must be an integer in [0, 2**32 - 1], got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
