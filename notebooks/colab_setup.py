# %% [markdown]
# # AnomalyInspect — Colab / Kaggle Setup Template
#
# Run these cells at the start of a Google Colab or Kaggle GPU session.
# Matches `docs/02_TRD.md` section 9 and uses the pinned `requirements-research.txt`.

# %%
"""Colab and Kaggle environment setup helpers for AnomalyInspect."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/Anomaly_inspect")
REPO_URL = "https://github.com/Hassan092-prog/Anomaly_inspect.git"
REPO_DIR = Path("/content/Anomaly_inspect")


def check_runtime() -> None:
    """Print Python version and GPU status."""
    print(f"Python: {sys.version}")
    subprocess.run(["nvidia-smi"], check=False)


def mount_drive_and_link_dirs(repo_dir: Path = REPO_DIR, drive_root: Path = DRIVE_ROOT) -> None:
    """Mount Google Drive in Colab and symlink persistent workspace directories."""
    if importlib.util.find_spec("google.colab") is None:
        print("Not running in Google Colab; skipping Drive mount.")
        return

    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount("/content/drive")
    for folder in ("data", "artifacts", "runs", "results"):
        target = drive_root / folder
        target.mkdir(parents=True, exist_ok=True)
        link = repo_dir / folder
        if link.is_symlink():
            continue
        if link.exists():
            # Preserve tracked files (e.g. results/.gitkeep) on first link, then clear dir.
            for item in link.iterdir():
                dest = target / item.name
                if not dest.exists():
                    item.replace(dest)
            shutil.rmtree(link)
        link.symlink_to(target)
        print(f"Linked {link} -> {target}")


def install_pinned_research_deps(repo_dir: Path = REPO_DIR) -> None:
    """Install pinned CUDA-compatible research dependencies into the runtime."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements-research.txt"],
        cwd=repo_dir,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"],
        cwd=repo_dir,
        check=True,
    )


def verify_anomalib_setup() -> None:
    """Verify PyTorch CUDA availability and expected Anomalib 2.7.1 model imports."""
    import anomalib
    import torch
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd, Patchcore, WinClip

    print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    print(f"anomalib={anomalib.__version__}")
    print("models=", Patchcore.__name__, EfficientAd.__name__, WinClip.__name__, Engine.__name__)


# %%
# Example Colab cell execution order:
# !git clone https://github.com/Hassan092-prog/Anomaly_inspect.git /content/Anomaly_inspect
# %cd /content/Anomaly_inspect
# from notebooks.colab_setup import (
#     check_runtime,
#     install_pinned_research_deps,
#     mount_drive_and_link_dirs,
#     verify_anomalib_setup,
# )
# check_runtime()
# mount_drive_and_link_dirs()
# install_pinned_research_deps()
# verify_anomalib_setup()
