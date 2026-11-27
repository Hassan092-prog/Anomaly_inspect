# 02. Technical Requirements Document (TRD)

> Version numbers below are starting points. Final pins live in `requirements-research.txt`, `requirements-research-cpu.txt`, and `requirements-backend.txt` (plus `package-lock.json` for frontend). Verified `anomalib==2.7.1` (`anomalib[clip]`) exports `Patchcore`, `EfficientAd`, `WinClip`, `Engine`, `MVTecAD`, and `Visa`.

## 1. Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | Matches Colab/Kaggle well |
| DL | PyTorch 2.x, torchvision | Pin to the CUDA build available on your runtime |
| AD library | Anomalib (`2.7.1`, `[clip]` extra) | Baselines: `Patchcore`, `EfficientAd`, `WinClip` |
| CV | OpenCV, albumentations, scikit-image | Corruptions, contours, post-processing |
| Vector search | faiss-cpu (or `torch.cdist` in chunks) | For custom per-layer memory banks and kNN |
| Metrics | scikit-learn, torchmetrics, own AUPRO | AUPRO is implemented and unit-tested by you |
| Backend | FastAPI, Pydantic v2, SQLAlchemy 2.0 async, asyncpg, Alembic | |
| DB | PostgreSQL 16 | |
| Frontend | React 18, TypeScript, Vite, TanStack Query, Recharts | Node 20 |
| Packaging | Docker, Docker Compose | Multi-stage builds |
| Quality | ruff, mypy (light), pytest, pytest-asyncio, httpx, vitest | |
| CI | GitHub Actions | Lint, tests, Docker build |

## 2. Data

| Dataset | Content | Use |
|---|---|---|
| MVTec AD | 15 categories (5 texture, 10 object), normal-only train, test with defects and pixel masks | Primary |
| VisA | 12 categories (multi-instance, complex structures), official split CSV, pixel masks | Secondary |

- Store under `data/` (git-ignored). Provide `scripts/download_data.md` with manual download instructions and expected folder layout; do not automate anything that violates a license.
- **Validation split:** carve 10-20% of each category's normal training images (fixed seed) for normalization statistics, threshold calibration, and confidence calibration. Report this fraction in every run config.
- **Few-shot subsets:** sample k in {1, 5, 10} normal training images, 3 seeds each, saved as index lists in `data/splits/` and reused by every model.

## 3. Pipeline requirements

### 3.1 Preprocessing
- Resize to 256, center-crop 224 (PatchCore default); ImageNet mean/std normalization. Configurable per model, recorded in the run config.
- Masks resized with nearest-neighbor; binarized at 0.5.

### 3.2 Models

| Model | Source | Training | Output |
|---|---|---|---|
| PatchCore | Anomalib | Feature extraction + coreset memory bank (no gradient training) | Anomaly map + score |
| EfficientAD (S, optionally M) | Anomalib | Student-teacher training per category; needs ImageNet-style penalty images (Anomalib handles Imagenette download) | Anomaly map + score |
| WinCLIP | Anomalib | None (zero-shot) and few-shot reference-image variant | Anomaly map + score |
| Per-layer PatchCore | Custom (`research/fusion/`) | Separate memory bank per backbone layer | One map per scale |
| AnomalyCLIP | External implementation | Stretch only; not in Anomalib's model set as of the last check, so verify before planning around it | |

### 3.3 Fusion (custom; see `04_RESEARCH_STRATEGY.md`)
Inputs: per-scale anomaly maps, validation statistics, confidence signals. Output: fused map, image score, confidence value.

### 3.4 Explanations
- Overlay: normalized map colorized (e.g. JET) alpha-blended with the image.
- Contour: threshold map, morphological cleanup, `cv2.findContours`, drawn on the image.
- Nearest-normal patches: for the top-scoring patch locations, return the nearest memory-bank patch's **source image id and patch coordinates**, so the UI can crop the reference. This requires storing patch provenance (image id, x, y) alongside the memory bank; plan for it at bank-build time. Coreset subsampling keeps provenance for retained items only. Because `data/` is not mounted inside the API container, either export lightweight cropped/retained reference thumbnails into `artifacts/{model}/{dataset}/{category}/references/` (respecting dataset license limits in `10_DECISIONS_AND_RISKS.md` R8) or return `reference_url = None` with metadata only.

## 4. Performance requirements

| Metric | Initial target | Measurement |
|---|---|---|
| End-to-end API latency, EfficientAD-S, CPU | <= 2 s | p50/p95 over 100 requests, warm |
| End-to-end API latency, PatchCore, CPU | <= 5 s | same |
| Model inference, GPU | <= 0.5 s | batch size 1, warm-up 10 runs, then 100 timed |
| Memory | Report only | Peak GPU memory; memory-bank size on disk |

Targets are placeholders; measure first, then revise and document in `10_DECISIONS_AND_RISKS.md`.

## 5. Artifacts and registry

```
artifacts/{model}/{dataset}/{category}/
  model.(pt|ckpt)         # or memory_bank.pt + provenance.npz for PatchCore
  references/             # optional cropped normal patch thumbnails for reference_url
  stats.json              # validation score stats (patch mean/std/percentiles + image-max mean/std)
  threshold.json          # default threshold(s) and how derived
  meta.json               # config hash, git SHA, library versions, created_at
```

A registry (`backend/app/registry.py`) lists available (model, dataset, category) combinations by scanning `artifacts/`, and loads models lazily with an LRU cache so memory stays bounded.

## 6. Reproducibility

- Single `seed.py` sets all RNGs; deterministic flags where feasible (document known nondeterminism, e.g. some CUDA ops).
- Each run writes `runs/{run_id}/config.yaml`, `env.txt` (library versions), `git_sha.txt`.
- All metric rows go to `results/*.csv` using the schema in `05_EVALUATION_PROTOCOL.md`.
- A `make report` target regenerates every table and figure from `results/` alone.

## 7. Security and safety

- Upload validation: content-type and magic-byte check, size limit, image decode with Pillow in a try/except, re-encode before storing.
- Generated filenames (UUID); never use user-supplied filenames in paths.
- CORS restricted to the frontend origin; no secrets in the repo; `.env.example` only.
- Rate limiting is optional (demo-only), but document the absence.

## 8. Deployment

- `docker-compose.yml` services: `db` (postgres), `api` (FastAPI + inference), `frontend` (static build served by nginx). Optional `worker` if inference moves off the API process.
- Volumes: `pgdata`, `uploads`, `artifacts` (read-only mount for the API).
- Health checks on `db` and `api`; `api` waits for `db`; migrations run on start.
- Demo: CPU-only image. Host on a free tier that supports Docker (verify current offerings), or ship a recorded walkthrough if memory limits block the bank.

## 9. Colab / Kaggle setup

Both platforms provide free GPU runtimes (T4 or P100) suitable for feature extraction and EfficientAD training. Keep these in mind:

- **Python version:** Colab defaults to 3.10 or 3.11; verify and document the runtime version in the notebook header.
- **Storage:** mount Google Drive for datasets and artifacts so they survive session restarts. Expected space: MVTec AD about 5 GB, VisA about 10 GB, artifacts about 1-2 GB.
- **Session limits:** save results and checkpoints frequently; use the resumable sweep orchestrator (T-013) so interrupted runs pick up where they left off.
- **Dependencies:** install from a pinned `requirements-research.txt` at the top of each notebook cell; verify Anomalib's installed version matches your pin.
- **A setup cell template** goes in `notebooks/colab_setup.py` (or the top of each research notebook) and is tracked in `T-002`.

## 10. Optional stretch

ONNX export for EfficientAD or the backbone, CPU latency comparison, and an INT8 note. Record as an addendum to the benchmark, not a requirement.
