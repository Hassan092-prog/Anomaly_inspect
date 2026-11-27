# AnomalyInspect: Open-World Visual Anomaly Detection & Explainable Inspection

> Learn only from **normal** images. Detect and localize **unseen** defects. Show how reliable the prediction is.

**Status:** Sprint 1 in progress (`T-001`–`T-002` complete: repo scaffold, CI, pinned research/backend environments, and Colab setup). Replace this line as sprints finish.

## What this is

1. **Research pipeline** that benchmarks three anomaly-detection families (PatchCore, EfficientAD, WinCLIP) on MVTec AD and VisA, under few-shot and distribution-shift conditions.
2. **Original contribution:** confidence-guided multi-scale anomaly fusion, ablated against mean/max/single-scale/fixed-weight fusion, with calibration and shift-robustness analysis.
3. **Inspection platform** (FastAPI + React + PostgreSQL + Docker) with heatmaps, nearest-normal-patch explanations, inspection history, and analytics.

## Documentation map

| Doc | Purpose | Read when |
|---|---|---|
| [AGENTS.md](AGENTS.md) | Rules and context for AI coding agents | Always (agents read this first) |
| [docs/00_IDEA_AND_GOALS.md](docs/00_IDEA_AND_GOALS.md) | Problem, idea, goals, non-goals, glossary | Starting out, explaining the project |
| [docs/01_PRD.md](docs/01_PRD.md) | Product requirements, user stories, screens | Building backend/frontend |
| [docs/02_TRD.md](docs/02_TRD.md) | Technical requirements, stack, targets | Making any technical choice |
| [docs/03_ARCHITECTURE.md](docs/03_ARCHITECTURE.md) | System design, diagrams, repo layout | Designing modules |
| [docs/04_RESEARCH_STRATEGY.md](docs/04_RESEARCH_STRATEGY.md) | Fusion method, hypotheses, experiment matrix | Doing research work |
| [docs/05_EVALUATION_PROTOCOL.md](docs/05_EVALUATION_PROTOCOL.md) | Metrics, splits, seeds, results schema | Writing or running any evaluation |
| [docs/06_API_AND_DATA_MODEL.md](docs/06_API_AND_DATA_MODEL.md) | Endpoints, schemas, DB tables | Backend work |
| [docs/07_ROADMAP.md](docs/07_ROADMAP.md) | 10-sprint Agile plan, gates, velocity tracking | Planning your sprint |
| [docs/08_AGENT_TASKS.md](docs/08_AGENT_TASKS.md) | Sprint backlogs, task definitions, Scrum templates | Executing sprints with AI agent |
| [docs/09_TESTING_AND_QUALITY.md](docs/09_TESTING_AND_QUALITY.md) | Test strategy, CI, definition of done | Before merging anything |
| [docs/10_DECISIONS_AND_RISKS.md](docs/10_DECISIONS_AND_RISKS.md) | ADR log, risk register, cut list | When a decision or problem arises |

## Repository structure

```
anomaly-inspect/
├── .github/workflows/ # CI pipeline (ruff, mypy, pytest)
├── docs/              # Full documentation set (11 files, see map above)
├── notebooks/         # Colab/Kaggle setup template (colab_setup.py)
├── scripts/           # Data download guides and utilities
├── research/          # Research pipeline: benchmarks, fusion, metrics, analysis
│   ├── common/        # Seed, config, results writer
│   ├── data/          # Dataset wrappers, splits, few-shot sampler
│   ├── metrics/       # AUROC, AUPRO, F1-max, ECE, risk-coverage
│   ├── benchmarks/    # Run scripts, sweep orchestration, profiling
│   ├── shift/         # Corruption pipeline
│   ├── fusion/        # Per-layer PatchCore, confidence signals, fusion
│   ├── explain/       # Overlay, contour, nearest-normal retrieval
│   ├── configs/       # One YAML per experiment run
│   └── analysis/      # Notebooks, plot scripts, experiment log
├── backend/           # FastAPI + SQLAlchemy + Alembic
├── frontend/          # React 18 + TypeScript + Vite (Sprint 9)
├── artifacts/         # Trained model artifacts (git-ignored)
├── data/              # Datasets (git-ignored)
├── runs/              # Per-run config/env snapshots (git-ignored)
├── results/           # Experiment result CSVs (committed)
├── pyproject.toml     # Package metadata, ruff, pytest, and mypy config
├── requirements-*.txt # Pinned deps (research, research-cpu, backend)
├── Makefile           # setup, setup-research, setup-backend, lint, test, bench, report, up
└── docker-compose.yml # One-command platform deployment (Sprint 10)
```

## Quickstart (fill in as built)

```bash
# 1. Local dev / research environment (Python 3.11; CPU by default, pass GPU=1 for CUDA)
make setup-research && make lint && make test
# 2. Download datasets (see docs/02_TRD.md, section Data)
# 3. Run one baseline (Sprint 2):  python -m research.benchmarks.run --model patchcore --dataset mvtec --category bottle
# 4. Run the platform (Sprint 10): docker compose up
```

## Results (fill in with real numbers only)

| Model | Dataset | Image AUROC | Pixel AUROC | AUPRO | Latency (ms) |
|---|---|---|---|---|---|
| PatchCore | MVTec AD | TBD | TBD | TBD | TBD |
| EfficientAD | MVTec AD | TBD | TBD | TBD | TBD |
| WinCLIP | MVTec AD | TBD | TBD | TBD | TBD |

## Datasets and licenses

- MVTec AD is licensed CC BY-NC-SA 4.0 (non-commercial). Do not redistribute it; the demo uses a few attributed sample images.
- VisA: check the current license in the official repository before use or redistribution.

## Citation and acknowledgements

PatchCore (Roth et al., 2022), EfficientAD (Batzner et al., 2024), WinCLIP (Jeong et al., 2023), Anomalib (Akcay et al.), MVTec AD (Bergmann et al., 2019), VisA (Zou et al., 2022). Verify exact references before publishing.
