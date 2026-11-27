# AGENTS.md: Context and rules for AI coding agents

Read this file first, then the docs named in your task. The human owner is a final-year CS (AI & ML) student building this as a flagship resume project. Correctness and explainability matter more than speed. The owner must be able to defend every line in an interview.

## Project in one paragraph

AnomalyInspect detects and localizes **unseen** industrial defects using models trained on **normal images only**. It has (a) a research pipeline benchmarking PatchCore, EfficientAD and WinCLIP on MVTec AD and VisA, including few-shot and distribution-shift evaluation, (b) an original **confidence-guided multi-scale fusion** method, and (c) an inspection platform (FastAPI, React, PostgreSQL, Docker) with explanations. Full context: `docs/00_IDEA_AND_GOALS.md`.

## Source of truth order

If documents conflict, trust them in this order: `docs/05_EVALUATION_PROTOCOL.md` (for anything about metrics/splits) > `docs/02_TRD.md` > `docs/06_API_AND_DATA_MODEL.md` > `docs/04_RESEARCH_STRATEGY.md` > `docs/01_PRD.md` > everything else. If you find a conflict, stop and report it instead of choosing silently.

## Hard rules (never violate)

1. **Never fabricate results.** Do not write numbers into any README, report, or table that were not produced by a run recorded in `results/`. Use `TBD` otherwise.
2. **Never change metric code, data splits, seeds, or the results schema** without an explicit instruction and a matching update to `docs/05_EVALUATION_PROTOCOL.md`.
3. **No test-set leakage.** Test images and masks are never used for fitting, normalization statistics, threshold selection (except the clearly labeled *oracle* threshold in benchmark tables), or hyperparameter tuning.
4. **Do not commit** datasets, model weights, memory banks, `.env` files, or secrets. Respect `.gitignore`.
5. **Do not silently add dependencies.** Propose them with a reason; pin versions.
6. **Do not rewrite working modules** to "improve" them. Make the smallest change that satisfies the task.
7. **Report uncertainty.** If a library API differs from what the docs assume (Anomalib's API changes between versions), say so and show what you verified.

## Conventions

- Python 3.11, type hints on public functions, `ruff` for lint/format, `pytest` for tests.
- Config-driven experiments: one YAML per run under `research/configs/`. Every run records config, git SHA, seed, and library versions next to its results.
- Seeds: always set via `research/common/seed.py` (Python, NumPy, PyTorch, CUDA).
- Results: append rows to `results/*.csv` using the schema in `docs/05_EVALUATION_PROTOCOL.md`. One row per (model, dataset, category, k_shot, seed, shift, severity, fusion).
- Backend: FastAPI + Pydantic v2, async SQLAlchemy 2.0, Alembic migrations. Business logic lives in `backend/app/services/`, not in route handlers.
- Frontend: React 18 + TypeScript + Vite; data fetching with TanStack Query; no business logic in components.
- Commit style: Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`). One task = one branch = one focused PR/commit set.

## Commands (update as they come to exist)

```bash
make setup           # create .venv (Python 3.11 via uv/venv) and install dev deps
make setup-research  # install pinned research deps (CPU by default; GPU=1 for CUDA)
make setup-backend   # install pinned backend API deps
make lint         # ruff check + ruff format --check + mypy
make format       # ruff check --fix + ruff format
make test         # pytest -m "not slow" (fast tests only)
make test-all     # includes slow/GPU tests
make bench ARGS="--model patchcore --dataset mvtec --category bottle"  # wired in T-010
make report       # regenerate tables/figures from results/ (wired in T-016)
make up           # docker compose up --build (wired in T-070)
```

## How to work on a task (Agile Sprint Workflow)

This project uses **Agile Scrum** with 2-week sprints. See `docs/07_ROADMAP.md` for the sprint plan and `docs/08_AGENT_TASKS.md` for sprint backlogs and Scrum ceremony templates.

1. **Sprint Planning:** at the start of each sprint, present the sprint goal, planned tasks, and risks. Wait for owner approval.
2. **Standup:** at the start of each session, report done/doing/blocked and sprint progress.
3. Read the task in `docs/08_AGENT_TASKS.md` and every doc it links.
4. Write a short plan (files to touch, tests to add, risks) and wait for approval if the task is marked `REVIEW GATE`.
5. Implement the minimum that meets the **Definition of Done**.
6. Run lint and tests. Paste the actual output in your summary.
7. Summarize: what changed, what you verified, what you did not verify, open questions.
8. **Sprint Review:** at the end of each sprint, demo completed work and report velocity.
9. **Sprint Retrospective:** log what went well, what didn't, and what to change in `docs/retros/sprint_N.md`.

## Things that are easy to get wrong here

- Anomalib's PatchCore concatenates multiple layers into one feature before building the memory bank. The fusion contribution needs **separate memory banks per layer**, so it uses custom code in `research/fusion/`, not Anomalib's model class.
- Image-level score for PatchCore is the **max** of the (smoothed) patch-score map, not the mean.
- AUPRO is computed per connected region and integrated up to an FPR limit (0.3 by default). Do not substitute pixel AUROC.
- MVTec AD `train/` contains only normal images; any validation split of normals must be carved from it with a fixed seed.
- Few-shot subsets are sampled from the **normal training images only**, with 3 seeds.
- Corruptions apply to **test images only**, photometric only (no geometric transforms that would misalign masks).
