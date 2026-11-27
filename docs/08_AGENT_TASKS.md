# 08. Agent Tasks — Sprint Backlog & Playbook

Each task is sized for one agent session. `REVIEW GATE` means: the agent writes a plan first and you approve before code. `OWNER-VERIFY` means you personally check the output (do not trust a green test alone).

## 1. Scrum Workflow for Agents

### Sprint Planning Prompt
At the start of each sprint, the agent reads this file and presents:

```
=== SPRINT N PLANNING ===
Sprint Goal: <from 07_ROADMAP.md>
Stories in this sprint: <task IDs and titles>
Dependencies satisfied: <yes/no, list any blockers>
Risks: <what could go wrong>
Plan: <order of tasks, estimated effort>
Ready to start? [waiting for owner approval]
```

### Daily Standup Format
Each agent session begins with:
```
--- Standup ---
✅ Done since last session: <tasks/subtasks completed>
🔄 Working on now: <current task>
🚫 Blocked by: <if any>
📋 Sprint progress: X/Y tasks done
```

### Sprint Review Format
At the end of each sprint:
```
=== SPRINT N REVIEW ===
Sprint Goal: <was it met? yes/no/partially>
Completed: <task IDs with evidence>
Not completed: <task IDs, reason, carried to next sprint>
Demo: <what can be shown working>
Metrics: <test results, lint output, reproduction checks>
Open questions: <anything for the owner>
```

### Sprint Retrospective Format
At the end of each sprint, save to `docs/retros/sprint_N.md`:
```
# Sprint N Retrospective
- **What went well:** <concrete wins>
- **What didn't go well:** <bottlenecks, surprises, bugs>
- **Action items for Sprint N+1:** <specific process or tooling changes>
- **Velocity:** X/Y tasks completed
```

## 2. Working Rules

1. One task per branch. Merge only after the owner reads the diff.
2. Ask the agent to **explain** non-obvious code (AUPRO, kNN scoring, fusion math) and then explain it back in your own words. If you cannot, you are not ready to put it on a resume.
3. When a result looks too good, assume a bug. Check for leakage first.
4. Keep the agent from "fixing" tests by weakening them. Tests encode the spec.
5. After any library upgrade, rerun the reproduction checks in `05_EVALUATION_PROTOCOL.md` section 6.
6. Commit docs updates together with the code that changes them.

## 3. Product Backlog (all tasks)

Status markers: `TODO` | `IN PROGRESS` | `DONE` | `BLOCKED` | `CUT`

### Sprint 1 — Project Foundations

| ID | Task | Status | Depends | Definition of Done |
|---|---|---|---|---|
| T-001 | Repo scaffold, Makefile, ruff/pytest config, `.gitignore`, `.env.example` | `DONE` | none | `make lint test` runs on an empty-but-valid tree; CI workflow green |
| T-002 | Environment files and pinned deps for research and backend | `TODO` | T-001 | Fresh env installs; versions pinned; Colab setup cell documented |
| T-003 | Seed utility and config loader (YAML to dataclass) | `TODO` | T-001 | Test proves same seed gives same tensor sequence |
| T-004 | Dataset wrappers for MVTec AD and VisA (images, masks, labels) | `TODO` | T-002 | Loader returns expected counts per category; masks aligned with images |
| T-005 | Split creation (val-normal, few-shot subsets) | `TODO` | T-004 | Deterministic JSON files; rerun is byte-identical; no overlap between val and train-fit |
| T-006 | Metrics: image AUROC, pixel AUROC, F1-max | `TODO` | T-001 | Unit tests for perfect/random/inverted predictors |
| T-007 | Metric: AUPRO (own implementation) `REVIEW GATE` `OWNER-VERIFY` | `TODO` | T-006 | Toy-case test matches hand-computed value; cross-check against a reference on one category within tolerance |
| T-008 | Results writer and schema validation | `TODO` | T-001 | Row appended matches schema in `05`; invalid row raises |

### Sprint 2 — First Baseline Working

| ID | Task | Status | Depends | Definition of Done |
|---|---|---|---|---|
| T-010 | PatchCore runner through Anomalib (fit, predict, evaluate) `REVIEW GATE` | `TODO` | T-004..T-008 | One category (bottle) produces all metrics and a results row |
| T-015 | Profiler (latency p50/p95, peak mem, bank size) | `TODO` | T-010 | Methodology matches `05` section 3.9; hardware recorded; PatchCore profiled (reused by S3 models) |

### Sprint 3 — All Baselines on MVTec

| ID | Task | Status | Depends | Definition of Done |
|---|---|---|---|---|
| T-011 | EfficientAD runner (S variant) | `TODO` | T-010 | One category produces all metrics; training time recorded |
| T-012 | WinCLIP runner (zero-shot and k-shot reference) | `TODO` | T-010 | Same as above; prompt set documented |
| T-013 | Sweep orchestrator over categories and datasets, resumable | `TODO` | T-010..T-012 | Interrupted sweep resumes without recomputing finished rows |
| T-016 | Aggregation and plotting scripts (`make report`) | `TODO` | T-013 | Tables and figures regenerate from CSV only |

### Sprint 4 — Robustness Evaluation

| ID | Task | Status | Depends | Definition of Done |
|---|---|---|---|---|
| T-014 | Few-shot sweep (k, seeds) using saved splits | `TODO` | T-005, T-013 | Rows for k = 1, 5, 10, full with 3 seeds |
| T-020 | Corruption pipeline from `recipes.yaml` | `TODO` | T-004 | Deterministic outputs; sample grid image saved; masks untouched |
| T-021 | Shifted evaluation runner | `TODO` | T-020, T-013 | Rows with shift and severity filled for all models |
| T-022 | Threshold analysis (per-category vs shared, FPR on shifted normals) | `TODO` | T-021 | Table and figure per `05` section 3.8 |

### Sprint 5 — Fusion Core

| ID | Task | Status | Depends | Definition of Done |
|---|---|---|---|---|
| T-030 | Per-layer PatchCore with provenance `REVIEW GATE` `OWNER-VERIFY` | `TODO` | T-004 | Separate banks per layer; provenance stored; unit test on tiny synthetic data |
| T-031 | Sanity check vs Anomalib PatchCore | `TODO` | T-010, T-030 | Normalized-mean fusion within ~1-2 points on 5 categories; deltas documented |
| T-032 | Normalization stats from val-normal | `TODO` | T-030 | `stats.json` written; test confirms no test data touched |
| T-033 | Confidence signals (sep, agree, focus) `REVIEW GATE` | `TODO` | T-032 | Pure functions with unit tests on synthetic maps |
| T-036 | Calibration, ECE, risk-coverage | `TODO` | T-006 | Tests on synthetic calibrated/miscalibrated scores |

### Sprint 6 — Fusion Ablations & Analysis

| ID | Task | Status | Depends | Definition of Done |
|---|---|---|---|---|
| T-034 | Fusion methods (single, mean, max, norm-mean, fixed, confidence-guided) | `TODO` | T-033 | All methods share one interface; tests for weight normalization |
| T-035 | Ablation runner and experiment log | `TODO` | T-034, T-014, T-021 | Grid from `04` section 4 runnable and resumable; log file updated |
| T-037 | Analysis notebook and figures (reliability, risk-coverage, gallery) | `TODO` | T-035, T-036 | Notebook runs end to end from `results/`; H1-H4 verdicts written |

### Sprint 7 — Explainability + Backend Skeleton

| ID | Task | Status | Depends | Definition of Done |
|---|---|---|---|---|
| T-040 | Overlay and contour functions | `TODO` | T-030 | Tests with synthetic blobs; visual check on 10 images |
| T-041 | Nearest-normal patch retrieval | `TODO` | T-030 | Planted-patch test returns the planted source; returns image id and coordinates |
| T-042 | CLIP zero-shot text label (optional) | `TODO` | T-040 | Prompt list documented; confidence scores returned |
| T-050 | FastAPI skeleton, config, `/health`, error format | `TODO` | T-001 | Test client passes; OpenAPI served |
| T-051 | DB models and Alembic migration (per `06`) | `TODO` | T-050 | `alembic upgrade head` works on empty DB; seed script idempotent |
| T-052 | Model registry and LRU loader with a stub model | `TODO` | T-050 | Unknown model returns 404; cache eviction tested |
| T-053 | `POST /inspections` full flow with stub model `REVIEW GATE` | `TODO` | T-051, T-052 | Persists inspection, prediction, files; invalid uploads rejected |

### Sprint 8 — Backend Complete

| ID | Task | Status | Depends | Definition of Done |
|---|---|---|---|---|
| T-054 | Real model adapters (PatchCore, EfficientAD, fusion) behind the interface | `TODO` | T-010, T-011, T-034, T-053 | Slow test runs one real prediction per adapter |
| T-055 | History and detail endpoints with filters and pagination | `TODO` | T-053 | Filter tests; ordering stable |
| T-056 | Analytics endpoints | `TODO` | T-055 | Queries match `06` section 5; tested on seeded data |
| T-057 | Explanation endpoints | `TODO` | T-041, T-053 | Returns nearest-normal refs |
| T-058 | Feedback and threshold update endpoints | `TODO` | T-053 | Tests included |
| T-059 | Benchmarks endpoint reading `results/results.csv` | `TODO` | T-008 | Read-only; handles missing file |

### Sprint 9 — Frontend

| ID | Task | Status | Depends | Definition of Done |
|---|---|---|---|---|
| T-060 | Vite + TS scaffold, API client, routing, layout | `TODO` | T-050 | Builds; lint clean |
| T-061 | Inspect page: upload, model/category, result viewer | `TODO` | T-053, T-060 | Works against real API; error states handled |
| T-062 | Threshold slider with canvas re-thresholding `OWNER-VERIFY` | `TODO` | T-061 | No network call on slider move; matches server label at default threshold |
| T-063 | History page with filters and detail view | `TODO` | T-055, T-060 | Pagination and empty state |
| T-064 | Dashboard charts | `TODO` | T-056, T-060 | Three charts plus summary stat; loading and error states |
| T-065 | Benchmarks and Models pages | `TODO` | T-059, T-060 | Tables render stored results |
| T-066 | Explanation tab (nearest normals) and feedback buttons | `TODO` | T-057, T-058, T-061 | Works end to end |

### Sprint 10 — Ship

| ID | Task | Status | Depends | Definition of Done |
|---|---|---|---|---|
| T-070 | Dockerfiles and Compose with health checks and volumes | `TODO` | T-054, T-065, T-066 | `docker compose up` from a clean clone works |
| T-071 | CI: lint, tests, Docker build | `TODO` | T-070 | Pipeline green on main |
| T-072 | README with diagram, tables, GIFs | `TODO` | all | Every number traceable to `results/` |
| T-073 | Technical report (4-6 pages) | `TODO` | T-037 | Includes protocol, results, threats to validity |
| T-074 | Demo hosting or recorded walkthrough | `TODO` | T-070 | Link works |
| T-075 | ONNX export and CPU latency (optional) | `TODO` | T-054 | Latency table added to benchmarks |

## 4. Current Sprint Backlog

> **Update this section at the start of each sprint.** Move the current sprint's tasks here with any additional notes, priorities, or adjustments.

### Active Sprint: S1 — Project Foundations (Weeks 1-2)

**Sprint Goal:** Get a working development environment where `make lint && make test` passes, datasets load correctly, all core metrics are implemented and tested, and the results writer can append valid rows.

**Priority order:**
1. T-001 — Repo scaffold (unblocks everything)
2. T-002 — Environment and deps (unblocks data loading)
3. T-003 — Seed utility (small, independent)
4. T-006 — Basic metrics (independent of data)
5. T-008 — Results writer (independent)
6. T-004 — Dataset wrappers (needs T-002)
7. T-007 — AUPRO (`REVIEW GATE` — plan first; toy tests need T-006, category cross-check needs T-004)
8. T-005 — Split creation (needs T-004)

**Blockers:** none (first sprint)

**Carry-over from previous sprint:** N/A

## 5. Sprint Review Checklist (use at end of every sprint)

- [ ] All planned tasks completed or explicitly carried over with reason
- [ ] `make lint && make test` still passes
- [ ] No test-set leakage introduced
- [ ] No tests weakened or deleted
- [ ] New dependencies justified and pinned
- [ ] Numbers in docs/README come from `results/` only
- [ ] Owner can explain the sprint's code
- [ ] Sprint velocity recorded in `07_ROADMAP.md`
- [ ] Retrospective logged in `docs/retros/sprint_N.md`
- [ ] Backlog updated: completed tasks marked `DONE`, next sprint's tasks confirmed

## 6. Prompt Template

```
=== SPRINT N — TASK <ID> ===
Task: <ID and title>
Sprint Goal: <from 07_ROADMAP.md>
Read first: AGENTS.md, <linked docs and sections>
Goal: <one sentence>
Constraints: <from AGENTS.md hard rules that apply; any extra>
Definition of done: <copy from the task table>
Process:
  1) Present standup (done / doing / blocked)
  2) Write a plan and list files you will touch
  3) Wait for approval if REVIEW GATE
  4) Implement the minimum that meets the DoD
  5) Run lint and tests and paste the real output
  6) Summarize: what changed, verified, not verified, open questions
Do not: add dependencies without asking, change metric code or splits, invent numbers.
```

## 7. Review Checklist (use on every task PR)

- [ ] Does the diff touch only what the task needs?
- [ ] Any test data used for fitting, normalization, thresholding, or tuning?
- [ ] Were any tests weakened or deleted?
- [ ] New dependencies justified and pinned?
- [ ] Numbers in docs or README come from `results/`?
- [ ] I can explain the core logic without reading it
