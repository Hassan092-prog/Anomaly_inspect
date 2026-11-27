# 07. Roadmap — Agile Sprint Plan

Assumes about 20 weeks at 10-15 hours/week, organized into **10 two-week sprints**. If you have less time, apply the cut list in `10_DECISIONS_AND_RISKS.md` first; do not compress the research sprints.

## Scrum Process

| Role | Who |
|---|---|
| **Product Owner** | Hassan (the human owner — sets priorities, accepts/rejects sprint deliverables) |
| **Scrum Master / Dev Agent** | AI coding agent (Hydrogen or successor — executes tasks, reports blockers) |

**Sprint cadence:**
- **Sprint Planning** (start of sprint): agent reads the sprint backlog from `08_AGENT_TASKS.md`, presents the sprint goal and planned stories, owner approves or reprioritizes.
- **Daily standups** (each session): agent summarizes what was done, what's next, and any blockers.
- **Sprint Review** (end of sprint): agent demos completed work, owner verifies against Definition of Done.
- **Sprint Retrospective** (end of sprint): document what went well, what didn't, and what to change in `docs/retros/sprint_N.md`.

**Artifacts:**
- **Product Backlog:** full task list in `08_AGENT_TASKS.md` section 3.
- **Sprint Backlog:** the tasks selected for the current sprint, in `08_AGENT_TASKS.md` section 4.
- **Increment:** working, tested code merged to `main` at the end of each sprint.
- **Sprint Board:** tracked via task status markers (`TODO`, `IN PROGRESS`, `DONE`, `BLOCKED`).

## Sprint Overview

| Sprint | Weeks | Sprint Goal | Key Deliverables | Gate |
|---|---|---|---|---|
| S1 | 1-2 | **Project foundations** | Repo scaffold, CI, data loaders, metrics with tests, results writer | `make lint && make test` green |
| S2 | 3-4 | **First baseline working** | PatchCore runner on one category; profiler built and run | One category produces all metrics and a results row |
| S3 | 5-6 | **All baselines on MVTec** | EfficientAD + WinCLIP runners; sweep orchestrator; full MVTec/VisA sweep; `make report` | 5 categories reproduce reference numbers ±1-2 pts |
| S4 | 7-8 | **Robustness evaluation** | Few-shot sweep; corruption pipeline; shifted evaluation | Few-shot curves and relative-drop tables exist |
| S5 | 9-10 | **Fusion core** | Per-layer PatchCore; sanity check; normalization; confidence signals | Sanity check passes; signal unit tests pass |
| S6 | 11-12 | **Fusion ablations & analysis** | All fusion methods; ablation runner; calibration/ECE; analysis notebook | Hypotheses H1-H4 have verdicts |
| S7 | 13-14 | **Explainability + backend skeleton** | Overlay, contour, nearest-normal retrieval; FastAPI skeleton with stub model | Explain module tested; API serves /health and /inspections |
| S8 | 15-16 | **Backend complete** | Real model adapters; all endpoints (analytics, feedback, benchmarks) | OpenAPI matches `06`; API tests pass |
| S9 | 17-18 | **Frontend** | All 5 screens; threshold slider; API integration | All screens work against real API; slider works client-side |
| S10 | 19-20 | **Ship** | Docker Compose; CI pipeline; README with real numbers; report; demo | Fresh clone → running app in <15 min |

> [!NOTE]
> **Overlap guidance:** Backend work (S7-S8) intentionally starts while fusion analysis wraps up, because the API uses a stub model initially. Use GPU-idle time for platform tasks. The inference interface from `03_ARCHITECTURE.md` is the contract between both halves.

## Sprint Details

### Sprint 1 — Project Foundations (Weeks 1-2)

**Sprint goal:** Get a working development environment where `make lint && make test` passes, datasets load correctly, all core metrics are implemented and tested, and the results writer can append valid rows.

**Backlog:** T-001, T-002, T-003, T-004, T-005, T-006, T-007, T-008

**Definition of Done:**
- [x] `make lint && make test` passes locally (`49 passed`, `ruff`/`mypy` clean; CI workflow configured in `.github/workflows/ci.yml`)
- [x] Metric tests pass (perfect predictor → 1.0, constant → ~0.5 AUROC, AUPRO toy case matches hand-computed value and real `mvtec/bottle` reference check passes)
- [x] `python -m research.data.make_splits --dataset mvtec --root data/mvtec_ad --category bottle` writes deterministic split files
- [x] Results writer appends a valid row matching the schema in `05`

**Sprint Review:** owner runs `make test` themselves, inspects split files, verifies AUPRO test logic.

---

### Sprint 2 — First Baseline Working (Weeks 3-4)

**Sprint goal:** Get PatchCore running end-to-end through Anomalib on one category (bottle), producing all metrics and a valid results row, and build the latency/memory profiler.

**Backlog:** T-010, T-015

**Definition of Done:**
- [ ] PatchCore on `bottle` produces image AUROC, pixel AUROC, AUPRO, F1-max, and a results CSV row
- [ ] Profiler records latency (p50/p95) and memory for PatchCore on documented hardware

**Sprint Review:** owner reads the results row, checks metrics against published PatchCore numbers for bottle, verifies no test leakage in the runner code.

---

### Sprint 3 — All Baselines on MVTec (Weeks 5-6)

**Sprint goal:** Complete EfficientAD and WinCLIP runners. Run all three models across MVTec and VisA categories. Build the resumable sweep orchestrator and aggregation scripts (`make report`).

**Backlog:** T-011, T-012, T-013, T-016

**Definition of Done:**
- [ ] EfficientAD-S and WinCLIP runners produce all metrics and valid results rows
- [ ] 5 MVTec categories reproduce reference numbers within ~1-2 points (deltas documented)
- [ ] Results for all 15 MVTec categories recorded; VisA on at least 6
- [ ] `make report` generates tables and figures from CSV alone
- [ ] Sweep orchestrator handles interruption and resumes

**Sprint Review:** owner checks reproduction deltas, runs `make report`, visually inspects generated figures.

---

### Sprint 4 — Robustness Evaluation (Weeks 7-8)

**Sprint goal:** Complete few-shot sweeps (`k = 1, 5, 10` across 3 seeds), build the corruption pipeline, run shifted evaluation for all models, and analyze threshold stability.

**Backlog:** T-014, T-020, T-021, T-022

**Definition of Done:**
- [ ] Corrupted test sets reproducible from recipe file; sample grid saved to `docs/figures/`
- [ ] Few-shot curves (metric vs k) per model plotted
- [ ] Relative-drop table per corruption and severity
- [ ] Shared-threshold experiment reported with FPR/FNR changes

**Sprint Review:** owner visually checks corruption samples, verifies masks are untouched, reviews threshold analysis.

---

### Sprint 5 — Fusion Core (Weeks 9-10)

**Sprint goal:** Build per-layer PatchCore with provenance, validate it against Anomalib's PatchCore, compute normalization stats, and implement all three confidence signals.

**Backlog:** T-030, T-031, T-032, T-033, T-036

**Definition of Done:**
- [ ] Per-layer banks built with provenance; unit tested on synthetic data
- [ ] Normalized-mean fusion within ~1-2 points of Anomalib PatchCore on 5 categories
- [ ] `stats.json` written from val-normal only (test confirms no test data touched)
- [ ] Confidence signals (sep, agree, focus) implemented as pure functions with unit tests
- [ ] Calibration + ECE + risk-coverage metrics implemented and tested

**Sprint Review:** owner verifies sanity-check deltas, reviews signal implementations and explains them back, checks for test leakage.

---

### Sprint 6 — Fusion Ablations & Analysis (Weeks 11-12)

**Sprint goal:** Implement all fusion methods, run the full ablation grid, generate reliability and risk-coverage plots, and state verdicts for hypotheses H1-H4.

**Backlog:** T-034, T-035, T-037

**Definition of Done:**
- [ ] All fusion methods share one interface; weight normalization tested
- [ ] Ablation grid from `04` section 4 runnable and resumable
- [ ] Signal ablation (sep / agree / focus / all) complete
- [ ] Reliability diagram and risk-coverage plots generated
- [ ] H1-H4 verdicts stated with evidence; decision rule from `04` section 8 applied
- [ ] Experiment log in `research/analysis/experiment_log.md` updated

**Sprint Review:** owner reviews hypothesis verdicts, checks that decision rules weren't rewritten post-hoc, can explain the fusion results.

---

### Sprint 7 — Explainability + Backend Skeleton (Weeks 13-14)

**Sprint goal:** Build the shared explain module (overlay, contour, nearest-normal retrieval). Stand up the FastAPI skeleton with DB models, registry, and the core inspection endpoint using a stub model.

**Backlog:** T-040, T-041, T-042 (optional), T-050, T-051, T-052, T-053

**Definition of Done:**
- [ ] Overlay and contour tested with synthetic blobs; visually checked on 10 images
- [ ] Nearest-normal retrieval returns correct source for a planted patch
- [ ] FastAPI serves `/health`; OpenAPI docs render
- [ ] `alembic upgrade head` works on empty DB; seed script idempotent
- [ ] `POST /inspections` with stub model persists inspection, prediction, and files
- [ ] Invalid uploads return clean 4xx errors

**Sprint Review:** owner uploads test images through the API, verifies explanation outputs, checks DB state.

---

### Sprint 8 — Backend Complete (Weeks 15-16)

**Sprint goal:** Wire real model adapters into the API. Implement all remaining endpoints: history, analytics, feedback, benchmarks, explanation.

**Backlog:** T-054, T-055, T-056, T-057, T-058, T-059

**Definition of Done:**
- [ ] Real model adapters (PatchCore, EfficientAD, fusion) pass slow tests
- [ ] All endpoints match `06_API_AND_DATA_MODEL.md`
- [ ] 100 sequential requests complete without memory growth (LRU cache works)
- [ ] Analytics queries match `06` section 5
- [ ] Feedback and threshold update endpoints tested

**Sprint Review:** owner runs the full API test suite, tries uploading through curl/Postman, checks analytics on seeded data.

---

### Sprint 9 — Frontend (Weeks 17-18)

**Sprint goal:** Build all five screens (Inspect, History, Dashboard, Benchmarks, Models) with full API integration. Threshold slider works client-side without network calls.

**Backlog:** T-060, T-061, T-062, T-063, T-064, T-065, T-066

**Definition of Done:**
- [ ] All screens in `01_PRD.md` section 6 exist and work against the real API
- [ ] Slider updates overlay and verdict without a network call
- [ ] Empty, loading, and error states handled on every page
- [ ] Basic accessibility pass (labels, focus order, alt text, contrast)

**Sprint Review:** owner clicks through every screen, tests edge cases (no data, slow model), verifies slider behavior with browser devtools.

---

### Sprint 10 — Ship (Weeks 19-20)

**Sprint goal:** Package everything into a one-command Docker deployment. Write the final README with real numbers, the technical report, and produce a demo.

**Backlog:** T-070, T-071, T-072, T-073, T-074, T-075 (optional)

**Definition of Done:**
- [ ] `docker compose up` from a fresh clone works in under 15 minutes
- [ ] CI pipeline: lint, tests, Docker build — all green on main
- [ ] Every number in README and report traceable to `results/`
- [ ] Report (4-6 pages) includes threats-to-validity section
- [ ] Walkthrough video or hosted demo link works
- [ ] Resume bullets updated with real numbers

**Sprint Review:** owner follows the README on a clean machine, verifies all numbers, reads the report end to end.

---

## Sprint Ceremonies Cadence

```
Week N (Mon):  Sprint Planning — review backlog, commit to sprint goal
Week N (daily): Standup — done / doing / blocked
Week N+1 (Fri): Sprint Review — demo increment, owner accepts/rejects
Week N+1 (Fri): Retrospective — log in docs/retros/sprint_N.md
Weekend:        Owner study — explain the sprint's code in own words
```

## Velocity Tracking

After each sprint, record actual vs planned story points (or task count) here:

| Sprint | Planned | Completed | Carried Over | Notes |
|---|---|---|---|---|
| S1 | 8 tasks | 8 tasks | 0 | All 8 completed (`T-001`–`T-008`); `mvtec/bottle` staged locally |
| S2 | 2 tasks | — | — | — |
| S3 | 4 tasks | — | — | — |
| S4 | 4 tasks | — | — | — |
| S5 | 5 tasks | — | — | — |
| S6 | 3 tasks | — | — | — |
| S7 | 7 tasks | — | — | — |
| S8 | 6 tasks | — | — | — |
| S9 | 7 tasks | — | — | — |
| S10 | 6 tasks | — | — | — |

Fill in after each sprint review. If velocity drops below 60% for two consecutive sprints, apply the cut list from `10_DECISIONS_AND_RISKS.md`.
