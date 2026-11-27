# 10. Decisions and Risks

## 1. Decision log (ADRs)

Format: ID, status, context, decision, consequences. Add new entries at the bottom; never rewrite history, supersede instead.

### ADR-001. Use Anomalib for baselines
**Status:** accepted. **Context:** PatchCore, EfficientAD and WinCLIP are available with consistent data/metric plumbing. **Decision:** use Anomalib for the three baselines; pin one version. **Consequences:** faster progress and trusted reference behavior; API churn between versions requires pinning and adapter wrappers.

### ADR-002. Custom per-layer PatchCore for the fusion work
**Status:** accepted. **Context:** Anomalib's PatchCore concatenates layers into one memory bank, which hides per-scale maps. **Decision:** implement a lightweight per-layer PatchCore in `research/fusion/` and validate it against Anomalib (sanity check). **Consequences:** more code to own and test; necessary for the contribution.

### ADR-003. WinCLIP is the required vision-language baseline; AnomalyCLIP is a stretch
**Status:** accepted. **Context:** WinCLIP ships in Anomalib; AnomalyCLIP would need an external implementation with its own dependencies and training/prompt-learning setup. **Decision:** required VLM baseline = WinCLIP; add AnomalyCLIP only if schedule allows. **Consequences:** resume bullets mention WinCLIP unless AnomalyCLIP is actually run. (This corrects an earlier assumption that Anomalib covered both.)

### ADR-004. Images on filesystem, metadata in PostgreSQL
**Status:** accepted. **Decision:** store paths in DB, files on a volume. **Consequences:** simple and fast; swap storage backend later behind the `storage` module.

### ADR-005. Inference runs in a threadpool inside the API process
**Status:** accepted for MVP. **Decision:** `asyncio.to_thread`; revisit with a worker (RQ/Celery) if concurrency matters. **Consequences:** simplest Compose; blocking risk under heavy load is documented.

### ADR-006. Validation split carved from normal training images
**Status:** accepted. **Decision:** 15% fixed-seed val-normal for stats, thresholds, calibration. **Consequences:** slightly less training data; avoids test leakage.

### ADR-007. Oracle metrics are labeled
**Status:** accepted. **Decision:** F1-max is oracle; deployment thresholds are normal-only. Both appear but are never conflated.

### ADR-008. Results are append-only CSV with a versioned schema
**Status:** accepted. **Decision:** `results/results.csv`, schema per `05`. **Consequences:** every figure regenerates from one file; schema changes require a protocol version bump.

### ADR-009. Dependency and tooling choices
**Status:** open. **To decide:** `uv` vs `pip-tools` for pinning; Celery vs none; hosting target for the demo. Record the choice here when made.

## 2. Risk register

| # | Risk | Likelihood | Impact | Mitigation | Trigger to act |
|---|---|---|---|---|---|
| R1 | Colab/Kaggle GPU limits slow benchmark sweeps | High | Medium | Cache features; run categories in resumable batches; EfficientAD on fewer categories if needed | Sweep time over plan by 30% |
| R2 | Anomalib version changes break adapters | Medium | Medium | Pin; wrapper layer; reproduction check after any upgrade | Any import or API error after install |
| R3 | Custom per-layer PatchCore does not match Anomalib | Medium | High | Sanity check early (T-031); debug before building fusion on top | Delta over about 2 points on a category |
| R4 | Fusion shows no improvement | Medium | Medium | Pre-registered decision rule; honest null-result analysis; signal diagnostics | After Sprint 6 ablations |
| R5 | Test leakage through tuning | Medium | Very high | Val-normal only; leakage test; code review checklist | Any result that looks "too good" |
| R6 | Agent-generated code is subtly wrong (metrics, splits) | High | Very high | Owner-verify tasks; independent tests; cross-checks vs reference implementations | Any metric mismatch |
| R7 | Memory bank too large for demo hosting | Medium | Medium | Smaller coreset ratio; fewer categories in the demo; EfficientAD for the hosted demo; recorded video fallback | Container memory exceeded |
| R8 | Dataset license constraints on demo | Medium | Low | Sample images only with attribution; no redistribution of full dataset | Before publishing |
| R9 | Schedule slip | High | Medium | Cut list below; keep S4-S6 (robustness & fusion) intact | Velocity <60% for 2 consecutive sprints |
| R10 | Scope creep (extra datasets, models, UI polish) | High | Medium | Non-goals in `00`; log ideas in a "later" list | New idea during a sprint |
| R11 | EfficientAD training instability or long runtime | Medium | Low | Use S variant; limit categories; document failures | Training diverges on a category |
| R12 | Overclaiming in resume/report | Medium | High | Numbers only from `results/`; claims reviewed against `04` decision rules | Before publishing |

## 3. Cut list (in order of what to cut first)

1. VisA down to 6 categories, then to a single smoke-test category.
2. CLIP text-label explanation (T-042).
3. Dashboard down to two charts.
4. ONNX stretch (T-075).
5. Benchmarks and Models pages (keep Inspect and History).
6. EfficientAD on a category subset (document which).
7. Hosted demo, replaced by a recorded walkthrough.

**Never cut:** metric correctness tests, the shift evaluation, the fusion ablations, the reproduction checks, the README and results traceability.

## 4. Ideas parking lot (do not build now)

Ensemble across models, per-pixel confidence weighting beyond the basic variant, active learning from feedback, adaptive thresholds from live data, additional datasets, multi-class unified models, real camera integration.

## 5. Dependency tracking

When a dependency is added or its version changes, update this section.

| Package | Pinned version | Purpose | Added in task |
|---|---|---|---|
| ruff | 0.17.6 | Lint and formatting | T-001 |
| pytest | 8.4.2 | Unit and integration test runner | T-001 |
| mypy | 1.20.2 | Light static type checking | T-001 |
| anomalib | TBD | Baselines (PatchCore, EfficientAD, WinCLIP) | T-002 |
| torch | TBD | Deep learning framework | T-002 |
| fastapi | TBD | Backend API | T-050 |
| sqlalchemy | TBD | ORM and async DB access | T-050 |
| react | TBD | Frontend | T-060 |

Fill in exact versions as they are pinned. This table is a quick reference; the lockfiles are the real source of truth.
