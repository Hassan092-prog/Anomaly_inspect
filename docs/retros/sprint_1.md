# Sprint 1 Retrospective

- **What went well:**
  - Completed all `8/8` foundational tasks (`T-001`–`T-008`) cleanly on `main` with strict lint (`ruff`, `mypy`) and fast CPU tests (`49 passed`) passing.
  - Caught and fixed subtle bugs before merge during per-task audits (RGBA mask alpha-channel leak, boolean resize crash, F1-max divide-by-zero warning, CSV truncated-row handling, and `connectivity=bool/float` validation).
  - Verified `anomalib==2.7.1` internals directly in `.venv` (correcting a mistaken 4-connectivity assumption to 8-connectivity for `AUPRO`) and cross-checked our `compute_aupro` against Anomalib's `_AUPRO` on real `mvtec/bottle` test masks (`83` images, within `1e-5`).
- **What didn't go well:**
  - Upstream MVTec AD auto-download in Anomalib currently returns HTTP 404, delaying real-dataset verification until manual `bottle.tar.xz` staging.
  - Some Anomalib class/API assumptions in early docs didn't match `2.7.1` (`Patchcore`/`EfficientAd`/`WinClip` naming, `_AUPRO` vs `AUPRO(fields=...)`, VisA `ValSplitMode.SAME_AS_TEST` default).
- **Action items for Sprint 2:**
  - Inspect Anomalib `2.7.1` `Patchcore` and `Engine` APIs in `.venv` before writing the `T-010` `[REVIEW GATE]` plan, explicitly enforcing our `data/splits/` (`train_fit`/`val`) with zero test leakage.
  - Build and test `T-010` and `T-015` on `homehp` CPU first (capturing real CPU latency/memory benchmarks) before running GPU timing on Jio Google AI Pro Colab.
  - Stage the remaining 14 MVTec AD categories (and VisA when ready) onto `homehp` / Google Drive ahead of Sprint 3 sweeps.
- **Velocity:** 8/8 tasks completed
