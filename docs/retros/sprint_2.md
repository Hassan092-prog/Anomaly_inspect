# Sprint 2 Retrospective

- **What went well:**
  - Completed `2/2` planned tasks (`T-010`, `T-015`) with clean lint (`ruff`, `mypy` across `24` files) and `60` passing fast tests.
  - Real `mvtec/bottle` PatchCore run produced clean, leak-free numbers (`image_auroc=1.0000`, `pixel_auroc=0.9799`, `aupro=0.9224`, `f1_max_image=1.0000`, `f1_max_pixel=0.7108`, `val_p99=32.28`) recorded in `results/results.csv`.
  - Saved standalone `artifacts/patchcore/mvtec/bottle/` (`model.pt`, `stats.json`, `threshold.json`, `meta.json`) reloads cleanly offline (`13,955` coreset vectors) and runs inference at `679.36 ms` p50 (`818.99 ms` p95) on `homehp` CPU, clearing the `<= 5s` CPU TRD target.
  - Caught subtle bugs during post-task audits (val split metadata mismatch check, few-shot artifact clobber guard, profiler timestamped `run_id`, checkpoint key/empty-bank validation, `threshold.json` lookup).
- **What didn't go well:**
  - Anomalib's `KCenterGreedy` sequential coreset loop took ~22 minutes on `homehp` CPU for `bottle` (`13,954` greedy steps), confirming multi-category and multi-seed sweeps cannot practically run on local CPU.
  - First standalone `profile_patchcore_mvtec_bottle` CSV row ran before `run_id` timestamping was added in `3fe59ea`.
- **Action items for Sprint 3:**
  - Stage full MVTec AD (~5 GB) and VisA (~10 GB) on Google Drive (`/MyDrive/Anomaly_inspect/data/`) before heavy `T-013` sweeps on Colab T4/L4 GPU.
  - Implement and unit-test `T-011` (`EfficientAd` S variant), `T-012` (`WinClip` zero-shot + few-shot), `T-013` (resumable sweep orchestrator), and `T-016` (`make report`) locally with offline synthetic tests before launching Colab sweeps.
  - Capture GPU `profile` rows (`T-015` on `cuda`) alongside CPU rows once Colab sweep runs execute.
- **Velocity:** 2/2 tasks completed
