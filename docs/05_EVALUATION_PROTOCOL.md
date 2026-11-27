# 05. Evaluation Protocol (source of truth for metrics, splits, seeds)

Changing anything here invalidates previous results. Version this file; bump the version and rerun when it changes.

**Protocol version:** 1.0

## 1. Splits

| Split | Source | Use |
|---|---|---|
| train-fit | Normal training images minus validation fraction (or the k-shot subset) | Fit/train the model |
| val-normal | 10-20% of normal training images, fixed seed (default 0.15, seed 0) | Normalization stats, thresholds, calibration, hyperparameters |
| test | Official test set (normal + anomalous, with masks) | Final metrics only |

- k-shot subsets are drawn from **train-fit** only. Seeds {0, 1, 2}. Saved to `data/splits/{dataset}/{category}/k{k}_seed{s}.json`.
- For k-shot runs, **val-normal stays the same** as the full run to keep thresholds/statistics comparable. If you instead derive val from the k-shot pool, state it explicitly and keep it consistent.
- VisA: use the official split CSV for the one-class protocol.

## 2. Preprocessing

Resize 256, center crop 224, ImageNet normalization (configurable per model, but recorded). Anomaly maps are upsampled to the cropped image resolution; masks use the same crop.

## 3. Metrics

All metrics computed per category, then macro-averaged across categories (unweighted mean). Report both.

### 3.1 Image AUROC
ROC AUC of image-level scores vs image labels (0 normal, 1 anomalous).

### 3.2 Pixel AUROC
ROC AUC over all pixels of all test images (flattened) using ground-truth masks. Normal images contribute all-zero masks.

### 3.3 AUPRO
Per-region overlap (PRO): for each connected anomalous region in the GT mask, the fraction covered by the thresholded prediction. Average over regions, plot against pixel FPR over thresholds, integrate up to **FPR = 0.3**, and normalize by 0.3. Implement and unit-test yourself (see `09_TESTING_AND_QUALITY.md`); cross-check against a reference implementation on one category.

### 3.4 F1-max
Maximum F1 over all thresholds, computed at image level and pixel level. This is an **oracle** threshold (it uses test labels). Label it as oracle in every table.

### 3.5 Thresholds for deployment ("normal-only threshold")
Derived from val-normal scores only: `t = percentile(val_scores, 99)` (default) or `mean + 3*std`. Record which. Used for the platform, shift FPR analysis, and the shared-threshold experiment. Never tune it on test.

### 3.6 Calibration (ECE) of image scores
Raw anomaly scores are not probabilities. Procedure:

1. Fit a monotonic mapping from score to probability. Without real anomalies in training, use one of: (a) a logistic mapping defined by val-normal statistics (center at the normal-threshold, scale by val std), or (b) Platt scaling fitted on val-normal plus synthetic anomalies. Choose one, document it, and use it for all models.
2. Compute ECE on the **test** set with 10 equal-width bins: ECE = Σ_b (n_b / N) · |acc_b − conf_b|.
3. Report the reliability diagram.

State clearly in the report that calibration is relative to this mapping.

### 3.7 Selective prediction
Rank images by confidence; at coverage c in {0.5 ... 1.0}, compute the error rate on retained images using the normal-only threshold for decisions. Report the area under the risk-coverage curve (lower is better).

### 3.8 Shift robustness
- **Relative drop:** (metric_clean − metric_shifted) / metric_clean per corruption and severity.
- **Threshold stability:** FPR on shifted val-normal images (corrupted with the same recipe) at the threshold fixed on clean val-normal.
- **Shared-threshold test:** one pooled threshold across categories (percentile over pooled val-normal scores, after per-category normalization if the model supports it) vs per-category thresholds. Report FPR/FNR change.

### 3.9 Efficiency
- Latency: batch size 1, 10 warm-up runs, then 100 timed runs, `torch.cuda.synchronize()` around GPU timings. Report p50 and p95, GPU and CPU separately. Record hardware.
- Memory: peak GPU memory (`torch.cuda.max_memory_allocated`), memory-bank size on disk (MB), model parameter count.
- Train/fit time per category.

## 4. Corruption recipes (test images only)

| Corruption | Severity levels (suggested) |
|---|---|
| Brightness | factor ±{0.1, 0.2, 0.3, 0.4, 0.5} |
| Contrast | factor {0.8, 0.6, 0.5, 0.4, 0.3} (and above 1) |
| Gaussian noise | sigma {0.02, 0.04, 0.06, 0.08, 0.10} (on [0,1] scale) |
| Gaussian blur | sigma {1, 2, 3, 4, 5} px |
| JPEG compression | quality {80, 60, 40, 20, 10} |

Photometric only; masks unchanged. Fixed seed per image for noise. Record the exact recipe in `research/shift/recipes.yaml` and hash it into the run config.

## 5. Results schema

One CSV row per evaluated condition. File: `results/results.csv` (append-only; never edit by hand).

| Column | Type | Notes |
|---|---|---|
| run_id | str | UUID or timestamp-slug |
| protocol_version | str | `1.0` |
| git_sha | str | |
| timestamp | str | ISO 8601 |
| model | str | `patchcore`, `efficientad_s`, `winclip`, `pc_layerwise`, ... |
| backbone | str | e.g. `wide_resnet50_2`, `clip_vit_b_16` |
| dataset | str | `mvtec`, `visa` |
| category | str | |
| k_shot | str/int | `full`, 0, 1, 5, 10 |
| seed | int | |
| shift | str | `none`, `brightness`, ... |
| severity | int | 0 for none |
| fusion | str | `none`, `mean`, `max`, `norm_mean`, `fixed_0.3_0.7`, `confidence`, ... |
| fusion_signals | str | e.g. `sep+agree+focus` |
| threshold_type | str | `oracle`, `val_p99`, ... |
| image_auroc | float | |
| pixel_auroc | float | |
| aupro | float | |
| f1_max_image | float | oracle |
| f1_max_pixel | float | oracle |
| ece | float | |
| risk_cov_area | float | |
| latency_ms_p50 | float | |
| latency_ms_p95 | float | |
| peak_mem_mb | float | |
| bank_size_mb | float | |
| device | str | hardware string |
| notes | str | |

Missing metrics are left empty, not zero. Zero is a valid measured value (e.g., ECE = 0 for a perfectly calibrated model); empty means "not computed for this run".

## 6. Reproduction checks (must pass before trusting any table)

1. On 5 MVTec categories, baseline PatchCore and EfficientAD land within about 1-2 points of Anomalib's published/reference numbers or your documented reasons.
2. Metric unit tests pass (perfect and random predictors, known toy cases for AUPRO).
3. Re-running a config with the same seed reproduces results within floating-point tolerance (document tolerances).
4. No test-derived quantity appears in any fitting or calibration code path (code review checklist item).
