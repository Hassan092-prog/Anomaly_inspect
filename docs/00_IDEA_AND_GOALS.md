# 00. Idea and Goals

## 1. Problem

Manufacturing defects are rare, varied, and often unknown in advance. Supervised defect detectors need labeled examples of every defect type, which usually do not exist at deployment time. Normal samples, however, are plentiful. The practical question is: **can we flag and localize defects we have never seen, using only normal images, and can we tell when the system's answer is unreliable?**

## 2. Idea

Build a system that:

1. Learns what "normal" looks like from normal-only images, using pretrained visual representations and normality-based models.
2. Scores new images and produces a pixel-level anomaly map that localizes the defect.
3. Estimates **how much to trust** each prediction, especially when data is limited or the imaging conditions shift (lighting, blur, noise, compression).
4. Explains each decision visually (heatmap, defect contour, nearest normal reference) and exposes it through a usable inspection application.

## 3. Why this is more than "I ran PatchCore"

| Typical repo | This project |
|---|---|
| One model, one dataset, headline AUROC | Three model families, two datasets, one shared protocol |
| Clean test set only | Few-shot axis and corruption/shift axis |
| Score only | Score + localization + reliability (calibration, selective prediction) |
| Notebook | Reproducible pipeline plus a deployed inspection app |
| No original idea | Confidence-guided multi-scale fusion with full ablations |

## 4. Goals

**G1. Benchmark (required).** Reproducible comparison of PatchCore, EfficientAD and WinCLIP on MVTec AD and VisA with image AUROC, pixel AUROC, AUPRO, F1-max, latency and memory.

**G2. Robustness study (required).** Quantify performance under limited normal data (k = 1, 5, 10, full) and under five corruption types at multiple severities, plus a shared-threshold cross-category test.

**G3. Fusion method (required, the contribution).** Design and ablate confidence-guided multi-scale fusion. Report honestly whether and where it helps.

**G4. Explainability (required, minimal).** Heatmap overlay with defect contour and nearest-normal-patch retrieval. CLIP text label is optional.

**G5. Platform (required, thin but real).** Upload, predict, view, history, analytics, model selection, one-command Docker run.

**G6. Documentation and communication (required).** README with results and diagrams, a 4-6 page technical report, and a demo (hosted or recorded).

## 5. Non-goals

- Training new backbones or a new foundation model.
- Beating published state of the art. A clean, honest, reproducible study is the target.
- Multi-user auth, billing, multi-tenant features, camera/PLC integration.
- Video or real-time streaming inference.
- Supporting datasets beyond MVTec AD and VisA.

## 6. Success criteria

The project is successful when all of these are true:

- [ ] `make bench` reproduces baseline numbers within about 1-2% AUROC of published or Anomalib reference values on at least 5 MVTec categories.
- [ ] A results table covers 3 models x 2 datasets x (clean, few-shot, shifted) from recorded runs.
- [ ] Fusion ablation table includes: single scales, mean, max, fixed weights, confidence-guided, with at least 3 seeds for few-shot settings.
- [ ] A calibration or reliability analysis exists (ECE and risk-coverage), with plots.
- [ ] `docker compose up` starts the app; a stranger can upload an image and see a heatmap and score.
- [ ] README and report contain only numbers traceable to files in `results/`.
- [ ] The owner can explain, from memory, every metric, the fusion method, and the system design.

A **null or negative fusion result is acceptable** if the analysis is rigorous (see `04_RESEARCH_STRATEGY.md`).

## 7. Assumptions (edit these)

- Timeline: about 20 weeks of part-time work (about 10-15 hours/week).
- Compute: free Colab/Kaggle GPU for research; CPU-only for the demo app.
- Work is done mainly through an AI coding agent (Antigravity) under owner review.
- Solo developer.

## 8. Glossary

| Term | Meaning |
|---|---|
| Open-world / unseen defects | Defect types not present in training; the model sees only normal data |
| One-class / normality-based | Learn a model of normal, flag deviations |
| Anomaly map | Per-pixel score indicating how abnormal a region is |
| Memory bank | Stored set of normal patch features used for nearest-neighbor scoring (PatchCore) |
| Coreset subsampling | Selecting a representative subset of patch features to shrink the memory bank |
| Student-teacher | EfficientAD-style: a student network imitates a teacher on normal data; discrepancy signals anomalies |
| WinCLIP | Zero/few-shot anomaly detection using CLIP with window-based region scoring and text prompts |
| Image AUROC | Area under ROC for image-level normal vs anomalous |
| Pixel AUROC | Area under ROC at pixel level using ground-truth masks |
| AUPRO | Area under per-region-overlap curve up to a false-positive-rate limit (default 0.3) |
| F1-max | Best F1 over all thresholds (oracle threshold; reporting only) |
| ECE | Expected Calibration Error: gap between predicted confidence and observed frequency |
| Risk-coverage | Error rate as a function of the fraction of predictions kept after rejecting the least confident |
| Distribution shift | Test conditions differ from training (brightness, blur, noise, compression, contrast) |
| Few-shot (k) | Only k normal training images available |
