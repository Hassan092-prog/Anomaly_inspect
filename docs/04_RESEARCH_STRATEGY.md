# 04. Research Strategy

## 1. Research question

Can per-image, per-scale **confidence estimates** be used to fuse multi-scale anomaly maps so that localization and prediction reliability improve over fixed fusion rules, particularly with **limited normal data** and under **distribution shift**?

## 2. Hypotheses

| ID | Hypothesis | Evidence that supports it |
|---|---|---|
| H1 | Confidence-guided fusion matches or beats mean fusion on clean data (pixel AUROC, AUPRO) | Non-negative paired difference across categories |
| H2 | The advantage grows under corruption | Larger gap at higher severities |
| H3 | Fused scores are better calibrated and support better selective prediction | Lower ECE, lower area under the risk-coverage curve |
| H4 | The advantage grows when k is small (few-shot) | Larger gap at k = 1, 5 than at full |

Each hypothesis can fail. Report failures with the same prominence as successes. A well-analyzed null result is a valid outcome.

## 3. Method

### 3.1 Per-scale maps
Use WideResNet-50 (ImageNet pretrained). Extract features from separate layers (default: layer2 and layer3; layer1 as optional third scale). For each layer l:

1. Extract patch features, apply local neighborhood aggregation (PatchCore-style average pooling), optional random projection.
2. Build a **separate memory bank** B_l from normal training images (coreset subsampling, default 10%; record provenance per bank item).
3. For a test image, compute per-patch nearest-neighbor distance to B_l, reshape to a grid, upsample to input resolution, Gaussian smooth. This yields S_l(x, y).

*Sanity check:* a mean of normalized per-layer maps should land close to Anomalib's PatchCore performance. Large gaps indicate a bug.

### 3.2 Normalization
Using only **validation normals**, compute per-layer statistics (mean, std, and e.g. 99th percentile of patch scores). Normalize: Ŝ_l = (S_l − μ_l) / σ_l (or a percentile-based variant). This makes scales comparable and is itself a baseline ("normalized mean").

### 3.3 Confidence signals (per image, per scale)

| Signal | Definition (initial) | Intuition |
|---|---|---|
| c_sep | Separation of the scale's peak score from the validation normal-score distribution, e.g. (max Ŝ_l − μ_val_max) / σ_val_max | Scale reports something far outside normal |
| c_agree | Agreement of scale l with other scales, e.g. correlation of Ŝ_l with the mean of the others, or overlap of top-q% regions | Independent scales agree on where the anomaly is |
| c_focus | 1 − normalized entropy of the softmax-normalized map | A localized, peaked map is more decisive than a diffuse one |

Combine: c_l = g(c_sep, c_agree, c_focus), starting with a simple average of rescaled signals, then ablate each signal individually.

### 3.4 Fusion
Weights w_l = softmax(c_l / τ) over scales (τ tuned on validation normals plus synthetic anomalies if used, never on test). Fused map: S = Σ_l w_l Ŝ_l. Per-pixel weights are an optional variant (confidence computed in a local window).

Image-level score: max of the fused map (consistent with PatchCore). Image confidence: a function of the fused weights and signals (e.g. the weighted mean confidence), used for selective prediction.

### 3.5 Optional learned variant (stretch)
Train a small logistic regression or tiny MLP mapping confidence signals to scale weights using **synthetic anomalies** (e.g. CutPaste-style) on training normals. No real anomalous test data may be used.

## 4. Experiment matrix

| Axis | Values |
|---|---|
| Datasets | MVTec AD (all 15 or at least 10 categories), VisA (all 12 or at least 6) |
| Fusion | single scale (each), mean, max, normalized mean, fixed weights (e.g. 0.5/0.5, 0.3/0.7), **confidence-guided** |
| Signal ablation | c_sep only, c_agree only, c_focus only, all |
| Few-shot k | 1, 5, 10, full (3 seeds for k < full) |
| Shift | none, brightness, contrast, gaussian noise, gaussian blur, JPEG; severities 1-5 (or 1, 3, 5 to save compute) |
| Hyperparameters | τ in {0.5, 1, 2}; number of scales in {2, 3} |

Prune the grid early. Run the full grid only on a subset of categories, then confirm the winning configuration on all categories.

## 5. Metrics for the study

- Localization: pixel AUROC, AUPRO (FPR <= 0.3).
- Detection: image AUROC, F1-max (oracle).
- Reliability: ECE of calibrated image scores (calibration fitted on validation normals, see `05_EVALUATION_PROTOCOL.md`), risk-coverage curve and area, FPR on shifted normals at a threshold chosen on clean validation.
- Efficiency: latency overhead of fusion vs mean fusion.

## 6. Statistical treatment

- Report macro-average across categories plus per-category tables in the appendix.
- For paired comparisons (fusion vs mean) use a paired test across categories (Wilcoxon signed-rank) and report effect size; with 15 categories, avoid overclaiming significance.
- For few-shot results, report mean ± std over 3 seeds.
- Keep a log of every configuration tried (including failures) in `research/analysis/experiment_log.md` so selection effects are visible.

## 7. Threats to validity (write these in the report)

1. **Validation leakage:** tuning τ or signal weights on test data. Mitigation: only validation normals and synthetic anomalies.
2. **Category cherry-picking:** report all categories run, state which were excluded and why.
3. **Corruption realism:** synthetic corruptions approximate real shift only partly.
4. **Baseline strength:** compare against well-tuned mean fusion and a fair normalized baseline.
5. **Small effect sizes:** gains may fall within seed noise; show variance.
6. **Reimplementation drift:** the custom per-layer PatchCore may differ from Anomalib's; show the sanity-check numbers.

## 8. Decision rules (pre-registered)

Decide before running the full study:

- Fusion is called "beneficial" on a condition if it improves the macro mean pixel AUROC or AUPRO over normalized mean by a margin above seed noise, without hurting image AUROC by more than 0.5 points.
- If no condition meets this, report the method as not improving over baselines and analyze why (which signals were uninformative, where scales disagreed).

## 9. Deliverables

1. Ablation tables and figures (`results/` plus `docs/figures/`).
2. Reliability plots: reliability diagram, risk-coverage curves.
3. Qualitative gallery: best and worst cases, including failure cases where fusion hurts.
4. A short method description (about 1 page) usable in the report and README.

## 10. Minimum viable experiment

If time is tighter than planned, run this subset first:

1. Per-layer PatchCore on **5 MVTec categories** (bottle, cable, hazelnut, metal_nut, screw — covering both texture and object types).
2. Fusion variants: single-scale (each), mean, confidence-guided (all signals).
3. k-shot: full and k = 5 (1 seed).
4. Shift: none and Gaussian noise severity 3.
5. Metrics: pixel AUROC, AUPRO, image AUROC.

This produces the core ablation table with about 30-40 runs. Expand to the full matrix only after this is clean and sanity-checked.
