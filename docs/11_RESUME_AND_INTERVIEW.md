# 11. Resume and Interview Preparation

Only use numbers that exist in `results/`. Replace every `X` before publishing. Remove any bullet you cannot defend.

## 1. Resume entry (template)

**AnomalyInspect: Open-World Visual Anomaly Detection & Explainable Inspection**
*Python, PyTorch, OpenCV, Anomalib, CLIP, FastAPI, React, PostgreSQL, Docker*

- Built an anomaly detection platform that localizes previously unseen industrial defects from normal-only training images, benchmarking PatchCore, EfficientAD, and WinCLIP on MVTec AD and VisA under a single reproducible protocol (image/pixel AUROC, AUPRO, latency).
- Evaluated robustness to limited data (k = 1/5/10 normal images) and distribution shift (brightness, contrast, noise, blur, JPEG), quantifying a X-point mean pixel-AUROC drop at severity 3 and the impact of shared vs per-category thresholds.
- Designed confidence-guided multi-scale anomaly fusion (separation, cross-scale agreement, map focus), ablated against single-scale, mean, max, and fixed-weight fusion; result: X change in AUPRO and X change in ECE (replace with the true outcome, including "no significant gain" if that is what you found).
- Engineered an end-to-end inspection system (FastAPI, React, PostgreSQL, Docker Compose) with heatmap and contour overlays, nearest-normal-patch explanations, inspection history, and analytics; X ms median CPU latency.

Shorter variant (if space is tight): keep bullets 1, 3, and 4.

## 2. One-line and 30-second pitches

- **One line:** "A system that learns only from normal images to detect and localize unseen defects, and measures how reliable those predictions are under limited data and distribution shift."
- **30 seconds:** problem (defects are rare and unknown), approach (normality-based models on pretrained features), contribution (confidence-guided multi-scale fusion with ablations and calibration analysis), proof (benchmarks on MVTec AD and VisA plus robustness tests), delivery (FastAPI/React/Docker app with explanations).

## 3. Interview questions you should be able to answer

**Fundamentals**
1. Why can't you just train a classifier for defects? What is the one-class setting?
2. How does PatchCore score an image? Why max over patches? What does coreset subsampling trade off?
3. How does EfficientAD work (student-teacher, why is it fast)? What is the role of the penalty images?
4. How does WinCLIP produce a localization map from CLIP, and why is it zero-shot?
5. Pixel AUROC vs AUPRO: why report both? What bias does pixel AUROC have with small defects?
6. What is F1-max and why is it an oracle metric?

**Your design**
7. Why separate memory banks per layer, and how did you validate your reimplementation?
8. Explain each confidence signal and why it should indicate reliability. What did the ablation show?
9. How did you avoid test leakage? Where do thresholds come from?
10. How did you calibrate raw anomaly scores, and what are the limits of that ECE?
11. What did the fusion do worse on, and why?
12. How do your results compare to published numbers, and what explains the gaps?

**Robustness and reliability**
13. Which corruption hurt most, and why does it hurt this model family?
14. Why does a shared threshold fail across categories?
15. What would you do to make the system reliable in a real factory (drift monitoring, recalibration)?

**Engineering**
16. Why a threadpool for inference? When would you move to a worker queue?
17. How does the model registry avoid loading everything into memory?
18. How does the slider re-threshold without a server call?
19. Walk through the data model. Why images on disk and metadata in the DB?
20. How is the project reproducible? Show me one number and trace it to a config and commit.

**Honesty and process**
21. How did you use AI coding agents, and how did you verify their output?
22. What would you do with another month?

## 4. Answers to prepare (outline per question)

For each question write 4-6 sentences in your own words: the idea, the formula or mechanism, your specific implementation choice, the evidence from your results, and the limitation. Store these in `docs/interview_notes.md` (create it yourself; writing them is the preparation).

## 5. Demo script (3 minutes)

1. 20 s: problem and idea.
2. 60 s: upload a defective bottle image, show heatmap, contour, nearest-normal references, move the slider.
3. 40 s: switch model, compare latency and score.
4. 40 s: dashboard and history.
5. 20 s: show the benchmark/fusion results table and name one honest limitation.

## 6. Publishing checklist

- [ ] README has the architecture diagram, results table, GIFs, and a 3-line quickstart
- [ ] Report (4-6 pages) in `docs/` with threats to validity
- [ ] All numbers traceable to `results/`
- [ ] Dataset license notes present; no dataset redistributed
- [ ] LinkedIn post: problem, what you built, one surprising finding, link
- [ ] Repo pinned on GitHub; topics set (anomaly-detection, computer-vision, pytorch, fastapi, react)
