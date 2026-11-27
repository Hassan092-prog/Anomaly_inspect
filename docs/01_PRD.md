# 01. Product Requirements Document (PRD)

## 1. Product summary

AnomalyInspect is a web application where a user uploads an image of an item, chooses a model and category, and receives an anomaly verdict, a localized heatmap, a confidence indication, and an explanation. All inspections are stored and summarized in a history view and an analytics dashboard.

## 2. Users

| Persona | Needs |
|---|---|
| **QA Inspector** (primary) | Fast pass/fail verdict, clear visual of where the defect is, ability to adjust sensitivity, history of past inspections |
| **ML Engineer / Reviewer** (secondary; also the interviewer) | Compare models, see scores and latency, inspect failure cases, understand reliability |

## 3. User stories

**Inspection**
- US-1: As an inspector, I upload an image and select a category so I get a verdict (normal / anomalous) with an anomaly score.
- US-2: I see a heatmap overlay and a contour around the predicted defect region.
- US-3: I can move a threshold slider and see the verdict and mask update without re-running the model.
- US-4: I see a confidence indicator so I know when to double-check manually.
- US-5: I can pick which model to use and see its latency.

**Explanation**
- US-6: I see the nearest normal reference patches so I understand what "normal" looks like for the flagged region.
- US-7 (optional): I see a zero-shot text label suggestion (e.g. scratch, crack, contamination).

**History and feedback**
- US-8: I browse past inspections with filters (category, model, verdict, date range).
- US-9: I open any past inspection and see exactly what was shown originally.
- US-10: I can mark a verdict as confirmed or overridden, to track agreement.

**Analytics**
- US-11: I see anomaly rate over time, score distribution by category, and model latency comparison.
- US-12: I see how often manual feedback disagrees with the model.

**Reviewer**
- US-13: I can open a Benchmarks page showing stored benchmark results (clean, few-shot, shifted) for each model.

## 4. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Upload PNG/JPG up to 10 MB; validate type and size | Must |
| FR-2 | Select model and category from available registered models | Must |
| FR-3 | Return image score, label, threshold used, anomaly map, overlay, contour | Must |
| FR-4 | Persist every inspection and prediction in PostgreSQL; images on a volume | Must |
| FR-5 | Interactive threshold slider (client-side re-thresholding of returned map) | Must |
| FR-6 | Inspection history with filters and pagination | Must |
| FR-7 | Analytics: anomaly rate over time, score distribution, latency by model | Must |
| FR-8 | Nearest-normal-patch explanation for PatchCore-family models | Must |
| FR-9 | Confidence indicator derived from fusion confidence signals | Should |
| FR-10 | Feedback (confirm/override) on verdicts | Should |
| FR-11 | Benchmarks page rendering stored results | Should |
| FR-12 | CLIP zero-shot text label for flagged region | Could |
| FR-13 | ONNX/CPU-optimized model option | Could |

## 5. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | Start everything with one command | `docker compose up` |
| NFR-2 | Inference latency (initial targets, revise after measurement) | CPU: EfficientAD-S <= 2 s, PatchCore <= 5 s; GPU <= 0.5 s |
| NFR-3 | Reproducibility | Every number traceable to a config + git SHA |
| NFR-4 | Reliability | Model load failure returns a clear error, not a crash |
| NFR-5 | Basic security | Upload validation, size limits, no path traversal, no secrets in repo |
| NFR-6 | Maintainability | Typed Python, lint clean, tests for core logic and API |
| NFR-7 | Accessibility basics | Keyboard-navigable controls, alt text, sufficient contrast |

## 6. Screens

1. **Inspect:** drag-and-drop upload, model and category selectors, Run button. Result panel: original | heatmap overlay | contour; score, label, confidence, latency; threshold slider; explanation tab (nearest normals); feedback buttons.
2. **History:** table with thumbnails, category, model, score, verdict, timestamp; filters; click opens detail.
3. **Dashboard:** anomaly rate over time (line), score distribution per category (histogram), latency by model (bar), feedback agreement (single stat).
4. **Benchmarks:** result tables and charts from `results/` (clean, few-shot, shifted).
5. **Models:** list of registered models and categories with stored default thresholds.

## 7. Out of scope

Authentication/roles, batch folder upload, video, camera integration, model training from the UI, mobile app, notifications.

## 8. Acceptance (product level)

- A new user runs the app, uploads a sample defective image, and sees a localized heatmap in under 10 seconds on a laptop CPU.
- Changing the threshold slider updates the overlay instantly.
- Reloading the app preserves history.
- The dashboard reflects inspections just performed.
