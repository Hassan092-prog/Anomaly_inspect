# 09. Testing and Quality

## 1. Principles

- Tests encode the spec. Agents may add tests; they may not weaken them to make code pass.
- Test pure logic heavily (metrics, fusion, signals, explanations). Test integration lightly but for real.
- Keep a **fast suite** (seconds, CPU, no datasets) for every commit and a **slow suite** (real models/data) run manually or nightly.

## 2. Test layers

| Layer | Scope | Tools | Examples |
|---|---|---|---|
| Unit (research) | metrics, splits, corruptions, signals, fusion, explain | pytest | Perfect predictor AUROC = 1; AUPRO toy case; weights sum to 1; contour of a synthetic blob |
| Property/sanity | invariants | pytest + hypothesis (optional) | Fusion with identical scales equals that scale; monotonic score to probability mapping |
| Unit (backend) | services, repositories, explain | pytest, pytest-asyncio | Threshold labeling; analytics SQL on seeded data |
| API | routers end to end with a stub model | httpx AsyncClient, test Postgres (container) | Upload valid image returns 201; invalid type returns 415/422; unknown model 404 |
| Slow integration | one real prediction per adapter | pytest marker `slow` | PatchCore adapter returns map of correct shape and finite score |
| Frontend | components and key flows | vitest, Testing Library, Playwright (optional, one smoke test) | Slider changes label without network call; empty states render |
| Reproduction | published/reference numbers | script + documented tolerances | 5-category baseline check |

## 3. Must-have metric tests

1. `image_auroc`: perfect = 1.0, inverted = 0.0, constant ≈ 0.5.
2. `pixel_auroc`: same, plus a case with an all-normal image mixed in.
3. `aupro`: a hand-built mask with two regions and a hand-built map where the correct value is computed by hand; verify normalization by the FPR limit; verify monotonic behavior as the FPR limit grows.
4. `f1_max`: known small example.
5. `ece`: perfectly calibrated synthetic data gives near 0; always-overconfident synthetic data gives a clearly positive value.
6. Split tests: val and train-fit disjoint; few-shot subsets are subsets of train-fit; deterministic across runs.
7. Leakage guard: a test that fitting/normalization functions never receive test-split paths (assert on the dataset split tag).

## 4. Backend and security tests

- Oversized upload rejected; non-image bytes with `.png` extension rejected; path traversal in filename ignored.
- Concurrent requests do not load duplicate model instances (cache lock).
- Migration up/down on a clean DB.
- API contract test: response matches the Pydantic schemas in `06`.

## 5. CI pipeline (GitHub Actions)

1. Set up Python and Node; cache dependencies.
2. `ruff check`, `ruff format --check`, light `mypy` on `research/metrics`, `research/fusion`, `backend/app`.
3. `pytest -m "not slow"` for research and backend (Postgres service container for API tests).
4. Frontend: lint, typecheck, vitest, build.
5. Build Docker images (no push) on main and PRs.
6. Optional nightly: slow suite on a small CPU-friendly fixture dataset.

## 6. Code quality standards

- Type hints on public functions; docstrings on metric and fusion functions stating shapes and units.
- No function over about 60 lines without a reason; no business logic in route handlers or React components.
- Logging via `logging`/structlog, not prints, in the backend; experiment scripts log config and key stats.
- Errors: raise specific exceptions in research code; map to the standard error format in the API.

## 7. Definition of Done (any task)

- [ ] Acceptance criteria in the task and sprint are met
- [ ] Lint and fast tests pass, with real output shown
- [ ] New logic has tests; no tests weakened
- [ ] Docs updated if behavior, schema, or protocol changed
- [ ] No secrets, data, or weights committed
- [ ] Owner reviewed the diff and can explain it

## 8. Manual QA checklist before release

- [ ] Fresh clone follows README to a running app
- [ ] Upload a normal image and a defective image for each of three categories
- [ ] Slider, history filters, and dashboard behave correctly
- [ ] Restart the stack; history persists
- [ ] Kill the DB briefly; API returns clean errors and recovers
- [ ] Browser console has no errors on any page

## 9. Test data management

- **Never commit real dataset images** to the repository. Tests use synthetic images (numpy arrays of known patterns, solid colors, random noise with a fixed seed).
- For API integration tests, use a small fixture set (3-5 images, about 100x100 px) generated programmatically in `conftest.py`.
- Slow tests that need real model artifacts use a pytest marker (`@pytest.mark.slow`) and are skipped in CI unless explicitly opted in.
