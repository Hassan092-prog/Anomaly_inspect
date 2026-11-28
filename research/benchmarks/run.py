"""CLI entrypoint for running single baseline benchmark experiments (``make bench``)."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from research.benchmarks.patchcore_runner import run_patchcore_experiment
from research.common.config import ExperimentConfig, load_config, validate_config
from research.common.results import ResultRow


def _apply_cli_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    data = cfg.data
    model = cfg.model
    if args.dataset is not None:
        data = replace(data, dataset=args.dataset)
    if args.category is not None:
        data = replace(data, category=args.category)
    if args.k_shot is not None:
        k_val: int | str = "full" if args.k_shot == "full" else int(args.k_shot)
        data = replace(data, k_shot=k_val)
    if args.model is not None:
        model = replace(model, name=args.model)

    updated = replace(
        cfg,
        data=data,
        model=model,
        seed=args.seed if args.seed is not None else cfg.seed,
        results_csv=str(args.results_csv) if args.results_csv is not None else cfg.results_csv,
    )
    validate_config(updated)
    return updated


def run_from_config(
    cfg: ExperimentConfig,
    *,
    device: str | None = None,
    write_results: bool = True,
) -> ResultRow:
    """Dispatch benchmark run for the configured model."""
    if cfg.model.name == "patchcore":
        return run_patchcore_experiment(cfg, device=device, write_results=write_results)
    raise ValueError(
        f"Unsupported benchmark model {cfg.model.name!r} (only 'patchcore' in Sprint 2)"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single AnomalyInspect benchmark experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("research/configs/patchcore_bottle.yaml"),
        help="Path to experiment YAML config.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override model name.")
    parser.add_argument(
        "--dataset", choices=("mvtec", "visa"), default=None, help="Override dataset."
    )
    parser.add_argument("--category", type=str, default=None, help="Override category.")
    parser.add_argument("--k-shot", type=str, default=None, help="Override k-shot ('full' or int).")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="Execution device.")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Optional path to results CSV output.",
    )
    parser.add_argument(
        "--no-results",
        action="store_true",
        help="Run fit/eval without appending to results CSV.",
    )
    return parser


def _main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)
    row = run_from_config(cfg, device=args.device, write_results=not args.no_results)
    print(
        f"[{row.model}] {row.dataset}/{row.category} (k={row.k_shot}, seed={row.seed}): "
        f"img_auroc={row.image_auroc:.4f}, px_auroc={row.pixel_auroc:.4f}, "
        f"aupro={row.aupro:.4f}, f1_img={row.f1_max_image:.4f}, "
        f"f1_px={row.f1_max_pixel:.4f}, bank={row.bank_size_mb:.2f}MB"
    )


if __name__ == "__main__":
    _main()
