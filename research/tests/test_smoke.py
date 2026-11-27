"""Smoke tests verifying the research package scaffold."""

import importlib


def test_research_packages_importable() -> None:
    """All scaffolded research subpackages should import cleanly."""
    modules = [
        "research",
        "research.common",
        "research.data",
        "research.metrics",
        "research.benchmarks",
        "research.shift",
        "research.fusion",
        "research.explain",
    ]
    for module_name in modules:
        assert importlib.import_module(module_name) is not None
