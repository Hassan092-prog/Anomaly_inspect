"""Smoke tests verifying the backend package scaffold."""

import importlib


def test_backend_packages_importable() -> None:
    """All scaffolded backend subpackages should import cleanly."""
    modules = [
        "backend",
        "backend.app",
        "backend.app.routers",
        "backend.app.services",
        "backend.app.repositories",
        "backend.app.models",
        "backend.app.schemas",
        "backend.app.storage",
    ]
    for module_name in modules:
        assert importlib.import_module(module_name) is not None
