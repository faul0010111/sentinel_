"""Phase 1 tests: every declared layer exists and is importable."""

from __future__ import annotations

import importlib

import pytest

LAYERS = [
    "ingestion",
    "validation",
    "preprocessing",
    "features",
    "models",
    "temporal",
    "graph",
    "detection",
    "risk",
    "explainability",
    "monitoring",
    "simulator",
    "streaming",
    "api",
]


@pytest.mark.parametrize("layer", LAYERS)
def test_layer_importable_and_documented(layer: str) -> None:
    module = importlib.import_module(f"cloudsentinel.{layer}")
    assert module.__doc__, f"{layer} must document its scope"
