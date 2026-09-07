"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _repo_root_cwd(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run every test from the repository root so config/default.yaml resolves."""
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("CS_CONFIG_FILE", os.getenv("CS_CONFIG_FILE", "config/default.yaml"))
    from cloudsentinel.settings import reload_settings

    reload_settings()
    yield
    reload_settings()
