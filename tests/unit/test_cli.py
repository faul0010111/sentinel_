"""Phase 1 tests: the CLI surface is importable and every command runs."""

from __future__ import annotations

from typer.testing import CliRunner

from cloudsentinel.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "cloudsentinel" in result.stdout


def test_config_validate() -> None:
    result = runner.invoke(app, ["config", "validate"])
    assert result.exit_code == 0
    assert "valid" in result.stdout


def test_config_show_json() -> None:
    result = runner.invoke(app, ["config", "show", "--json"])
    assert result.exit_code == 0
    assert "behavioral" in result.stdout
