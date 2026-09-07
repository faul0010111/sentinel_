"""Phase 3 tests: the simulate CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cloudsentinel.cli import app

runner = CliRunner()


def test_scenarios_command_lists_all_ten() -> None:
    result = runner.invoke(app, ["scenarios"])
    assert result.exit_code == 0
    assert "ai_agent_identity_abuse" in result.stdout


def test_simulate_writes_dataset(tmp_path: Path) -> None:
    output = tmp_path / "events.parquet"
    result = runner.invoke(
        app,
        [
            "simulate",
            "-o",
            str(output),
            "--identities",
            "12",
            "--days",
            "6",
            "--events-per-day",
            "10",
            "--attack-ratio",
            "0.25",
            "--seed",
            "3",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert output.exists()
    assert (tmp_path / "events.parquet.manifest.json").exists()
    assert "SYNTHETIC DATA EXPERIMENT" in result.stdout
