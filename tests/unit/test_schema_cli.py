"""Phase 2 tests: schema CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from cloudsentinel.cli import app

runner = CliRunner()


def test_schema_example_is_valid_json() -> None:
    result = runner.invoke(app, ["schema", "example"])
    assert result.exit_code == 0
    assert "AssumeRole" in result.stdout


def test_schema_export_writes_json_schema(tmp_path: Path) -> None:
    result = runner.invoke(app, ["schema", "export", "-o", str(tmp_path)])
    assert result.exit_code == 0
    schema = json.loads((tmp_path / "cloud_event.schema.json").read_text())
    assert schema["properties"]["privilege_level"]
    assert (tmp_path / "alert.schema.json").exists()


def test_schema_validate_flags_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"identity_id": "x"}\n')
    result = runner.invoke(app, ["schema", "validate", str(bad)])
    assert result.exit_code == 1
    assert "quarantined" in result.stdout


def test_schema_validate_missing_file() -> None:
    result = runner.invoke(app, ["schema", "validate", "nope.jsonl"])
    assert result.exit_code == 2
