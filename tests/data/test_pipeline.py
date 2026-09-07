"""Phase 4 tests: quality gate, transforms and the pipeline itself."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cloudsentinel.pipeline import PipelineError, run_pipeline, write_data_lake
from cloudsentinel.preprocessing import deduplicate, enrich, normalize
from cloudsentinel.schema import events_to_dataframe
from cloudsentinel.simulator import CloudSimulator, SimulationConfig
from cloudsentinel.validation import check_drift, population_stability_index, validate_dataset

SMALL = SimulationConfig(
    n_identities=12, days=6, events_per_identity_per_day=12, attack_ratio=0.3, seed=21
)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return events_to_dataframe(CloudSimulator(SMALL).run().events)


def test_pipeline_produces_enriched_dataset(raw: pd.DataFrame) -> None:
    frame, result = run_pipeline(raw)

    assert result.report.passed
    assert result.output_rows == len(raw)
    assert {"hour", "is_weekend", "is_business_hour", "is_hosting_asn", "bytes_total"} <= set(
        frame.columns
    )
    assert frame["timestamp"].is_monotonic_increasing


def test_duplicates_are_removed(raw: pd.DataFrame) -> None:
    doubled = pd.concat([raw, raw.head(50)], ignore_index=True)
    _, result = run_pipeline(doubled)
    assert result.duplicates_removed == 50
    assert result.output_rows == len(raw)


def test_quality_gate_fails_loudly_in_strict_mode(raw: pd.DataFrame) -> None:
    broken = raw.copy()
    broken.loc[broken.index[:5], "bytes_out"] = -10

    with pytest.raises(PipelineError, match="quality gate failed"):
        run_pipeline(broken, strict=True)


def test_permissive_mode_continues(raw: pd.DataFrame) -> None:
    broken = raw.copy()
    broken.loc[broken.index[:5], "bytes_out"] = -10
    _, result = run_pipeline(broken, strict=False)
    assert not result.report.passed
    assert result.output_rows == len(raw)


def test_enrichment_flags_business_hours_and_origin(raw: pd.DataFrame) -> None:
    frame = enrich(normalize(raw))
    assert frame.loc[frame["is_business_hour"], "hour"].between(8, 18).all()
    assert not frame.loc[frame["is_business_hour"], "is_weekend"].any()
    assert frame["is_corporate_ip"].mean() > 0.9


def test_deduplicate_keeps_first(raw: pd.DataFrame) -> None:
    frame, removed = deduplicate(pd.concat([raw, raw], ignore_index=True))
    assert removed == len(raw)
    assert len(frame) == len(raw)


def test_data_lake_is_partitioned_by_date(raw: pd.DataFrame, tmp_path: Path) -> None:
    frame, _ = run_pipeline(raw)
    partitions = write_data_lake(frame, tmp_path / "lake")

    assert partitions == frame["timestamp"].dt.date.nunique()
    assert list((tmp_path / "lake").glob("event_date=*"))
    assert len(pd.read_parquet(tmp_path / "lake")) == len(frame)


def test_psi_detects_a_shifted_distribution() -> None:
    reference = pd.Series(range(1000))
    assert population_stability_index(reference, reference) < 0.01
    assert population_stability_index(reference, reference + 800) > 0.25


def test_drift_check_reports_per_column(raw: pd.DataFrame) -> None:
    frame, _ = run_pipeline(raw)
    half = len(frame) // 2
    shifted = frame.iloc[half:].copy()
    shifted["bytes_out"] = shifted["bytes_out"] * 500

    results = check_drift(frame.iloc[:half], shifted, ("bytes_out",))
    assert results[0].status.value == "fail"
    assert results[0].value is not None and results[0].value > 0.25


def test_validate_dataset_flags_missing_columns() -> None:
    report = validate_dataset(pd.DataFrame({"foo": [1, 2]}))
    assert not report.passed
    assert any(check.name == "required_columns" for check in report.failures)
