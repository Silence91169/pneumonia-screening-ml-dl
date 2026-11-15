"""Regression checks for claims that previously exceeded available evidence."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_split_script_scopes_duplicate_check_to_exact_hashes() -> None:
    source = (PROJECT_ROOT / "scripts" / "prepare_data.py").read_text(encoding="utf-8")
    assert "ZERO DATA LEAKAGE CONFIRMED" not in source
    assert "patient-level separation is not established" in source


def test_app_does_not_synthesize_benchmark_results() -> None:
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "Dynamic Threshold Simulation" not in source
    assert "tp_b" not in source
    assert "No prediction was generated" in source
