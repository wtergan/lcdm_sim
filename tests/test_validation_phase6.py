from pathlib import Path

import pytest

pytest.importorskip("h5py")


def test_run_validation_suite_and_json_report(tiny_config, tmp_path):
    from lcdm_sim.simulation import run_simulation
    from lcdm_sim.validation import run_validation_suite, save_validation_report_json

    run_dir = Path(tmp_path) / "run"
    result = run_simulation(
        tiny_config,
        output_dir=run_dir,
        num_snapshots=3,
        history_stride=1,
        save_snapshots=True,
    )

    report = run_validation_suite(result, tiny_config)
    assert report.ok is True
    assert report.summary["num_checks"] >= 6
    assert report.summary["num_failed"] == 0
    assert any(c["name"] == "step_durations_nonnegative" for c in report.checks)

    out = Path(tmp_path) / "validation_report.json"
    save_validation_report_json(report, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_validate_run_directory_and_reference_compare(tiny_config, tmp_path):
    from lcdm_sim.simulation import run_simulation
    from lcdm_sim.validation import validate_run_directory

    run_a = Path(tmp_path) / "run_a"
    run_b = Path(tmp_path) / "run_b"
    _ = run_simulation(
        tiny_config, output_dir=run_a, num_snapshots=3, save_snapshots=True
    )
    _ = run_simulation(
        tiny_config, output_dir=run_b, num_snapshots=3, save_snapshots=True
    )

    report = validate_run_directory(run_a, reference_run_dir=run_b)
    assert report.ok is True
    assert report.reference is not None
    assert "final_a_abs_diff" in report.reference
    assert report.reference["final_a_abs_diff"] == pytest.approx(0.0)
