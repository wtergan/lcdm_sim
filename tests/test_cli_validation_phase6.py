import json
from pathlib import Path

import pytest

pytest.importorskip("h5py")


def test_cli_validate_command_writes_report(tiny_config, tmp_path, capsys):
    from lcdm_sim.cli import main
    from lcdm_sim.simulation import run_simulation

    run_dir = Path(tmp_path) / "run_cli"
    _ = run_simulation(
        tiny_config, output_dir=run_dir, num_snapshots=3, save_snapshots=True
    )

    code = main(["validate", "--run-dir", str(run_dir)])
    assert code == 0
    captured = capsys.readouterr()
    assert "validation" in captured.out.lower()

    report_path = run_dir / "metrics" / "validation_report.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "ok" in data
    assert "summary" in data
