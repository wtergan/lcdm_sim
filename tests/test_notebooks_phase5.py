import json
from pathlib import Path

NOTEBOOKS = [
    Path("notebooks/01_intro_parameters.ipynb"),
    Path("notebooks/02_initial_conditions.ipynb"),
    Path("notebooks/03_cic_forces_and_diagnostics.ipynb"),
    Path("notebooks/04_full_pm_simulation.ipynb"),
]


def _load_notebook(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_sources(nb):
    for c in nb.get("cells", []):
        yield c.get("cell_type"), "".join(c.get("source", []))


def test_refactored_teaching_notebooks_exist_and_parse():
    for nb_path in NOTEBOOKS:
        assert nb_path.exists(), f"missing notebook: {nb_path}"
        nb = _load_notebook(nb_path)
        assert nb.get("nbformat", 0) >= 4
        assert len(nb.get("cells", [])) >= 3


def test_refactored_notebooks_import_package_apis_and_avoid_core_redefinitions():
    forbidden_defs = [
        "def generate_grf",
        "def initial_conditions_from_density",
        "def deposit_density",
        "def solve_poisson_potential",
        "def leapfrog",
    ]
    required_tokens = {
        "01_intro_parameters.ipynb": [
            "from lcdm_sim.config",
            "from lcdm_sim.cosmology",
        ],
        "02_initial_conditions.ipynb": ["from lcdm_sim.grf", "from lcdm_sim.zeldovich"],
        "03_cic_forces_and_diagnostics.ipynb": [
            "from lcdm_sim.cic",
            "from lcdm_sim.diagnostics",
        ],
        "04_full_pm_simulation.ipynb": [
            "from lcdm_sim.simulation",
            "from lcdm_sim.validation",
        ],
    }

    for nb_path in NOTEBOOKS:
        nb = _load_notebook(nb_path)
        joined = "\n".join(src for _, src in _cell_sources(nb))
        assert "lcdm_sim" in joined
        for token in required_tokens[nb_path.name]:
            assert token in joined, f"{token} missing in {nb_path.name}"
        for bad in forbidden_defs:
            assert bad not in joined, f"found core redefinition {bad} in {nb_path.name}"


def test_reproducibility_doc_exists():
    doc = Path("docs/reproducibility.md")
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "PYTHONPATH=src" in text
    assert "pytest -q tests" in text
