"""Structural checks on the design (NFR2)."""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def modules_loaded_by(module):
    """Import a module in a fresh interpreter and list the top-level packages it loads."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        f"import {module}; "
        "print('\\n'.join(sorted({name.split('.')[0] for name in sys.modules})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(SRC)],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.split())


def test_decision_engine_loads_neither_the_interface_nor_the_ml_framework():
    loaded = modules_loaded_by("sensoryaware.decision")
    assert "streamlit" not in loaded
    assert "tensorflow" not in loaded
    assert "tensorflow_hub" not in loaded


def test_default_rules_load_neither_the_interface_nor_the_ml_framework():
    loaded = modules_loaded_by("sensoryaware.config")
    assert not loaded & {"streamlit", "tensorflow", "tensorflow_hub"}


def test_model_module_defers_loading_tensorflow_until_used():
    loaded = modules_loaded_by("sensoryaware.model")
    assert "tensorflow" not in loaded
