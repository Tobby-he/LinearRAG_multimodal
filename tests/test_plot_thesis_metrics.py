from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import plot_thesis_metrics as plotter


def test_module_imports():
    assert hasattr(plotter, "main")

