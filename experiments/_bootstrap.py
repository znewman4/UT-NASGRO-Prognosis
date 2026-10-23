"""Lets the numbered scripts run from a fresh clone without `pip install -e .`.

Summary tables go to results/summary/, or to <dir>/summary/ when the run is
redirected with --out <dir> (smoke tests), so trial runs never mix with results.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
SUMMARY = ROOT / "results" / "summary"
if "--out" in sys.argv:
    SUMMARY = Path(sys.argv[sys.argv.index("--out") + 1]) / "summary"
