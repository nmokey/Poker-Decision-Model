"""Community Cloud entry point with its own CPU-only dependency lock."""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "apps/demo.py"), run_name="__main__")
