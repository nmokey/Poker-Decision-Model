"""Community Cloud entry point with its own CPU-only dependency lock."""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Serve the current checkout after code-only updates, including new package modules.
# Community Cloud may retain the installed wheel until dependency inputs change.
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "apps/demo.py"), run_name="__main__")
