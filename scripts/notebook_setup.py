"""Shared notebook runtime setup. Executed with runpy after locating a checkout.

Notebook callers supply ROOT and EXTRAS. Only Colab installs dependencies; local
users select the project environment as their kernel. No training runs here.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from poker_decision.paths import project_root

ROOT = project_root(Path(globals()["ROOT"]))
EXTRAS = globals()["EXTRAS"]
IN_COLAB = (
    importlib.util.find_spec("google") is not None
    and importlib.util.find_spec("google.colab") is not None
)
if IN_COLAB:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", f"{ROOT}[{EXTRAS}]"],
        check=True,
    )
os.chdir(ROOT)
print("Project:", ROOT)
