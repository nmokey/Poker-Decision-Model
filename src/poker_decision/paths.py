"""Project-wide locations, including read compatibility for earlier artifacts."""

from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Find this repository from a notebook, script, or nested working directory."""
    start = (start or Path.cwd()).resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/poker_decision").is_dir():
            return candidate
    raise FileNotFoundError("Run inside a Poker-Decision-Model checkout or supply --root")


def prepared_directory(root: Path) -> Path:
    """Prefer the task-based dataset; reuse an audited legacy split if it exists.

    Never move or rewrite existing datasets just to change a directory name.
    If both exist, the canonical location takes precedence, including on errors.
    """
    canonical = root / "data/processed/pokerbench"
    legacy = root / "data/processed/pokerbench_meeting1"
    if canonical.exists() or not (legacy / "manifest.json").is_file():
        return canonical
    return legacy


def matrix_directory(root: Path) -> Path:
    """Locate new or historical matrix checkpoints for packaging."""
    canonical = root / "artifacts/decision_matrix"
    legacy = root / "artifacts/week2_matrix"
    return legacy if not canonical.exists() and legacy.exists() else canonical
