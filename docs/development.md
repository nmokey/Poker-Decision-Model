# Development setup and proposed conventions

Meeting 1 implements the PokerBench data contract, preprocessing, grouped splits,
CPU candidate baselines, and evaluation. Meeting 2 adds the joint transformer and
training walkthrough. The demo remains a placeholder. See the
[Meeting 1](meeting1-integration.md) and [Meeting 2](meeting2-integration.md) integration notes.

## Dependencies

Use a `src/` Python package and Python 3.12 as the team default. `pyproject.toml`
declares dependency ranges; `uv.lock` records exact versions. CI uses `--locked`
to detect stale lockfiles. NumPy and Pydantic are the proposed core dependencies.
Optional extras separate model work (torch, transformers), analysis (pandas,
sklearn, notebooks, plots), demo (Streamlit), and an LLM client (OpenAI).

Use `uv add PACKAGE`, `uv add --optional ml PACKAGE`, or `uv add --dev PACKAGE`.
Commit the manifest and lockfile together. Upgrade deliberately with
`uv lock --upgrade-package PACKAGE`, then run checks. Avoid a second requirements
file. Choose the poker variant and data source before adding simulator/solver
packages. Document accelerator-specific PyTorch setup when training begins.

Reference: [uv dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)
and [locking](https://docs.astral.sh/uv/concepts/projects/sync/).

## Package boundaries

- `data.py`: answer-free input schema, loading, preprocessing, validation, and grouped splits.
- `pipeline.py`: reproducible preparation, inspection, output manifests, and CLI.
- `models/`: encoder and shared candidate scorer; no generative vocabulary head.
- `baselines.py`: simple baselines and eventual LLM comparison.
- `evaluation.py`: quality, calibration, and performance measurements.
- `apps/`: interactive presentation of package functionality.
- `configs/`: reviewed experiment settings and provenance.
- `notebooks/`: exploration; reusable logic moves into the package.

## Notebooks in GitHub

Keep one maintained walkthrough notebook per meeting. Meeting 1 is teaching material
that shows raw rows, decision examples, validation, metrics, and mistakes. Put
reusable preprocessing, splitting, training, and evaluation in the Python package
with tests; the notebook and command-line runner should call that same code.
Future training scripts and the demo should import the package directly.

Meeting 2 calls `models/encoder.py` for joint tokenization, candidate pooling and
scoring, and `training.py` for training, saved-split loading, batched prediction,
and self-contained checkpoints. Install both `analysis` and `ml` extras for this
notebook and full mypy checks. Core-only mypy excludes the optional ML modules;
the separate ML CI job checks all modules and runs offline randomly initialized
BERT tests without fetching pretrained weights.

Before committing a notebook, restart its kernel and run all cells, then clear
outputs and execution counts. Commit the notebook source and supporting package
changes together. Keep downloaded data, generated datasets, split manifests,
metrics, and model files in the ignored output directories. Preserve links to the
team's original notebooks in the integration notes instead of keeping competing
copies of the pipeline in the repository.

Keep data, weights, credentials, and generated reports out of Git. Pin backbone
revisions and save tokenizer settings when models are introduced. Use opponent
history available before each decision; group related hands/sessions in splits.

## Testing strategy

Pytest and coverage are configured; Ruff handles formatting/lint and mypy handles
strict package typing. CI runs on Python 3.12 and 3.13. Add meaningful tests as
behavior is implemented: schema failures, split leakage, variable option counts,
permutation equivariance, legal-option masks, gradient flow, known metric values,
tiny training loss reduction, and checkpoint round trips. Keep routine tests
CPU-only and offline; separate pretrained downloads and live API calls.

Future evaluation should include agreement, log loss, Brier score, calibration,
latency, throughput, and opponent-history ablations. Fit calibration on validation
only and preserve an untouched test set. Record hardware and workload when
benchmarking; compare identical candidate options across systems.

## Scientific integrity checks

`scenarios.py` defines answer-free grouping; `cross_validation.py` fixes fold
memberships and settings before fitting, refits every baseline/model, and evaluates
held-out folds only after the fixed training budget. `diagnostics.py` measures
reliability, confusion/sizing errors, training-only menu association and conditional
paired grouped intervals. `source_experiments.py` keeps source label semantics separate.

New preparation writes train/validation file hashes. Legacy manifests are checked
against pinned cached raw rows; missing/corrupted legacy provenance requires explicit
re-preparation, not a silent new split. Never use the outer CV folds for early stopping,
calibration fitting, or reporting an unbiased score after tuning against them.

Run `uv sync --locked --extra analysis --extra ml --extra experiments` and
`uv run --locked --extra analysis --extra ml --extra experiments pytest` for all
offline safeguards. No test downloads pretrained weights or data. Full training
belongs on Colab; the unit tests use tiny randomly initialized BERTs.
