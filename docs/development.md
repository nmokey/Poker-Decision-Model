# Development setup and conventions

Meeting 1 implements the PokerBench data contract, preprocessing, grouped splits,
CPU candidate baselines, and evaluation. Meeting 2 adds the joint transformer and
training walkthrough. PocketSolver serves the saved model and Groq baseline;
see [deployment notes](deployment.md). See the
[Meeting 1](meeting1-integration.md) and [Meeting 2](meeting2-integration.md) integration notes.

## Dependencies

Use a `src/` Python package and Python 3.12 as the team default. `pyproject.toml`
declares dependency ranges; `uv.lock` records exact versions. CI uses `--locked`
to detect stale lockfiles. NumPy and Pydantic are the core dependencies.
Optional extras separate model work (torch, transformers), analysis (pandas,
sklearn, notebooks, plots), demo (Streamlit), and a live LLM client (Groq plus dotenv).

Use `uv add PACKAGE`, `uv add --optional ml PACKAGE`, or `uv add --dev PACKAGE`.
Commit the manifest and lockfile together. Upgrade deliberately with
`uv lock --upgrade-package PACKAGE`, then run checks. Avoid a second requirements
file. Choose the poker variant and data source before adding simulator/solver
packages. The completed GPU matrix used three RTX 3090s; see
[training instructions](week2-training.md) for the locked environment and settings.

Reference: [uv dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)
and [locking](https://docs.astral.sh/uv/concepts/projects/sync/).

## Package boundaries

- `data.py`: answer-free input schema, loading, preprocessing, validation, and grouped splits.
- `pipeline.py`: reproducible preparation, inspection, output manifests, and CLI.
- `models/`: encoder and shared candidate scorer; no generative vocabulary head.
- `baselines.py`: uniform, option-prior, and learned linear candidate baselines.
- `training.py`: fitting, inference, data provenance, and portable checkpoints.
- `cross_validation.py`: fixed-budget scenario-grouped evaluation and verified resume.
- `experiments.py`: training-only tiny checks and per-backbone matrix execution.
- `scripts/run_week2_matrix.py`: three-GPU queues and combined metric export.
- `evaluation.py`: quality, calibration, and performance measurements.
- `apps/`: interactive presentation of package functionality.
- `configs/`: pointers to canonical settings; reviewed results live in `docs/results/`.
- `notebooks/`: exploration; reusable logic moves into the package.

## Notebooks in GitHub

Keep one maintained walkthrough notebook per meeting. Meeting 1 is teaching material
that shows raw rows, decision examples, validation, metrics, and mistakes. Put
reusable preprocessing, splitting, training, and evaluation in the Python package
with tests; the notebook and command-line runner should call that same code.
The training runner and demo import the package directly.

Meeting 2 calls `models/encoder.py` for joint tokenization, candidate pooling and
scoring, and `training.py` for training, saved-split loading, batched prediction,
and self-contained checkpoints. Install both `analysis` and `ml` extras for this
notebook and full mypy checks. Core-only mypy excludes the optional ML modules;
the separate ML CI job checks all modules and runs offline randomly initialized
BERT, DistilBERT, and ModernBERT tests without fetching pretrained weights.
The core-only command excludes `training.py`, `cross_validation.py`,
`experiments.py`, and `models/encoder.py`; the ML job checks all package modules.

For executable notebook changes, validate the relevant workflow in its supported
environment, then clear outputs and execution counts. For prose-only changes,
validate notebook schema and code-cell preservation without restarting training.
Commit notebook source and supporting package changes together. Keep downloaded data, generated datasets, split manifests,
example-level predictions, and model files in the ignored output directories.
Publish reviewed aggregate metrics and provenance in `docs/results/` and explain
their interpretation in `docs/week2-results.md`. Preserve links to the
team's original notebooks in the integration notes instead of keeping competing
copies of the pipeline in the repository.

Keep raw data, weights, credentials, and unreviewed run artifacts out of Git. Pin
backbone revisions and save tokenizer settings with checkpoints. Use opponent
history available before each decision; group related hands/sessions in splits.

## Testing strategy

Pytest and coverage are configured; Ruff handles formatting/lint and mypy handles
strict package typing. CI runs on Python 3.12 and 3.13. Add meaningful tests as
behavior is implemented: schema failures, split leakage, variable option counts,
candidate-order sensitivity, legal-option masks, gradient flow, known metric values,
tiny training loss reduction, and checkpoint round trips. Keep routine tests
CPU-only and offline; separate pretrained downloads and live API calls.

The completed grouped experiment reports agreement, log loss, Brier score, and
reliability diagnostics. Further work includes retraining across seeds, independent
evaluation, calibration, serving latency/throughput, and opponent-history ablations.
Fit calibration on validation only and preserve an untouched test set. Record hardware and workload when
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
runs on a local CUDA GPU or Colab GPU; unit tests use tiny random encoders.
The [completed results](week2-results.md) are a separate empirical artifact, not
unit-test scores. Preserve the recorded training revision and package versions
when resuming; current application changes can produce a new protocol hash.
