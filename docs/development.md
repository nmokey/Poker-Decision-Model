# Development setup and conventions

The package supports audited PokerBench preparation, shared-head transformer
training, grouped evaluation, Leduc/real Hold’em history experiments and PocketSolver
serving. See the [documentation index](README.md), [current context](project-context.md)
and [deployment guide](deployment.md) for task boundaries and entry points.

## Dependencies

Use a `src/` Python package and Python 3.12 as the team default. `pyproject.toml`
declares dependency ranges; `uv.lock` records exact versions. CI uses `--locked`
to detect stale lockfiles. NumPy and Pydantic are the core dependencies.
Optional extras separate model work (torch, transformers), analysis (pandas,
sklearn, SciPy, threadpoolctl, notebooks and plots), simulation/source controls
(OpenSpiel and PokerKit), demo (Streamlit), and live LLM calls (Groq plus dotenv).

Use `uv add PACKAGE`, `uv add --optional ml PACKAGE`, or `uv add --dev PACKAGE`.
Commit the manifest and lockfile together. Upgrade deliberately with
`uv lock --upgrade-package PACKAGE`, then run checks. Avoid a second requirements
file. OpenSpiel/PokerKit are already declared in the `experiments` extra.
The completed GPU matrix used three RTX 3090s; see
[training instructions](week2-training.md) for the locked environment and settings.

Reference: [uv dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)
and [locking](https://docs.astral.sh/uv/concepts/projects/sync/).

## Package boundaries

- `data.py`: answer-free input schema, loading, preprocessing, validation, and grouped splits.
- `pipeline.py`: reproducible preparation, inspection, output manifests, and CLI.
- `models/`: encoder and shared candidate scorer; no generative vocabulary head.
- `baselines.py`: uniform, option-prior, and learned linear candidate baselines.
- `prepared.py`, `paths.py`: shared split loading, provenance checks and artifact compatibility.
- `tasks.py`: source/opponent adapters to the shared input interface, with explicit task questions.
- `training.py`: shared learning checks, holdout fitting, optimizers, inference and checkpoints.
- `cross_validation.py`: fixed-budget scenario-grouped evaluation and verified resume.
- `experiments.py`: training-only tiny checks and per-backbone matrix execution.
- `scripts/run_decision_matrix.py`: three-GPU queues and combined metric export.
- `evaluation.py`: reference-action probability metrics and deterministic tie handling.
- `diagnostics.py`: reliability/error diagnostics and conditional grouped intervals.
- `opponents/`: chronological Leduc data, causal features, forecasters and run artifacts.
- `holdem_history.py`, `holdem_integration.py`, `holdem_scaling.py`: causal Pluribus views, matched CPU model pairs, and fixed-validation training sizes.
- `holdem_ablations.py`: matched CUDA encoder/head and history comparisons, microbatch settings, recorded selections and training-seed repeats.
- `source_experiments.py`: separate observed-play and teacher-policy tasks.
- `demo_models.py`, `demo_results.py`: serving catalogue and checked result-table data.
- `apps/`: interactive presentation of package functionality.
- `configs/`: pointers to canonical settings; reviewed results live in `docs/results/`.
- `notebooks/`: exploration; reusable logic moves into the package.

## Notebooks in GitHub

Keep one maintained walkthrough per workflow. The data notebook shows raw rows,
decision examples, validation, metrics, and mistakes. Put
reusable preprocessing, splitting, training, and evaluation in the Python package
with tests; the notebook and command-line runner should call that same code.
The training runner and demo import the package directly.

The decision-transformer notebook calls `models/encoder.py` for joint tokenization, candidate pooling and
scoring, and `training.py` for training, saved-split loading, batched prediction,
and self-contained checkpoints. Install both `analysis` and `ml` extras for this
notebook and full mypy checks. Core-only mypy excludes the optional ML modules;
the separate ML CI job checks all modules and runs offline randomly initialized
BERT, DistilBERT, and ModernBERT tests without fetching pretrained weights.
The core-only command excludes `training.py`, `cross_validation.py`,
`experiments.py`, and `models/encoder.py`; the ML job checks all package modules.
Hold’em imports PyTorch lazily. Mypy permits that optional import when PyTorch is
absent and checks its installed types in the ML environment.

For executable notebook changes, validate the relevant workflow in its supported
environment, then clear outputs and execution counts. For prose-only changes,
validate notebook schema and code-cell preservation without restarting training.
Commit notebook source and supporting package changes together. Keep downloaded data, generated datasets, split manifests,
example-level predictions, and model files in the ignored output directories.
Publish reviewed aggregate metrics and provenance in `docs/results/`, explain
their interpretation in the study's results report, and link both from the
[results index](results/README.md). Preserve links to the
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
reliability diagnostics. History experiments cover simulated Leduc forecasts and
real Hold’em transformer inputs; selected Hold’em pairs have three training-seed
repeats. Saved local CPU latency measurements appear in the demo results table.
Further work includes independent evaluation, fitted calibration, representative
hosted latency/throughput, and measuring whether inferred opponents improve Hero's responses.
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
runs on a local CUDA GPU or Colab GPU for PokerBench and GPU ablations; the bounded
Hold’em integration/scaling workflows use CPU. Unit tests use tiny random encoders.
The [completed results](results/README.md) are separate empirical artifacts, not
unit-test scores. Preserve the recorded training revision and package versions
when resuming; current application changes can produce a new protocol hash.

## Opponent forecasting extension

`opponents/data.py`, `generate.py`, `models.py`, `evaluation.py` and `experiment.py`
own the opponent observer contract, optional OpenSpiel simulation, causal history,
CPU forecasters and reproducible selection/evaluation artifacts. The opponent-modeling notebook calls
these functions. Keep this observed-action task separate from PokerBench decision
labels. The analysis extra declares SciPy and threadpoolctl directly for the
masked objective and bounded CPU fitting. See [the experiment guide](opponent-experiments.md).

## Commands and checks

Run commands from the repository root with the indicated optional extras retained
on `uv run`. Each task exposes `--help` without launching training.

| Command after `python -m poker_decision` | Extras | Purpose / compute |
| --- | --- | --- |
| `prepare` | `analysis` | Pinned PokerBench preparation and CPU baselines |
| `train --backbone bert_mini` | `analysis`, `ml` | Introductory holdout training; CUDA |
| `evaluate --backbone bert_mini` | `analysis`, `ml` | Fixed scenario-fold comparison; CUDA |
| `opponents --help` | `analysis`, `experiments` | Chronological Leduc forecasts; CPU |
| `holdem STEP` | `analysis`, `ml`, `experiments` | Downloads, preparation, forecasts, checks, CPU comparisons and reports |
| `holdem-scale` | `analysis`, `ml`, `experiments` | Nested training sizes with fixed validation; CPU |
| `holdem-ablate STEP` | `analysis`, `ml`, `experiments` | Fixed-membership model/history comparisons; CUDA for fits |

For example, inspect the ablation options with:

```sh
uv run --locked --extra analysis --extra ml --extra experiments python -m poker_decision holdem-ablate --help
```

The [notebook guide](../notebooks/README.md), [PokerBench training guide](week2-training.md),
[Leduc guide](opponent-experiments.md), [Hold’em guide](meeting5-integration.md) and
[GPU guide](meeting5-ablations.md) describe preparation and artifact requirements.

Core CI installs no optional dependencies:

```sh
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy --exclude 'src/poker_decision/(training|cross_validation|experiments|models/encoder)\.py$'
```

Full offline checks include neural and simulator safeguards:

```sh
uv sync --locked --extra analysis --extra ml --extra experiments
uv run --locked --extra analysis --extra ml --extra experiments pytest
uv run --locked --extra analysis --extra ml --extra experiments mypy
uv lock --check
```

See the [test guide](../tests/README.md) for dependency-specific coverage. Experiment
test counts are historical verification records; they are not model-quality metrics.
