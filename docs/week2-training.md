# Decision-model GPU comparison

This reproduces the PokerBench reference-action matrix. Hold’em feature studies
use a separate [GPU ablation protocol](meeting5-ablations.md); the
[results index](results/README.md) maps both to their recorded evidence.

The local runner executes the same scenario-grouped development evaluation as
[evaluation notebook](../notebooks/evaluation.ipynb). Four pinned pretrained encoders (`bert_mini`, `minilm`, `distilbert`,
`modernbert`) each get independent `linear` and `mlp` heads across five fixed folds:
40 full fits. All eight combinations completed successfully; see the
[complete accuracy/log-loss report](week2-results.md),
[CSV](results/week2-development.csv), and [provenance JSON](results/week2-development.json).
Three epochs per fit were fixed before seeing results, matching the
Week 2 main training budget; [evaluation notebook](../notebooks/evaluation.ipynb)'s preview defaults to one epoch.
Batch size 16, encoder LR 2e-5, head LR 1e-3, weight decay 0.01, seed 42 and
MLP width 64 stay identical. Compute and parameter counts differ across encoders.

Use the matching training revision and locked environment. Prepare data once
only if the saved manifest is absent, then run each backbone:

```sh
uv sync --locked --extra analysis --extra ml
# Optional, only if no prepared split exists:
uv run --locked --extra analysis python -m poker_decision prepare --skip-baseline
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m poker_decision.experiments --backbone bert_mini --output artifacts/week2_matrix
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m poker_decision.experiments --backbone minilm --output artifacts/week2_matrix
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m poker_decision.experiments --backbone distilbert --output artifacts/week2_matrix
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u -m poker_decision.experiments --backbone modernbert --output artifacts/week2_matrix
```

On three GPUs, queue MiniLM after BERT Mini on GPU 0; the other two may run
concurrently. Do not prepare the dataset concurrently. Existing verified data is
reused by the experiment runner. It accepts `--epochs` and `--output`; changing
the epoch budget creates a different recorded experiment.

Before full fits, each encoder/head must learn 32 distinct multi-option examples
from the first fold's training side, using the notebook's independent optimizer
and up to 100 epochs. At least 95% training accuracy is required. No held-out score
selects this diagnostic's stopping point. All full fits then start fresh.

Each backbone writes `preflight_protocol.json`, `folds.json`, `tiny-*.json`,
`result.json`, and `metrics.csv` in a named folder under `artifacts/week2_matrix`.
The existing CV runner saves hashed protocol folders containing fold histories,
held-out probabilities, portable checkpoints and reload checks, per-street metrics,
reliability diagnostics, error examples, and paired intervals against the linear
baseline. Completed folds can be resumed by rerunning the same command and code;
incomplete folds retrain. Tiny checks repeat on restart. The four published runs
all completed five folds and passed post-run artifact verification.

These are development results on the 10,000-example working pool. The original
Week 2 holdout overlaps related scenarios and selects epochs by validation loss;
it cannot supply an independent final estimate. Grouped folds prevent the specified
scenario/context overlap but do not establish all possible forms of independence.
A single seed cannot measure retraining variability. All eight combinations should
be reported, including failed fits. Choosing settings after viewing these scores
requires nested selection or a new held-out evaluation for a final estimate.

To run the three queues and collect `all_metrics.csv` automatically:

```sh
.venv/bin/python -u scripts/run_decision_matrix.py --output artifacts/week2_matrix
```

Per-backbone `.log` and `-status.json` files record progress and exit codes. A final
`matrix_status.json` distinguishes complete results from failed combinations.

## Pre-run verification (October 28, 2026)

Independent code inspection confirmed answer-free inputs, candidate masking,
raw-logit cross-entropy, fresh fold fits, fixed-budget held-out evaluation and
portable checkpoint reloads. The real-data audit validated 63,200 preflop and
500,000 postflop source rows with zero parsing failures. The default 10,000-row
pool reproduced five folds with zero shared scenario groups or exact contexts.
The historical holdout reproduced 515 shared scenario groups and 91.8% of
validation rows having a training scenario group, supporting use of grouped CV.

Before launching pretrained training: 92 offline tests passed; three optional
Pluribus/Leduc tests skipped because their optional libraries were not installed.
Ruff lint/format, package mypy, and actual computations on all three RTX 3090s
passed. These checks establish executable safeguards, not proof of all scientific
assumptions. Full pretrained results are separate artifacts, not offline test scores.

## Historical revision, completed outputs, and safe verification

All recorded training-package source hashes match commit `d7f0e23`. To reproduce
or resume those runs, use a dedicated checkout at that revision with its locked
ML/analysis environment and the original data, settings, device, and output path.
Later package additions, even for the demo, change the protocol fingerprint and
create a new run. The current branch is suitable for new experiments, but is not
a substitute for the original protocol when reusing historical fold artifacts.

The runner performs training when matching completed outputs are absent; invoking
it is not a read-only result check. Inspect saved reports and compare their hashes,
coverage, and recomputed metrics directly when verifying an old experiment under
a newer checkout. Do not relax resume checks to force acceptance of changed code.

The completed local matrix has per-backbone `result.json`/`metrics.csv` files, a
combined `all_metrics.csv`, `completion_summary.json`, and `verification.json`.
The original multi-job launcher was interrupted and individual runs were resumed;
its final `matrix_status.json` is absent for this historical run. Final completion
is established by all four result files, five verified fold reports per backbone,
and 10,000 held-out predictions per model, not by the interrupted launcher's exit.

Run long jobs in a host-managed persistent session and keep artifacts on durable
storage. A background user service can survive chat interruptions but may stop at
logout or host restart. Check the process/service and saved fold reports rather
than assuming a queued job is still running. An incomplete fold retrains on resume.

Aggregate metrics/provenance are versioned in `docs/results/`; raw datasets,
example-level predictions, and model weights remain outside Git history. These
results cover PokerBench reference-action agreement only. Pluribus replay and
Leduc teacher-policy generation remain separate optional experiments.

Current code also provides `python -m poker_decision evaluate --backbone NAME`.
Its task-based default output is `artifacts/decision_matrix`; specify `--output`
to choose a run location. The commands above retain the historical matrix path
for comparison. Code changes create new protocol hashes; reproduce or resume the
recorded experiment at its recorded source revision. See
[artifact compatibility](architecture.md#existing-artifacts-and-reproducibility).
