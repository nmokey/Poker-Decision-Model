# Week 2 GPU matrix

The local runner executes the same scenario-grouped development evaluation as
notebook 03. Four pinned pretrained encoders (`bert_mini`, `minilm`, `distilbert`,
`modernbert`) each get independent `linear` and `mlp` heads across five fixed folds:
40 full fits. Three epochs per fit are fixed before seeing results, matching the
Week 2 main training budget; notebook 03's preview defaults to one epoch.
Batch size 16, encoder LR 2e-5, head LR 1e-3, weight decay 0.01, seed 42 and
MLP width 64 stay identical. Compute and parameter counts differ across encoders.

Install the locked environment, prepare data once, then run each backbone:

```sh
uv sync --locked --extra analysis --extra ml
uv run --locked --extra analysis --extra ml python -m poker_decision.pipeline --skip-baseline
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m poker_decision.experiments --backbone bert_mini
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m poker_decision.experiments --backbone minilm
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m poker_decision.experiments --backbone distilbert
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u -m poker_decision.experiments --backbone modernbert
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
incomplete folds retrain. Tiny checks repeat on restart.

These are development results on the 10,000-example working pool. The original
Week 2 holdout overlaps related scenarios and selects epochs by validation loss;
it cannot supply an independent final estimate. Grouped folds prevent the specified
scenario/context overlap but do not establish all possible forms of independence.
A single seed cannot measure retraining variability. All eight combinations should
be reported, including failed fits. Choosing settings after viewing these scores
requires nested selection or a new held-out evaluation for a final estimate.

To run the three queues and collect `all_metrics.csv` automatically:

```sh
.venv/bin/python -u scripts/run_week2_matrix.py
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
