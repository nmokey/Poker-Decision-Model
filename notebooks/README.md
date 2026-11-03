# Notebooks

Start with [Meeting 1: PokerBench pipeline](01_pokerbench_pipeline.ipynb), consolidated
from the team's eight unique Colab notebooks. It audits the preflop and postflop
training CSVs, constructs validated decision examples, makes a fixed-seed context
group split, and evaluates three exact-candidate CPU baselines. No GPU or API key.

From the repository root:

```sh
uv sync --locked --extra analysis
uv run --extra analysis jupyter lab notebooks/01_pokerbench_pipeline.ipynb
```

Select the project environment, restart the kernel, and run all cells. Defaults:
5,000 working examples per source, seed 42, 20% of distinct contexts for validation.
All source rows are validated before sampling. Source revisions and hashes are
pinned. Postflop contexts exclude future board cards. Any parse failure is saved
and stops preparation. The official test set is never loaded.

Reusable logic lives in `src/poker_decision/`, including a matching command-line
workflow: `uv run --extra analysis python -m poker_decision.pipeline`.
Generated data, split manifests, metrics, and five errors stay in ignored
`data/processed/pokerbench_meeting1/` and `artifacts/pokerbench_meeting1/`.

See [integration notes](../docs/meeting1-integration.md) for source contributions and
limitations. Keep notebook outputs cleared in Git and import the installed package
rather than modifying Python's import path.

## Meeting 2

Continue with [Meeting 2: single-pass decision model](02_pokerbench_decision_model.ipynb),
consolidated from five shared Colabs and the local `week2.ipynb` submission.

```sh
uv sync --locked --extra analysis --extra ml
uv run --locked --extra analysis --extra ml jupyter lab notebooks/02_pokerbench_decision_model.ipynb
```

The notebook explains the input adapter, joint serialization, candidate token spans,
mean pooling, shared head, candidate padding, and raw-logit cross-entropy. It requires
a tiny training-set learning check, resets the model for real training, selects a
checkpoint by validation log loss, reloads it, compares all baselines, shows five
mistakes, and measures candidate-order sensitivity. Encoder revisions are pinned.

Use a local CUDA GPU or Colab GPU runtime for pretrained training. CPU environments
support inspection and offline tests. The Colab bootstrap clones the repository
only if needed; the maintained code is published on `main`. Use the
recorded historical training revision to reproduce the completed matrix. Set
`POKER_SMOKE=1` before launching Jupyter; this selects 128 train / 64 validation rows
for quick verification. All baselines are refitted on that selected training subset. Normal runs reuse
the complete saved Meeting 1 membership. If absent, preparation uses Meeting 1 defaults.

Generated checkpoints, split copies, training history, comparisons, probabilities,
and errors live under `artifacts/pokerbench_meeting2/<backbone>-<timestamp>/`.
No official test data is loaded and no tokenization failures are silently skipped.
See [Meeting 2 integration notes](../docs/meeting2-integration.md).

## Scientific evaluation extensions

- [Scenario evaluation](03_scenario_evaluation.ipynb): inspect historical scenario
  overlap; save five fixed grouped folds; refit all models; compare linear/MLP heads
  under identical settings. Training is disabled by default and guarded for a CUDA GPU.
  Completed folds are resumed only after checking protocol, prediction and weight
  hashes and recomputing metrics. Use persistent output storage or download before
  Colab ends. Resuming historical outputs requires their original code and versions.
- [Alternative data sources](04_alternative_data_sources.ipynb): opt-in PHH/Pluribus
  replay and Leduc mixed-policy target generation, requiring `--extra experiments`.
  Neither silently changes PokerBench inputs, labels or memberships.

The main Meeting 2 notebook now supports `HEAD_TYPE="linear"` or `"mlp"`, with the
choice persisted in portable checkpoints (old linear checkpoints remain readable).
Its added diagnostics characterize reliability and uncertainty without fitting a
calibrator on evaluation rows. See [scientific audit](../docs/scientific-audit.md).

## Swapping encoders and sweeping settings

Notebook 02 has one configuration block for `BACKBONE`, head type/width, encoder
and head learning rates, weight decay, epochs, batch size, token limit, seed, and
separate tiny-check settings. Choose `bert_mini`, `minilm`, `distilbert`, or
`modernbert`; tokenizer and encoder share a pinned revision. Restart the runtime
and run all cells after edits. ModernBERT uses PyTorch SDPA without optional
compilation. Architecture compatibility is checked offline with tiny BERT,
DistilBERT, and ModernBERT models; this does not establish pretrained-model quality.

Notebook 03 exposes every `CVConfig` setting and an optional bounded `GRID`.
`sweep_configs` checks settings before fitting, keeps fold seed/count fixed, and
rejects duplicate or oversized grids. The preview prints trial and encoder-fit
counts; `RUN_SWEEP=False` initially. Four encoders with two heads over five folds
cost 40 encoder fits. Start with one head or a smaller grid. Each trial has its own
hashed protocol and saved configuration. Use persistent storage for long Colab runs.

Sweeps are exploratory development comparisons. A winner selected from those
scores needs an untouched final test or nested grouped selection for an independent
estimate. The sweep does not implement nested CV or automatically select a winner.

For the complete GPU matrix outside Jupyter, see [Week 2 training](../docs/week2-training.md).

## Completed matrix and interpretation

All eight combinations of BERT Mini, MiniLM, DistilBERT, and ModernBERT with linear
or MLP heads have completed five grouped folds, three epochs per fit: 40 full fits.
The [complete table](../docs/week2-results.md) includes accuracy and log loss, with
[CSV](../docs/results/week2-development.csv) and
[JSON provenance](../docs/results/week2-development.json). ModernBERT + linear
has the highest observed scores (71.92%, 0.6078), close to DistilBERT + linear
(71.78%, 0.6164). The linear candidate baseline scores 69.69%, 1.0025.

These results come from the fixed-budget grouped protocol, not notebook 02's
checkpoint-selected holdout. Notebook 03's preview still defaults to one epoch;
the published matrix fixed three epochs before training. Changing preview settings
does not alter the recorded results. Notebook outputs remain cleared in Git;
pretrained experiment results are published as reviewed documentation artifacts.
