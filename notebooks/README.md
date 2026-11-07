# Notebook guide

The repository maintains one executable walkthrough per workflow. Reusable logic
lives in the Python package, so notebook and command-line runs use the same code.
Names describe tasks rather than meetings. Run only the workflow you need;
notebooks exchange saved, audited artifacts through the package.

| Workflow | Notebook | Optional extras | Compute |
| --- | --- | --- | --- |
| Source audit and CPU baselines | [PokerBench pipeline](data_pipeline.ipynb) | `analysis` | CPU |
| Joint state/question/action scoring | [Decision transformer](decision_model.ipynb) | `analysis`, `ml` | CUDA GPU for pretrained training |
| Scenario folds and controlled head comparison | [Scenario evaluation](evaluation.ipynb) | `analysis`, `ml` | CUDA GPU for enabled pretrained fits |
| Observed-play replay and solver-policy targets | [Alternative sources](data_sources.ipynb) | `analysis`, `experiments` | Follow the notebook's source/runtime guards |
| Chronological opponent histories and forecasts | [Opponent forecasting](opponent_modeling.ipynb) | `analysis`, `experiments` | CPU, local or Colab |

## Setup

Run from the repository root, using its environment as the notebook kernel:

```sh
uv sync --locked --extra analysis --extra experiments
uv run --locked --extra analysis --extra experiments jupyter lab
```

Add `--extra ml` to both commands for neural-model work. Keep extras on `uv run`
to retain the dependencies. The neural notebooks support local CUDA or Colab GPU;
the forecasting notebook supports CPU. Colab bootstraps require a checkout
containing the corresponding package modules. No hosted-model key is needed for
these research notebooks.

## PokerBench pipeline and decision scoring

The pipeline audits checksum-pinned training CSVs before selecting its default
5,000 examples per source. It saves source provenance, validated context/menus,
context-grouped membership, baseline metrics and errors. Targets remain separate
from `DecisionInputs`; future board cards are excluded. The official test is never
loaded. [Data integration details](../docs/meeting1-integration.md).

The transformer walkthrough jointly encodes the context, question and all action
options, pools candidate token spans, and applies a shared linear or MLP head.
It calls the same `learning_check` and `fit_holdout` routines as the CLI, then checks checkpoint reload, baseline
comparison and candidate-order probes. `BACKBONE` and `HEAD_TYPE` select pinned
encoders/heads. BERT Mini is the teaching default; the live demo defaults to
ModernBERT linear. `POKER_SMOKE=1` reduces the introductory data sizes to 128/64;
pretrained training still requires the supported GPU runtime.

The introductory holdout chooses checkpoints on validation loss and has related
scenario overlap. Scenario evaluation instead uses five fixed scenario folds,
fresh model fits and a fixed epoch budget. `RUN_CV=False` and `RUN_SWEEP=False`
keep expensive fits disabled until configured. The bounded grid is development
selection, not nested CV or an unbiased final estimate.

All four encoders and both heads have completed 40 grouped fits on the same 10,000
examples. [Recorded results](../docs/week2-results.md) and
[training/reproduction instructions](../docs/week2-training.md) preserve that
specific protocol. Resuming requires matching source/settings and checked
checkpoints/predictions; changes can create a new run path.

## Separate source and opponent tasks

Alternative sources preserve their own label meanings: Pluribus labels are
observed action categories; Leduc solver labels include teacher distributions,
argmax actions and sampled actions. Teacher exploitability is not a measurement of
a learned student's strategy. Neither source silently replaces PokerBench rows.

Opponent forecasting generates repeated-player sessions and predicts the next
observed legal action from the observer's information. Five policy families,
priors, Naive Bayes, logistic variants, equity controls, history windows and a
Bayesian filter share one contract. Whole opponent identities stay in a split;
histories are built before the current target is consumed. The default uses
completed earlier hands, and an alternative permits earlier current-hand actions.

`RUN_FINAL=False` leaves test scoring explicit after reviewing saved validation
choices. C can instead use grouped training CV; history-window and Bayesian choices
remain on validation. Report all predeclared windows, denominators, switch slices
and paired opponent-cluster uncertainty. [Run guide](../docs/opponent-experiments.md),
[recorded results](../docs/meeting4-results.md), and
[contribution map](../docs/meeting4-integration.md).

## Artifact conventions

Keep notebook outputs and execution counts cleared in Git. Validate executable
changes in the supported environment; prose-only edits require schema/code-cell
checks without repeating training. Source data, generated sessions, split files,
per-example predictions and weights stay in ignored `data/` or `artifacts/` paths.
Reviewed aggregate evidence is versioned under `docs/results/`.

See the [development guide](../docs/development.md),
[scientific audit](../docs/scientific-audit.md), and
[documentation index](../docs/README.md). Forecasts currently run separately from
Hero's decision model; improved prediction alone does not establish better play.

## Connections and shared code

[Project architecture](../docs/architecture.md) maps every notebook to its package
components and saved artifacts. Setup is shared in `scripts/notebook_setup.py`.
Data preparation, training and evaluation all load the same audited split through
`prepared.load_prepared`; none requires another notebook's kernel variables.

The source and opponent walkthroughs demonstrate `tasks.source_task` and
`tasks.opponent_task`, which expose their records through the decision scorer's
input/target interface. Each task carries its own question, label meaning and
external grouping. Pass `question=task.question` when encoding another task.
This connection enables model reuse; no neural opponent-training result is claimed.

New outputs use `data/processed/pokerbench`, `artifacts/baselines`,
`artifacts/decision_model`, `artifacts/scenario_evaluation`,
`artifacts/decision_matrix`, `artifacts/alternative_sources`, and `artifacts/opponents`.
The package still reads an existing `data/processed/pokerbench_meeting1` dataset
when the canonical directory is absent, with the same provenance checks.
