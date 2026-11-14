# Experiment configuration locations

Each study has its own protocol. Use the [results index](../docs/results/README.md)
to match settings to the reported target, split and source revision.

## PokerBench encoder/head comparison

The completed transformer-comparison settings are declared by `CVConfig` in
[`cross_validation.py`](../src/poker_decision/cross_validation.py) and the fixed
matrix runner [`experiments.py`](../src/poker_decision/experiments.py). The decision-model notebook
exposes its holdout settings; the evaluation notebook exposes grouped-CV settings and grid
previews. Avoid maintaining a competing configuration file here.

The completed matrix used four pinned encoders, linear/MLP heads, five folds,
seed 42, three fixed epochs, batch size 16, maximum length 512, encoder LR 2e-5,
head LR 1e-3, weight decay 0.01, and MLP width 64. Model and tokenizer revisions,
source hashes, dependency versions, data fingerprints, and protocol hashes are
recorded in the [provenance JSON](../docs/results/week2-development.json).

See [training instructions](../docs/week2-training.md) for reproduction and verified
resume, and [all accuracy/log-loss results](../docs/week2-results.md) for the
completed comparison. Raw run artifacts and model files remain ignored. Never
store credentials in experiment settings.

## Hold’em history integration and ablations

The introductory CPU integration and training-size workflows expose sample sizes
and output locations through their CLI. They retain the published BERT Mini linear
checkpoint and fixed optimizer settings. See the [Hold’em guide](../docs/meeting5-integration.md).

Hold’em GPU ablations use [holdem_ablations.json](holdem_ablations.json), a
separate fixed-membership follow-up to the Meeting 5 integration. It configures
the encoder/head pairs, history windows, training epochs/seeds and GPU microbatches.
The declared encoder/head, window and training-budget comparisons are a staged
18-fit sequence, not a Cartesian product of every JSON list. The completed study
adds four separately recorded three-epoch replication fits. Sampling remains at
seed 42; `training_seeds` controls model fitting, with effective batch 16 and
microbatch 2. Changing settings creates a new protocol rather than resuming an
old fit. See [the ablation protocol](../docs/meeting5-ablations.md) and
[all 22 results](../docs/meeting5-ablation-results.md) for selection and limits.

## Leduc opponent forecasting

Opponent generation uses `Config` and history features use `FeatureSpec` in
[`opponents/data.py`](../src/poker_decision/opponents/data.py). Selection/evaluation
settings live in [`opponents/experiment.py`](../src/poker_decision/opponents/experiment.py)
and the opponent-modeling notebook. Data generation and predictor settings are saved with source,
software and input hashes; model selection never uses final-test scores.
See the [forecasting guide](../docs/opponent-experiments.md) for policy families,
visibility, history timing/windows and validation/grouped-training choices.
