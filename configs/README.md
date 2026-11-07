# Experiment configuration locations

The completed transformer-comparison settings are declared by `CVConfig` in
[`cross_validation.py`](../src/poker_decision/cross_validation.py) and the fixed
matrix runner [`experiments.py`](../src/poker_decision/experiments.py). Notebook 02
exposes its holdout settings; notebook 03 exposes grouped-CV settings and grid
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


Opponent generation uses `Config` and history features use `FeatureSpec` in
[`opponents/data.py`](../src/poker_decision/opponents/data.py). Selection/evaluation
settings live in [`opponents/experiment.py`](../src/poker_decision/opponents/experiment.py)
and notebook 05. Data generation and predictor settings are saved with source,
software and input hashes; model selection never uses final-test scores.
See the [forecasting guide](../docs/opponent-experiments.md) for policy families,
visibility, history timing/windows and validation/grouped-training choices.
