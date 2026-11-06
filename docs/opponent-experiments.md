# Meeting 4: canonical opponent forecasting

Use [notebook 05](../notebooks/05_opponent_history.ipynb) for the walkthrough. The
same code runs from `poker_decision.opponents`. This is a separate Leduc
**next observed opponent action** task. PokerBench remains the source for Hero's
reference-action imitation task; the deployed transformer is not retrained or
connected to these forecasts.

## Run the experiment

Install the existing optional simulator and analysis environment:

```sh
uv sync --locked --extra analysis --extra experiments
uv run --locked --extra analysis --extra experiments jupyter lab notebooks/05_opponent_history.ipynb
```

Generation and training use CPU. No API key, GPU, PokerBench download, pretrained
weights, or paid service is needed. Local and Colab runs are supported. In Colab,
the notebook bootstrap requires a repository revision containing this integration.
Notebook outputs stay cleared in Git; generated data/results stay in `artifacts/`.

The command-line workflow separates generation, selection, and test evaluation:

```sh
uv run --locked --extra analysis --extra experiments python -m poker_decision.opponents.experiment generate --seed 20261106
```

Use the printed data directory in the next command. `DATA_DIR` and `SELECTION_DIR`
below are placeholders for the printed absolute or repository-relative paths:

```sh
uv run --locked --extra analysis --extra experiments python -m poker_decision.opponents.experiment select --data DATA_DIR
uv run --locked --extra analysis --extra experiments python -m poker_decision.opponents.experiment evaluate --data DATA_DIR --selection SELECTION_DIR --final
uv run --locked --extra analysis --extra experiments python -m poker_decision.opponents.experiment audit --data DATA_DIR
```

`select` reads training/validation labels, never test labels. It chooses C from
0.1/1/10 on validation by default. `--selection-method grouped_cv` instead chooses
C in five folds of **training opponent instances**, keeping repeated sessions
together. In both modes, the primary summary window and Bayesian parameters use
validation; all predeclared windows are retained for test reporting. Fitted models
use training rows only. `evaluate` requires explicit `--final`. The notebook has
`RUN_FINAL=False` by default.

A new configuration, source hash, or recorded runtime version gets a new run path.
Incomplete paths fail explicitly. Cached datasets, selected numeric models, and
final prediction/report artifacts are checksum-checked. Existing final results are
reused after verification; changing evaluation settings does not silently overwrite
them. An exposed test set remains a development result if later choices use it.

## Data and visibility

Each record has `inputs`, a separate `target_index`, and outer session/opponent,
hand/decision order and split metadata. The target index is local to the legal menu.
The engine action categories are fold, check/call, and bet/raise. No illegal action
gets probability or is converted to another action.

`inputs.public` includes board rank, round, acting seat, pot/contributions, and the
ordered public betting prefix of the current hand. Optionally `observer_rank`
contains the observing player's **own** card. `--public-only` excludes that card
from every predictor and skips equity models. No opponent private card,
unrevealed board, outcome, identity, true policy, change point, or seed is a model
feature. Audit artifacts contain hidden replay information and must not be merged
into inputs.

The generator creates independent seeded opponent instances within known behavior
families. All repeated sessions for an instance belong to one 60/20/20 split,
stratified by family and switch direction **before dealing**. Default: 50 opponent
instances, 100 sessions, 60 hands/session. Seats alternate by hand, and history
resets at session boundaries. Held-out instances come from known families; this
is not an unseen-family evaluation.

Independent chance, observer and opponent RNG streams are recorded. The full
engine action trace allows every recorded decision, legal menu, and terminal
return to be replayed. The no-history baseline includes current-hand betting
information. Past target actions can become feedback only after they were observed;
that is online representation updating, not fitting weights on test labels.

## Retained simulator options

Pass `--policy-family NAME` to `generate`. These are canonical adaptations of the
submitted ideas, not byte-identical reproductions of the historical datasets.

| Family | Retained contribution |
| --- | --- |
| `weighted` (default) | Rank/pair-dependent log weights, tight/passive/aggressive/balanced, instance perturbations, switches in both directions |
| `categorical` | Card-independent passive/aggressive legal-action weights and switches from the simple modular prototype |
| `table` | Explicit strength-conditioned tight/aggressive probability tables and switching opponents |
| `residual` | Aggression/fold-style archetypes from the Bayesian notebook, with hidden private strength affecting behavior |
| `cfr` | Exported approximate CFR+ policy plus tight/loose distortions and two-way switches, adapted from the linked Colab |

`--observer-policy cfr` uses the same teacher as the observer across all sessions;
otherwise the observer is uniform. `--cfr-iterations` sets a fixed teacher budget
(default 500). The complete table and measured convergence curve are retained,
with C++ policy values copied before solver destruction. No convergence target or
strength claim is inferred from the iteration count. Teacher exploitability is
not predictor exploitability or evidence that the modified opponents are GTO.

## Retained prediction methods

| Method | Role |
| --- | --- |
| Uniform / smoothed prior | Legal-normalized simple controls; the prior uses training labels only |
| Naive Bayes, with/without history | The simple prototype's probabilistic token approach, with discretized numeric features and a train-only vocabulary |
| Masked logistic, with/without history | Primary matched comparison; standardization fitted on training rows; cross-entropy normalized over legal actions during training |
| Separate logistic per legal menu | Alternative from the smaller archive; unseen/sparse menus use a declared smoothed training-prior fallback |
| Binary candidate scorer + softmax | The linked Colab's distinct candidate-scoring objective, with equal positive/negative weight per decision; retained as a control alongside the conditional objective |
| Equity with/without history | Exact showdown equity versus uniformly random unknown physical cards; matched round/menu cues on both sides avoid a context-feature confound |
| Bayesian style filter | 25 hypotheses over aggression/folding offsets from the no-history forecast, with validation-selected temperature and per-hand reset hazard |

The main history comparison reports windows 4/16/64 and all prior decisions.
`FeatureSpec` explicitly controls timing, opportunity matching, and window order:

- `completed_hands` (default) uses only completed earlier hands in summaries and
  Bayesian updates. This avoids attributing within-hand hidden-card evidence to a
  persistent style. Current-hand betting is still available to the baseline.
- `all_prior` includes earlier observed opponent actions of the current hand.
- Primary summaries take the last W eligible decisions **before** matching round
  and exact legal menu. The menu-model alternative takes the last 10 matching-menu
  opportunities **after** matching and does not require matching round.
- Summaries retain counts, opportunity denominators, and +1-smoothed rates. Empty
  histories have zero counts and uniform legal rates. All-history and rolling
  variants are explicit; their numeric models are re-estimated on this contract.

The Bayesian filter predicts before feedback, resets by session, never receives
true style/switch metadata, and matches the declared history timing. The hazard
accounts for elapsed hands, including hands without an opponent decision. It is
Bayes inference conditional on a chosen 25-style likelihood model, not a validated
measurement of a person's psychological traits. A sequential neural encoder was
not provided by these submissions and remains future work.

## Evaluation and completion conditions

Report the same test decisions for every method: log loss, exact and tie-expected
accuracy, Brier score, decision/session/opponent denominators, session-macro loss,
all predeclared windows, history and switch slices, and five inspected predictions.
Accuracy uses the package's lexical action tie rule. The first example plus extreme
help/hurt examples and seeded samples are illustrative, not a representative sample.

Paired bootstrap intervals resample whole **opponents**, including all their
sessions, and condition on this simulation and these fitted models. Small groups
are flagged as descriptive. History-size and switch slices describe different
situations/stages; they do not establish that collecting more history causally
improves predictions. The default main comparison keeps model architecture,
current-state information, training population and held-out decisions matched.

Data is complete when deterministic generation, identity separation, legality and
full engine replay pass. Forecasting is complete when both matched baselines have
saved validation choices, valid held-out predictions, denominators, uncertainty,
and inspected successes/failures. A negative history result still completes the
experiment. Connecting the forecast to Hero decisions requires separate response
targets or repeated-play reward evaluation.

Run the safeguards with:

```sh
uv run --locked --extra analysis --extra experiments pytest tests/test_opponents.py --no-cov
```

See [reconciliation and historical results](meeting4-integration.md) for which
submitted results were reproduced and which remain source-only claims.
