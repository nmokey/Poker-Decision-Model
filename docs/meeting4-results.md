# Meeting 4 canonical development result

Executed November 6, 2026 with the maintained
[notebook 05](../notebooks/05_opponent_history.ipynb), including explicit final
scoring. These are a **new one-seed controlled experiment**, separate from the
[original submitted results](meeting4-integration.md). All models below receive
the same test decisions; no test score selected settings or the reported history
window. Future tuning based on this inspected run requires new evaluation data.

Recorded numerical implementation: commit `130cb81367a8d053f417a4871160644ad605193e`.
Later optional-import typing annotations do not change its runtime behavior.

## Protocol

- Seed 20261106, weighted card-dependent opponent family; tight, passive,
  aggressive, balanced and switching profiles.
- 50 opponent instances, two sessions each, 60 hands/session. All repeated
  sessions stay in the same split. Seats alternate by hand; history resets per
  session. Switch points are seeded and hidden from predictors.
- Uniform observer policy; predictors see the observer's own private rank,
  public board, contributions/pot, legal menu and current-hand public prefix.
  Opponent cards and audit metadata are excluded.
- History and Bayesian feedback use completed earlier hands. Opportunity
  summaries match betting round and exact legal menu. All models are fitted on
  training rows; C and the primary history window use validation log loss.
- 12,711 opponent decisions across 6,000 hands. Full engine replay verified every
  recorded decision and hand. Test: 2,506 decisions, 20 sessions, 10 opponents.
- Python 3.12.3, NumPy 2.5.3, OpenSpiel 2.0.2, scikit-learn 1.9.1, SciPy 1.18.1.
  [Aggregate protocol and results](results/meeting4-canonical.json) include the
  implementation hash, dataset-manifest hash, all validation candidates and slices.

## Held-out results

| Model | Accuracy | Log loss | Role |
| --- | ---: | ---: | --- |
| Uniform over legal actions | 38.43% | 0.8231 | Deterministic lexical tie rule; tie-expected accuracy is reported separately |
| Smoothed training prior | 55.23% | 0.8404 | Training-only legal-normalized control |
| Masked logistic, no history | 59.26% | 0.7741 | Main current-situation baseline |
| Same model + window 4 | 63.77% | 0.7251 | Predeclared history comparison |
| Same model + window 16 | 69.31% | 0.6606 | Predeclared history comparison |
| Same model + window 64 | 69.87% | 0.6373 | Validation-selected primary history model |
| Same model + all history | 68.95% | 0.6435 | Predeclared history comparison |
| Binary candidate scorer, no history | 59.26% | 0.8048 | Alternative objective |
| Binary candidate scorer + history | 70.43% | 0.7142 | Alternative objective; window 20 |
| Per-menu logistic, no history | 58.90% | 0.7736 | Alternative architecture |
| Per-menu logistic + history | 71.67% | 0.6298 | Latest 10 comparable menu opportunities |
| Naive Bayes, no history | 43.81% | 1.2253 | Simple token-feature control |
| Naive Bayes + history | 58.30% | 2.0995 | More accurate but substantially worse probability quality |
| Equity + opportunity cues | 55.23% | 0.7945 | Reduced representation |
| Equity + opportunity cues + history | 68.52% | 0.6618 | Matched reduced-representation ablation |
| Bayesian filter on the main no-history model | 73.34% | 0.6157 | Temperature 1.3, per-hand hazard 0.005, selected on validation |

The selected summary improves accuracy by **10.61 percentage points** and reduces
log loss by **0.1368**. Paired whole-opponent bootstrap (1,000 resamples):
history-minus-no-history log-loss interval **[-0.1928, -0.0720]** and accuracy
interval **[+3.42, +18.97] percentage points**. These intervals condition on the
fitted models and this simulated population; they do not measure retraining
variability or sensitivity to the generator design.

The Bayesian model has the best observed test score among these configurations.
This is a descriptive comparison on one simulated population, not an independent
selection of a generally best model. The Naive Bayes results also illustrate why
accuracy alone is insufficient: its history variant improves top-action agreement
while becoming more confidently wrong on some decisions.

## Behavior changes remain difficult

In the first five hands after a switch, only **40 decisions from four sessions
and two opponent instances** are available. No-history loss is **0.7410**, the
selected history model is **1.1390**, and the Bayesian model is **1.0210**. This
slice is explicitly descriptive. Both history methods can lag after a switch,
even though they improve overall.

This supports a tractable next experiment about memory and changing policies.
It does not establish human tilt detection, GTO play, or better Hero actions.
No existing PokerBench checkpoint was modified. Connecting forecasts to the
Hero policy still needs response labels or repeated-play reward evaluation.
