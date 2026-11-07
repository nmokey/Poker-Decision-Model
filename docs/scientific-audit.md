# Scientific integrity audit

Initial review: October 28, 2026. Transformer completion: November 2, 2026.
Opponent-forecasting update: November 6, 2026.

## Scope and conclusion

The maintained Meeting 2 notebook and its late-submission extensions were reviewed
for input/label separation, related-example isolation, fit/evaluation boundaries,
checkpoint selection, reproducibility, metric semantics, and source-specific claims.
The implementation has executable safeguards; this is not proof that all dataset
biases, label errors, or forms of dependence have been eliminated.

The full pretrained-model matrix has now completed on three local RTX 3090 GPUs:
four pinned encoders × two heads × five folds, with three fixed epochs per fit.
All eight models have exactly 10,000 held-out predictions. Model fitting followed
training-only tiny-set checks; full folds started fresh. Saved checkpoint hashes,
prediction hashes, fold metrics, pooled metrics, diagnostics, and coverage were
verified after completion. See [all accuracy and log-loss scores](week2-results.md)
and the [recorded settings and provenance](results/week2-development.json).

ModernBERT + linear has the highest observed accuracy (71.92%) and lowest log loss
(0.6078), versus 69.69% / 1.0025 for the linear candidate baseline. DistilBERT +
linear is close at 71.78% / 0.6164; a single seed does not establish a reliable
advantage for ModernBERT. This is development CV on the inspected working pool.
Notebook training remains opt-in and requires a local CUDA GPU or Colab GPU;
offline model tests use tiny randomly initialized encoders. Historical holdout
scores below and in the integration notes remain distinct from these CV results.

## Findings and fixes

### The original holdout measures a narrower kind of generalization

On the original 8,000/2,000 saved partitions, exact-context overlap is zero. Under
the new answer-free scenario definition, 515 groups are shared; **91.8% of validation
rows have a scenario group present in training**. That is not proof of answer leakage.
It means an exact-context holdout is not an unseen-scenario benchmark.

The stress-test grouping uses the full preflop betting history, and the sorted
ranks of the first three board cards for postflop examples, including suit variants
and later streets. Five deterministic grouped folds have these sizes:

| Fold | Training rows | Held-out rows | Shared scenario groups | Shared exact contexts |
| --- | ---: | ---: | ---: | ---: |
| 1 | 8,070 | 1,930 | 0 | 0 |
| 2 | 7,889 | 2,111 | 0 | 0 |
| 3 | 8,067 | 1,933 | 0 | 0 |
| 4 | 7,911 | 2,089 | 0 | 0 |
| 5 | 8,063 | 1,937 | 0 | 0 |

Grouping never reads targets. Approximate street/action-label stratification is
used only for fold allocation, not as a predictive feature. All duplicates and
conflicting labels with identical context stay together. Every source row is held
out exactly once. This grouping is deliberately broad; it is not a claim that all
hands, opponent sessions, or semantically related situations are independent.

### Content checks now protect saved targets as well as membership

Previously, saved membership checks did not reject every edit to a context,
candidate menu, or label if IDs stayed unchanged. New preparation writes hashes
of complete train/validation JSONL files and writes the manifest last. Loading
checks those hashes, pinned source provenance, IDs, and context separation.
Legacy manifests without file hashes are compared against the selected rows of
checksum-verified cached raw CSVs. Missing or inconsistent provenance stops loading;
it does not silently create a different split. These hashes detect changed files,
not intentional coordinated changes to both data and provenance.

### Cross-validation fits are isolated and fixed-budget

`evaluation.ipynb` saves protocol and memberships before fitting. Each
fold refits the option prior and linear baseline, then initializes a fresh pinned
encoder, head, and optimizer for each selected head architecture. Only fold-training
inputs and labels reach fitting. Tokenization is stateless and label-free.

Outer-fold predictions are computed after the fixed epoch budget; they cannot
select epochs, calibration, or an early-stopping checkpoint in this runner. Linear
and nonlinear heads use the same pool, folds, backbone and training settings. They
are comparisons of alternative heads, not claims that complexity necessarily helps.
The joint all-options architecture remains distinct from the source notebook's
context/candidate pair architecture.

Resumption now requires matching protocol, source membership, prediction hashes,
checkpoint-file hashes, model coverage and the recorded reload check. Metrics are
recomputed from held-out probabilities before accepting saved reports. Exactly one
held-out prediction is required per source row/model. Pooled log loss is checked
against the row-weighted fold losses, including unequal fold sizes.

### Evaluation is not training or calibration

The original Meeting 2 walkthrough selects a checkpoint using validation log loss;
its reported validation performance is a development estimate, not a final test.
The current CV pool has also been inspected previously. Choosing the best head or
other settings from these results and quoting the same scores as unbiased final
performance would be inappropriate. Use inner folds or a new held-out evaluation
for subsequent tuning. Some team notebooks have already inspected official-test
examples; do not describe that test as pristine for the whole team's history.

Inference uses eval mode and disables gradient tracking. Tests verify that it
neither changes weights nor accumulates gradients. Candidate padding receives
zero probability; cross-entropy receives raw logits. Changing metadata/labels does
not change encoded inputs. Existing parsing tests exclude future board cards,
reference answers, and arbitrary extra source fields.

### Statistics have explicit meanings and limits

Reliability bins and ECE concern agreement with the reference labels, not a
solver's mixed strategy. Binning is fixed in advance and counts are reported.
Confusion counts distinguish action-type errors from same-type amount errors.
Paired bootstrap comparisons resample whole specified groups and use the exact
same examples/predictions for both models. Their percentile bounds condition on
these fitted models and groups; they do not include retraining uncertainty or
establish independence of all scenarios. Fold SD is descriptive, not a confidence
interval. Neither statistic justifies a general superiority claim by itself.

Menu/target association is computed using only supplied training contexts and a
seeded representative per duplicate context. Within-street shuffled-label values
are an exploratory reference: related-scenario dependence limits formal p-value
interpretation. Legal-action restrictions can naturally make menus predictive;
association is not, by itself, evidence of leakage or dataset construction errors.

### Alternative sources remain separate tasks

`data_sources.ipynb` has opt-in source experiments:

- Pluribus: strict PokerKit replay, acting-player cards and prior public information
  only; menus are built before the observed label is read. Exact sizing is
  abstracted. Replay failures are saved and block modeling exports. Shared hands
  or identical contexts are joined into connected split components. Player/session
  dependence and longitudinal opponent-feature construction still need investigation.
- Leduc: acting-player information-state inputs, approximate CFR+ teacher
  distributions, argmax labels and sampled actions saved separately. Exploitability
  measures the teacher only. No learned neural student's exploitability is claimed,
  and no student train/test split is silently invented.

Hidden-card mutation controls check both sources. Public data availability and a
publisher license do not independently establish every upstream redistribution
right. Full archive replay and full solver-generation runs are not verified by the
small offline fixtures used here.

## Executable checks

At the initial October 28 audit: **93 tests passed, 89% overall statement coverage**,
with Ruff lint/format checks, full package mypy, and offline lockfile consistency checks passing.
CV runner coverage is 91%, diagnostics 96%, and saved-split/training helpers 91%.
Coverage measures executed code, not scientific correctness or proof of no leakage.
These counts describe that audit environment, not a permanent suite total. The
pretraining checks later passed 92 tests with three optional-source skips; the
[tests guide](../tests/README.md) explains dependency-specific checks. The GPU runner
is excluded only from core-only typing; the ML CI job checks the full package.

Tests cover:

- Label/future-card exclusion and exact candidate normalization.
- Scenario-key independence from targets, action menus, suits and later streets.
- Zero cross-fold group/context/ID overlap, complete coverage, deterministic
  memberships, and conflicting duplicates kept together.
- Spy-verified training memberships for both baselines and neural training; fresh
  models and empty-state optimizers for every fold/head.
- Known reliability/error counts, probability metrics, whole-cluster resampling,
  paired differences and seeded menu-association diagnostics.
- Unchanged weights/gradients during evaluation; correct masking and gradient flow.
- Linear/MLP checkpoint reloads; rejection of corrupted split content, predictions,
  stored metrics and checkpoint weights.
- Hidden-card controls, strict replay failure accounting, connected hand/context
  splits, and separate Leduc teacher target semantics.
- Notebook schema, syntax, cleared outputs, training-disabled defaults and CUDA training guards.

At the initial audit, the scenario and source notebooks also executed from fresh
local Jupyter kernels with their default training/download/solver flags disabled. The actual
10,000-row grouping/provenance audit completed without training a model.

## What remains an empirical question

Reference-action agreement does not establish profitable or unexploitable poker.
Single labels do not recover full mixed strategies. Tests cannot prove that the
reference action is optimal, menus exhaust legal actions, or all dataset shortcuts
have disappeared. The grouped protocol has now completed. The next empirical
steps are retraining across seeds, measuring serving tradeoffs, and defining an
independent evaluation or nested grouped selection before claiming a model winner.


### Encoder swaps and development sweeps

All four requested encoder choices are pinned, including ModernBERT-base at
`8949b909ec900327062f0ebf497f51aef5e6f0c8`. The main notebook centralizes optimizer,
head, training and tiny-check settings and records them in metadata. ModernBERT
loads through SDPA with optional reference compilation disabled. Offline tests
exercise actual tiny BERT, DistilBERT and ModernBERT forward/backward and checkpoint
APIs for both head types, verify matching tokenizer/encoder revisions, and check
that the displayed learning rates and decay reach AdamW. Full pretrained results
are reported separately in the [completed experiment report](week2-results.md).

The optional the evaluation notebook grid validates settings, prints fit count, preserves fold
membership/seed, and saves every trial through the existing protocol runner.
It explicitly produces development comparisons, not nested model selection.
Winning sweep scores cannot be reported as independent final test performance.


## Opponent forecasting safeguards and evidence

The opponent-modeling notebook and `poker_decision.opponents` add a separate controlled Leduc
next-observed-action task. All repeated sessions for an opponent stay in one
split and optional training CV group; intervals resample whole opponents. Labels,
true profiles, switch times, hidden opponent cards, outcomes and future actions
are excluded from predictive features. Causal summaries are recorded before
feedback and default to completed earlier hands. Bayesian updates use the same
feedback timing; an `all_prior` mode permits earlier actions of the current hand.

The canonical run replayed all 6,000 hands and 12,711 decisions. Tests exercise
current/future-target mutation, session isolation, hidden-card invariance,
hand-worked opportunity denominators, empty history, exact equity against engine
returns, legal normalization, numeric reloads, grouped selection and corrupt-cache
rejection. Validation or grouped training selects regularization as declared;
validation selects the primary history window and Bayesian settings. Explicit
final scoring preserves all predeclared windows, counts, examples and switch slices.

[Forecast results](meeting4-results.md) concern one simulated population: history
improves overall prediction but can hurt immediately after a switch. Small slices
are descriptive. The [submission reconciliation](meeting4-integration.md) separates
independently reproduced results from missing-source claims and records an
unresolved byte-regeneration mismatch in one submitted archive. It does not
combine incomparable source scores into a leaderboard.

No learned sequence encoder, human tilt detector, fusion with the existing Hero
policy or improvement in poker payoff was established by this forecasting result.
Any subsequent tuning on these exposed test results needs a new evaluation design.
