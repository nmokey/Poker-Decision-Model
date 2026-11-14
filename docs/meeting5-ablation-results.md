# Meeting 5: history ablation results

Completed November 13, 2026 on temescal's RTX 3090 GPUs 1 and 2. All 22 fits
completed and their saved predictions were independently rescored. The
[aggregate evidence](results/meeting5-ablations.json) records exact metrics,
paired slices/intervals, source and checkpoint hashes, selection and seed repeats.

These GPU experiments retain the same 4,096 training and 1,024 validation
decisions as the earlier learning curve. The original 18-fit sequence covers
model/head pairs, history windows, wording, opponent assignment, training epochs
and training-seed repeats. The [protocol and configuration](meeting5-ablations.md)
describe the declared selection rules. All comparisons are development analysis
on an already inspected session split.

## Encoder and head comparison

One epoch, seed 42, identical published starting weights within each pair.
History uses the last 64 completed observed actions per opponent.

| Encoder / head | Accuracy without | Accuracy with | Log loss without | Log loss with |
| --- | ---: | ---: | ---: | ---: |
| BERT Mini / linear | 73.34% | 72.36% | 0.610718 | 0.663062 |
| BERT Mini / MLP | 72.66% | 71.29% | 0.628483 | 0.657909 |
| DistilBERT / linear | 73.54% | 72.85% | 0.605690 | 0.632416 |
| DistilBERT / MLP | 72.85% | 72.56% | 0.624120 | 0.603948 |

History worsens log loss in three configurations and improves it in one.
DistilBERT / MLP was selected for the feature ablations by the smallest paired
history-minus-control log-loss change, rather than absolute accuracy.

## Feature and training ablations

All rows use DistilBERT / MLP and seed 42. The one-epoch no-history baseline is
shared across the window, wording and assignment comparisons.

| Input / training | Accuracy | Log loss |
| --- | ---: | ---: |
| No history, 1 epoch | 72.85% | 0.624120 |
| Recent 16 actions, 1 epoch | 72.66% | 0.593773 |
| Recent 64 actions, 1 epoch | 72.56% | 0.603948 |
| Recent 256 actions, 1 epoch | 73.24% | 0.623461 |
| Recent 64, explicit rate labels, 1 epoch | 73.05% | 0.626151 |
| Recent 64, swapped opponent assignments, 1 epoch | 73.24% | 0.605169 |
| No history, 3 epochs | 74.12% | 0.571766 |
| Recent 64 actions, 3 epochs | 75.68% | 0.571764 |

The short window has the strongest one-seed log-loss benefit. Increasing the
window to 256 does not strengthen it. Spelling out the rate labels reduces the
maximum DistilBERT input from 391 to 371 tokens but does not improve log loss;
this changes both serialization and token positions.

The assignment control rotates the same visible profile values among active
opponent seats. It changes 597 of 1,024 validation inputs; one-opponent situations
and identical profiles can remain unchanged. Its result is close to the correctly
assigned recent-64 version, so this experiment provides little evidence that the
gain relies on precise opponent assignment. The control retains historical
information and does not test removing all behavioral signal.

Three epochs improve the baseline substantially. History adds 16 correct
predictions at that budget on seed 42, while its log loss is essentially identical.
That accuracy finding motivated a separately recorded four-fit replication,
in addition to the original 18-fit plan.

## Short-window training-seed repeats

Recent-16 history, one epoch, identical source rows in every run. Seed 42 selected
this configuration; seeds 43 and 44 repeat the frozen pair.

| Training seed | Accuracy without | Accuracy with | Log loss without | Log loss with |
| --- | ---: | ---: | ---: | ---: |
| 42 | 72.85% | 72.66% | 0.624120 | 0.593773 |
| 43 | 72.66% | 74.12% | 0.611310 | 0.607349 |
| 44 | 72.75% | 74.22% | 0.608970 | 0.612612 |

The mean paired accuracy change is **+0.91 percentage points**, with a sample
standard deviation of 0.96 points across three training seeds. Mean log-loss
change is **-0.010222**, with a sample standard deviation of 0.017838.
Neither metric improves in every seed. These are descriptive training-seed
comparisons on the same inspected validation set, not three independent data
replications or a final-test confidence interval.

## Additional three-epoch replication

This separate four-fit follow-up was added after observing the seed-42 accuracy
gain. It repeats recent-64 history at three epochs and leaves the original
short-window/log-loss selection unchanged.

| Training seed | Accuracy without | Accuracy with | Log loss without | Log loss with |
| --- | ---: | ---: | ---: | ---: |
| 42 | 74.12% | 75.68% | 0.571766 | 0.571764 |
| 43 | 73.05% | 73.24% | 0.588257 | 0.594167 |
| 44 | 74.61% | 74.12% | 0.564821 | 0.571884 |

The mean accuracy change is **+0.42 percentage points**, with a sample standard
deviation of 1.04 points. Mean log loss is **0.004324 worse** with history,
with a sample standard deviation of 0.003791. The striking seed-42 accuracy gain
does not persist uniformly, and probability quality worsens in two repeats.

## Interpretation

The original BERT Mini result is specific to its configuration. A larger encoder
with an MLP and short history can show a modest average benefit, but the effect
depends on representation, memory, training budget and seed. The assignment
control also limits evidence for precise opponent-specific use of the profiles.
These experiments support a working integration with conditional, variable
predictive benefits; they do not demonstrate a consistently improved policy.

More training improves the no-history baseline's log loss in every repeated seed.
History's additional contribution is smaller and mixed. The targets remain human
observed action categories, rather than PokerBench reference actions or payoff.

## Verification and provenance

- All 22 fits retain the same ordered source IDs, targets, legal menus and base
  states. Recent-64 replay matches the historical paired rows exactly; other
  windows change only the profile information. No inputs were truncated or dropped.
- Every saved checkpoint reproduces validation tokenization and probabilities
  exactly after reload. Independent rescoring verifies every aggregate, paired
  contrast and session-bootstrap interval from the saved probabilities.
- Source files match experiment commit `7561eb4`. The earlier CPU learning curve
  remains separate: these GPU fits use float32 and microbatches of two, accumulating
  an effective batch of 16. The first-epoch checks are saved with the evidence.
- All 169 full-environment tests pass; core-only checks pass 112 tests with 33
  expected optional-dependency skips. Full/core mypy, Ruff and lockfile checks pass.
- One additional seed-43 worker could not allocate a GPU context after another
  workload filled GPU 0. It failed before training and was rerun on GPU 2 with
  unchanged settings. Every reported result is from a successful, reload-verified fit.

Weights, raw samples, predictions and worker logs stay in ignored
`artifacts/holdem_ablations_gpu/`. The source/configuration and aggregate evidence
are versioned. Session intervals condition on the saved fits and do not account
for adaptive configuration selection; three training seeds on one inspected split
are not an independent final evaluation or an unseen-player test.
