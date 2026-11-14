# Results index

Updated November 13, 2026. These are separate experiments with different targets,
populations and evaluation protocols. Compare methods within a study; scores
across studies do not form a common leaderboard. Higher accuracy and lower log
loss are better; accuracy changes below are percentage points.

## Findings by study

| Study | Evaluation | Main finding | Detailed report |
| --- | --- | --- | --- |
| PokerBench reference-action imitation | 10,000 examples, five scenario folds, 40 encoder/head fits | ModernBERT linear: 71.92% accuracy / 0.6078 loss; DistilBERT linear: 71.78% / 0.6164 | [Model comparison](../week2-results.md) |
| Controlled Leduc opponent forecasting | 2,506 test decisions from 10 held-out simulated opponents | Matched no-history → recent-64: 59.26% → 69.87%, loss 0.7741 → 0.6373; Bayesian filter: 73.34% / 0.6157 | [Forecast results](../meeting4-results.md) |
| Pluribus Hold’em history features | Fixed 4,096 training / 1,024 validation decisions for GPU comparisons | Short-history DistilBERT/MLP: mean +0.91 accuracy points / −0.0102 loss across three training seeds; effects vary by seed and configuration | [Latest GPU results](../meeting5-ablation-results.md) |

The project first built reference-action scoring and controlled opponent forecasts,
then reused the history summaries as transformer inputs on real Hold’em histories.
This verifies the architectural connection. The Hold’em target is an observed human
action category, with bet sizing abstracted; it is not a PokerBench reference action.
Learned forecaster probabilities are not fed into the transformer.

## Hold’em experiment stages

Read the stages in order when interpreting how the conclusion changed. The early
small-sample result is preserved rather than substituted for the larger studies.

| Stage | Protocol | Finding | Report / evidence |
| --- | --- | --- | --- |
| Initial integration check | 128 training / 64 validation, BERT Mini linear, one epoch on CPU, seed 42 | Transformer accuracy 57.81% → 62.50% with history: three more correct decisions, too small to establish reliable benefit | [Report](../meeting5-results.md) / [aggregate](meeting5-canonical.json) |
| Training-size comparison | Nested 128 / 1,024 / 4,096 training sets, fixed 1,024 validation decisions; BERT Mini linear, one epoch on CPU, seed 42 | At 4,096: 73.14% → 72.46%, loss 0.621108 → 0.651398 with history; public forecaster shows a small separate gain | [Report](../meeting5-scaling.md) / [aggregate](meeting5-scaling.json) |
| GPU architecture and feature ablations | Same 4,096 / 1,024 rows; 22 fits varying encoder/head, memory, wording, assignment, epochs and training seed | Conditional DistilBERT/MLP gains; three-seed checks do not improve both metrics consistently | [Report](../meeting5-ablation-results.md) / [aggregate](meeting5-ablations.json) |

The GPU study includes the original 18-fit sequence and a separately recorded
four-fit replication prompted by the three-epoch accuracy result. Its two repeated
DistilBERT/MLP comparisons are:

| History / training | Mean accuracy without → with | Mean paired accuracy change | Mean paired loss change |
| --- | ---: | ---: | ---: |
| Recent 16 actions, one epoch | 72.75% → 73.67% | +0.91 points | −0.010222 |
| Recent 64 actions, three epochs | 73.93% → 74.35% | +0.42 points | +0.004324 |

These means use training seeds 42, 43 and 44 on the same inspected validation set;
they are not independent data replications. The swapped-profile control performs
similarly to correctly assigned recent-64 profiles, limiting evidence for precise
opponent-specific use. Additional training improves the no-history baseline’s loss
in every repeated seed. See the [ablation protocol](../meeting5-ablations.md) for
selection rules and the [integration guide](../meeting5-integration.md) for replay,
visibility and session grouping.

## Evidence catalogue

| Record | Contents |
| --- | --- |
| [PokerBench JSON](week2-development.json), [CSV](week2-development.csv) | All model scores, folds, settings and provenance; [reproduction](../week2-training.md) |
| [Serving latency](week2-serving-latency.json) | Separate local CPU timings on 64 hands, three repeats per model; [measurement details](../week2-results.md#serving-latency) |
| [Leduc canonical run](meeting4-canonical.json) | All methods, selected settings, slices, paired opponent intervals and replay provenance; [run guide](../opponent-experiments.md) |
| [Hold’em initial run](meeting5-canonical.json) | Original 128/64 paired forecasts and transformer results |
| [Hold’em publication checks](meeting5-publication.json) | Refactored notebook execution, exact prediction parity and publication-time source/check hashes |
| [Hold’em training-size run](meeting5-scaling.json) | Fixed-validation learning curve, nested membership, paired session intervals and reload checks |
| [Hold’em GPU run](meeting5-ablations.json) | All 22 fits, selections, training-seed repeats, slices/intervals, independent rescoring and source/checkpoint hashes |
| [Leduc submissions](meeting4-submissions.json), [Hold’em submissions](meeting5-submissions.json) | Source inventory and contribution provenance, distinct from canonical experiment results |

The [additional Week 7 notebook review](../meeting5-additional-submission.md)
records an interesting engineered-card baseline, but its reported scores use a
different protocol and were not independently reproduced. They are not added to
the verified comparisons above.

## Interpretation boundaries

PokerBench comparisons use one training seed on an inspected development pool.
Leduc tests held-out simulated opponent instances from known families. Hold’em
holds out session groups, while the same humans can recur across splits; selected
settings have three training-seed repeats, and all tuning uses exposed validation
data. Paired bootstrap intervals condition on saved fits and do not account for
adaptive selection or all sources of dependence.

Reference-action agreement and observed-action prediction do not establish
calibration, GTO mixing, human tilt detection or profitable play. The hosted demo
uses the PokerBench models. See the [scientific audit](../scientific-audit.md) and
[project context](../project-context.md) for the scope of each claim. Raw samples,
per-decision predictions, logs and weights stay outside Git; this directory stores
reviewed aggregates and provenance.
