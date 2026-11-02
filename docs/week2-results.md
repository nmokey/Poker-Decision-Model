# Week 2 development results

Snapshot: November 2, 2026. Six of eight transformer/head combinations have complete
five-fold results. ModernBERT has three saved completed folds; its aggregate scores
are pending and are not compared with full-pool results below. The interrupted run
has been restarted from saved folds.

All completed models use the same 10,000 examples, five scenario-grouped folds,
seed 42, three fixed epochs, batch size 16, encoder LR 2e-5, head LR 1e-3,
weight decay 0.01, and MLP width 64. Held-out folds never select epochs.
There is zero specified scenario-group or exact-context overlap across folds.

| Encoder | Head/model | Exact accuracy | Log loss (lower is better) |
| --- | --- | ---: | ---: |
| minilm | linear | 69.57% | 0.6849 |
| minilm | mlp | 68.83% | 0.6764 |
| baseline | uniform | 41.28% | 0.8551 |
| baseline | option_prior | 57.64% | 0.7623 |
| baseline | linear_candidate | 69.69% | 1.0025 |
| bert_mini | linear | 68.96% | 0.6722 |
| bert_mini | mlp | 69.35% | 0.6656 |
| distilbert | linear | 71.78% | 0.6164 |
| distilbert | mlp | 71.60% | 0.6213 |
| modernbert | linear | pending | pending |
| modernbert | mlp | pending | pending |

DistilBERT with the linear head leads the completed models on accuracy and log loss.
Its accuracy exceeds the linear candidate baseline by 2.09 percentage points.
The saved paired group-bootstrap interval for this difference is approximately
+0.85 to +3.23 percentage points, conditional on the fitted models and specified
groups. It does not include retraining uncertainty. BERT Mini and MiniLM do not
improve exact accuracy over the linear baseline in these runs, although all six
transformers have lower log loss. Lower log loss measures reference-label
probability quality; it does not establish calibration or solver mixing frequencies.

These are development comparisons on an inspected working pool, with one random
seed. Choosing a winner using these scores requires a new untouched evaluation
or nested grouped selection for an independent final estimate. Reference-action
agreement does not establish profitable or unexploitable play.

The [CSV](results/week2-development.csv) contains the complete aggregate metric
fields. The [JSON](results/week2-development.json) records full protocol hashes for
completed runs, data fingerprints, pinned encoders, settings, dependency/code
versions, per-street metrics, and paired intervals. Raw datasets, example-level
predictions and model checkpoints remain in ignored local artifact folders.
See [training instructions](week2-training.md) to reproduce or resume the runs.
