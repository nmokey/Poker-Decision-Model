# Meeting 5: larger training samples

Completed November 13, 2026 on **temescal**. The original server-run metrics were
verified again: the 64 transformer predictions reproduce exactly, and the original
forecaster probabilities reproduce within 1e-10. Then the same workflow was run
with 1,024 and 4,096 training decisions. Original outputs were preserved.

**More training data substantially improves prediction, but the current opponent
summaries do not improve the transformer in these runs.** The larger forecaster
has a modest history benefit. These are observed-human-action development results,
not optimal-strategy or poker-profit measurements.

## Fixed validation comparison

All rows below use the **same 1,024 validation decisions**, including the original
64. Training sets are nested and retain the original 128 decisions. The 53/14
numeric-session-group split, source histories, seed 42, public starting BERT Mini
checkpoint, one epoch, batch 16, learning rates, weight decay, CPU, and four threads
stay fixed. No validation-based epoch or hyperparameter selection was used.

### Transformer

| Training decisions | Accuracy, no history | Accuracy, history | Log loss, no history | Log loss, history |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 55.27% | 56.84% | 1.028083 | 1.063747 |
| 1,024 | 71.78% | 71.88% | 0.680133 | 0.701715 |
| 4,096 | 73.14% | 72.46% | 0.621108 | 0.651398 |

At 4,096 examples, history gives seven fewer correct predictions and raises log
loss by 0.030290. The descriptive paired 95% session-cluster interval for the
accuracy change is -2.51 to +1.04 percentage points; for log loss it is -0.000180
to +0.066453. This does not establish a history benefit.

### Public-state forecaster

| Training decisions | Accuracy, no history | Accuracy, history | Log loss, no history | Log loss, history |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 63.87% | 62.60% | 1.391140 | 1.421411 |
| 1,024 | 67.77% | 67.19% | 0.773322 | 0.776228 |
| 4,096 | 68.07% | 69.34% | 0.705272 | 0.700675 |

At 4,096 examples, history adds 13 correct predictions and lowers log loss by
0.004597. Its descriptive paired session-cluster log-loss interval is -0.008054
to -0.001070. This is a small conditional benefit for forecasting, separate from
the result of appending opponent summaries to the transformer.

## Direct check on the original 64 decisions

| Transformer training decisions | No-history accuracy | History accuracy |
| ---: | ---: | ---: |
| 128 | 57.81% | 62.50% |
| 1,024 | 64.06% | 64.06% |
| 4,096 | 71.88% | 70.31% |

The initial 57.81% / 62.50% result was real; its apparent advantage for history
does not persist with the larger training sets. The original 64 rows remain
a supplementary small sample; the 1,024-row set is the primary comparison here.

## Verification and reproduction

- The original 128-example transformer checkpoints were reevaluated; fresh model
  pairs were fine-tuned for 1,024 and 4,096 examples. Both new pairs reproduce
  validation tokenization and probabilities exactly after checkpoint reload.
- Every saved prediction was independently checked against its target and source
  ID; recomputed metrics match within 1e-12. Train membership is nested, validation
  membership is identical, and session groups do not overlap.
- All 143 offline tests pass. Repository lint/format checks and full mypy checks
  pass (24 source files).
- Total experiment time was about 11.6 minutes on CPU, including source replay,
  sampling, refits, evaluation, checkpoint writes, and reload checks.

```sh
.venv/bin/python -m poker_decision holdem-scale
```

The runner refuses to overwrite an existing output directory; use `--out NEW_PATH`
for another run. See [the integration guide](meeting5-integration.md#larger-training-samples)
for size controls. Local outputs are in `artifacts/holdem_meeting5_scaling/`;
[the reviewed aggregate](results/meeting5-scaling.json) includes exact scores,
protocol, hashes, versions, and paired intervals. Weights and decision-level data
stay in ignored artifacts.

These runs use one seed and an inspected validation set with the same human
aliases across different sessions. Session intervals condition on the saved fits
and do not capture retraining or unseen-player uncertainty. A larger one-epoch
sample also entails more optimizer steps, so the curve is not an equal-compute
comparison. The injected feature remains a history summary; learned forecaster
probabilities are not fed to the transformer.

Fresh runs of the refactored published workflow use `artifacts/holdem_scaling/`
by default; the historical server output path above is preserved as provenance.
