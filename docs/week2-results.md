# Week 2 development results

PokerBench reference-action comparison. See the [results index](results/README.md)
for the separate Leduc and Hold’em studies. The serving-latency measurements below
use a different workload from this quality evaluation.

Completed: November 2, 2026. All eight transformer/head combinations have complete
five-fold results: 40 full encoder fits. Each model has exactly one held-out
prediction for each of the same 10,000 examples.

All transformer fits use five scenario-grouped folds, seed 42, three fixed epochs,
batch size 16, encoder LR 2e-5, head LR 1e-3, weight decay 0.01, and MLP width 64.
Held-out folds never select epochs. There is zero specified scenario-group or exact-context
overlap across folds.

| Encoder | Head/model | Exact accuracy | Log loss (lower is better) |
| --- | --- | ---: | ---: |
| baseline | linear_candidate | 69.69% | 1.0025 |
| baseline | option_prior | 57.64% | 0.7623 |
| baseline | uniform | 41.28% | 0.8551 |
| bert_mini | linear | 68.96% | 0.6722 |
| bert_mini | mlp | 69.35% | 0.6656 |
| distilbert | linear | 71.78% | 0.6164 |
| distilbert | mlp | 71.60% | 0.6213 |
| minilm | linear | 69.57% | 0.6849 |
| minilm | mlp | 68.83% | 0.6764 |
| modernbert | linear | 71.92% | 0.6078 |
| modernbert | mlp | 71.38% | 0.6158 |

ModernBERT with the linear head has the highest observed accuracy (71.92%) and
lowest log loss (0.6078). Its accuracy exceeds the linear candidate baseline by
2.23 percentage points. The saved paired group-bootstrap interval for this
difference is approximately +1.01 to +3.33 percentage points, conditional on the
fitted models and specified groups; it excludes retraining uncertainty.

DistilBERT with the linear head is close at 71.78%, a gap of only 0.14 percentage
points (14 decisions out of 10,000). That small observed gap does not establish a
reliable advantage for ModernBERT. MLP heads do not consistently improve results.
BERT Mini and MiniLM do not improve exact accuracy over the linear baseline in
these runs, although all eight transformers have lower log loss. Lower log loss
measures reference-label probability quality; it does not establish calibration
or solver mixing frequencies.

These are development comparisons on an inspected working pool, with one random
seed. Choosing a winner using these scores requires a new untouched evaluation
or nested grouped selection for an independent final estimate. Reference-action
agreement does not establish profitable or unexploitable play.

The [CSV](results/week2-development.csv) contains the complete aggregate metric
fields. The [JSON](results/week2-development.json) records full protocol hashes,
data fingerprints, pinned encoders, settings, dependency/code versions, per-street
metrics, and paired intervals. Raw datasets, example-level predictions and model
checkpoints remain in ignored local artifact folders. See
[training instructions](week2-training.md) to reproduce or resume the runs.

All recorded training-package source hashes match commit `d7f0e23`. Later demo
changes can create a new protocol fingerprint; use the historical training revision
when resuming these artifacts. See the [documentation index](README.md) for the
audit, notebook walkthroughs, current demo, and remaining work.

## Serving latency

Measured November 3, 2026 using the eight published fold-1 serving checkpoints.
This is a separate local CPU measurement, not the latency of Streamlit hosting or
of the five-fold training evaluation above.

| Encoder | Head | Average latency (ms) |
| --- | --- | ---: |
| ModernBERT | Linear | 234.84 |
| ModernBERT | MLP | 214.55 |
| DistilBERT | Linear | 84.96 |
| DistilBERT | MLP | 88.07 |
| MiniLM | Linear | 44.36 |
| MiniLM | MLP | 45.36 |
| BERT Mini | Linear | 7.75 |
| BERT Mini | MLP | 7.71 |

Every combination processes the same seeded sample of 64 multi-option held-out
hands three times (192 measurements), with five excluded warm-up calls. Model
order is seeded and shuffled; hand order is shuffled identically across models
within each repeat. Batch size is one, with two PyTorch CPU threads on an AMD Ryzen
Threadripper PRO 5955WX. Python 3.12.3, PyTorch 2.15.0+cpu, Transformers 4.57.6.
Timing includes validation, tokenization, batch preparation, CPU inference and
probability validation; it excludes loading, warm-up, queueing, network and UI.
There is no response cache. These figures do not isolate head execution time;
small differences between heads include run variability and independently fitted
encoders. No full-dataset Groq accuracy/log-loss evaluation is recorded.

The [latency report](results/week2-serving-latency.json) retains every measurement,
repeat means, median/p95, hand IDs, hardware/settings and checkpoint/source hashes.
The Results matrix view joins it to the original development CSV and highlights
the selected encoder/head. Reproduce with unpacked deployment assets:

```bash
uv run --extra ml --extra demo python scripts/benchmark_demo_models.py \
  --assets-root artifacts/deployment/pocketsolver-models-v2 \
  --output /tmp/week2-serving-latency.json
```

The script refuses to overwrite an existing report. New hardware or hosting
measurements should be saved separately and clearly labeled.
