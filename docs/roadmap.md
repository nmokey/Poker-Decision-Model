# Roadmap and current status

## Completed by November 2, 2026

- Audited 63,200 preflop and 500,000 postflop source rows without parsing failures.
- Prepared the pinned 10,000-example working pool and verified five scenario folds.
- Trained all four encoders with linear and MLP heads: 40 fresh fits, three epochs
  each, with saved checkpoints and one held-out prediction per row/model.
- Published [accuracy, log loss, and provenance](week2-results.md) for all eight
  combinations. ModernBERT + linear leads the observed scores at 71.92% / 0.6078;
  its 0.14-point accuracy lead over DistilBERT is small and based on one seed.
- Integrated PocketSolver with all eight trained encoder/head choices, a
  ModernBERT linear default and a live Groq baseline;
  see [deployment](deployment.md). Demo results use a selected fold checkpoint.

## Next empirical work

Repeat training across seeds, measure serving latency/memory on the intended host,
and define an independent evaluation or nested grouped selection before claiming a
winning model. Candidate-order augmentation, validation-only calibration, and
opponent-history ablations remain separate experiments. Opponent-conditioned
labels and histories are not implemented by adding descriptions to existing labels.

## Original eight-week plan

| Week | Work | Reviewable milestone |
| --- | --- | --- |
| 1 | Choose game variant and legal decisions; run scaffold | Examples and data contract |
| 2 | Data, opponent summaries, group splits | Versioned split manifests |
| 3 | Uniform, heuristic, LLM baselines | Shared metrics and saved outputs |
| 4 | Tokenizer, training, checkpoints | Tiny overfit test and trained scorer |
| 5 | Opponent conditioning, multiple questions | History ablation and unseen options |
| 6 | Calibration and benchmarking | Reliability plots, held-out metrics, p50/p95 |
| 7 | Connect model and comparison to demo | Editable opponent profile and probabilities |
| 8 | Freeze versions and rehearse | Reproducible report and presentation |

Suggested ownership: data/environment, model/training, baselines/evaluation, and
demo/integration. Every group owns its tests. Agree on record and prediction
contracts before dividing work.
