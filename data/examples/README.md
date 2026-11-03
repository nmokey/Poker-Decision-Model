# Example data

The implemented contract is `DecisionInputs(context, candidates)`, with labels
stored separately as `target_index` in `DecisionExample`. Source parsing and
answer/future-card exclusion are documented in the
[Meeting 1 integration notes](../../docs/meeting1-integration.md).

Use tiny, clearly labeled synthetic fixtures for tests. Do not commit raw hand
histories, full processed datasets, or generated model predictions here. Reviewed
aggregate experiment metrics and provenance live under
[`docs/results/`](../../docs/results/); the complete accuracy/log-loss comparison
is in the [Week 2 report](../../docs/week2-results.md).
