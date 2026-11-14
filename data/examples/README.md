# Example data

The implemented contract is `DecisionInputs(context, candidates)`, with labels
stored separately as `target_index` in `DecisionExample`. Source parsing and
answer/future-card exclusion are documented in the
[Meeting 1 integration notes](../../docs/meeting1-integration.md).

Use tiny, clearly labeled synthetic fixtures for tests. Do not commit raw hand
histories, full processed datasets, or generated model predictions here. Reviewed
aggregate experiment metrics and provenance live under
the [results index](../../docs/results/README.md), which separates the PokerBench,
Leduc and real Hold’em studies and links their accuracy/log-loss reports.


Opponent forecasting has a distinct Leduc contract: visible `inputs`, separate
`target_index`, and outer identity/order/split metadata. Audit-only files hold
hidden replay details and policy settings. Generated datasets and per-example
forecasts belong in ignored run directories. See the
[forecasting guide](../../docs/opponent-experiments.md); these labels are not
interchangeable with PokerBench Hero reference actions. The
[Hold’em adapter](../../docs/meeting5-integration.md) joins public forecasts and
acting-player views by source ID, preserving separate labels and causal
completed-hand histories. Keep these raw paired rows in ignored artifacts too.
