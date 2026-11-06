# Roadmap and current status

Updated November 6, 2026. The work is organized by capability; the
[original proposal](archive/original-proposal.md) retains the initial calendar.

## Implemented

| Capability | Current evidence |
| --- | --- |
| PokerBench data and decision contract | Pinned source audit, separate inputs/targets, exact action amounts, saved splits |
| Neural decision scoring | Four encoders with linear/MLP heads, portable checkpoints, tiny learning checks |
| Grouped development comparison | 40 fits on five fixed scenario folds; all eight models reported on 10,000 examples |
| Interactive inference | Live PocketSolver with all trained model choices, Groq comparison and saved CPU results table |
| Opponent forecasting | Chronological Leduc sessions, repeated-identity splits, five policy families, causal summaries and Bayesian updates |
| Forecast evaluation | Matched baselines, history-window ablations, validation/grouped-training selection, replay and paired opponent-cluster reporting |

See [decision-model results](week2-results.md), [forecast results](meeting4-results.md)
and [deployment](deployment.md). Each result has its own population and target;
their scores are not directly comparable.

## Next research milestones

1. **Test adaptation to changing behavior.** Compare memory/decay or a learned
   sequence encoder on a fixed protocol and newly reserved opponent instances.
   Done when the comparison reports predictive quality, post-switch behavior and
   uncertainty for the same held-out decisions, including negative findings.
2. **Connect forecasts to Hero decisions.** Define response targets or a controlled
   repeated-play payoff experiment. Done when a policy using inferred opponent
   behavior is compared with a fixed baseline over many hands, without access to
   simulator-only identities or hidden state.
3. **Strengthen external evidence.** Repeat training across seeds and evaluate on
   new policies or suitable chronological real histories. Done when results state
   which generalization is tested and preserve data visibility and grouping rules.
4. **Complete independent quality and performance measurements.** Freeze choices,
   fit any calibration on separate validation data, and measure representative
   hosted latency, coverage and provider usage. Done when saved protocols and
   predictions support the reported claims.

The current demo and notebook results are reviewable artifacts. Human tilt
inference, GTO mixing, unexploitable strategy and profitable play are not yet
established outcomes.
