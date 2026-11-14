# Roadmap and current status

Updated November 13, 2026. The work is organized by capability; the
[original proposal](archive/original-proposal.md) retains the initial calendar.

## Implemented

| Capability | Current evidence |
| --- | --- |
| PokerBench data and decision contract | Pinned source audit, separate inputs/targets, exact action amounts, saved splits |
| Neural decision scoring | Four encoders with linear/MLP heads, portable checkpoints, tiny learning checks |
| Grouped development comparison | 40 fits on five fixed scenario folds; all eight models reported on 10,000 examples |
| Interactive inference | Live PocketSolver with all trained model choices, Groq comparison and saved CPU results table |
| Opponent forecasting | Chronological Leduc sessions, repeated-identity splits, five policy families, causal summaries and Bayesian updates |
| Hold’em history integration | Full Pluribus replay, matched forecast/Hero views, BERT Mini integration and fixed-validation 128/1,024/4,096-example comparison |
| Hold’em feature ablations | 22 GPU fits: encoder/head, history window/format/assignment and epochs; two selected pairs repeated across three training seeds, with mixed benefits |
| Forecast evaluation | Matched baselines, history-window ablations, validation/grouped-training selection, replay and paired opponent-cluster reporting |

The [results index](results/README.md) maps each stage to its protocol, findings and
verified evidence. The Hold’em integration is complete as an observed-action
experiment: summaries enter the existing neural architecture. It does not yet
measure better strategic responses. The [demo](deployment.md) serves PokerBench
checkpoints and remains separate from the history-feature experiments.

## Project closeout

Use the existing reports for the write-up and demo presentation. Preserve the
negative BERT Mini results, the conditional DistilBERT/MLP effects and the training-seed
variability rather than selecting the largest observed gain. Distinguish model
quality from the separately measured CPU serving latency. Current scope does not
require another model family or a new policy-learning experiment.

## Research beyond the current project

1. **Test adaptation to changing behavior.** Compare memory/decay or a learned
   sequence encoder on a fixed protocol and newly reserved opponent instances.
   Done when the comparison reports predictive quality, post-switch behavior and
   uncertainty for the same held-out decisions, including negative findings.
2. **Measure better responses to inferred opponents.** Define response targets or a controlled
   repeated-play payoff experiment. Done when a policy using inferred opponent
   behavior is compared with a fixed baseline over many hands, without access to
   simulator-only identities or hidden state.
3. **Strengthen external evidence.** Extend the selected Hold’em seed repeats to
   a frozen independent evaluation, and test new policies or unseen human players.
   Real histories are already used; session holdout alone does not test unseen
   people. Done when results state which generalization is tested and preserve
   data visibility and grouping rules.
4. **Complete independent quality and performance measurements.** Freeze choices,
   fit any calibration on separate validation data, and measure representative
   hosted latency, coverage and provider usage. Done when saved protocols and
   predictions support the reported claims.

The current demo and notebook results are reviewable artifacts. Human tilt
inference, GTO mixing, unexploitable strategy and profitable play are not yet
established outcomes.
