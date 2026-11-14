# Meeting 1 integration

This records the original data/model integration. For the current project,
including the Leduc and real Hold’em history workflows, see
[project context](project-context.md), the [results index](results/README.md)
and the [documentation index](README.md). Historical findings below retain their
original scope and verification status.

The consolidated notebook and package combine eight unique shared Colab notebooks.
The PokerBench integration link was supplied twice. The last notebook contains a
heading and an empty code cell. The originals were read without editing them.

## Source contributions

| Notebook | Useful contribution | Integration decision |
| --- | --- | --- |
| [ACM AI Poker Project](https://colab.research.google.com/drive/1Pfq5WDgVviizHcnkQDgv6g9Rq76tfhhe) | Menu/label correlation audit, conflicts, option-set prior, tie-aware uniform reference | Keep the diagnostics and training-only prior; omit official-test inspection and test-derived weighting |
| [pokerbench_ingest.ipynb](https://colab.research.google.com/drive/1Dv9HFhPQ6KoWkjvh2znvkkn22cMkA9o7) | Pre/postflop builders, visible-board reconstruction, answer-independent inputs, saved provenance | Adopt visible-board validation and explicit input boundaries; stop on parse failures; preserve source IP/OOP without inferred seats/stacks |
| [AMCAI.ipynb](https://colab.research.google.com/drive/1eTjaj_6qQfL_U386W4EItgjrVoE1csyp) | Raw EDA, per-source split, legal-candidate majority baseline | Keep inspection and candidate-only prediction; replace a column denylist with explicit state fields to exclude future cards and added answers |
| [pokerdecisionmodel.ipynb](https://colab.research.google.com/drive/1sHBTFJOu81VLt0CdIzg8TouMXvU7UXmQ) | Revision tracking, exact-size candidates, hashed context/action interactions, shared linear scorer | Pin a fixed revision and hashes, preserve cards and sizes, add pair interactions; remove test evaluation and split strictly by context regardless of menu |
| [PokerBench_integration_copy.ipynb](https://colab.research.google.com/drive/1GlZPGc1ixxpLy00Ewxhy3QCxWG2fo07n) | Small-sample workflow, train-only preprocessing, emphasis on preserving amounts | Use a small working sample after full-source validation; candidate scoring replaces a fixed action head/regression adapter |
| [Maksim Poker Project Notebook.ipynb](https://colab.research.google.com/drive/1VrCEunvE3SMQgSW-K_eh8UCEYoiQN4TQ) | Readable preflop example builder, row-level failures, context-group split | Keep the clear validation and split structure; interpret blank history consistently and stop on failures |
| [pokerbench_data_ingestion.ipynb](https://colab.research.google.com/drive/1Jjn-fJj6TMYQ5pegGuD20VoAWqOxVfAr) | Raw rows, types, missingness, label counts, source index handling | Keep the EDA for training CSVs only; identify examples by source kind and zero-based CSV data-row offset |
| [ACM AI Projects W1.ipynb](https://colab.research.google.com/drive/1GHX6o-6mzOQirpwDJlrc1yZ5bD1CBCCC) | Heading only; no implementation yet | No executable material to merge |

The existing repository's preflop notebook supplied checksum verification and a
case-preserving baseline walkthrough. Its reusable logic now lives in the package;
the notebook covers both training CSVs and exact candidate actions.

## Implemented contract

`DecisionInputs(context, candidates)` is the complete prediction input. A separate
`DecisionExample` carries `source_id`, `street`, and `target_index`. Saved JSONL is
flat for readability: `source_id`, `street`, `context`, `candidates`, `target_index`.
Consumers must explicitly construct inputs from the two input fields. No answer
text, action-type target, source ID, or target index is passed to prediction.

Candidate parsing uses `ast.literal_eval`, requires a nonempty list of strings,
canonicalizes syntax and numeric spelling, rejects normalized duplicates, and
requires the reference action to match exactly once. It never inserts a missing
answer or changes candidate order based on the target. Numeric preflop actions
become explicit raises; postflop bets and raises remain distinct. Decimal action
amounts are preserved without rounding. Card rank/suit case stays intact.

Contexts use explicit state fields. Empty preflop history means no recorded prior
action. Pot size and source counters are validated but retained as source fields;
we do not claim to reconstruct the pot or number of active players. Postflop IP/OOP
stays as provided. We do not infer stack sizes, real seats, or missing actions.

Postflop boards are rebuilt from flop cards plus `dealcards` markers in the action
history. Visible dealt cards must agree with visible board columns and the current
street. Cards must be valid and unique, including Hero's cards. Future board
columns and `aggressor_position` never enter inputs. This prevents future-card
leakage but does not certify that every supplied history follows all poker rules.

## Reproducibility and failure reporting

Only the two named training CSVs are fetched, from revision
`7ac61f961c81a50fc0f667820b2fb0e432dfec0d`. Their SHA-256 values are in `data.py`.
Downloads are checked before replacing the cache; cached bytes are checked again.
No generic Hub split inference, prompt/CSV row join, official-test download,
official-test label-mix weighting, or test evaluation is part of this workflow.

Every source row is either a validated example or a saved row-level parse failure.
Any failure stops preparation before splitting/training. Raw inspection reports
include three rows, inferred column types, missing cells, blank strings, and exact
action frequencies. Blanks are counted separately from absent cells.

All rows are audited before selecting a fixed-seed working sample (5,000 per source
by default). No duplicates are removed. Conflicting context/menu labels are
reported. The split shuffles sorted distinct contexts with seed 42 and assigns 20%
of contexts to validation, keeping identical contexts together even when menus or
labels differ. The split is unstratified; inspect actual per-street/action counts.

The saved manifest records source revision/hashes/counts, diagnostics, sampling
settings, seed, code hashes, Python version, and exact membership. Separate JSONL
files hold train/validation records. Baseline reports include dependency versions,
model settings, metrics, and five fixed-seed mistakes. Outputs are ignored by Git.

## Baseline design

Three models use the same examples and candidates:

1. Uniform probabilities over the supplied candidates.
2. A smoothed action-type prior conditional on the unordered multiset of menu
   types, fitted on training labels only. Multiple sizes share their type's mass;
   unseen menus use uniform probabilities. This measures menu-only predictability.
3. A shared candidate scorer using a stateless feature hash and binary logistic
   regression over explicit context/action interactions. Each decision has 0.5
   positive weight and 0.5 total negative weight. Single-option decisions are
   omitted from training. Model fitting uses only training examples.

Explicit interaction features matter: a linear score with only additive context
features would add the same context contribution to every candidate, canceling
when ranking. Here action-type and exact-action interactions allow context to
change the ranking. The scorer can evaluate action amounts unseen during fitting.
Prediction accepts answer-free inputs and normalizes scores within each menu.

Metrics include exact-candidate accuracy, action-type macro-F1 (sizing ignored),
log loss, Brier score, single-option frequency, and multi-option exact accuracy,
overall and by street. Deterministic predictions break ties lexically by action,
independent of menu order. A separate `tie_expected_accuracy` reports expected
accuracy under uniform choice among tied maxima. The five mistakes use the same
deterministic prediction rule and show context, menu, target, prediction, and
candidate probabilities. Fewer than five errors is handled safely.

## Historical exact-context development run

Both full training CSVs passed parsing and validation: **63,200 preflop** and
**500,000 postflop** rows, with **zero failures**. The full-source audit found 101
preflop context/menu groups with conflicting labels and 6,187 repeated exact
postflop examples. They remain in the data and identical contexts stay together.

Default sample: 10,000 examples, split into 8,000 training and 2,000 validation
examples. The consolidated notebook also executed from a fresh kernel starting in
`notebooks/`. Reference results on Python 3.12.13, NumPy 2.5.3, sklearn 1.9.1:

| Model | Exact accuracy | Tie-expected accuracy | Multi-option accuracy | Log loss |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 43.10% | 45.10% | 40.20% | 0.8436 |
| Option prior | 59.00% | 59.00% | 56.91% | 0.7462 |
| Linear candidate scorer | 72.90% | 72.90% | 71.52% | 0.7919 |

The learned model improves accuracy but has worse log loss than the prior;
probability quality needs work. Scores vary across streets, and the preflop score
is much higher than the postflop scores. The notebook exposes those breakdowns.
These are development results on the stated working sample, not official-test
results or evidence of profitable play. Versions and exact membership accompany
local generated results; numerical variation across environments is possible.

## Limits and next steps

Exact-context splitting does not isolate related scenarios or suit-equivalent
hands. The completed grouped evaluation described below tests broader scenario
families and reports per-street metrics. Source candidate menus may encode construction artifacts; the
candidate-only prior and menu/label cross-tab help reveal this. These checks
validate consistency, not that the supplied menu exhausts legal actions.

Single reference actions do not reveal solver mixing probabilities, and normalized
baseline scores are not calibrated solver strategies. No opponent identities or
longitudinal histories are introduced. The implemented shared neural scorer reuses
the same input contract, split manifest, and evaluation functions. Neural training,
tokenization checks, grouped evaluation, and PocketSolver are now implemented;
opponent adaptation remains future work.

## Subsequent grouped evaluation

The baseline results above describe the original exact-context holdout. The
[completed Week 2 matrix](week2-results.md) uses five broader scenario-grouped folds
and evaluates 10,000 held-out predictions per model. Its linear candidate baseline
scores 69.69% accuracy and 1.0025 log loss; these figures have a different denominator
from the historical holdout and should not be treated as a controlled change in
model quality. All eight pretrained transformer/head combinations are reported
with accuracy and log loss in the linked report.
