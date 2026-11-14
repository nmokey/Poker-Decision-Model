# Review: additional Week 7 Hold'em notebook

Source: [Week 7 Poker Model](https://colab.research.google.com/drive/1jHlk6UYUI9kOKGQzb2fo6N4wyhE4cBpm),
reviewed November 13, 2026. This is a substantive additional study, not a duplicate
of the original quick integration. Notebook SHA-256:
`b2fcf6e89f3b682093c3ab2778e7a7d81bdabdc08a3ae4541767b8c89235edbb`.

## Meaningful additions

- **An engineered-card boosted-tree baseline.** It encodes Hero's own rank/suit
  information, made-hand category and public situation, then fits histogram
  gradient boosting. This is a useful new reference for the text architecture.
- **Frozen BERT Mini/MiniLM representations with fresh heads.** Candidate
  representations are cached without fitting on Pluribus; linear/MLP heads and
  normalization are fitted separately inside each fold. This allows substantially
  cheaper grouped comparisons. It differs from our end-to-end one-epoch fine-tunes.
- **Nested session-group selection and calibration.** Source implements five outer
  folds and three inner folds, shared history/control settings and temperature
  selection by log loss, then a final development-only selection before evaluation.
- **Broader evaluation.** The expanded forecast task uses 66,599 development and
  9,588 held-out decisions; the paired Hero comparison uses 4,096/1,024. Fixed
  chronological forward checks, held-alias/session stress checks, learning curves,
  errors, portable exports and target-free inference checks are included.
- **A different history representation.** Cumulative plus recent-20 summaries
  replace the canonical recent-64 summary. Numeric groups carry histories across
  suffix directories, whereas our conservative adapter resets memory per
  directory. These are protocol changes, not interchangeable implementations.

The original notebook section separately reports a 20-group 128/64 integration
and fixed-setting grouped CV. Its expanded section is the main new contribution.

## Recorded results, not independently reproduced here

| Task/pipeline | History | Reported accuracy | Reported log loss |
| --- | --- | ---: | ---: |
| Hero engineered-card trees | No | 82.42% | 0.388070 |
| Hero engineered-card trees | Yes | 81.84% | 0.404729 |
| Hero frozen BERT Mini + linear head | No | 72.95% | 0.619969 |
| Hero frozen BERT Mini + MLP head | No | 73.24% | 0.605234 |
| Hero frozen MiniLM + linear head | No | 72.17% | 0.614668 |
| Hero frozen MiniLM + MLP head | No | 73.14% | 0.620165 |
| Public forecasting trees | No | 70.89% | 0.626823 |
| Public forecasting trees | Yes | 70.66% | 0.627548 |

The saved nested-CV report also favors Hero trees (81.67% / 0.386571 without
history; 80.74% / 0.405366 with history). Both primary pipelines again show no
history benefit. The architecture/representation comparison is informative
inside this notebook, but its 82.42% is not directly comparable with our 73.14%
fine-tuned transformer: split, samples, engineered features, history, calibration,
and encoder training differ.

## What was checked

The notebook was downloaded and source modules were extracted without executing
its setup or launching training. Source inspection confirms explicit public/own-card
features, completed-hand observer-specific snapshots before labels, transactional
history updates, fold-local head normalization, separate development/test indices,
and development-only settings selection in the driver. These are code-review
findings; they do not establish that the inaccessible exported artifact exactly
matches this source.

The linked [model/data/results archive](https://drive.google.com/file/d/1rOC3MwdhsRZsx_07E5nsYVL59p8x1dlT/view)
returned a not-found/access error through connected Drive and could not be opened
through the public web tool. The notebook's current output contains a link to that
archive, not the advertised embedded recovery ZIP. Consequently predictions,
selected settings, hashes and fitted models could not be independently recovered
and rescored. No claim of locally reproduced 82.42% accuracy is made.

## The test groups are not untouched across the project

We independently reconstructed the notebook's split from the pinned raw archive,
its declared seed `20261113`, and its exact group-selection procedure. Its held-out
groups are:

`105, 64, 65, 73, 76, 77, 91, 94, 97`.

Comparison with the actual sampled rows of our completed 4,096/1,024 server run
shows eight groups (`105, 64, 65, 76, 77, 91, 94, 97`) already occurred in training;
`73` already occurred in validation. None is newly untouched project-wide.
The notebook excludes its own earlier 20-group cohort from its final test; that
does not account for our broader experiments.

This overlap does not by itself show within-notebook model-fitting leakage: its
new models use original PokerBench encoder checkpoints, and the inspected driver
keeps its own development/test fits separate. It means we should describe its
scores as another development study rather than an independent final project test.

## Integration decision

Record this as an additional contribution with its own protocol and verification
status. The first reusable piece worth testing in our existing 4,096/1,024 split is
the engineered-card tree baseline; that would provide a direct comparison with
the already verified transformer. The nested-selection machinery is useful for a
separate expanded evaluation. Preserve the distinction between frozen-head and
end-to-end training, and retain our verified results as their own experiment.
This review does not rerun the large nested study or replace the canonical pipeline.

