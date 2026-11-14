# Meeting 5: integrated Hold'em results

Stage 1: the original 128/64 integration check. The [results index](results/README.md)
connects this historical result to the [training-size study](meeting5-scaling.md)
and [latest 22-fit GPU comparisons](meeting5-ablation-results.md).

The combined notebook completed every step on the real, checksum-verified PHH v1
archive and the published trained BERT Mini linear checkpoint. This is a small
integration result, not evidence that history reliably improves poker decisions.
The exact protocol, code/source hashes, versions, memberships, and aggregate
verification are in [the saved result](results/meeting5-canonical.json).

The refactored notebook was rerun before publication: both model pairs reproduce
all original validation probabilities exactly, and all 164 publication-time tests passed.
[Publication evidence](results/meeting5-publication.json) records publication-time source
hashes and checks separately from the historical run below.

The [training-size follow-up](meeting5-scaling.md) preserves this baseline and
compares larger training samples on a fixed, expanded validation set.

## Matched comparisons

Both pairs use the same 128 training / 64 validation observed human decisions,
seed 42. The archive contains 10,000 hands, 91,356 betting decisions and 76,187
human decisions across 92 directories. Histories reset per directory; the split
keeps related directories together in 67 numeric groups: 53 training and 14
validation. History uses the last 64 co-present, completed-hand decisions.

| Model | Input version | Validation accuracy | Validation log loss |
| --- | --- | ---: | ---: |
| forecaster | without_history | 67.19% | 1.083420 |
| forecaster | with_history | 64.06% | 1.082508 |
| decision | without_history | 57.81% | 0.964243 |
| decision | with_history | 62.50% | 0.943463 |

The forecaster loses two correct predictions while log loss changes very little.
The transformer gains three correct predictions and lowers log loss slightly.
On 64 decisions, these mixed descriptive changes establish that the feature can
be prepared, trained, scored, and saved through our architecture; they do not
establish a reliable performance gain. There was no settings search or epoch
selection on this validation set. Both transformer versions used the same
checkpoint, one epoch, seed, sample, candidate menus, optimizer settings, and CPU.

## Historical integration verification

- The complete notebook executed successfully; the source notebook is stored with
  cleared outputs and execution counts. Its executed copy stays in ignored artifacts.
- All 192 examples encode in both versions without truncation and return valid
  candidate probabilities. Maximum lengths: 183 tokens without history, 383 with
  history, below the 512-token limit.
- Both saved fine-tunes reproduce the tokenized validation inputs and probabilities
  exactly after reload (maximum absolute difference 0.0).
- All 141 offline repository tests passed. Full configured mypy, repository Ruff
  lint, and formatting checks passed. Tests include real replay mutations,
  private-card exclusion, observer visibility, seat rotation, failed-hand rollback,
  suffix-session reset/grouping, legal-menu alignment, and provenance checks.
- The CLI and notebook call the same implementation. The local explorer reads
  saved paired validation probabilities without retraining.

No weights or generated decision-level data are committed. Local outputs live in
`artifacts/holdem_meeting5/`; open `explorer.html` for each validation decision or
`results.md` for the compact comparison. Follow [the integration guide](meeting5-integration.md)
to reproduce, inspect the submission comparison, or change output locations.

The new feature is the opponent's **completed-hand summary**, not a learned
forecaster output. PokerBench reference-action imitation, controlled Leduc
forecasting, and this Pluribus human-action check remain distinct evaluations.
