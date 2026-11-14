# Meeting 5 integration

Members extended Leduc opponent-history forecasting to real Texas Hold'em and
tested the same information as an extra input to our existing decision model.
The maintained workflow is [Hold’em integration notebook](../notebooks/holdem_opponent_integration.ipynb),
backed by shared package code and a matching command-line runner.

## Submission comparison

These implementations use different splits, seeds, histories, and menus. Their
recorded scores are not a leaderboard.

| Submission | What was completed | Pieces retained | Integration changes |
| --- | --- | --- | --- |
| `Meeting 5 Hold'em.ipynb` | Full archive, matched 128/64 sample, both forecasters, both one-epoch BERT Mini fits, reloads, optional profile swap; saved outputs show both pairs completed | Four-task walkthrough, independent checkpoint loads, explicit artifacts and token checks | Replace embedded helper/schema adapters with the package; group suffix directories for splitting; bound observer-specific histories; exclude optional swap experiment from required workflow |
| `holdem_opponent_integration.ipynb` | All 10,000 hands, Leduc summaries, causal replay mutations, both 128/64 comparisons, reloads | Direct `history_summary` reuse, 64-decision window before matching, completed-hand timing, observer visibility, active-opponent summaries, observed-action question | Standardize 75/25 split to 80/20; read structured public state instead of parsing text; retain actual PokerKit menus |
| `holdem_history.py`, `test_holdem_history.py`, `run_holdem_integration.py` | Bounded replay, unique paired sample, individual CLI steps, causal/visibility tests, download/check/compare code | Stepwise runner, identical base contexts, separate labels, checksum checks, provenance, failed-hand transaction boundary | Adapt to this checkout's APIs; replace 10-group/30-hand default with full replay; retain optional debug bounds; replace custom unbounded history with Leduc summaries |
| `meeting5-holdem.zip` | Forecaster, Hero adapter, model comparison, shared data/results, independent checks, explanatory analysis, explorer | Source-ID join, candidate-index metrics, per-decision evidence, replay checks, local explorer idea | Consolidate duplicate pipelines; replace seed 0 and 60/20/20 split with meeting protocol; remove sibling-folder dependencies; display saved predictions without refitting |
| [Shared Colab](https://colab.research.google.com/drive/1_tsqHrjjEjCE3KzdnIsENBY0qDXfWPjf) | Earlier PokerBench work plus metrics, published checkpoint loading, training update, save/reload, paired constructed 16/8 examples | Ragged probability checks, checkpoint identity, before/after and reload checks | Its Meeting 5 pair uses constructed examples, not a completed Pluribus-history experiment; use the real shared sample |
| [Additional Week 7 Colab](https://colab.research.google.com/drive/1jHlk6UYUI9kOKGQzb2fo6N4wyhE4cBpm) | Expanded forecasting, engineered-card trees, frozen BERT Mini/MiniLM heads, nested session CV, calibration and stress checks | Strong new tabular baseline and separate expanded-evaluation design | [Additional review](meeting5-additional-submission.md): different protocol; reported scores not locally reproduced; its test groups overlap our prior experiments |

[Submission inventory](results/meeting5-submissions.json) records attachment hashes
and bounded saved outputs. Historical outputs are inspected claims, not exact
local reproductions. The combined protocol is rerun independently. Original files
remain at their supplied attachment locations; duplicate pipelines, weights,
notebook execution metadata, and generated datasets are not copied into Git.

## Canonical protocol

- Verify the pinned PHH v1 archive and replay all 10,000 hands before sampling.
  Build both complete input views before reading the action. Public forecasts
  never read private cards; Hero sees only its own cards. Only a successfully
  completed hand updates memory.
- Key memory by observer/target alias within each original directory, following
  seat rotations. Keep the last 64 co-present completed-hand decisions, then
  reuse Leduc's `history_summary` with the same street, legal menu, and facing-bet
  status. The extra facing-bet match matters because PokerKit allows folding
  when checking is free; menu alone does not distinguish those opportunities.
- Reset history per directory; keep related letter-suffix directories together
  when splitting. The source has consecutive numbers across suffixes (`100`:
  hands 0–70; `100b`: 71–183), but this conservative adapter does not assume
  visibility across roster changes or a global chronology across groups.
- Seed 42 shuffles numeric session groups; ceiling(20%) go to validation. Join
  views by unique source ID, then sample 128 training / 64 validation human
  decisions. Groups never straddle the split. Repeated aliases may occur in both
  splits: this tests new sessions, not unseen people. Earlier validation hands
  update visible history without fitting parameters on validation labels.
- Reuse `Forecaster(kind="masked_logistic", c=1.0)`, identical public features,
  and optionally the 11 Leduc summary features. Forecasts observe as Pluribus.
  Both model views retain PokerKit's actual legal menus.
- Reuse `pluribus_inputs` for Hero's base state/candidates. Append only summaries
  of other active players, as observed by Hero. Empty history is explicit.
  Removing that one section gives the control; cards, candidates, order, and
  target remain identical.
- Independently reload the same published BERT Mini linear checkpoint for each
  version. Ask “What action will the acting player take?” Fine-tune one epoch:
  batch 16, encoder LR 2e-5, head LR 1e-3, weight decay 0.01, seed 42, CPU.
  Fail on inputs over the model budget, at most 512 tokens; never truncate.
- Score local candidate indices on the same 64 decisions. Save before/after
  probabilities, settings, IDs, hashes, numeric forecasters, and both checkpoints.
  Re-encode with the saved tokenizer and verify probabilities after reload.
  Build the explorer from saved predictions without another fit.

The previous feasibility helper keeps its tuple return interface and optional
bounded Hero probes. Its earlier pilot scores used different memory, session,
and feature definitions; those remain historical evidence.

## Run it

~~~sh
uv sync --locked --extra analysis --extra ml --extra experiments
uv run --locked --extra analysis --extra ml --extra experiments \
  jupyter lab notebooks/holdem_opponent_integration.ipynb
~~~

Equivalent steps with the project Python environment:

~~~sh
python -m poker_decision holdem download-data
python -m poker_decision holdem download-model
python -m poker_decision holdem prepare
python -m poker_decision holdem forecast
python -m poker_decision holdem check
python -m poker_decision holdem compare
python -m poker_decision holdem report
~~~

`scripts/run_holdem_integration.py STEP` calls the same entry point.
`--out PATH` selects an output directory; `--checkpoint PATH` supplies an existing
saved model. Preparation supports `--max-sessions` and `--hands-per-session` for
debugging; defaults replay the full archive. The downloader supports this
checkout's single-model release manifest and the newer multi-model catalog.
No new dependency is introduced. The notebook supports local runtimes and Colab;
a fresh public clone needs these changes shared first.

Ignored `artifacts/holdem_integration/` holds `shared_sample.json`,
`preparation.json`, `example.md`, `input_checks.json`, comparison JSON/CSV files,
portable forecasters, paired validation probabilities, `checkpoints/`,
`summary.json`, `results.md`, and `explorer.html`. Membership and integration-code
hashes are checked before downstream steps. Changed samples/code require fresh
preparation and comparisons.

## Meaning and limits

The transformer consumes **history summaries**, not learned forecaster
probabilities. The experiments share completed-hand behavioral information and
the existing joint state/question/options encoder and candidate head. This is
an extra input, not a learned forecast-to-decision stack.

Targets are human observed action categories with bet sizing abstracted.
PokerBench's reference labels, splits, and deployed benchmark stay separate.
The 64 inspected validation decisions do not establish reliable improvement,
generalization to unseen players, optimal strategy, or poker profit. Do not
average the submitted scores or pick an implementation because its score is
higher. See [canonical results](meeting5-results.md) for the integrated rerun.


## Larger training samples

`poker_decision.holdem_scaling` compares nested training samples of 128, 1,024,
and 4,096 decisions on a fixed 1,024-decision validation set. The validation set
starts with the original 64 decisions, and every larger training set includes
the original 128. Separate seeded sampling streams keep validation independent
of training size. The split groups, source histories, initial checkpoint, one
training epoch, optimizer settings, and CPU execution stay fixed. More training
examples also mean more optimizer updates; this is a learning curve rather than
a comparison at equal compute.

```sh
.venv/bin/python -m poker_decision holdem-scale
```

Use `--train-sizes 128 1024 4096`, `--validation-size 1024`, and `--out NEW_PATH`
to set the sample sizes and output location. The baseline run must already exist;
its source rows, initial checkpoint, and saved predictions are verified before
reuse. The runner refuses to overwrite an existing scaling output directory.
It reevaluates the original 128-example transformer checkpoints and trains fresh
pairs for the larger samples, then reports both the expanded validation set and
the original 64-row subset. Descriptive paired intervals resample validation
session groups and condition on the saved fits. See [larger-run results](meeting5-scaling.md).

## Publication on the refactored repository

The published notebook is `holdem_opponent_integration.ipynb`; it uses the shared
`notebook_setup.py`, task-based CLI and `training.decision_optimizer`. Existing
Meeting 4 functionality is already on main and is reused directly. Historical
experiment hashes and original artifact paths remain unchanged in the reviewed
result records; [publication checks](results/meeting5-publication.json) record a
fresh notebook execution and exact prediction parity for both 128/64 model pairs.
All 164 tests, full/core type checks, lint and formatting passed. Default fresh outputs
are `artifacts/holdem_integration` and `artifacts/holdem_scaling`. No model weights
or generated decision-level data are included in the code change.
