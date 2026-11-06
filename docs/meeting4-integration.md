# Meeting 4 submission reconciliation

Integrated November 6, 2026. The maintained entry points are
[notebook 05](../notebooks/05_opponent_history.ipynb) and
[`poker_decision.opponents`](../src/poker_decision/opponents). See the
[run guide](opponent-experiments.md) for commands, retained settings and completion
conditions. This extends the project with a separate supervised opponent
forecasting task. PokerBench labels and existing trained Hero decision models
retain their original meaning.

## Source inventory and contribution map

All eleven attached files and the linked Colab were inspected. Source SHA-256
fingerprints and bounded aggregate evidence live in
[`meeting4-submissions.json`](results/meeting4-submissions.json). Names and filename
numbers do not establish authorship or chronology. No member is assigned credit
based only on Colab execution metadata. The attachment names identify submissions
unambiguously for later attribution.

| Submission | Distinct contribution | Canonical treatment |
| --- | --- | --- |
| `Week6_Project.zip` | Dataclass input/target contracts, chronological validation, grouped splitting, simple legal-normalized Naive Bayes pair, sample and 10,000-hand generated data | Shared contract and categorical policy variant; Naive Bayes controls retained. Use OpenSpiel because the default local engine is absent. Avoid storing full growing histories in every row. |
| `opponent-leduc-experiment.zip` | Card-conditioned probability tables, fixed CFR observer, legal-menu logistic models, validation search, smoothed cumulative/recent summaries, shuffle control, saved per-decision predictions | Table policy option, CFR observer option, per-menu logistic pair, matching/window controls, and opponent-cluster reporting |
| `results.md` | The smaller archive's saved report | Exact duplicate of `opponent_leduc/results/results.md`; retain one aggregate record |
| `05_leduc_opponent_history_experiment.ipynb` | Self-contained card-dependent log-weight generator, repeated identities, completed-hand histories, windows 4/16/64, six regression tests, exhaustive summary audit, opponent-cluster intervals and figures | Default weighted generator, identity grouping, completed-hand timing, multi-window comparison, source ideas moved into tested package functions |
| `Leduc Opponent Experiment` | JSON notebook despite extensionless filename; residual style archetypes, true masked logistic objective, online 25-style Bayesian filter and reset hazard, belief trajectories | Residual policy option, matched masked logistic models, Bayesian filter, validation selection and notebook belief plot |
| `04_alternative_data_sources.ipynb` | Existing source notebook plus a chronological event/history extension | Keep existing 04 for distinct Pluribus/teacher tasks; route opponent forecasting to 05. Its imported `opponent_history` module was not attached. |
| `05_opponent_forecast.ipynb` | Proposed PokerBench-aligned names, chronological CFR/tight/loose generation and 2,000-decision-per-style datasets | Preserve controlled CFR deviations and compact observer contract; imported generator module and referenced CSV datasets were not attached |
| `06_opponent_unconditional_baseline.ipynb` | Training-only constant priors, whole-session splits and legal-normalized controls for each known style | Pooled train-only prior in the common suite; historical known-style results retained separately |
| `07_opponent_logistic_regression.ipynb` | Exact visible-card showdown equity, opportunity-based summaries, numeric parameter export, validation-only model/window choices | Equity features, matched opportunity controls and portable model export; imported `opponent_features` module and CSV datasets were not attached |
| `comparison_report.md`, `metric.csv` | Additional 373-decision summary comparison and generation report | Keep as submitted claims; the referenced generator/comparison implementation and matching dataset were not included |
| [Poker W6 Opponent Dataset (Colab)](https://colab.research.google.com/drive/15ETJliWKqkfARHxXAz0IQCiipWN_12Gx?usp=sharing) | Exported CFR teacher, five CFR-derived opponent profiles, separate RNG streams, public-only whitelist, full engine replay, hashed runs, training-group CV, binary candidate scorer, descriptive post-switch analysis | CFR policy option, replay/provenance safeguards, candidate scorer, optional grouped-training C selection; original recorded results kept separate |

## Scientific decisions

The submissions differ in opponent policies, observer policies, private information
available to the observer, history timing/window definition, sample sizes and
objectives. Their accuracies are **not a model leaderboard**. The canonical suite
makes those choices explicit and compares models on the same generated data.

Primary models optimize conditional cross-entropy over the legal menu during
training. The binary candidate objective, post-mask per-menu models, Naive Bayes,
priors and equity representations remain alternative controls. Equity is against
uniform unknown cards, not the opponent's true strength or inferred range. Both
strength models now receive the same current opportunity cues, addressing a
confound acknowledged in notebook 07.

Repeated sessions of an opponent stay together for splitting, optional training
CV, and bootstrap uncertainty. Default behavior summaries exclude current-hand
targets; an `all_prior` mode retains the other submissions' timing. Baseline
current-hand public information is identical across the main pair. Bayesian
feedback timing follows the same choice. Its hazard uses elapsed hands, including
hands with no modeled decision.

The existing archive's shuffle test is useful descriptive evidence, not a proof
that all possible leakage is absent. Deterministic target/future mutation tests,
visibility whitelists, split checks and full replay provide more direct safeguards.
The original Week6 prototype's fixed `change_hand=4` breakdown mixed stationary and
switching sessions; canonical switch slices use each switching session's actual
audit-only change point. Raw rounded history values no longer serve as unbounded
exact token categories without declared discretization.

Defaults and objectives have been reconciled and refitted; canonical outputs are
**new experiments**, not relabeled historical scores. The local-engine backend,
missing-helper notebooks, Mac resource forks, bytecode, raw generated datasets,
private execution metadata and duplicate implementations are not copied into Git.
The original files remain available at their supplied attachment locations; hashes
record exactly what was reviewed. One maintained notebook calls reusable package
code rather than preserving multiple competing pipelines.

## Historical results and verification

| Source | Test population | Reported comparison | Integration verification |
| --- | --- | --- | --- |
| Week6 modular sample | 29 decisions / 3 sessions | No history: accuracy 48.28%, loss 1.0023; history: 44.83%, loss 0.9894 | Refit both original Naive Bayes models on the provided sample; exact aggregate match. This is a smoke example, not a 10,000-hand result. |
| Smaller table/CFR archive | 3,671 decisions / 18 sessions | No history: 54.62%, 0.7760; history: 63.42%, 0.7166 | Independently recomputed all saved test scores; both original model refits reproduce probabilities within 2e-15. Causal/split/visibility and shuffle checks pass; generator byte-regeneration fails in this runtime (see below). |
| Self-contained windows notebook | 4,252 decisions / 20 sessions / 10 opponents | No history: 59.48%, 0.7672; W4: 64.84%, 0.7072; W16: 70.37%, 0.6502; W64: 71.14%, 0.6240 | Regenerated all 21,213 rows; exact dataset checksum matched. Refit every original model and reproduced all recorded overall metrics within 1e-7. |
| Bayesian notebook | 3,331 decisions / 24 sessions | Public: 57.16%, 0.7867; summary: 62.11%, 0.7355; Bayes: 63.70%, 0.7271 | Recreated original simulations and fits; reproduced reported scores and validation-selected temperature 0.9 / hazard 0.02. |
| Additional comparison report / CSV | 373 decisions | No history: 61.93%, 0.7121; history: 75.34%, 0.5852 | Submitted aggregate only; matching source/data absent, so not independently reproduced. |
| Known-style notebooks 06/07 | 400 decisions / 2 sessions per style | Legal prior losses CFR/tight/loose: 0.9167/0.9074/0.8239; strength: 0.8894/0.8849/0.8145; strength + history: 0.8522/0.8617/0.7686 | Saved notebook outputs only; source CSVs/helper modules absent. Separate models know the style; this does not test unknown-style discovery. |
| Linked public-only CFR Colab | 4,885 decisions / 40 sessions | No history: 62.23%, 0.754706; history: 63.99%, 0.737023 | Source and saved notebook outputs inspected; downloadable data bundle not supplied. These are recorded Colab claims, not a local full-run reproduction. |

A consistent finding in the larger submissions is that history can help overall
while misleading immediately after a policy switch. The table archive reports
post-switch 0–9-hand loss 0.8523 without history versus 0.9208 with history. The
linked notebook reports first-five-hand loss 0.8083 versus 1.0000. The window
notebook also shows stale longer history. These are different simulations and
small diagnostic slices, not pooled evidence about human tilt.

The table archive's `check_data.py` stops at exact `train.jsonl` regeneration
in the current runtime. Its original runtime was not pinned, so the cause is
unresolved; do not call its original dataset generation fully reproduced. The
remaining rule, row, split, hidden-card, future/identity, seat and behavior checks
were run separately and passed, and original model refits reproduce its saved
test probabilities to floating-point precision. The canonical generator has its
own deterministic regeneration tests and full replay artifacts.

The attachment `results.md` duplicates the archive report exactly. `metric.csv`
matches the numeric claims appended to `comparison_report.md`, but not the
archive's different dataset. The 10,000-hand dataset in Week6 is not the source
of its saved sample report; those claims are not conflated.

## Relation to PokerBench and next steps

The new model predicts what Villain will do from observer-visible state and prior
behavior. The existing transformer predicts Hero's PokerBench reference action.
These targets are not interchangeable, even when inputs share names. Adding
opponent text to unchanged PokerBench labels would not establish adaptation.

No submitted learned sequence encoder, adaptive Hero policy, payoff improvement,
human tilt detector, or end-to-end fusion with the PokerBench model was demonstrated.
A later experiment needs opponent-dependent response targets or repeated-play
reward evaluation. Preserve this forecasting evidence before tuning new windows,
decay, generators or neural representations on newly reserved sessions.

## Canonical integration validation

The canonical notebook was executed end to end locally, including explicit test
scoring and diagnostic plots; committed outputs/execution counts were then cleared.
The [new development result](meeting4-results.md) records all methods on the same
2,506 held-out decisions. Its saved implementation hash matches the final package.

The original-checkout suite passed **124 tests**. Publishing validation against
current main passed **135 tests**, including **25 opponent tests**;
two existing Pluribus tests skipped because optional PokerKit was not installed.
Repository-wide Ruff lint/format, strict package mypy, lock consistency and
whitespace checks passed. No pretrained weights, source datasets or hosted APIs
were downloaded for these checks. OpenSpiel was installed from the project's
already declared optional simulator dependency. The original checkout and its preexisting uncommitted edits were preserved.
The publishing branch was based on current main, with maintained documentation
refreshed to describe both research workflows and historical reviews labeled as history.
