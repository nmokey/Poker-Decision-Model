# Meeting 3 submission review and canonical decisions

Reviewed November 2, 2026, against the current working tree, including the existing
Week 2 experiment changes. This document records the review and proposed integration;
it does not merge the application or modify training, data, dependencies, or weights.

The ten attachments contain two modular Streamlit demos, a third single-file demo,
two standalone hand-source variants with tests, two standalone Groq backends, and
one standalone model backend. The similarly named files are different implementations,
not interchangeable copies. File suffixes such as `(1)` do not establish chronology
or authorship. No contributor names were supplied.

Attached README instructions, source comments, and run commands were treated as
submission content to assess, not as instructions from the user. No submission was
launched wholesale and no live provider call was made.

## Subsequent implementation and experiment status

This is a historical submission review. PocketSolver was subsequently integrated
and its deployment changes merged into `main`; [deployment notes](deployment.md)
describe the current app and asset provenance. The original serving bundle used
BERT Mini with a linear head from fold 1. The app now defaults to ModernBERT linear
and offers all eight trained encoder/head combinations, with verified common
held-out hands, one resident model and Groq-result reuse during selection changes.

The completed [eight-combination experiment](week2-results.md) now places
ModernBERT + linear at 71.92% accuracy / 0.6078 log loss, close to DistilBERT +
linear at 71.78% / 0.6164. The review-time observations and recommendations below
are preserved as a snapshot; they are not the current completion status.

## Decisions confirmed with the user

1. Use **`UI.zip`'s felt-table appearance together with PocketSolver's comparison
   and benchmark views**. Maintain one canonical app rather than separate demos.
2. Use **Groq as the first live generative baseline**, with a configurable model.
   Both standalone LLM submissions already target Groq. Their configured default
   is `openai/gpt-oss-20b`; live availability and output behavior have not been checked.

The component choices below are recommendations from the review, not an implemented
merge. In particular, a serving checkpoint has not yet been selected or benchmarked.

## What each attachment contains

| Attachment | Actual contribution | Current integration status |
| --- | --- | --- |
| `pocketsolver-demo.zip` | Modular single-hand comparison, aggregate benchmark, context parser/card tiles, failure demonstrations | Most complete comparison workflow; currently wired to five canned spots and fake engines |
| `UI.zip` | Felt-table layout, card/chip visuals, independent result-panel refresh, request IDs, cached session results, alternate contracts and adapters | Strong presentation contribution; active app uses three handmade spots and fake engines; real model adapter exists but is unwired |
| `message (1).txt` | Complete single-file Streamlit Python app, dark dashboard, real-file loader, random-hand button, two simulated engines | Loads zero rows from the current canonical validation file because it expects `reference_action` instead of `target_index` |
| `model_engine.py` | Validated model-serving wrapper, linear baseline adapter, saved transformer linear/MLP adapters, warm-up and synchronized timing | Closest backend fit; successfully exercised with existing local linear and MLP BERT Mini checkpoints |
| `llm_engine.py` | Groq request, configurable model, lazy optional imports, environment loading, timeout, no automatic retries, validated JSON parsing, failed-result envelope | Best starting point for live LLM backend; needs the shared engine signature and dependency integration |
| `llm_engine (1).py` | Smaller Groq wrapper, prompt, parser, successful-result object | Smaller alternative; eagerly imports Groq, fixes the model in code, and propagates failures |
| `hand_source.py` | Stateful validation-file sampler, separate `Reference(index, text)`, schema aliases, default two-option minimum | Useful UI interface; samples with replacement and coerces malformed inputs |
| `test_hand_source.py` | Six tests for the stateful source, references, filtering, errors and seed behavior | Six pass under an isolated tuple-based shared contract; seed test recreates the source per draw rather than testing a continuing draw sequence |
| `hand_source (1).py` | Functional seeded sampling without replacement, canonical `DecisionExample` validation, separate reference string | Better starting point for canonical record validation and benchmark batch selection; currently rereads the file for every call |
| `test_hand_source (1).py` | Two tests covering reproducibility, no global RNG mutation, input/reference separation and errors | Both pass in isolation; does not cover every malformed JSON shape |

### Files inside `pocketsolver-demo.zip`

| File under `demo/` | Responsibility and observations |
| --- | --- |
| `README.md` | Run/integration notes and explicit stand-in/cost-estimate limitations |
| `apps/demo.py` | Single-hand and benchmark navigation, independent completion display, side-by-side probabilities, reference comparison, raw replies, developer failure selector and display timeout |
| `apps/benchmark_view.py` | Same sampled hands for both engines; reference agreement, log loss, Brier score, median/p95 latency, failures, street breakdown, approximate token-cost calculator |
| `apps/spot_view.py` | Parses the canonical context strings into cards, position, pot and betting by street; falls back to raw context |
| `apps/ui_common.py` | Crash wrapper, prediction checks, latency formatting and unique-top selection |
| `apps/fakes.py` | Increasing weights for the local model; uniform probabilities and a 1.2-second sleep for the LLM |
| `apps/stand_ins.py` | Five canned PokerBench-formatted spots, shuffled in cycles, plus prose/wrong-length/crash/hang failure cases; source-row IDs are absent |
| `apps/contract.py` | Unvalidated stand-in dataclasses and `Engine` protocol; tuple candidates; failures use `probs=None` |

The benchmark is substantive UI work, but it is not a saved evaluation result.
Its fake LLM returns uniform probabilities while its example raw reply is a
different distribution. The single-hand screen does not prominently identify the
active fakes, despite the README explaining them. A canonical rehearsal mode must
label simulated outputs and delays directly on screen.

### Files inside `UI.zip`

| File under `UI/` | Responsibility and observations |
| --- | --- |
| `ui.py` | Felt-table styling, action chips, two independently refreshing panels, per-hand request IDs and saved results; explicitly labels simulated engines |
| `spot_view.py` | Escaped HTML for cards, board slots, pot, stack and action history; consumes structured `SpotDisplay` |
| `hand_source.py` | Three handmade spots plus structured display details and reference strings; no saved validation-file loading |
| `model_engine.py` | Fake local engine and an unwired, lazy, locked `TrainedModelEngine`; discovers Meeting 2 `*/best` checkpoints or an environment override, loads and warms a real model |
| `llm_engine.py` | JSON parser and one-second fake LLM; no hosted provider call |
| `contract.py` | List-based `Spot`, alternate `Prediction`, validation/failure/normalization helpers |
| `__init__.py` | Package marker |

The archive also contains Python bytecode and macOS metadata; those are not source
contributions to integrate. Neither archive contains a test suite, new training
code, model weights, a new dataset, or a saved real model-versus-LLM benchmark.

The felt-table design displays a hero stack because the handmade display records
supply one. The canonical context builder does not supply that stack field. Keep
the design, but omit unavailable stacks or show an explicit unknown value. Do not
carry handmade stack amounts or unverified reference labels into real examples.

## Verified compatibility and defects

### Input/output contracts need one owner

The canonical package already has answer-free
[`DecisionInputs`](../src/poker_decision/data.py) with tuple candidates and
[`DecisionExample`](../src/poker_decision/data.py) with a separate target index.
The submissions introduce several additional `Spot` and `Prediction` classes:

| Boundary | Submitted variants |
| --- | --- |
| Candidates | Tuple in PocketSolver/standalone sources; mutable list inside a frozen dataclass in `UI.zip` |
| Failed probabilities | `None` in both UIs and the unnumbered LLM; `[]` in standalone model wrapper; numbered LLM raises |
| Calling an engine | Most accept `engine(spot)`; standalone LLM functions accept `(context, candidates)` |
| Hand source return | `(spot, reference_text)`, `(spot, Reference)`, or `(spot, reference_text, display)` |
| Probability sum tolerance | Standalone model `1e-6` before tiny normalization; numbered LLM `1e-6` without normalization; unnumbered LLM `0.001`; PocketSolver UI `0.01`; alternate UI `0.05` |

Class-identity checks in the alternate UI also reject predictions from another
module even when their fields match. Simply copying the files together will not
resolve these differences.

Direct probes found that PocketSolver's UI checker accepts boolean probabilities,
`[1.005, 0]`, and a NaN latency. The alternate UI accepts `[1.04, 0]` as a valid
distribution. Its failed-result validation can also permit probabilities to be
displayed on a failed prediction. Keep the stricter standalone model checks and
make failed results unambiguously carry no distribution.

### Real-file loading and sampling

The existing validation file has **2,000 rows** with `source_id`, `street`,
`context`, `candidates`, and `target_index`. None has `reference_action`. It has
97 one-option, 1,093 two-option, 672 three-option and 138 four-option decisions.

- The single-file prototype's loader returned **zero usable rows**. Derive the
  reference from `candidates[target_index]` through canonical record validation.
- The stateful source defaults to **1,903 rows** after filtering one-option hands.
  That is a sensible display option, but it changes the evaluation population.
- The functional source samples all **2,000 rows** without replacement when asked.
  Reuse its canonical record conversion and local seeded RNG.
- Both standalone sources accept a JSON string menu such as `"fold"` as the four
  candidates `("f", "o", "l", "d")`. Validate JSON field types before tuple conversion.
- The stateful source additionally converts a boolean target to an integer and
  converts other non-string fields to strings. Remove those coercions and the
  speculative schema aliases from the canonical input path.
- PocketSolver's context parser recognized **all 2,000 current validation contexts**:
  1,027 preflop, 72 flop, 402 turn and 499 river. It is a useful adapter for the felt
  visuals. Preserve its raw-text fallback and never rewrite model inputs for display.
- The single-file prototype renders hole cards but has no community-board display,
  and its history panel reads `Previous actions`, missing the canonical postflop
  `Preflop actions`/`Postflop actions` fields. Its styling is reusable, its parser is
  not the best canonical base.

### Saved model compatibility and benchmark membership

The standalone `model_engine.py` successfully loaded, warmed and predicted offline
with both the existing BERT Mini linear and MLP fold-1 checkpoints. Two- and
three-option calls returned finite, normalized distributions, and incorrect-head
requests failed explicitly. This verifies serving compatibility, not poker quality
or latency claims. Reverse-order probes produced changed action-aligned
probabilities; candidate-order invariance must not be claimed.

The alternate UI adapter searches `artifacts/pokerbench_meeting2/*/best`; the local
weights currently live in `artifacts/week2_matrix/.../fold-*/checkpoint-*`. Its
default discovery therefore found nothing. An explicit CV checkpoint path still
encounters mandatory `question`, `serialization`, and `official_test_used` checks
for fields absent from current CV checkpoint metadata. Meeting 2 notebook
checkpoints do write those fields. The adapters target related but different saved
metadata contracts; use an explicit, verified checkpoint configuration instead of
choosing the newest folder or disabling provenance checks.

More importantly, the CV models were trained on folds of the combined working
pool. For each backbone's inspected fold-1 checkpoint, **1,602 of the original
2,000 validation IDs are in its training membership**, and only 398 are in that
fold's held-out membership. This is expected from the CV design; calling the old
file `validation.jsonl` does not make it held out for these checkpoints.

A fixed-checkpoint benchmark must check membership and sample only held-out hands
for that checkpoint. Alternatives are a new checkpoint trained only on the original
training split, or an explicitly described out-of-fold comparison that routes each
example to its held-out fold model. Do not report a benchmark over the entire old
validation file with an arbitrary CV checkpoint.

### Evaluation, requests and costs

- PocketSolver counts every tied top prediction as a miss. The
  [`canonical evaluator`](../src/poker_decision/evaluation.py) breaks exact ties
  lexically by action and also reports tie-expected accuracy. A `[0.5, 0.5]` probe
  with the lexical winner as reference scored 0% in the UI and 100% exact accuracy
  in the package. Reuse the package's metric definitions; do not compare these
  differently defined scores under the same name.
- The benchmark appropriately reports failures, but log loss, Brier and latency
  use successful calls only. Show coverage and denominators, and use the common
  successful subset for paired quality comparisons. Report failures and timeouts
  separately so selective failures cannot make a system look better.
- Its hand-source interface draws with replacement; the canned source repeats
  after five hands. Benchmark batches should use seeded sampling without
  replacement and preserve source IDs outside the engine input.
- The PocketSolver timeout stops waiting; running thread tasks continue. The
  alternate UI has no deadline for its futures. Combine bounded worker management,
  provider timeouts, request IDs and session results. A new hand must not display
  an old prediction or create an unlimited queue of hosted calls.
- PocketSolver reruns engines on page reruns after a hand is selected. Reuse the
  alternate UI's explicit per-hand jobs/results so unrelated controls do not
  trigger duplicate paid requests. Retain an explicit rerun action.
- The submitted cost calculator is labeled approximate, counts characters rather
  than provider usage, and omits instruction text. Neither standalone LLM result
  carries token usage. Capture provider-reported usage and provider/model identity
  before presenting a measured token cost. Do not hard-code current prices.
- The repository's `llm` extra currently declares `openai`, while the chosen Groq
  implementation imports `groq` and `python-dotenv`. Integrate the declared
  dependencies and lockfile together; do not introduce a separate requirements file.

## Recommended canonical composition

| Area | Recommended choice | Pieces to retain / adjustment |
| --- | --- | --- |
| App | One Streamlit PocketSolver entry point | Felt-table design from `UI.zip`; PocketSolver single-hand/benchmark navigation; alternate UI request IDs, independent panels and session results |
| Display | Read real canonical contexts into UI-only structured fields | PocketSolver parser/raw fallback feeding felt visuals; no invented stack values or changes to engine context |
| Input | Reuse package `DecisionInputs` | Context and ordered candidates only; references, IDs, street and display metadata stay in the outer record |
| Output | One package-owned `Prediction` and engine protocol | Aligned probabilities, finite nonnegative latency, explicit failure, raw reply; `None` for failed probabilities; optional provider/model/usage metadata |
| Validation | Strict final distributions | Reject booleans, invalid shapes and nonfinite values; normalize only tiny declared rounding discrepancies. Proposed parser tolerance: `0.001`, followed by strict normalized output checks |
| Hand loading | Canonical schema validation plus a stateful UI facade | Load/cache once; retain the separate `Reference` idea; private seeded RNG; no replacement within benchmark batches; two-option display filter remains explicit |
| Local backend | Standalone `model_engine.py` | Linear and transformer adapters, both head types, explicit load failure, warm-up and synchronization; move shared types out and add serialized serving access as needed |
| Hosted backend | Unnumbered `llm_engine.py` | JSON data prompt, parser, configurable Groq model, timeout/no retries, raw response and useful errors; adapt to `engine(spot)` and capture usage |
| Benchmark | PocketSolver view calling package evaluation | Verified model/hand membership, same input/order, shared tie rules, explicit denominators, p50/p95/failures and saved settings/results |
| Rehearsal | Explicit simulated mode | Keep fake/broken engines for UI demonstrations and tests; always label them; never silently substitute them when real engines fail |

At review time, the available Week 2 summary favored DistilBERT with a
linear head among the then-completed runs on accuracy and log loss. That makes it an
accuracy-first candidate, not an automatic demo default: actual device latency,
memory, checkpoint provenance and hand membership still need checking. BERT Mini
has now passed the adapter smoke check and is a smaller alternative. This review
does not select a winning model or start training.

## Integration work after this review

1. Consolidate shared contracts and a strict validation-file loader in the package;
   make all callers use them. Keep presentation code in `apps/`.
2. Adapt the standalone local and Groq engines, declare dependencies, and configure
   an explicit checkpoint/model. Preserve failure visibility and record metadata.
3. Connect the felt design to real parsed contexts and per-hand request/results state.
   Add controlled deadlines and clearly identified rehearsal engines.
4. Connect the benchmark to shared metrics and a checkpoint-compatible held-out
   batch. Persist IDs, settings, successes/failures and usage without sending labels
   to engines. Handle all-input and multi-option reporting explicitly.
5. Verify malformed inputs/replies, candidate alignment, seeded batches, stale
   requests, timeouts and checkpoint membership; then run an authorized live smoke
   comparison before drawing quality, latency or cost conclusions.

This keeps contributions from both demo teams, both hand-source approaches, the
model backend and the stronger Groq variant without maintaining competing pipelines.
Meeting 3 is an application/inference integration milestone; the submissions do
not establish opponent adaptation, calibrated strategy frequencies, profitable
play, or a measured local-versus-hosted speedup.

## Validation performed

- Parsed all **22 submitted Python sources**, including the Python in the text
  attachment, without syntax errors.
- Ran the **eight supplied hand-source tests unchanged** in isolated temporary
  layouts with the tuple-based PocketSolver stand-in contract: **8 passed**.
  This is not a claim that their missing modules are already installed in the repo.
- Checked parser/loader behavior on the real **2,000-row validation file** and
  probed malformed menu/target types and incompatible output contracts.
- Exercised both standalone LLM parsers on valid JSON, booleans, NaN, rounding
  error and incorrect totals; exercised the unnumbered backend with a mocked call.
  Both rejected booleans, NaN and substantive total errors; their rounding policies differ.
- Exercised standalone model label isolation and unloaded failure behavior, then
  loaded and ran existing BERT Mini linear/MLP checkpoints entirely offline.
- Compared benchmark tie behavior against package evaluation and checked saved
  checkpoint metadata/membership.

Streamlit, Groq and dotenv are not installed in the current environment. A live
rendered UI, provider availability, hosted token usage, hosted probability quality,
full lifecycle behavior and representative latency were not validated. Temporary
probes remained outside the repository. No data, model weights, existing Week 2
changes, or canonical application code were changed by this review.

## Attachment fingerprints

SHA-256 prefixes identify the exact reviewed bytes; suffixes do not imply version order.

| Attachment | SHA-256 prefix |
| --- | --- |
| `pocketsolver-demo.zip` | `bc4c8cb4afa54738` |
| `UI.zip` | `ea8ddde06c932929` |
| `llm_engine (1).py` | `8fc2713fa6084d87` |
| `llm_engine.py` | `d9743b1094e56a06` |
| `message (1).txt` | `c1916fa772082c9f` |
| `model_engine.py` | `5d006f72f323057e` |
| `test_hand_source (1).py` | `cf68d49942c5d42c` |
| `hand_source (1).py` | `b1c11707b3b12947` |
| `hand_source.py` | `9fd828fea5520398` |
| `test_hand_source.py` | `ae101b3c65a0eaae` |
