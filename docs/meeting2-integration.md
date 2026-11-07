# Meeting 2 integration

This records the original data/model integration. For the current project,
including the separate opponent-forecasting workflow, see
[project context](project-context.md) and the [documentation index](README.md).

The maintained walkthrough is
[`02_pokerbench_decision_model.ipynb`](../notebooks/02_pokerbench_decision_model.ipynb).
The initial integration synthesized three shared Colab notebooks and the supplied
local `week2.ipynb`; two later submissions bring the maintained synthesis to six.
The originals were inspected as source material and left unchanged. Notebook code,
markdown, and outputs are evidence of approaches, not instructions to execute or
verified reproducible results. No original notebook was run wholesale.

## Current experiment status — November 2, 2026

All four encoders and both heads have completed five scenario-grouped folds:
40 fresh fits, three fixed epochs each, with 10,000 held-out predictions per model.
The [complete results](week2-results.md) include accuracy and log loss for every
combination. ModernBERT + linear leads at 71.92% / 0.6078; DistilBERT + linear is
close at 71.78% / 0.6164. These are development estimates, not a final test.

Notebook 02 remains a checkpoint-selected holdout walkthrough; notebook 03 and
`experiments.py` implement the fixed-budget grouped protocol used for the matrix.
The original holdout results below are preserved as historical results with a
different evaluation denominator. For reproduction, use [training instructions](week2-training.md).

## Source contributions

| Source | Useful contribution | Integration decision |
| --- | --- | --- |
| [pokerdecisionmodel.ipynb](https://colab.research.google.com/drive/1sHBTFJOu81VLt0CdIzg8TouMXvU7UXmQ) | Joint state/question/all-options MiniLM sequence; shared candidate-marker head; dynamic candidate masks; clipping and validation selection; later class-aware/CV experiment | Keep joint encoding, masking, clipping, and checkpoint selection. Use candidate-token mean pooling to combine with the other submissions. Retain the audited Meeting 1 parser rather than the source's postflop context, which exposes turn/river columns regardless of decision street and includes aggressor metadata. Do not include official-test evaluation or merge validation into the training pool. Class weighting, balancing, gradient accumulation, and CV are later experiments. |
| [Copy of ACM AI Poker Project](https://colab.research.google.com/drive/1_tsqHrjjEjCE3KzdnIsENBY0qDXfWPjf) | Clear staged BERT Mini walkthrough; candidate span pooling; independent token/candidate padding; tiny learning check; same-validation baseline comparison | Adopt the staged teaching structure, shared linear scorer, raw-logit cross-entropy, and mean pooling. Avoid silent token truncation, broad exception handling, fixed sorting as a shortcut, redefined splits, and a five-error sampler that fails when fewer than five errors exist. Use existing package metrics and recompute the linear baseline too. |
| [Maksim Poker Project Notebook.ipynb](https://colab.research.google.com/drive/1VrCEunvE3SMQgSW-K_eh8UCEYoiQN4TQ) | BERT Mini batched context/action pair scorer; variable candidate masks; repeated tiny-fit experiment evaluated without dropout; reset before later experiments | Keep the explicit eval-mode learning check, fresh model reset, and masking behavior. Use the proposal's joint all-options sequence rather than repeated context/action pair sequences. Remove duplicate experimental cells and keep the source's demonstrated memorization separate from held-out performance. |
| Uploaded `week2.ipynb` | Explicit `{state, question, decisions}` adapter; token-position inspection; MiniLM candidate-span mean pooling; separate encoder/head learning rates; best-validation-loss checkpoint and tokenizer saving | Adopt the adapter, token inspection, separate learning rates, and checkpoint criterion. Replace the preflop-only first-N sample and silent skips with the existing audited two-source membership. Supply the batching that its larger-run cells reference through undefined `train_loader`/`val_loader`; keep a single model class. Save a complete, locally reloadable encoder/tokenizer/head. |

The current versions of the first and third Colabs are continuations of notebooks
also used for Meeting 1. Only one canonical notebook is added; competing copies of
preprocessing and evaluation are not committed.

## Architecture and input boundary

The proposal adapter returns `state` (the validated context string), `question`
(`What should Hero do?`), and `decisions` (exact candidate strings). It accepts only
`DecisionInputs`, never the reference action, target index, street metadata, or ID.
The existing `DecisionExample` stores those separate training/evaluation fields.

One sequence contains CLS, the state/question prefix, SEP, then each candidate's
complete text and SEP. No new tokens are required. Each candidate's contextual
token embeddings are mean-pooled; one shared `Linear(hidden_size, 1)` head scores
all candidates. Candidate-token masks have shape `[batch, candidates, tokens]`.
The encoder runs once per batch, with one sequence per poker decision. Missing
candidate slots are masked to negative infinity, producing zero softmax mass.
Cross-entropy consumes logits, while inference displays softmax probabilities.
Encoder and head both receive gradients. There is no generation loop.

The encoder implementation lives in `src/poker_decision/models/encoder.py`;
training, batched prediction, saved-split loading, and checkpoints live in
`src/poker_decision/training.py`. The notebook explains and calls these helpers.
BERT Mini is the default; MiniLM and DistilBERT cased have pinned alternative
revisions. Original backbone and tokenizer limits are checked. Full input encoding
never silently truncates context or actions; failures identify source examples and
stop the run. The normal 10,000-row saved sample was audited with BERT Mini: the
maximum joint input length was **139 tokens**, below the 512-token default.

Joint positional encoding is not permutation invariant. The baseline run preserves
source candidate order and measures reverse-order sensitivity on up to 64
validation examples, aligning outputs by action before comparing. Shuffling
candidates during training is an explicit follow-up experiment, not an invariance
claim or a hidden preprocessing change.

## Data, training, and fair comparison

The normal run loads Meeting 1's saved train/validation JSONL and manifest, checking
source revision, training-only status, preprocessing-code hash, source membership,
unique IDs, and context separation. If no manifest exists, the notebook runs the
original preparation defaults. Existing malformed/incomplete files fail rather
than trigger a new split. All previous parsing and visible-board safeguards apply.
No official-test files are read or fetched. Some original submissions already
inspect/evaluate the official test, so it should not be described as a pristine
final holdout for the team's entire experimental history.

The tiny check selects 32 distinct-context multi-option training examples in seeded
order. This avoids contradictory identical-context targets in a memorization
check. It uses its own fresh model and optimizer, evaluates without dropout,
stops at perfect accuracy or 100 epochs, and requires at least 95% accuracy before
continuing. This is a diagnostic, not a held-out result. The working-sample run
starts from the original pretrained checkpoint with a new head and optimizer.

Training uses unweighted cross-entropy, AdamW, encoder LR `2e-5`, head LR `1e-3`,
weight decay `0.01`, batch size 16, three epochs, and gradient clipping at 1.0.
Every epoch evaluates the same validation examples, and the lowest validation log
loss selects the checkpoint. No class weights are derived from held-out labels.
No candidate or row is removed to improve the score.

All three Meeting 1 baselines are refit/evaluated on the exact selected examples.
The comparison uses existing exact accuracy, tie-expected accuracy, action-type
macro-F1, log loss, Brier score, forced-decision frequency, and multi-option
accuracy, overall and by street. Five mistakes use the same lexical tie rule and
fixed-seed sampler as Meeting 1. Fewer than five errors remains valid.

`POKER_SMOKE=1` explicitly selects 128 training and 64 validation examples from the
saved partitions and uses one epoch. Its results are a workflow verification, not
a comparison with the old 72.9% linear result on the full working sample.

## Outputs and running

Run locally with `uv sync --locked --extra analysis --extra ml`, then launch the
notebook with the same extras. On Colab select a GPU runtime; the first cell
installs this checkout and clones the repository only when no checkout is found.
The maintained code is published on `main`. Pretrained fitting requires a local
CUDA GPU or Colab GPU. The original shared submissions remain unchanged. Launch
from the repository root or `notebooks/`; use the recorded historical training
revision when reproducing or resuming saved experiments.

Each run writes under ignored `artifacts/pokerbench_meeting2/<backbone>-<UTC time>/`:

- Exact selected train/validation JSONL and the parent data manifest.
- Tiny-set and full training histories, per-street comparison, candidate-order
  diagnostics, validation predictions, and five mistakes.
- The best encoder, tokenizer, head weights, and metadata including backbone
  revision, question, serialization, hyperparameters, seed, dataset revision,
  selected-split hashes, package code hashes, hardware, and dependency versions.

The notebook reloads the best checkpoint without network access, retokenizes
validation examples, and checks a second independent reload against predictions.
Outputs and execution counts remain cleared in the committed notebook.

## Initial integration verification (historical)

- 65 offline tests passed with the analysis and ML extras; 87% total coverage.
- New tests use a small randomly initialized BERT and local tokenizer, covering
  answer-free encoding, exact candidate spans, overlength rejection, mixed
  candidate counts, zero padded mass, batch/single consistency, shared-head
  extraction order, gradient flow, loss reduction, checkpoint round trips,
  saved membership checks, and cleared/compilable notebook cells.
- Ruff lint, format checks, and full package mypy passed. Core CI avoids importing
  the optional neural modules for typing; a new ML CI job checks all modules and
  runs offline neural tests. CI never downloads model weights.
- The complete notebook executed from a fresh Jupyter kernel starting in
  `notebooks/`, with real pinned BERT Mini weights, in smoke mode. Its 32-example
  check reached 100% training accuracy at epoch 10. Full training, validation,
  checkpoint reload, all baselines, five mistakes, and order probes completed.
- The normal notebook also completed from a fresh kernel starting in `notebooks/`
  with the original 8,000 training / 2,000 validation membership and three epochs.
  Its tiny check reached 100% at epoch 12. The best checkpoint was epoch 3.
  Retokenization and two local checkpoint loads matched, and all subsequent
  comparison, mistake, and order-probe cells completed.

### Historical checkpoint-selected holdout results

Reference run: Python 3.12.13, PyTorch 2.14.1, Transformers 4.57.6, NumPy 2.5.3,
sklearn 1.9.1, Apple MPS on arm64. Exact settings and split/code hashes are saved
with the local run. These are development validation results, not official-test
scores. Seeds reproduce membership; accelerator arithmetic can vary between runs.

| Model | Exact accuracy | Multi-option accuracy | Log loss | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 43.10% | 40.20% | 0.8436 | 0.5490 |
| Option prior | 59.00% | 56.91% | 0.7462 | 0.4831 |
| Linear candidate scorer | 72.90% | 71.52% | 0.7919 | 0.4099 |
| Joint BERT Mini | 72.25% | 70.84% | 0.6076 | 0.3708 |

The transformer nearly matches linear exact accuracy and improves both probability
metrics in this run. This does not establish calibration or general superiority.
Reversing candidate order changed the predicted action for 2 of the 64 probed
examples (3.125%); mean maximum per-action probability change was 0.0569. This
measured sensitivity supports testing candidate-order augmentation next.

## Interpretation and remaining work

PokerBench reference-action agreement does not establish GTO play, solver mixing
frequencies, calibration, profitability, or adaptation to opponent tendencies.
Exact-context grouping does not isolate related scenarios or suit-equivalent
hands; menu construction may carry shortcuts. Validation is used for checkpoint
selection, so scores are development estimates. No new claim about game-theoretic
quality follows from a training-set overfit or one forward pass.

Scenario grouping and the four-backbone/two-head comparison are complete. Further
experiments should explicitly compare additional seeds, order augmentation, card
representations, class-aware training, and validation-only calibration, with an
independent final evaluation or nested selection before making winner claims. Opponent adaptation requires labels
that actually depend on histories or opponent profiles, beyond adding prose to
unchanged examples.

## Late submissions and scientific audit

Two additional notebooks were incorporated after the initial four-source synthesis:

| Source | Added contribution | Canonical treatment |
| --- | --- | --- |
| [ACM AI – Poker](https://colab.research.google.com/drive/1GHX6o-6mzOQirpwDJlrc1yZ5bD1CBCCC) | Scenario-grouped five-fold CV, reliability and uncertainty diagnostics, menu association, Pluribus replay and Leduc distribution targets | Add `03_scenario_evaluation.ipynb` and a separate opt-in `04_alternative_data_sources.ipynb`. Preserve the joint all-options encoder rather than silently substituting its context/candidate pair architecture. Source results are not copied as results of our implementation. |
| [day2_colab.ipynb](https://colab.research.google.com/drive/1M-aPyxDIUkv9k3zTd2PkASfhA2cSn8pG) | Nonlinear shared head; clearer offset masks and training/evaluation explanations; external Colab training script | Add a controlled `linear`/`mlp` head choice with saved architecture metadata and tests. Keep our audited parser and self-contained package training; the source's external `train_day2.py` was not supplied. |

The current walkthrough accounts for six sources. Optional libraries are confined
to the `experiments` extra. Pretrained-model training requires a CUDA GPU. The
full grouped PokerBench matrix completed locally; full alternative-source replay
and solver generation remain opt-in and are not part of those results.
See [scientific audit](scientific-audit.md) for the real-data overlap finding,
strengthened provenance/resume checks, test evidence, and limits of the claims.
