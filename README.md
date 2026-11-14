# Poker Decision Model

A small transformer that scores supplied poker actions in one forward pass, plus
experiments that learn an opponent's tendencies from earlier play. Built by ACM AI
Fall 2026, with a live [PocketSolver demo](https://www.nmokey.com/poker/).

The project studies two connected questions:

1. Can a compact model imitate solver-derived decisions over changing action menus?
2. Does an opponent's observed history help predict their next action, especially
   when their behavior changes?

One shared Python package supports both tasks. The first workflow uses PokerBench and a pretrained encoder with a shared action
scorer. The second uses chronological Leduc sessions and statistical forecasting
models. A Pluribus Hold’em workflow now tests the same history summaries as extra
transformer inputs for observed human actions. Better-response or payoff claims
still require opponent-dependent response targets or repeated-play evaluation.

## How it works

| Workflow | Input | Target | Current implementation |
| --- | --- | --- | --- |
| Hero decision scoring | Poker situation, question, candidate actions | PokerBench reference action | Joint transformer encoding with a shared linear or MLP head |
| Hold’em history integration | Hero cards, public state and completed-hand opponent summaries | Observed human action category | Matched forecasters and BERT Mini fine-tunes, plus nested training-size comparisons |
| Opponent forecasting | Observer-visible situation and earlier opponent actions | Opponent's next observed legal action | Priors, Naive Bayes, logistic variants, equity controls, history windows and Bayesian style updates |

The decision scorer returns probabilities aligned to the supplied action menu,
including exact bet/raise amounts. It does not generate answer tokens. Reference
labels stay outside inference inputs. All candidate tokens share the same scoring
head; unsupported padding choices receive zero probability.

The forecasting workflow makes predictions before revealing the current action.
Opponent identities stay together across splits, histories reset by session, and
engine traces support replay checks. Five simulated policy families and explicit
visibility/history settings preserve different experimental approaches.

## Results so far

- **Decision scoring:** four pretrained encoders × two heads × five scenario
  folds, totaling 40 fits on the same 10,000-example working pool. ModernBERT with
  a linear head has the highest observed accuracy, **71.92%**, and log loss
  **0.6078**; DistilBERT linear is close at **71.78% / 0.6164**.
  [All model scores and provenance](docs/week2-results.md).
- **Opponent forecasting:** in the recorded controlled Leduc experiment, accuracy
  rose from **59.3% without history to 69.9% with a validation-selected history
  window**; a Bayesian filter reached **73.3%**. All methods used the same 2,506
  test decisions. History sometimes hurt immediately after a behavior switch.
  [All methods, uncertainty and limitations](docs/meeting4-results.md).

- **Hold’em integration:** on the same 1,024 validation decisions, increasing training
  from 128 to 4,096 examples raised transformer accuracy from 55.3% to 73.1%
  without history. The history version reached 72.5%; these summaries did not
  improve the transformer. [Verified learning curve](docs/meeting5-scaling.md),
  [workflow and source review](docs/meeting5-integration.md).

These are distinct, one-seed development experiments with different labels and
populations. Action agreement does not establish a GTO mixed strategy, calibrated
probabilities or profitable play. Simulated behavior changes do not establish
human tilt detection. See the [scientific audit](docs/scientific-audit.md).

## Get started

Use Python 3.12 (3.13 is supported) and [uv](https://docs.astral.sh/uv/).
For the data, evaluation and opponent-forecasting notebooks:

```sh
uv sync --locked --extra analysis --extra experiments
uv run --locked --extra analysis --extra experiments jupyter lab
```

Select the project kernel and choose a walkthrough:

| Notebook | Purpose |
| --- | --- |
| [PokerBench pipeline](notebooks/data_pipeline.ipynb) | Audit pinned source data, create input/target records and compare CPU baselines |
| [Transformer decision scorer](notebooks/decision_model.ipynb) | Tokenize state/question/options, train a shared head, save and reload checkpoints |
| [Scenario evaluation](notebooks/evaluation.ipynb) | Fixed scenario folds, controlled head comparisons and reliability/error diagnostics |
| [Alternative data sources](notebooks/data_sources.ipynb) | Separate Pluribus replay and Leduc teacher-policy experiments |
| [Hold’em history integration](notebooks/holdem_opponent_integration.ipynb) | Build causal Pluribus histories, compare matched forecasters and decision models |
| [Opponent forecasting](notebooks/opponent_modeling.ipynb) | Generate chronological sessions, compare history representations and inspect behavior changes |

The notebooks are task-based views of the package. Preparation feeds both decision
training and grouped evaluation; source and opponent adapters reuse the same input
interface with explicit task questions. [Architecture and artifact flow](docs/architecture.md)
explain these connections and how existing saved runs remain usable.

The same workflows are available without Jupyter (add the appropriate extras):

```sh
uv run --locked --extra analysis python -m poker_decision prepare
uv run --locked --extra analysis --extra ml python -m poker_decision train --backbone bert_mini
uv run --locked --extra analysis --extra ml python -m poker_decision evaluate --backbone bert_mini
uv run --locked --extra analysis --extra experiments python -m poker_decision opponents --help
uv run --locked --extra analysis --extra ml --extra experiments python -m poker_decision holdem --help
uv run --locked --extra analysis --extra ml --extra experiments python -m poker_decision holdem-scale --help
```

The PokerBench `train`/`evaluate` commands require CUDA. Hold’em `holdem compare`
and `holdem-scale` are bounded CPU workflows. `evaluate` runs the fixed scenario
comparison; `train` uses validation loss to select an introductory holdout checkpoint.
Use `python -m poker_decision TASK --help` for that workflow's options.

Pretrained transformer training also needs `--extra ml` and a local CUDA GPU or
Colab GPU. Opponent forecasting runs on CPU. No API key is needed for either
training workflow. See [notebook setup](notebooks/README.md),
[transformer reproduction](docs/week2-training.md), and
[opponent experiment commands](docs/opponent-experiments.md).

The PokerBench pipeline downloads only checksum-pinned training CSVs, audits all
63,200 preflop and 500,000 postflop rows, then uses 5,000 examples per source by
default. The official test is not loaded. Scenario-grouped evaluation is stronger
than the introductory context split; both remain development evidence. Raw data,
checkpoints and example-level generated outputs stay in ignored directories.

## PocketSolver

The [live demo](https://www.nmokey.com/poker/) serves all eight trained encoder/head
combinations and defaults to ModernBERT + linear. It compares predictions with a
live Groq model on the same 256 hands held out by every serving checkpoint. The
results table shows saved development scores and separately measured CPU latency;
click a row to select that model. One model stays resident in server memory.

Groq calls require a key and have public request limits. The current demo displays
PokerBench decision predictions; the opponent-forecasting experiment runs in its
own notebook and command-line workflow. [Deployment and local serving](docs/deployment.md)
cover assets, setup, model selection and measurement boundaries.

## Development

Core parsing and metrics need only the base environment. Optional extras separate
analysis, neural models, simulation, the demo and hosted LLM calls:

```sh
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy --exclude 'src/poker_decision/(training|cross_validation|experiments|models/encoder)\.py$'
```

For all offline model and simulator safeguards:

```sh
uv sync --locked --extra analysis --extra ml --extra experiments
uv run --locked --extra analysis --extra ml --extra experiments pytest
uv run --locked --extra analysis --extra ml --extra experiments mypy
```

Tests use synthetic fixtures and tiny randomly initialized encoders, with no
pretrained downloads or live API calls. Reviewed aggregate evidence is versioned
under `docs/results/`. Start with the [documentation index](docs/README.md),
[development guide](docs/development.md), [test guide](tests/README.md), or
[current roadmap](docs/roadmap.md). Historical contribution reviews remain linked
from the documentation index.
