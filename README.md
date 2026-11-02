# Poker Decision Model

ACM AI Fall 2026. A context-conditioned poker decision model project, starting with
an audited PokerBench pipeline and CPU baselines over variable candidate actions.

## Run Meeting 1

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --locked --extra analysis
uv run --extra analysis jupyter lab notebooks/01_pokerbench_pipeline.ipynb
```

Select the project environment, restart the kernel, and run all cells. Or run the
same workflow from the repository root without Jupyter:

```sh
uv run --extra analysis python -m poker_decision.pipeline
```

This downloads only the pinned preflop and postflop **training** CSVs (about 90 MB),
checks their hashes, inspects and validates every row, and selects 5,000 examples
per source for a quick CPU run. It saves context-grouped train/validation subsets,
provenance, exact-candidate metrics, and five mistakes. The official test set is
never loaded. No GPU or API key is needed.

Use `--kind preflop` for the smaller source, `--skip-baseline` for data preparation,
or `--sample-per-kind 0` to use all validated rows. Outputs live in ignored
`data/processed/pokerbench_meeting1/` and `artifacts/pokerbench_meeting1/`.

Inputs are `DecisionInputs(context, candidates)`; labels are stored separately as
`target_index`. Exact action amounts are preserved. Postflop contexts include only
cards already dealt. See [integration notes](docs/meeting1-integration.md) for the
source notebook comparison, data contract, baseline design, and limitations.

## Run Meeting 2

The canonical [Meeting 2 notebook](notebooks/02_pokerbench_decision_model.ipynb)
combines four submissions into a joint state/question/options transformer, shared
candidate head, required tiny-set learning check, training, checkpoint reload, and
same-split comparison with all three CPU baselines.

```sh
uv sync --locked --extra analysis --extra ml
uv run --locked --extra analysis --extra ml jupyter lab notebooks/02_pokerbench_decision_model.ipynb
```

BERT Mini is the pinned default; MiniLM, DistilBERT cased, and ModernBERT-base are alternatives. A GPU is
recommended; CPU works for small experiments. It reuses the saved Meeting 1 split
or prepares it if absent. Nothing is silently truncated or removed. Outputs go to
an individual ignored `artifacts/pokerbench_meeting2/` run folder. For a smaller
end-to-end check, launch Jupyter with `POKER_SMOKE=1`; this uses 128 training and 64
validation examples and recomputes the baselines. See the notebook for Colab setup
and [integration notes](docs/meeting2-integration.md) for attribution and limits.

## Stronger evaluation and optional sources

[Scenario evaluation](notebooks/03_scenario_evaluation.ipynb) adds five-fold scenario
splits, a controlled linear/MLP head comparison, reliability/error diagnostics,
paired grouped intervals, and verified resumable outputs. Review the protocol,
then enable `RUN_CV` on a Colab GPU. Every fold starts fresh; held-out folds never
select epochs. This remains development evaluation on an inspected pool.

[Alternative sources](notebooks/04_alternative_data_sources.ipynb) keeps Pluribus
observed-play replay and Leduc solver-policy generation separate from PokerBench.
Install `--extra experiments` for those opt-in sections. Teacher exploitability
is not evidence of a learned model's exploitability.

Pretrained-model training supports local CUDA GPUs and Colab. See the
[Week 2 GPU matrix](docs/week2-training.md) for all four encoders and both heads.
See [saved development results](docs/week2-results.md) for completed metrics and run status.
Offline checks use data audits and small randomly initialized models. See the [scientific audit](docs/scientific-audit.md)
for verified safeguards, current limitations, and the original split's scenario overlap.

## Development

Python 3.12 is the team default; 3.13 is supported. Core parsing and evaluation
need no optional ML dependencies. Install extras as needed: `analysis` (pandas,
sklearn, JupyterLab, plots), `ml` (torch, transformers), `demo` (Streamlit), and
`llm` (OpenAI SDK). Keep extra flags on `uv run` to retain those dependencies.

```sh
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy --exclude 'src/poker_decision/(training|cross_validation|experiments|models/encoder)\.py$'
# With ML dependencies installed, check all modules:
uv run --locked --extra analysis --extra ml mypy
```

The learned-baseline tests run when the analysis extra is installed; CI checks
the core environment, learned baseline, and offline neural contracts. Raw data and
generated reports stay out of Git.

## Layout

```text
src/poker_decision/   data contract, preprocessing, baselines, evaluation, workflow
notebooks/           consolidated Meeting 1 walkthrough
apps/                demo placeholder
tests/               offline parsing, leakage, splitting, metrics, model tests
data/raw/            local source training CSVs (ignored)
data/processed/      generated datasets, failure reports, manifest (ignored)
artifacts/           generated metrics and mistakes (ignored)
configs/             future experiment settings
docs/                project context, development notes, integration decisions
```

See [project context](docs/project-context.md), [development guide](docs/development.md),
and [roadmap](docs/roadmap.md). Neural training is introduced in Meeting 2;
opponent adaptation and the demo remain future work. Scores measure reference-action agreement.
