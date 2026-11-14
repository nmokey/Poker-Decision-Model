# Poker Decision Model

A compact transformer that scores supplied poker actions in one forward pass,
plus experiments that predict opponents’ actions from earlier play. Built by
ACM AI Fall 2026. Try the [PocketSolver demo](https://www.nmokey.com/poker/).

The project connects three studies through one Python package:

| Study | What we built | Result |
| --- | --- | --- |
| PokerBench decision scoring | Four encoders × linear/MLP heads, evaluated on five scenario folds | ModernBERT linear: **71.92% accuracy**, **0.6078 log loss** |
| Leduc opponent forecasting | Causal histories, matched statistical models and Bayesian updates | Matched accuracy: **59.26% → 69.87%** with history |
| Pluribus Hold’em integration | Opponent summaries added to the existing transformer, followed by 22 GPU fits | Modest, configuration-dependent gains; neither accuracy nor log loss improves across every repeated seed |

See the **[results index](docs/results/README.md)** for protocols, complete tables
and verified evidence. These studies predict different targets. Hold’em injects
history summaries and predicts observed human actions; it does not establish
better strategic decisions or poker profit.

## Get started

Use Python 3.12 (3.13 is supported) and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked --extra analysis --extra experiments
uv run --locked --extra analysis --extra experiments jupyter lab
```

Choose a walkthrough in the [notebook guide](notebooks/README.md): data preparation,
decision modeling, scenario evaluation, alternative sources, opponent forecasting
or Hold’em integration. Add `--extra ml` to both commands for neural work.
PokerBench training and Hold’em GPU ablations require CUDA; the small Hold’em
integration and statistical forecasts run on CPU. Research notebooks need no API key.

The same workflows run through `python -m poker_decision TASK --help`.
See the [command guide](docs/development.md#commands-and-checks) and
[GPU ablation protocol](docs/meeting5-ablations.md) for reproduction.

## Demo and development

PocketSolver serves all eight PokerBench encoder/head combinations, defaults to
ModernBERT linear, and optionally compares them with Groq on shared held-out hands.
It displays development quality scores alongside separately measured CPU latency.
Hold’em history models remain research artifacts. [Run or deploy the demo](docs/deployment.md).

```sh
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Tests use synthetic fixtures and tiny random encoders without pretrained downloads
or live API calls. Raw data, predictions and weights stay in ignored directories;
reviewed aggregates live under `docs/results/`.

[Documentation](docs/README.md) · [Architecture](docs/architecture.md) ·
[Development and full checks](docs/development.md) · [Tests](tests/README.md) ·
[Scientific limits](docs/scientific-audit.md)
