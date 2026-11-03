# Tests

Run `uv run --locked pytest` for core parsing and metrics. Add `--extra analysis`
for learned baselines and grouped split/statistical checks, `--extra ml` for tiny
random-BERT and CV tests, and `--extra experiments` for PokerKit/OpenSpiel controls.
CI includes all extras in its ML job. No test downloads data or pretrained weights.

```sh
uv run --locked --extra analysis --extra ml --extra experiments pytest
```

Scientific regressions cover label/future-card exclusion, exact-context and scenario
isolation, duplicate/conflicting labels staying together, complete held-out coverage,
training-only fit membership, fresh model/optimizer initialization, known metric and
reliability values, whole-cluster paired resampling, deterministic folds, and saved
split/prediction/checkpoint corruption. Validation inference cannot update weights
or accumulate gradients. Both heads round-trip through portable checkpoints.

Source controls change hidden opponent cards and confirm identical acting-player
inputs. Pluribus hand/context connected components cannot cross splits, replay
failures are recorded, and Leduc keeps teacher distributions, argmax labels and
sampled actions distinct. Its tiny API test does not train a neural model.

Notebooks are checked for syntax, schema and cleared outputs; new training sections
require a CUDA GPU when enabled. Disabled default paths were also verified in
fresh kernels during the original notebook audit. These are implementation checks,
not proof of GTO play or absence of all possible dataset biases. See [scientific audit](../docs/scientific-audit.md).

## CI environments and experiment evidence

Core CI runs on Python 3.12 and 3.13 without optional training dependencies. Its
mypy command excludes `training.py`, `cross_validation.py`, `experiments.py`, and
`models/encoder.py`; the ML job installs the ML/analysis/experiments extras and
checks all package modules. A local full-ML pass does not substitute for verifying
the core-only environment. Follow the commands in the [root README](../README.md).

The [completed experiment report](../docs/week2-results.md) is separate from the
offline suite: 40 pretrained encoder fits with saved checkpoints and held-out
predictions. Completion verification checked checkpoint/prediction hashes,
recomputed fold and pooled metrics/diagnostics, and confirmed 10,000 held-out
predictions per model. Unit-test or coverage counts do not establish model quality.
