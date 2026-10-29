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
are guarded for Colab. Validation-only default paths are also executed in fresh
kernels. These are implementation checks, not proof of GTO play or absence of all
possible dataset biases. See [scientific audit](../docs/scientific-audit.md).
