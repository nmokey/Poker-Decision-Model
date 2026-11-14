# Meeting 5: matched GPU history ablations

This follow-up tests whether the earlier history result depends on the encoder,
head, history window, text format, training duration, or opponent assignment.
It preserves the verified 4,096 training and 1,024 validation decision IDs, their
order, targets, current states, menus and session groups. The validation set has
already been inspected; all selection here is development analysis.

## Declared experiment sequence

1. Compare BERT Mini and DistilBERT, each with its published linear and MLP head,
   with and without recent-64 history: eight independent fine-tunes. Each pair
   starts from identical weights and settings. Select the architecture with the
   smallest history-minus-control validation log-loss change.
2. On that architecture, compare recent-16 and recent-256 history; explicit
   fold/call/raise field names with the same values; a profile-assignment control;
   and three epochs with and without history: six additional fits. Keep all other
   settings fixed. The assignment control rotates the already-visible profiles
   among active opponent seats within each decision. It preserves their values,
   uses no donor decisions and never accesses future actions or other splits.
3. Select the real-history configuration with the smallest paired log-loss
   change, excluding the assignment control. Freeze it and repeat its history
   and no-history pair with training seeds 43 and 44: four additional fits.
   Sampling and split membership remain fixed at seed 42 throughout.

The default plan has 18 fits. The two selection decisions and all candidates
are recorded, including configurations that do not help. Seed repeats assess
training variability on this split; they are not an independent final test.

## Settings and execution

[The configuration](../configs/holdem_ablations.json) declares encoders, heads,
windows, epochs, seeds, learning rates and memory settings. The default uses an
effective batch size of 16, microbatches of two with weighted gradient accumulation,
encoder LR 2e-5, head LR 1e-3, weight decay 0.01 and float32 CUDA training.
Every partial batch is weighted by its actual size. The offline equivalence test
compares updates with a full batch using dropout disabled. With dropout enabled,
microbatch runs constitute a new protocol; the original CPU learning-curve fits
are preserved separately.

```sh
uv sync --locked --extra analysis --extra ml --extra experiments
python -m poker_decision holdem-ablate prepare \
  --out artifacts/holdem_ablations \
  --reference-out artifacts/holdem_scaling/train_04096
python -m poker_decision holdem-ablate suite \
  --out artifacts/holdem_ablations --gpu 0
python -m poker_decision holdem-ablate report --out artifacts/holdem_ablations
```

`--checkpoint-root` selects cached published asset directories. Missing bundles
are downloaded from the existing release catalog; manifests and checkpoint hashes
are verified. `prepare` refuses an existing output directory. `suite` can resume
completed fits only with unchanged source and sample fingerprints. It requires
CUDA and does not fall back to CPU. A failed worker stops the suite and leaves
its log available for diagnosis.

Replay uses only completed, co-observed hands and the same directory boundaries
as the original experiment. The default window is checked against the original
full paired rows, and every window must retain the exact base state, targets and
candidate menus. Windows count opponent actions, not hands or matched examples.
Explicit field names preserve the numeric values. All inputs must fit the model's
budget without truncation or dropped examples.

Ignored artifacts contain the fixed samples, released-checkpoint provenance,
individual worker logs, saved fine-tunes, validation probabilities and a summary.
Every saved model is reloaded and re-encoded to verify probabilities and tokenizer
behavior. The report independently recomputes each aggregate from probabilities,
reports street/history-availability slices and paired session-cluster intervals.
The intervals condition on the saved fits and do not correct for adaptive
configuration selection. Only aggregate evidence is intended for version control.

The earlier [learning curve](meeting5-scaling.md) remains a separate experiment.
This study still predicts observed human actions; it does not measure improved
strategic responses or poker profit. Numeric feature fusion is a possible later
architecture change and is outside this matched text-feature sweep.
