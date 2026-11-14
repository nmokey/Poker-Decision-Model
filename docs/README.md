# Documentation

Updated November 13, 2026. Start with the **[results index](results/README.md)**
for the project’s findings, study boundaries and aggregate evidence. Meeting notes
preserve contribution history; task guides describe the maintained workflows.

## Project and workflows

| Need | Guide |
| --- | --- |
| Current implementation and how the studies connect | [Project context](project-context.md), [architecture](architecture.md) |
| Notebook setup and walkthroughs | [Notebook guide](../notebooks/README.md) |
| PokerBench transformer reproduction and verified resume | [Training guide](week2-training.md) |
| Leduc histories, forecasters and commands | [Forecasting guide](opponent-experiments.md) |
| Pluribus histories and matched transformer integration | [Hold’em workflow](meeting5-integration.md) |
| Hold’em architecture, history and training-seed experiments | [GPU ablation protocol](meeting5-ablations.md) |
| Commands, dependencies and checks | [Development](development.md), [tests](../tests/README.md), [configurations](../configs/README.md) |
| Demo, model catalogue and hosting | [Deployment](deployment.md) |
| Claim boundaries and remaining research | [Scientific audit](scientific-audit.md), [roadmap](roadmap.md) |

## Results and evidence

The [results index](results/README.md) separates the PokerBench comparison, serving
latency, controlled Leduc forecasting, and three Hold’em experiment stages. Each
study links its report, protocol and machine-readable evidence. An additional
submitted notebook is reviewed separately because its scores were not reproduced.

Raw source rows, generated sessions, per-decision predictions and weights remain
in ignored directories or checksum-pinned model release assets. Historical hashes,
scores and verification counts remain tied to their recorded revisions.

## Contribution history

- [Data pipeline reconciliation](meeting1-integration.md)
- [Transformer architecture reconciliation](meeting2-integration.md)
- [Demo submission review](meeting3-review.md)
- [Opponent experiment reconciliation](meeting4-integration.md)
- [Hold’em integration and member reconciliation](meeting5-integration.md)
- [Additional Week 7 study review](meeting5-additional-submission.md)
- [Original proposal](archive/original-proposal.md)
