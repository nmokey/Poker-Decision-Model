# Documentation

Updated November 13, 2026. The project combines a trained PokerBench decision
scorer, a controlled Leduc opponent-forecasting workflow and the live PocketSolver
demo. Use the task-based guides below; meeting notes preserve contribution history.

| Topic | Read this |
| --- | --- |
| Shared components, notebook connections and artifact migration | [Architecture](architecture.md) |
| Current data, results and scientific boundaries | [Project context](project-context.md) |
| Setup and executable walkthroughs | [Notebook guide](../notebooks/README.md) |
| All eight transformer/head results | [Decision-model comparison](week2-results.md) |
| Transformer reproduction and verified resume | [Training guide](week2-training.md) |
| Opponent histories, models and commands | [Forecasting guide](opponent-experiments.md) |
| Forecasting results and behavior changes | [Controlled Leduc results](meeting4-results.md) |
| Real Hold’em histories and decision-feature integration | [Hold’em workflow](meeting5-integration.md), [verified results](meeting5-results.md), [training-size comparison](meeting5-scaling.md) |
| Hold’em model/head, memory, format and training-seed comparisons | [GPU ablation protocol](meeting5-ablations.md), [recorded comparisons](meeting5-ablation-results.md) |
| Data leakage, metrics and scientific limits | [Scientific audit](scientific-audit.md) |
| Package development and checks | [Development](development.md), [tests](../tests/README.md) |
| Demo, serving catalogue and hosting | [Deployment](deployment.md) |
| Remaining research milestones | [Roadmap](roadmap.md) |

Reviewed aggregate metrics and protocols live under [results](results/). Raw
source rows, generated sessions, example-level predictions and weights remain in
ignored directories or checksum-pinned model release assets.

## Contribution history

- [Data pipeline reconciliation](meeting1-integration.md)
- [Transformer architecture reconciliation](meeting2-integration.md)
- [Demo submission review](meeting3-review.md)
- [Opponent experiment reconciliation](meeting4-integration.md)
- [Hold’em integration and member reconciliation](meeting5-integration.md)
- [Additional Week 7 study review](meeting5-additional-submission.md)
- [Original proposal](archive/original-proposal.md)

The decision-model and forecasting scores concern different targets and datasets.
The original comparisons use one seed. Selected Hold’em configurations also have
three training-seed repeats on a fixed inspected validation set. These results
do not establish profitable play or human psychological-state inference.
