# Documentation

Updated November 2, 2026. The Week 2 matrix is complete: four pinned pretrained
encoders × two heads × five scenario-grouped folds, with three fixed epochs per
fit. Every model has 10,000 held-out predictions. PocketSolver offers all eight
trained fold-1 checkpoints and defaults to ModernBERT + linear. Its shared
256-hand demo sample is distinct from the aggregate comparison.

| Topic | Read this |
| --- | --- |
| All eight accuracy and log-loss scores | [Completed results](week2-results.md) |
| Aggregate metrics and reproducibility evidence | [CSV](results/week2-development.csv), [JSON](results/week2-development.json) |
| GPU setup, commands, and verified resume | [Training instructions](week2-training.md) |
| Leakage safeguards and interpretation limits | [Scientific audit](scientific-audit.md) |
| Environment and checks | [Development](development.md), [tests](../tests/README.md) |
| Data and original baseline integration | [Meeting 1](meeting1-integration.md) |
| Neural architecture and historical holdout | [Meeting 2](meeting2-integration.md) |
| Current demo and model assets | [Deployment](deployment.md) |
| Historical submission review | [Meeting 3 review](meeting3-review.md) |
| Remaining work and original plan | [Roadmap](roadmap.md), [project context](project-context.md) |
| Maintained walkthroughs | [Notebooks](../notebooks/README.md) |

ModernBERT + linear has the highest observed accuracy (71.92%) and lowest log loss
(0.6078). DistilBERT + linear is close (71.78%, 0.6164). These are one-seed
development comparisons, not an independent final estimate or a claim of GTO play.
