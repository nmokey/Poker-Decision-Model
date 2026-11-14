# Project context

Updated November 13, 2026. This is the current implementation brief. The
[original proposal](archive/original-proposal.md) preserves the early goals and
examples as historical planning material.

## Purpose

Study structured poker decisions with a compact model that scores supplied
alternatives directly, and investigate how observed opponent behavior can inform
future decisions. The system has connected PokerBench, Leduc and Pluribus workflows and a live
inference demo. See the [results index](results/README.md) for the study sequence
and evidence, and the [documentation index](README.md) for workflow guides.

## Decision scoring

The canonical input is `DecisionInputs(context, candidates)`. A separate
`DecisionExample` holds source ID, street and target index. The neural adapter adds
`What should Hero do?` and jointly encodes the state, question and complete action
menu. Candidate tokens are mean-pooled and a shared linear or MLP head produces
one logit per option. Encoder and head are trained together; candidate padding is
masked. Inference uses one forward pass without answer generation. Joint position
encoding makes the model sensitive to option order.

PokerBench provides solver-derived reference actions, including exact amounts.
The parser uses visible cards and source betting context, keeping labels and future
board cards out of inputs. Pinned training CSVs contain 63,200 preflop and 500,000
postflop rows; the default development pool samples 5,000 from each. No official
test data is loaded by the maintained workflows.

The introductory 8,000/2,000 context split is retained for teaching. Its related
scenario overlap limits generalization claims. Five fixed scenario-grouped folds
provide the stronger development comparison, with fresh fits and no held-out
selection of epochs. All four encoders and both heads completed 40 fits on the
same 10,000 rows. ModernBERT linear has the highest observed accuracy, 71.92%, and
log loss 0.6078; DistilBERT linear is close. These are inspected one-seed
comparisons, not an independent final estimate. [Results](week2-results.md).

## Opponent forecasting

`poker_decision.opponents` and the opponent-modeling notebook predict the opponent's next observed
legal action in two-player Leduc. These labels differ from Hero's PokerBench
reference decisions. The observer sees public state, current-hand actions and
optionally its own card. Opponent hidden cards, policy parameters, identity,
future actions, outcomes and change points are excluded from model features.

Five simulated policy families support passive/aggressive behavior, card-dependent
weights/tables, residual style archetypes and approximate CFR-derived opponents.
Chronological sessions have independent RNG streams; repeated opponents stay in
one split. Histories reset per session, and all hands can be replayed through the
game engine. Default summaries and Bayesian updates use completed earlier hands;
`all_prior` additionally permits earlier actions of the current hand.

The suite preserves priors, Naive Bayes, matched masked logistic models, per-menu
and binary candidate scorers, exact visible-card equity controls, multiple history
windows and an online Bayesian filter. Model choices use validation or grouped
training CV as declared. Explicit final scoring reports paired opponent-cluster
intervals, counts, history/switch slices and examples.

The recorded canonical run contains 12,711 decisions in 100 sessions from 50
opponent instances. On 2,506 held-out decisions, the main history model reaches
69.9% accuracy versus 59.3% without history; the Bayesian filter reaches 73.3%.
History hurts in the small immediate post-switch slice. These are controlled
simulation results. See [run guide](opponent-experiments.md),
[complete results](meeting4-results.md), and [source reconciliation](meeting4-integration.md).

## Hold’em history integration

The [Hold’em walkthrough](../notebooks/holdem_opponent_integration.ipynb) transfers
causal Leduc summaries to Pluribus histories and appends the same type of
information to the decision scorer’s context. A public forecaster observes as
Pluribus and summarizes the target human's prior actions; the transformer sees
the acting human’s own cards and summaries of other active opponents. Both inputs
precede their separate observed-action label. Histories use completed co-observed
hands, reset per directory and keep related session directories in one split group.

On a fixed 1,024-decision validation set, increasing training from 128 to 4,096
examples raised the no-history BERT Mini from 55.3% to 73.1% accuracy. History
reached 72.5% and worsened log loss. The 4,096-example forecaster had a modest
history gain. These one-seed development results demonstrate the integration;
they do not show an improved reference-action policy. [Results and protocol](meeting5-scaling.md).

A [GPU ablation follow-up](meeting5-ablation-results.md) compares BERT Mini and
DistilBERT with linear/MLP heads, then varies the history window, wording, opponent
assignment and training duration. The 22 completed fits include three-seed checks
of selected DistilBERT/MLP pairs. Recent-16 history at one epoch averages +0.91
accuracy points and −0.0102 loss; recent-64 at three epochs averages +0.42 points
but +0.0043 worse loss. Neither metric improves uniformly across seeds. Swapping
profile assignments has a similar result to correct assignments in its comparison,
limiting evidence for precise opponent-specific use. The sampling and session
split remain fixed, and additional selection is recorded as development analysis.

## Live PocketSolver

The [demo](https://www.nmokey.com/poker/) defaults to ModernBERT linear and serves
all eight fine-tuned encoder/head combinations on CPU. A common 256-hand sample
is held out by every serving checkpoint. Selecting a model retains the current
hand and its Groq result; only one trained model is resident. The compact results
table supports click-to-select and shows saved development scores plus a separate
CPU latency benchmark. The website embeds the Streamlit application.

Groq receives the same context and action menu as the local model. References stay
outside both engines; provider failures remain visible. The bounded live benchmark
uses common successful coverage. Full provider usage accounting and a representative
hosted quality/speed study remain open. [Deployment details](deployment.md).

## Boundaries and next steps

The existing transformer is trained on one question type. A new question or menu
being syntactically accepted does not establish generalization. Softmax alone does
not calibrate probabilities or recover a solver's mixed strategy. Reference-action
agreement and opponent prediction accuracy do not demonstrate profitable play.

Learned forecast outputs still run separately from Hero decision inference. The
Hold’em experiment injects summaries and predicts observed actions; improving
Hero’s responses still requires response labels or repeated-play reward evaluation.
Real human histories and selected training-seed repeats are now tested; unseen-human
generalization, learned history encoders, fitted calibration and an independent
final policy evaluation remain future work. [Roadmap](roadmap.md).
