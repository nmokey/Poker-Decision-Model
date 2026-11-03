# Project context and original proposal

## Current status — November 2, 2026

The implemented system uses the audited PokerBench contract, joint all-options
encoding, and a shared linear or MLP candidate head. The completed four-encoder /
two-head comparison uses five fixed scenario folds and three epochs per fit.
[Results](week2-results.md) report accuracy and log loss for all eight combinations:
ModernBERT + linear leads at 71.92% / 0.6078, close to DistilBERT + linear at
71.78% / 0.6164. These are one-seed development estimates on 10,000 examples.

PocketSolver defaults to ModernBERT + linear, with selectors for all four encoders
and both trained heads. Each selection restores its own fold-1 checkpoint; the
same 256 hands are held out by every serving model. Switching models reuses the
hand's live Groq result, and only one trained model is resident in server memory.
See [deployment](deployment.md). Calibration, prior-hand opponent
adaptation, profitable play, and unexploitable strategy have not been established.
The proposal below preserves the original project goals rather than claiming that
all requested capabilities are implemented. See the [documentation index](README.md)
for the maintained implementation, audit, and reproduction instructions.

## Original planning context

---

# ACM AI Quarter Project Context

I’m organizing an 8-week ACM AI project for a team of roughly 5–10 students. We meet once per week for about 2 hours. The project should have clear milestones, concrete technical work, and a polished final demo.

## High-level project idea

We want to build a **Jev-inspired poker decision model**.

The motivation is that generative LLMs are often used to make simple structured decisions, even though they generate text autoregressively. Decision models instead try to map context directly to a probability distribution over actions in one forward pass.

We are **not using Jev itself**. We are trying to recreate the core architectural idea in a narrow domain.

The project should be:

- non-autoregressive at inference time
- decision-oriented rather than text-generative
- small and fast
- conditioned on context
- able to output calibrated probabilities over dynamically supplied options
- demoable against a general-purpose LLM baseline

## Important architecture direction

We do **not** want a basic fixed classifier like:

\[
\text{poker state} \to [\text{fold},\text{call},\text{raise}]
\]

That would be too much like a standard policy network/classifier.

Instead, we want something closer to modern open-source Jev alternatives:

\[
(\text{state},\text{question},\text{candidate options})
\to
\text{transformer encoder}
\to
\text{shared option scorer}
\to
P(\text{options})
\]

Example:

State:
- Hero: A♠ J♠
- Board: J♦ 8♣ 3♥
- Pot: 6 BB
- Villain bets
- opponent tendencies/history

Question:
- “What should Hero do?”

Options:
- Fold
- Call
- Raise

Output:
- Fold: 0.04
- Call: 0.31
- Raise: 0.65

The model should use a small pretrained transformer backbone as an encoder, not as an autoregressive generator. We can replace/ignore the LM vocabulary head and attach a shared dynamic option-scoring head.

A reasonable architecture:

1. Concatenate or jointly encode:
   - poker state
   - natural-language question
   - candidate answer text
2. Obtain contextual embeddings from a small transformer.
3. Represent the decision/query and each option.
4. Score each option using a shared compatibility/pointer-style head, e.g.

\[
s_i = (W_q h_q)^T(W_o h_i)
\]

5. Apply softmax:

\[
P_i = \frac{e^{s_i/T}}{\sum_j e^{s_j/T}}
\]

6. Optionally temperature-calibrate probabilities on a held-out validation set.

No token-by-token generation should happen at inference time.

## Why poker

Poker is visually compelling and naturally decision-oriented.

However, we do **not** want the project to simply learn a mathematically optimal static poker policy. That would mostly be policy distillation and could be somewhat contrived.

The more interesting version conditions on **opponent behavior/history**.

The state should include:
- hole cards
- community cards
- betting history
- position
- pot size / stacks
- opponent history or tendencies

Possible opponent features:
- aggression rate
- fold rate
- bluff rate
- VPIP-like frequency
- recent action sequences
- prior hands / behavioral summary

This lets the same game state produce different decisions depending on the opponent.

Conceptually:

\[
\pi_\theta(a \mid s, h_{\text{opponent}})
\]

rather than only:

\[
\pi(a \mid s)
\]

This makes the learned model actually useful for context-dependent decisions.

## Possible model questions

The same model should ideally handle multiple decision types using dynamic questions/options:

### Action choice
Question:
- “What should Hero do?”

Options:
- fold
- call
- raise

### Opponent prediction
Question:
- “What is Villain most likely to do next?”

Options:
- fold
- call
- raise

### Binary decision
Question:
- “Is Villain likely bluffing?”

Options:
- yes
- no

### Hand characterization
Question:
- “How strong is Hero’s hand in this situation?”

Options:
- weak
- marginal
- strong
- very strong

This is important because it makes the model more Jev-like than a fixed output classifier.

## Data

Potential data sources include:
- public poker hand histories
- strong computer-poker agent logs
- synthetic toy data for early development
- solver-generated labels where useful

For the main quarter project, we should avoid spending too much time scraping and cleaning data.

A useful training record might look like:

```python
{
    "state": "...",
    "question": "What should Hero do?",
    "options": ["fold", "call", "raise"],
    "target": [0.05, 0.35, 0.60]
}
```

If only one observed action is available, the target could be one-hot. If solver or aggregate behavior probabilities are available, use the full target distribution.

## Evaluation

We care about:

- decision accuracy / agreement
- cross-entropy or log loss
- probability calibration
- Brier score
- latency
- throughput
- gameplay quality / behavior where appropriate

Main baseline:
- a general-purpose LLM prompted with the same state and options

Core demo claim:

> A small decision model can make structured decisions in one forward pass much faster than an autoregressive LLM.

## Final demo

The final demo should be interactive.

Possible interface:
- show a poker hand/state
- show opponent profile/history
- ask a question
- show candidate options
- display our model’s probabilities instantly
- compare side-by-side with an LLM response and latency
- optionally let users play against the model

Example:

```
POCKETSOLVER

Hero: A♠ J♠
Board: J♦ 8♣ 3♥
Opponent: aggressive, high bluff rate

Question:
What should Hero do?

Fold   4%
Call  31%
Raise 65%

Decision time: 3.1 ms
```

Then compare with a general-purpose LLM taking much longer.

## Tentative 8-week structure

Week 1:
- understand problem and architecture
- build toy poker environment / inspect data

Week 2:
- construct dataset pipeline
- create state/question/options format

Week 3:
- establish simple baselines
- heuristic / LLM baseline

Week 4:
- implement first Jev-style option-scoring model

Week 5:
- improve model and opponent conditioning

Week 6:
- evaluate calibration, latency, and behavior

Week 7:
- build final interactive UI / integration

Week 8:
- final benchmarking, demo, presentation

## Candidate application exercise

We also created a toy applicant exercise.

Candidates receive a synthetic poker-like dataset with:
- current game-state features
- opponent-behavior features
- some categorical/string features
- mild missingness/noise
- target = fold/call/raise

They are asked to:
1. explore the data
2. build a model that outputs \(P(\text{fold}), P(\text{call}), P(\text{raise})\)
3. choose and justify evaluation metrics
4. probe model behavior under changed opponent conditions
5. improve one aspect of the system
6. reflect on how they used AI tools

The exercise is intentionally ambiguous because we want to evaluate how candidates direct AI agents, investigate problems, and make modeling choices—not just whether they can follow a recipe.

## Overall framing

The best concise description of the project is:

> Build a Jev-inspired poker decision model: a small pretrained transformer with a non-generative decision head that maps poker state, opponent context, a natural-language question, and dynamically supplied options directly to calibrated decision probabilities.

The project should stay practical and product-oriented rather than becoming open-ended ML research. The goal is to build a concrete working system in 8 weeks.

---
