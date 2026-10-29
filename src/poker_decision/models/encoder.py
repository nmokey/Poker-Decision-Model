"""Joint state/question/options encoder and shared scorer; requires the ml extra."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from poker_decision.data import DecisionInputs

QUESTION = "What should Hero do?"
BACKBONES = {
    "bert_mini": (
        "google/bert_uncased_L-4_H-256_A-4",
        "387825ce42dbb39b87911cdf8e383ee3b25184f8",
    ),
    "minilm": (
        "microsoft/MiniLM-L12-H384-uncased",
        "44acabbec0ef496f6dbc93adadea57f376b7c0ec",
    ),
    "distilbert": (
        "distilbert/distilbert-base-cased",
        "6ea81172465e8b0ad3fddeed32b986cdcdcffcf0",
    ),
    "modernbert": (
        "answerdotai/ModernBERT-base",
        "8949b909ec900327062f0ebf497f51aef5e6f0c8",
    ),
}


def model_input(inputs: DecisionInputs, question: str = QUESTION) -> dict[str, object]:
    """The proposal's input shape. Never accepts a target or source ID."""
    return {"state": inputs.context, "question": question, "decisions": list(inputs.candidates)}


@dataclass(frozen=True)
class EncodedDecision:
    input_ids: tuple[int, ...]
    candidate_spans: tuple[tuple[int, int], ...]


def encode_inputs(
    inputs: DecisionInputs, tokenizer: Any, *, max_length: int = 512, question: str = QUESTION
) -> EncodedDecision:
    """One sequence retaining every state and candidate token; fail rather than truncate."""
    if not question.strip():
        raise ValueError("Empty question")
    if any(getattr(tokenizer, f"{name}_token_id") is None for name in ("cls", "sep", "pad")):
        raise ValueError("This serialization requires CLS, SEP and PAD tokens")
    ids = [int(tokenizer.cls_token_id)]
    prefix = f"State: {inputs.context}\nQuestion: {question}\nDecisions:"
    ids.extend(tokenizer.encode(prefix, add_special_tokens=False))
    ids.append(int(tokenizer.sep_token_id))
    spans = []
    for candidate in inputs.candidates:
        tokens = tokenizer.encode(candidate, add_special_tokens=False)
        if not tokens:
            raise ValueError(f"Candidate has no tokens: {candidate!r}")
        start = len(ids)
        ids.extend(tokens)
        spans.append((start, len(ids)))
        ids.append(int(tokenizer.sep_token_id))
    limit = min(max_length, tokenizer.model_max_length)
    if len(ids) > limit:
        raise ValueError(f"Input needs {len(ids)} tokens; limit is {limit}. No truncation applied.")
    return EncodedDecision(tuple(ids), tuple(spans))


def collate_decisions(rows: Sequence[EncodedDecision], *, pad_token_id: int) -> dict[str, Tensor]:
    """Pad tokens and candidate spans independently. Labels stay separate."""
    if not rows:
        raise ValueError("Empty batch")
    length = max(len(row.input_ids) for row in rows)
    count = max(len(row.candidate_spans) for row in rows)
    ids = torch.full((len(rows), length), pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    token_mask = torch.zeros((len(rows), count, length), dtype=torch.bool)
    for b, row in enumerate(rows):
        if not row.candidate_spans:
            raise ValueError("No candidate spans")
        ids[b, : len(row.input_ids)] = torch.tensor(row.input_ids)
        attention[b, : len(row.input_ids)] = 1
        for k, (start, end) in enumerate(row.candidate_spans):
            if not 0 <= start < end <= len(row.input_ids):
                raise ValueError("Invalid candidate span")
            token_mask[b, k, start:end] = True
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "candidate_token_mask": token_mask,
    }


class PokerDecisionModel(nn.Module):
    """One encoder sequence per decision; mean-pool each option's contextual tokens."""

    def __init__(self, encoder: Any, *, head_type: str = "linear", head_width: int = 64) -> None:
        super().__init__()
        if head_type not in {"linear", "mlp"} or head_width < 1:
            raise ValueError("Choose a linear or mlp head with a positive hidden width")
        self.encoder = encoder
        self.head_type = head_type
        self.head_width = head_width
        hidden = int(encoder.config.hidden_size)
        self.head = (
            nn.Linear(hidden, 1)
            if head_type == "linear"
            else nn.Sequential(nn.Linear(hidden, head_width), nn.ReLU(), nn.Linear(head_width, 1))
        )

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, candidate_token_mask: Tensor
    ) -> Tensor:
        if candidate_token_mask.ndim != 3:
            raise ValueError("Expected candidate mask with shape [batch, candidates, tokens]")
        if candidate_token_mask.shape[0] != input_ids.shape[0] or (
            candidate_token_mask.shape[2] != input_ids.shape[1]
        ):
            raise ValueError("Candidate mask and input shapes disagree")
        if (candidate_token_mask & ~attention_mask.bool().unsqueeze(1)).any():
            raise ValueError("Candidate span includes padding")
        valid = candidate_token_mask.any(dim=-1)
        if not valid.any(dim=-1).all():
            raise ValueError("Every decision needs at least one candidate")
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        weights = candidate_token_mask.to(hidden.dtype)
        pooled = torch.bmm(weights, hidden) / weights.sum(-1, keepdim=True).clamp_min(1)
        logits: Tensor = self.head(pooled).squeeze(-1)
        return logits.masked_fill(~valid, float("-inf"))


def load_pretrained(
    name: str, revision: str, *, head_type: str = "linear", head_width: int = 64
) -> tuple[Any, PokerDecisionModel]:
    """Load a pinned encoder without the vocabulary prediction head."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)  # type: ignore[no-untyped-call]
    # ModernBERT's optional compilation/FlashAttention path adds runtime dependencies.
    # SDPA works with the standard Colab PyTorch install and persists in checkpoints.
    options = (
        {"attn_implementation": "sdpa", "reference_compile": False}
        if name == BACKBONES["modernbert"][0]
        else {}
    )
    encoder = AutoModel.from_pretrained(name, revision=revision, **options)
    return tokenizer, PokerDecisionModel(encoder, head_type=head_type, head_width=head_width)
