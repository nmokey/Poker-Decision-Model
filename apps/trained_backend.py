"""Model backend: Spot -> Prediction with ordered probabilities and latency.

Integration: load a model once, optionally call engine.warm_up(spot), then call
prediction = engine(spot). The caller supplies an object with context: str and
candidates: tuple[str, ...]. No hand source, reference answer or UI is required.

Requires poker_decision for input validation and model adapters. Checkpoint
loading additionally requires torch, transformers and a saved model directory.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from poker_decision.data import DecisionInputs

if TYPE_CHECKING:
    from poker_decision.baselines import LinearCandidateScorer

Predictor = Callable[[DecisionInputs], Sequence[float]]


@dataclass(frozen=True)
class Spot(DecisionInputs):
    """Visible context and supplied legal candidates, without labels.

    The provider is responsible for poker-rule legality. Input validation checks
    nonempty context and nonempty, distinct candidate strings.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.context, str):
            raise ValueError("Context must be a string")
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(candidate, str) for candidate in self.candidates
        ):
            raise ValueError("Candidates must be a tuple of strings")
        super().__post_init__()


@dataclass
class Prediction:
    """Probabilities aligned to Spot.candidates, plus wall-clock latency.

    Successful distributions sum to one. Failures carry no distribution and
    explain the problem in note so the UI can display an error.
    """

    probs: list[float]
    latency_ms: float
    ok: bool = True
    note: str = ""
    raw: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("Latency must be finite and nonnegative")
        if not self.ok:
            if self.probs or not self.note.strip():
                raise ValueError("Failed predictions need empty probabilities and a note")
            return
        if any(isinstance(p, bool) or not isinstance(p, Real) for p in self.probs):
            raise ValueError("Probabilities must be numbers, not strings or booleans")
        if not self.probs or any(not math.isfinite(p) or not 0 <= p <= 1 for p in self.probs):
            raise ValueError("Invalid candidate probabilities")
        if not math.isclose(sum(self.probs), 1.0, abs_tol=1e-8):
            raise ValueError("Candidate probabilities must sum to one")

    def validate_for(self, spot: Spot) -> None:
        """Check validity and candidate alignment for a model result."""
        self.__post_init__()
        if self.ok and len(self.probs) != len(spot.candidates):
            raise ValueError("Prediction length must match Spot.candidates")


def validate_probabilities(values: Sequence[float], count: int) -> list[float]:
    """Validate this engine's output before returning it to any caller."""
    if count < 1 or len(values) != count:
        raise ValueError("Expected one probability for every candidate, in the supplied order")
    if any(isinstance(p, bool) or not isinstance(p, Real) for p in values):
        raise ValueError("Probabilities must be numbers, not strings or booleans")
    probs = [float(p) for p in values]
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probs):
        raise ValueError("Probabilities must be finite and between zero and one")
    total = math.fsum(probs)
    if total <= 0 or not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Probabilities must sum to one; substantive errors are not normalized")
    return [p / total for p in probs]


class ModelEngine:
    """Initialize once before serving. __call__(spot: Spot) -> Prediction.

    Only context and ordered candidates reach the predictor. Latency covers input
    preparation, synchronization, prediction and probability validation. Neural
    adapters include tokenization and device transfers. Loading, startup warm-up
    and rendering are separate from subsequent measured calls. Failed calls also
    report elapsed milliseconds. No reference label is accepted.
    """

    def __init__(
        self,
        predictor: Predictor | None = None,
        *,
        name: str = "Decision model",
        synchronize: Callable[[], None] | None = None,
        error: str = "",
    ) -> None:
        self.name = name
        self._predictor = predictor
        self._synchronize = synchronize
        self._error = error or ("No model has been loaded" if predictor is None else "")
        self.warmed_up = False

    @property
    def ready(self) -> bool:
        return self._predictor is not None and not self._error

    def __call__(self, spot: Spot) -> Prediction:
        start = perf_counter()
        try:
            if self._error or self._predictor is None:
                raise RuntimeError(self._error)
            # Revalidate inputs while preserving all text, amounts and candidate order.
            validated = Spot(spot.context, tuple(spot.candidates))
            inputs = DecisionInputs(validated.context, validated.candidates)
            if self._synchronize is not None:
                self._synchronize()
            values = self._predictor(inputs)
            if self._synchronize is not None:
                self._synchronize()
            probs = validate_probabilities(values, len(inputs.candidates))
            return Prediction(probs, (perf_counter() - start) * 1000)
        except Exception as error:
            return Prediction(
                [],
                (perf_counter() - start) * 1000,
                ok=False,
                note=f"{self.name}: {type(error).__name__}: {error}",
            )

    def warm_up(self, spot: Spot) -> Prediction:
        """Explicit startup call; exclude its returned timing from steady-state comparisons."""
        prediction = self(spot)
        self.warmed_up = prediction.ok
        return prediction

    @classmethod
    def from_linear(
        cls, model: "LinearCandidateScorer", *, warmup_spot: Spot | None = None
    ) -> "ModelEngine":
        """Wrap a fitted baseline; data loading and fitting remain the caller's responsibility."""
        name = "Linear candidate scorer (local CPU)"
        try:
            if not model.fitted:
                raise ValueError("Provide a fitted LinearCandidateScorer before serving")
            engine = cls(model.predict, name=name)
            if warmup_spot is not None:
                warmup = engine.warm_up(warmup_spot)
                if not warmup.ok:
                    engine._error = warmup.note
            return engine
        except Exception as error:
            return cls(name=name, error=f"Cannot initialize linear model: {error}")

    @classmethod
    def from_checkpoint(
        cls,
        directory: str | Path,
        *,
        device: str = "cpu",
        expected_head: str | None = None,
        warmup_spot: Spot | None = None,
    ) -> "ModelEngine":
        """Restore a saved joint encoder + linear/MLP head, entirely offline."""
        name = "Joint transformer"
        try:
            import torch

            from poker_decision.models.encoder import QUESTION, collate_decisions, encode_inputs
            from poker_decision.training import load_checkpoint

            path = Path(directory)
            required = ["encoder/config.json", "tokenizer/tokenizer_config.json", "head.pt"]
            if not all((path / item).is_file() for item in [*required, "metadata.json"]):
                raise ValueError(
                    "Expected the canonical checkpoint folder with encoder/, tokenizer/, "
                    "head.pt and metadata.json. Older paired MiniLM notebook checkpoints "
                    "use a different architecture and cannot be substituted."
                )
            if device == "auto":
                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )
            if device not in {"cpu", "cuda", "mps"}:
                raise ValueError("Choose cpu, cuda, mps or auto")
            if expected_head not in {None, "linear", "mlp"}:
                raise ValueError("Expected head must be linear or mlp")
            tokenizer, model, metadata = load_checkpoint(path)
            if expected_head is not None and model.head_type != expected_head:
                raise ValueError(
                    f"Checkpoint has a {model.head_type} head; expected {expected_head}. "
                    "Use a checkpoint trained with the requested head."
                )
            model = model.to(device).eval()
            question = metadata.get("question", QUESTION)
            max_length = metadata.get("max_length", 512)
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Checkpoint question must be a nonempty string")
            if question != QUESTION:
                raise ValueError(
                    "This model adapter requires the shared question: What should Hero do?"
                )
            if type(max_length) is not int or max_length < 1:
                raise ValueError("Checkpoint token limit must be a positive integer")
            max_length = min(max_length, model.encoder.config.max_position_embeddings)
            name = (
                f"{metadata.get('backbone', model.encoder.config.model_type)} "
                f"+ {model.head_type} head (local {device})"
            )

            def predict(inputs: DecisionInputs) -> list[float]:
                encoded = encode_inputs(inputs, tokenizer, max_length=max_length, question=question)
                batch = {
                    key: value.to(device)
                    for key, value in collate_decisions(
                        [encoded], pad_token_id=tokenizer.pad_token_id
                    ).items()
                }
                with torch.inference_mode():
                    logits = model(**batch)
                    return logits.detach().cpu().double().softmax(-1)[0].tolist()  # type: ignore[no-any-return]

            synchronize: Callable[[], None] | None = None
            if device == "cuda":
                synchronize = torch.cuda.synchronize
            elif device == "mps":
                synchronize = torch.mps.synchronize
            engine = cls(predict, name=name, synchronize=synchronize)
            if warmup_spot is not None:
                warmup = engine.warm_up(warmup_spot)
                if not warmup.ok:
                    engine._error = warmup.note
            return engine
        except Exception as error:
            return cls(name=name, error=f"Cannot initialize transformer: {error}")

    @classmethod
    def from_mlp_checkpoint(
        cls, directory: str | Path, *, device: str = "cpu", warmup_spot: Spot | None = None
    ) -> "ModelEngine":
        """Require a trained transformer + MLP checkpoint; never initialize a random head."""
        return cls.from_checkpoint(
            directory, device=device, expected_head="mlp", warmup_spot=warmup_spot
        )


DecisionModelWrapper = ModelEngine
