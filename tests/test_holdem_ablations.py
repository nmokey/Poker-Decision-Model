"""Matched experiment plans and causal history transformations."""

import copy
import json
from pathlib import Path

import pytest

from poker_decision.holdem_ablations import (
    Fit,
    architecture_fits,
    examples,
    feature_fits,
    profile_coverage,
    profile_parts,
    transform_context,
)
from poker_decision.holdem_history import PROFILE_SECTION

CONTEXT = (
    "Hero cards: Ah As. Public state."
    + PROFILE_SECTION
    + "seat 1: H=4; facing bet n=3, F/C/R=0.50/0.25/0.25; no bet: no comparable decisions"
    + " | seat 4: H=2; facing bet n=1, F/C/R=0.25/0.25/0.50; no bet n=2, F/C/R=0.20/0.40/0.40"
)


def test_swapping_preserves_causal_profile_set_and_seat_labels():
    swapped = transform_context(CONTEXT, "decision-1", "swapped")
    base, profiles = profile_parts(CONTEXT)
    changed_base, changed = profile_parts(swapped)
    assert base == changed_base
    assert [p[0] for p in profiles] == [p[0] for p in changed]
    assert sorted(p[1] for p in profiles) == sorted(p[1] for p in changed)
    assert profiles != changed
    assert swapped == transform_context(CONTEXT, "decision-1", "swapped")
    single = CONTEXT.split(" | ")[0]
    assert transform_context(single, "decision-1", "swapped") == single


def test_explicit_format_preserves_values_and_base_state():
    text = transform_context(CONTEXT, "decision-1", "explicit")
    assert text.split(PROFILE_SECTION)[0] == CONTEXT.split(PROFILE_SECTION)[0]
    assert "fold=0.50 call=0.25 raise=0.25" in text
    assert "hands=4" in text and "count=3" in text
    assert "no comparable decisions" in text
    assert profile_coverage(CONTEXT) == {"profiles": 2, "comparable_opportunities": 6}
    with pytest.raises(ValueError, match="Unknown"):
        transform_context(CONTEXT, "x", "future")


def test_variants_preserve_ids_candidates_and_label_independent_inputs():
    row = dict(
        hero=dict(
            source_id="a",
            street="preflop",
            context=CONTEXT,
            candidates=["fold", "call", "raise"],
            target_index=1,
        )
    )
    data = {"train": [row]}
    changed = copy.deepcopy(data)
    changed["train"][0]["hero"]["target_index"] = 2
    for fit in (
        Fit("bert_mini", "linear"),
        Fit("bert_mini", "linear").control(),
        Fit("bert_mini", "linear", representation="explicit"),
        Fit("bert_mini", "linear", representation="swapped"),
    ):
        original = examples(data, "train", fit)[0]
        altered = examples(changed, "train", fit)[0]
        assert original.inputs == altered.inputs
        assert original.source_id == "a" and original.inputs.candidates == ("fold", "call", "raise")
        assert original.target_index == 1 and altered.target_index == 2


def test_default_plan_has_eight_architecture_and_six_feature_fits():
    settings = json.loads((Path(__file__).parents[1] / "configs/holdem_ablations.json").read_text())
    architecture = architecture_fits(settings)
    assert len(architecture) == len({f.key for f in architecture}) == 8
    selected = Fit("bert_mini", "linear")
    features = feature_fits(selected, settings)
    assert len(features) == len({f.key for f in features}) == 6
    assert {f.window for f in features if f.representation == "compact"} == {16, 64, 256}
    assert selected.control().window == 0 and selected.control().representation == "none"
    assert all(f.seed == 42 for f in architecture + features)
