"""Unit tests for data/relabel.py's relabel() logic.

Run with: python -m pytest tests/test_relabel.py -v
"""
from data.relabel import relabel


def test_gsm8k_always_expensive_even_if_cheap_correct():
    row = {"source": "gsm8k", "cheap_correct": True, "exp_correct": True}
    assert relabel(row)["tier"] == "expensive"


def test_gsm8k_always_expensive_when_cheap_wrong():
    row = {"source": "gsm8k", "cheap_correct": False, "exp_correct": True}
    assert relabel(row)["tier"] == "expensive"


def test_arc_cheap_when_cheap_model_correct():
    row = {"source": "arc", "cheap_correct": True, "exp_correct": False}
    assert relabel(row)["tier"] == "cheap"


def test_mmlu_expensive_when_only_exp_model_correct():
    row = {"source": "mmlu", "cheap_correct": False, "exp_correct": True}
    assert relabel(row)["tier"] == "expensive"


def test_mmlu_discarded_when_both_wrong():
    row = {"source": "mmlu", "cheap_correct": False, "exp_correct": False}
    assert relabel(row) is None


def test_arc_discarded_when_both_wrong():
    row = {"source": "arc", "cheap_correct": False, "exp_correct": False}
    assert relabel(row) is None


def test_unknown_source_discarded():
    row = {"source": "unknown_source", "cheap_correct": True, "exp_correct": True}
    assert relabel(row) is None


def test_relabel_is_idempotent():
    """Running relabel twice on its own output must not change the tier.

    relabel() overwrites row["tier"] but leaves cheap_correct/exp_correct
    untouched, so re-running it on already-relabeled rows (as would happen
    if data/relabel.py is accidentally run twice against the same file)
    should be a no-op.
    """
    row = {"source": "arc", "cheap_correct": True, "exp_correct": False}
    once = relabel(dict(row))
    twice = relabel(dict(once))
    assert once["tier"] == twice["tier"]