"""Unit tests for the cost/quality metric functions in models/baseline.py.

Imports matplotlib with the non-interactive "Agg" backend before importing
baseline.py, since baseline.py creates pyplot figures at module scope and
there's no display in a test/CI environment.

Run with: python -m pytest tests/test_baseline_metrics.py -v
"""
import matplotlib
matplotlib.use("Agg")

from models.baseline import cost_per_correct, savings_vs_always_expensive, COST


def test_cost_per_correct_cheap_hit_is_free():
    # true=cheap, pred=cheap -> correct, and cheap tier costs $0
    cpc, total_cost, correct = cost_per_correct(["cheap"], ["cheap"], [500])
    assert total_cost == 0.0
    assert correct == 1
    assert cpc == 0.0


def test_cost_per_correct_expensive_pred_always_counts_correct():
    # pred="expensive" counts correct regardless of true label, by the
    # documented assumption that the expensive tier always answers right
    y_true = ["cheap", "expensive"]
    y_pred = ["expensive", "expensive"]
    tokens = [100, 100]
    cpc, total_cost, correct = cost_per_correct(y_true, y_pred, tokens)
    assert correct == 2
    assert total_cost == COST["expensive"] * 200 / 1_000_000


def test_cost_per_correct_under_routing_not_counted_correct():
    # true=expensive, pred=cheap: under-routing, must NOT count as correct
    cpc, total_cost, correct = cost_per_correct(["expensive"], ["cheap"], [100])
    assert correct == 0
    assert cpc == float("inf")


def test_cost_per_correct_mixed_batch():
    y_true = ["cheap", "expensive", "cheap"]
    y_pred = ["cheap", "expensive", "expensive"]
    tokens = [100, 200, 100]
    cpc, total_cost, correct = cost_per_correct(y_true, y_pred, tokens)
    expected_cost = (
        COST["cheap"] * 100 / 1_000_000
        + COST["expensive"] * 200 / 1_000_000
        + COST["expensive"] * 100 / 1_000_000
    )
    assert total_cost == expected_cost
    assert correct == 3  # row0: cheap hit, row1: exp pred, row2: over-route
    assert cpc == expected_cost / 3


def test_savings_vs_always_expensive_matches_hand_calc():
    y_pred = ["cheap", "cheap", "expensive"]
    tokens = [100, 100, 100]
    savings = savings_vs_always_expensive(y_pred, tokens)
    always_exp_cost = COST["expensive"] * 300 / 1_000_000
    router_cost = COST["expensive"] * 100 / 1_000_000
    expected = (always_exp_cost - router_cost) / always_exp_cost * 100
    assert savings == expected


def test_savings_vs_always_expensive_all_cheap_is_100_percent():
    savings = savings_vs_always_expensive(["cheap", "cheap"], [100, 200])
    assert savings == 100.0


def test_savings_vs_always_expensive_zero_tokens_guarded():
    # always_exp_cost == 0 must return 0.0, not raise ZeroDivisionError
    assert savings_vs_always_expensive(["cheap"], [0]) == 0.0