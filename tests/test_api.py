"""Tests for api/main.py: request validation, rule-based overrides, and
train/serve feature parity against data/features.py.

api/main.py loads the XGBoost model, label encoder, and DistilBERT
checkpoint at import time. The DistilBERT weights (models/saved/bert_router_hf/
model.safetensors) are gitignored for size, so a fresh clone that hasn't run
models/bert_router.py won't have them yet. This whole file skips cleanly in
that case instead of erroring out, so it stays safe to run in CI or on a
machine that only just cloned the repo.

Run with: python -m pytest tests/test_api.py -v
"""
import pytest

try:
    import api.main as api_main
    from fastapi.testclient import TestClient
except Exception as e:  # noqa: BLE001 - deliberately broad, see module docstring
    api_main = None
    _import_error = e
else:
    _import_error = None

pytestmark = pytest.mark.skipif(
    api_main is None,
    reason=(
        f"api.main could not be loaded, likely missing a dependency or the "
        f"trained model artifacts under models/saved/: {_import_error!r}"
    ),
)


@pytest.fixture(scope="module")
def client():
    return TestClient(api_main.app)


# ── endpoint tests ───────────────────────────────────────────────────────

def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_models_endpoint_lists_both_routers(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    assert resp.json()["routers"] == ["xgboost", "distilbert"]


def test_route_rejects_invalid_router_name(client):
    resp = client.post("/route", json={"prompt": "hello", "router": "not_a_real_router"})
    assert resp.status_code == 400


def test_route_rejects_threshold_above_one(client):
    resp = client.post("/route", json={"prompt": "hello", "threshold": 1.5})
    assert resp.status_code == 400


def test_route_rejects_threshold_at_zero(client):
    resp = client.post("/route", json={"prompt": "hello", "threshold": 0.0})
    assert resp.status_code == 400


def test_route_xgboost_returns_well_formed_response(client):
    resp = client.post(
        "/route",
        json={"prompt": "What is the capital of France?", "router": "xgboost"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tier"] in ("cheap", "expensive")
    assert body["model"] in api_main.MODEL_NAMES.values()
    assert 0.0 <= body["confidence"] <= 1.0


# ── rule_based_override ──────────────────────────────────────────────────

def test_rule_override_short_factual_prompt_is_cheap():
    prompt = "What color is the sky?"
    features = api_main.extract_features(prompt)
    assert api_main.rule_based_override(prompt, features) == "cheap"


def test_rule_override_math_calc_prompt_is_expensive():
    prompt = "Calculate the integral of x^2 from 0 to 1."
    features = api_main.extract_features(prompt)
    assert api_main.rule_based_override(prompt, features) == "expensive"


def test_rule_override_none_for_long_non_math_prompt():
    prompt = (
        "Describe the major causes of the French Revolution and explain how "
        "Enlightenment philosophy shaped the political demands of the "
        "revolutionaries between 1789 and 1799."
    )
    features = api_main.extract_features(prompt)
    assert api_main.rule_based_override(prompt, features) is None


# ── train/serve parity (regression test for the sentence_count skew fix) ──

def test_extract_features_matches_training_extraction_short_prompt():
    """api/main.py's extract_features() must produce the same feature
    values as data/features.py's extract() for the same prompt. This is
    the regression test for the sentence_count truncation bug: before the
    fix, prompts over 1000 characters got a different sentence_count at
    serve time than at train time.
    """
    from data.features import extract as train_extract

    prompt = "What is the boiling point of water in Celsius?"
    train_feats = train_extract(prompt)
    serve_feats = api_main.extract_features(prompt)[0]

    for i, col in enumerate(api_main.FEATURE_COLS):
        assert serve_feats[i] == pytest.approx(train_feats[col]), (
            f"{col} mismatch: train={train_feats[col]} serve={serve_feats[i]}"
        )


def test_extract_features_matches_training_extraction_long_prompt():
    """Same parity check, with a prompt over 1000 characters -- exactly
    the length the sentence_count truncation bug affected."""
    from data.features import extract as train_extract

    prompt = "This is a sentence about a topic. " * 40
    assert len(prompt) > 1000  # guard: this test is pointless if it isn't
    train_feats = train_extract(prompt)
    serve_feats = api_main.extract_features(prompt)[0]

    for i, col in enumerate(api_main.FEATURE_COLS):
        assert serve_feats[i] == pytest.approx(train_feats[col]), (
            f"{col} mismatch: train={train_feats[col]} serve={serve_feats[i]}"
        )