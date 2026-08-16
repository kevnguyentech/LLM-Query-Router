"""Shared constants for cost and model-tier configuration.

Single source of truth for data/collect.py, api/main.py, models/baseline.py,
and eval/cost_quality.py so training, evaluation, and serving cost estimates
can't silently drift from each other.
"""

COST = {
    "cheap": 0.0,   # Groq free tier (actual)
    # Benchmark price, not Groq's actual rate for llama-3.3-70b-versatile.
    # Used to compute relative cost savings vs a GPT-4o-class model.
    # Swap this for Groq's published rate if you want real billing estimates.
    "expensive": 2.50,  # USD per 1M tokens, GPT-4o-class benchmark
}

MODEL_NAMES = {
    "cheap":     "llama-3.1-8b-instant",
    "expensive": "llama-3.3-70b-versatile",
}