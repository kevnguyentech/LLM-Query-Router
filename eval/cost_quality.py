import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

LABELS_PATH  = Path("data/labels.jsonl")
FEATURES_PATH = Path("data/features.csv")
MODELS_DIR   = Path("models/saved")
PLOTS_DIR    = Path("eval/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COST

FEATURE_COLS_NO_SRC = [
    "token_count", "sentence_count", "avg_word_length",
    "has_math", "has_code", "digit_count", "unique_word_ratio",
    "named_entity_count", "flesch_reading_ease", "choice_count",
    "starts_with_what", "starts_with_why", "starts_with_how",
    "starts_with_which", "starts_with_calc",
]

# ── cost helpers ──────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, token_counts):
    """Returns (cost_per_correct, quality_pct, savings_pct)."""
    total_cost    = 0.0
    total_correct = 0

    for true, pred, tokens in zip(y_true, y_pred, token_counts):
        total_cost += COST[pred] * tokens / 1_000_000
        if pred == "expensive" or (pred == "cheap" and true == "cheap"):
            total_correct += 1

    always_exp_cost = sum(COST["expensive"] * t / 1_000_000 for t in token_counts)
    quality_pct  = total_correct / len(y_true) * 100
    savings_pct  = (always_exp_cost - total_cost) / always_exp_cost * 100 if always_exp_cost > 0 else 0.0
    cpc          = total_cost / total_correct if total_correct > 0 else float("inf")
    return cpc, quality_pct, savings_pct

# ── strategy: always cheap / always expensive / random ───────────────────────

def baseline_strategies(y_true, token_counts):
    strategies = {}

    # always cheap
    pred_cheap = ["cheap"] * len(y_true)
    cpc, q, s = compute_metrics(y_true, pred_cheap, token_counts)
    strategies["Always Cheap"]     = {"quality": q, "savings": s, "cpc": cpc}

    # always expensive
    pred_exp = ["expensive"] * len(y_true)
    cpc, q, s = compute_metrics(y_true, pred_exp, token_counts)
    strategies["Always Expensive"] = {"quality": q, "savings": s, "cpc": cpc}

    # random
    rng = np.random.default_rng(42)
    pred_rand = rng.choice(["cheap", "expensive"], size=len(y_true)).tolist()
    cpc, q, s = compute_metrics(y_true, pred_rand, token_counts)
    strategies["Random"]           = {"quality": q, "savings": s, "cpc": cpc}

    return strategies

# ── strategy: XGBoost with confidence threshold sweep ────────────────────────

def xgb_threshold_sweep(xgb, le, X, y_true, token_counts):
    proba      = xgb.predict_proba(X)
    exp_idx    = list(le.classes_).index("expensive")
    exp_proba  = proba[:, exp_idx]

    points = []
    for threshold in np.linspace(0.1, 0.9, 50):
        preds = [
            "expensive" if p >= threshold else "cheap"
            for p in exp_proba
        ]
        _, q, s = compute_metrics(y_true, preds, token_counts)
        points.append({"threshold": threshold, "quality": q, "savings": s})
    return points

# ── strategy: DistilBERT with confidence threshold sweep ─────────────────────

def bert_threshold_sweep(model, tokenizer, prompts, y_true, token_counts, le):
    model.eval()
    exp_idx = list(le.classes_).index("expensive")
    all_proba = []

    batch_size = 32
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        enc   = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        all_proba.extend(proba[:, exp_idx].tolist())

    points = []
    for threshold in np.linspace(0.1, 0.9, 50):
        preds = [
            "expensive" if p >= threshold else "cheap"
            for p in all_proba
        ]
        _, q, s = compute_metrics(y_true, preds, token_counts)
        points.append({"threshold": threshold, "quality": q, "savings": s})
    return points

# ── plot ──────────────────────────────────────────────────────────────────────

def plot_pareto(strategies, xgb_points, bert_points, path):
    fig, ax = plt.subplots(figsize=(9, 6))

    # threshold sweep curves
    xgb_q  = [p["quality"]  for p in xgb_points]
    xgb_s  = [p["savings"]  for p in xgb_points]
    bert_q = [p["quality"]  for p in bert_points]
    bert_s = [p["savings"]  for p in bert_points]

    ax.plot(xgb_s,  xgb_q,  label="XGBoost (threshold sweep)", color="steelblue",  linewidth=2)
    ax.plot(bert_s, bert_q, label="DistilBERT (threshold sweep)", color="darkorange", linewidth=2)

    # baseline scatter points
    markers = {"Always Cheap": "v", "Always Expensive": "^", "Random": "s"}
    colors  = {"Always Cheap": "green", "Always Expensive": "red", "Random": "gray"}
    for name, vals in strategies.items():
        ax.scatter(
            vals["savings"], vals["quality"],
            marker=markers[name], color=colors[name],
            s=120, zorder=5, label=name,
        )
        ax.annotate(
            name,
            (vals["savings"], vals["quality"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )

    ax.set_xlabel("Cost Savings vs Always-Expensive (%)", fontsize=12)
    ax.set_ylabel("Quality (% Correct Answers)", fontsize=12)
    ax.set_title("Router Pareto Frontier: Cost Savings vs Quality", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(40, 105)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # load data
    with open(LABELS_PATH) as _f:
        rows = [json.loads(l) for l in _f]
    prompts      = [r["prompt"] for r in rows]
    y_true       = [r["tier"]   for r in rows]
    df           = pd.read_csv(FEATURES_PATH)
    token_counts = df["token_count"].values
    X_no_src     = df[FEATURE_COLS_NO_SRC].values

    with open(MODELS_DIR / "label_encoder.pkl", "rb") as _f:
        le = pickle.load(_f)

    # baseline strategies
    print("Computing baseline strategies...")
    strategies = baseline_strategies(y_true, token_counts)
    for name, vals in strategies.items():
        print(f"  {name:20s}  quality={vals['quality']:.1f}%  savings={vals['savings']:.1f}%")

    # XGBoost sweep
    print("\nRunning XGBoost threshold sweep...")
    with open(MODELS_DIR / "xgb_no_source_flags.pkl", "rb") as _f:
        xgb = pickle.load(_f)
    xgb_points = xgb_threshold_sweep(xgb, le, X_no_src, y_true, token_counts)

    # DistilBERT sweep
    print("Running DistilBERT threshold sweep...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODELS_DIR / "bert_tokenizer"))
    model = DistilBertForSequenceClassification.from_pretrained(
        str(MODELS_DIR / "bert_router_hf")
    ).to(DEVICE)
    print("DistilBERT checkpoint loaded successfully.")
    bert_points = bert_threshold_sweep(model, tokenizer, prompts, y_true, token_counts, le)

    # print summary table
    print("\nXGBoost Pareto points (sample):")
    for p in xgb_points[::10]:
        print(f"  threshold={p['threshold']:.2f}  quality={p['quality']:.1f}%  savings={p['savings']:.1f}%")

    print("\nDistilBERT Pareto points (sample):")
    for p in bert_points[::10]:
        print(f"  threshold={p['threshold']:.2f}  quality={p['quality']:.1f}%  savings={p['savings']:.1f}%")

    # plot
    print("\nGenerating Pareto plot...")
    plot_pareto(
        strategies, xgb_points, bert_points,
        path=PLOTS_DIR / "pareto_frontier.png",
    )
    print("\nDone. Main deliverable: eval/plots/pareto_frontier.png")

if __name__ == "__main__":
    main()