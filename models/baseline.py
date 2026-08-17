import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier

FEATURES_PATH = Path("data/features.csv")
MODELS_DIR    = Path("models/saved")
PLOTS_DIR     = Path("eval/plots")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COST

FEATURE_COLS_WITH_SRC = [
    "token_count", "sentence_count", "avg_word_length",
    "has_math", "has_code", "digit_count", "unique_word_ratio",
    "named_entity_count", "flesch_reading_ease", "choice_count",
    "starts_with_what", "starts_with_why", "starts_with_how",
    "starts_with_which", "starts_with_calc",
    "src_mmlu", "src_arc", "src_gsm8k",
]

FEATURE_COLS_NO_SRC = [
    "token_count", "sentence_count", "avg_word_length",
    "has_math", "has_code", "digit_count", "unique_word_ratio",
    "named_entity_count", "flesch_reading_ease", "choice_count",
    "starts_with_what", "starts_with_why", "starts_with_how",
    "starts_with_which", "starts_with_calc",
]

# ── cost metric ───────────────────────────────────────────────────────────────

def cost_per_correct(y_true, y_pred, token_counts):
    """
    Simulate routing cost. For each prediction:
    - Pay cost[predicted_tier] per token
    - Count as correct only if predicted tier matches true tier
      OR if we over-routed (sent cheap prompt to expensive -- still correct)
    """
    total_cost    = 0.0
    total_correct = 0

    for true, pred, tokens in zip(y_true, y_pred, token_counts):
        tier_cost     = COST[pred] * tokens / 1_000_000
        total_cost   += tier_cost
        # correct if: routed to expensive (always answers correctly by definition)
        # or routed to cheap and cheap was the right tier
        if pred == "expensive" or (pred == "cheap" and true == "cheap"):
            total_correct += 1

    cpc = total_cost / total_correct if total_correct > 0 else float("inf")
    return cpc, total_cost, total_correct

def savings_vs_always_expensive(y_pred, token_counts):
    always_exp_cost = sum(
        COST["expensive"] * t / 1_000_000 for t in token_counts
    )
    router_cost = sum(
        COST[p] * t / 1_000_000 for p, t in zip(y_pred, token_counts)
    )
    if always_exp_cost == 0:
        return 0.0
    return (always_exp_cost - router_cost) / always_exp_cost * 100

# ── plotting ──────────────────────────────────────────────────────────────────

def plot_confusion(cm, labels, title, path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
    )
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

def plot_feature_importance(model, feature_names, path):
    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importance)), importance[idx])
    plt.xticks(range(len(importance)), [feature_names[i] for i in idx], rotation=45, ha="right")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ── train + eval ──────────────────────────────────────────────────────────────

def evaluate(name, model, X_test, y_test, le, token_counts):
    y_pred_enc = model.predict(X_test)
    y_pred     = le.inverse_transform(y_pred_enc)
    y_true     = le.inverse_transform(y_test)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
    plot_confusion(
        cm, le.classes_,
        title=f"{name} -- Confusion Matrix",
        path=PLOTS_DIR / f"{name.lower().replace(' ', '_')}_cm.png",
    )

    cpc, total_cost, correct = cost_per_correct(y_true, y_pred, token_counts)
    savings = savings_vs_always_expensive(y_pred, token_counts)

    print(f"  Cost per correct answer : ${cpc:.6f}")
    print(f"  Total routing cost      : ${total_cost:.4f}")
    print(f"  Correct answers         : {correct}/{len(y_true)}")
    print(f"  Savings vs always-exp   : {savings:.1f}%")

    return {
        "name":       name,
        "f1_macro":   f1_score(y_true, y_pred, average="macro"),
        "cpc":        cpc,
        "savings_pct": savings,
    }

def main():
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} rows.")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    le = LabelEncoder()
    y  = le.fit_transform(df["label"].values)
    print(f"Classes: {le.classes_}")

    token_counts = df["token_count"].values

    results = []

    for experiment, feature_cols in [
        ("with_source_flags", FEATURE_COLS_WITH_SRC),
        ("no_source_flags",   FEATURE_COLS_NO_SRC),
    ]:
        print(f"\n{'#'*60}")
        print(f"  EXPERIMENT: {experiment}")
        print(f"{'#'*60}")

        X = df[feature_cols].values

        X_train, X_test, y_train, y_test, tc_train, tc_test = train_test_split(
            X, y, token_counts,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        print(f"Train: {len(X_train)}  Test: {len(X_test)}")

        # ── Logistic Regression ──
        with mlflow.start_run(run_name=f"lr_{experiment}"):
            lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
            cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring="f1_macro")
            print(f"\nLR 5-fold CV F1: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

            lr.fit(X_train, y_train)
            mlflow.log_param("model", f"logistic_regression_{experiment}")
            mlflow.log_metric("cv_f1_mean", cv_scores.mean())

            res = evaluate(f"LR [{experiment}]", lr, X_test, y_test, le, tc_test)
            mlflow.log_metrics({"f1_macro": res["f1_macro"], "savings_pct": res["savings_pct"]})
            results.append(res)

            with open(MODELS_DIR / f"lr_{experiment}.pkl", "wb") as _f:
                pickle.dump(lr, _f)
            with open(MODELS_DIR / "label_encoder.pkl", "wb") as _f:
                pickle.dump(le, _f)

        # ── XGBoost ──
        with mlflow.start_run(run_name=f"xgb_{experiment}"):
            cheap_idx      = le.transform(["cheap"])[0]
            exp_idx        = le.transform(["expensive"])[0]
            scale_pos_w    = np.sum(y_train == cheap_idx) / max(np.sum(y_train == exp_idx), 1)
            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_w,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )
            cv_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring="f1_macro")
            print(f"\nXGB 5-fold CV F1: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

            xgb.fit(X_train, y_train)
            mlflow.log_param("model", f"xgboost_{experiment}")
            mlflow.log_metric("cv_f1_mean", cv_scores.mean())

            res = evaluate(f"XGBoost [{experiment}]", xgb, X_test, y_test, le, tc_test)
            mlflow.log_metrics({"f1_macro": res["f1_macro"], "savings_pct": res["savings_pct"]})
            results.append(res)

            if experiment == "no_source_flags":
                plot_feature_importance(
                    xgb, feature_cols,
                    path=PLOTS_DIR / "xgb_feature_importance_no_src.png",
                )
            else:
                plot_feature_importance(
                    xgb, feature_cols,
                    path=PLOTS_DIR / "xgb_feature_importance_with_src.png",
                )

            with open(MODELS_DIR / f"xgb_{experiment}.pkl", "wb") as _f:
                pickle.dump(xgb, _f)

    # ── summary ──
    print(f"\n{'='*60}")
    print("  SUMMARY -- all experiments")
    print(f"{'='*60}")
    summary = pd.DataFrame(results).set_index("name")
    print(summary[["f1_macro", "cpc", "savings_pct"]].to_string())
    print("\nKey: f1_macro = routing accuracy, savings_pct = % cost saved vs always-expensive")
    print("\nAll models saved to models/saved/")
    print("All plots saved to eval/plots/")

if __name__ == "__main__":
    main()