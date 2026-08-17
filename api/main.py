import pickle
import torch
import tiktoken
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import textstat, spacy, re
import numpy as np

# ── setup ─────────────────────────────────────────────────────────────────────

MODELS_DIR = Path("models/saved")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nlp = spacy.load("en_core_web_sm")
enc = tiktoken.get_encoding("cl100k_base")

# load artifacts once at startup
le      = pickle.load(open(MODELS_DIR / "label_encoder.pkl", "rb"))
xgb     = pickle.load(open(MODELS_DIR / "xgb_no_source_flags.pkl", "rb"))

tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODELS_DIR / "bert_tokenizer"))
bert      = DistilBertForSequenceClassification.from_pretrained(
    str(MODELS_DIR / "bert_router_hf")
).to(DEVICE)
bert.eval()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COST, MODEL_NAMES

FEATURE_COLS = [
    "token_count", "sentence_count", "avg_word_length",
    "has_math", "has_code", "digit_count", "unique_word_ratio",
    "named_entity_count", "flesch_reading_ease", "choice_count",
    "starts_with_what", "starts_with_why", "starts_with_how",
    "starts_with_which", "starts_with_calc",
]

# ── feature extraction (mirrors data/features.py) ────────────────────────────

def extract_features(prompt: str) -> np.ndarray:
    lower = prompt.lower()
    doc     = nlp(prompt)
    ent_doc = nlp(prompt[:1000])
    words   = prompt.split()

    feats = {
        "token_count":         len(enc.encode(prompt)),
        "sentence_count":      len(list(doc.sents)),
        "avg_word_length":     sum(len(w.strip(".,!?;:")) for w in words) / max(len(words), 1),
        "has_math":            int(bool(re.search(r"[\+\*\/\=\^\%\$]|\d-\d|\d+\.\d+|\bsin\b|\bcos\b|\blog\b|\bsqrt\b", prompt))),
        "has_code":            int(bool(re.search(r"```|def |import |class |for |while |if |return |print\(", prompt))),
        "digit_count":         len(re.findall(r"\d", prompt)),
        "unique_word_ratio":   len(set(re.findall(r"\b\w+\b", lower))) / max(len(re.findall(r"\b\w+\b", lower)), 1),
        "named_entity_count":  len(ent_doc.ents),
        "flesch_reading_ease": max(-50.0, min(120.0, textstat.flesch_reading_ease(prompt))),
        "choice_count":        len(re.findall(r"\b[A-D]\.", prompt)),
        "starts_with_what":    int(bool(re.search(r"\bwhat\b",    lower))),
        "starts_with_why":     int(bool(re.search(r"\bwhy\b",     lower))),
        "starts_with_how":     int(bool(re.search(r"\bhow\b",     lower))),
        "starts_with_which":   int(bool(re.search(r"\bwhich\b",   lower))),
        "starts_with_calc":    int(bool(re.search(r"\b(calculate|compute|solve|find)\b", lower))),
    }
    return np.array([feats[c] for c in FEATURE_COLS]).reshape(1, -1)

def rule_based_override(prompt: str, features: np.ndarray) -> str | None:
    """
    Hard rules for clear-cut cases that the learned model may miss
    due to training distribution mismatch.
    Returns a tier string if a rule fires, None otherwise.
    """
    f = dict(zip(FEATURE_COLS, features.flatten().tolist()))

    # very short factual question with no math/code -- clearly cheap
    if f["token_count"] < 20 and f["choice_count"] == 0 and f["has_math"] == 0 and f["has_code"] == 0:
        return "cheap"

    # explicit math/solve prompt -- clearly expensive
    if f["has_math"] == 1 and f["starts_with_calc"] == 1:
        return "expensive"

    return None

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLM Query Router",
    description="Routes prompts to the cheapest capable model tier.",
    version="1.0.0",
)

class RouteRequest(BaseModel):
    prompt: str
    router: str = "xgboost"   # "xgboost" or "distilbert"
    threshold: float = 0.5    # probability threshold for "expensive"

class RouteResponse(BaseModel):
    tier: str
    model: str
    confidence: float
    estimated_cost_usd: float
    savings_vs_expensive_usd: float
    router_used: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest):
    if req.router not in ("xgboost", "distilbert"):
        raise HTTPException(status_code=400, detail="router must be 'xgboost' or 'distilbert'")
    if not 0.0 < req.threshold < 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    exp_idx = list(le.classes_).index("expensive")
    X       = extract_features(req.prompt)

    # check hard rules first
    override = rule_based_override(req.prompt, X)
    if override is not None:
        tier       = override
        confidence = 1.0
        token_count = int(X[0][0])   # already computed in extract_features
        est_cost    = COST[tier] * token_count / 1_000_000
        exp_cost    = COST["expensive"] * token_count / 1_000_000
        savings     = exp_cost - est_cost
        return RouteResponse(
            tier=tier,
            model=MODEL_NAMES[tier],
            confidence=confidence,
            estimated_cost_usd=round(est_cost, 8),
            savings_vs_expensive_usd=round(savings, 8),
            router_used="rule_override",
        )

    if req.router == "xgboost":
        proba    = xgb.predict_proba(X)[0]
        exp_prob = float(proba[exp_idx])

    else:  # distilbert
        enc_input = tokenizer(
            req.prompt,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            logits = bert(**enc_input).logits
        proba    = torch.softmax(logits, dim=1).cpu().numpy()[0]
        exp_prob = float(proba[exp_idx])

    tier       = "expensive" if exp_prob >= req.threshold else "cheap"
    confidence = exp_prob if tier == "expensive" else 1.0 - exp_prob

    token_count = int(X[0][0])   # already computed in extract_features
    est_cost    = COST[tier] * token_count / 1_000_000
    exp_cost    = COST["expensive"] * token_count / 1_000_000
    savings     = exp_cost - est_cost

    return RouteResponse(
        tier=tier,
        model=MODEL_NAMES[tier],
        confidence=round(confidence, 4),
        estimated_cost_usd=round(est_cost, 8),
        savings_vs_expensive_usd=round(savings, 8),
        router_used=req.router,
    )

@app.get("/models")
def list_models():
    return {
        "cheap":     MODEL_NAMES["cheap"],
        "expensive": MODEL_NAMES["expensive"],
        "routers":   ["xgboost", "distilbert"],
    }