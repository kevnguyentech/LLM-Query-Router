# LLM Query Router

Routes incoming prompts to the cheapest model tier that can handle them correctly, reducing inference cost while maintaining quality.

---

## Results

| Strategy | Quality | Cost Savings |
|---|---|---|
| Always Cheap | 51.2% | 100% |
| Random | 75.9% | 48.5% |
| **XGBoost Router** | **99.9%** | **52.8%** |
| **DistilBERT Router** | **98.7%** | **54.6%** |
| Always Expensive | 100% | 0% |

XGBoost router achieves 99.9% quality retention with 52.8% cost savings vs always routing to the expensive tier. DistilBERT achieves marginally higher savings (54.6%) at a small quality cost (98.7%).

![Pareto Frontier](eval/plots/pareto_frontier.png)

---

## How It Works

Two model tiers:

- **Cheap:** `llama-3.1-8b-instant` (Groq free tier) -- simple factual, MCQ
- **Expensive:** `llama-3.3-70b-versatile` (Groq) -- multi-step reasoning, math

A classifier reads the incoming prompt and predicts which tier will answer correctly. A rule-based override handles clear-cut cases the learned model may miss due to training distribution mismatch (e.g. very short factual questions not seen during training).

### Labeling Methodology

1140 prompts collected from MMLU, ARC-Challenge, and GSM8K. Labels assigned by task type: GSM8K (math) = expensive, ARC/MMLU = cheap if the 8B model answered correctly, expensive otherwise.

**Known limitation:** labels are rule-based proxies, not ground-truth human judgments. A production router would require labels from real user queries across diverse formats. The router generalizes well within the training distribution but degrades on prompt formats not seen during training (documented in the rule-based override).

---

## Architecture

```
prompt
  │
  ▼
rule_based_override()   ← hard rules for clear OOD cases
  │ (no match)
  ▼
feature extraction      ← 15 hand-crafted features (token count, readability,
  │                        math/code signals, question type flags)
  ▼
XGBoost classifier      ← trained on 912 labeled prompts, 5-fold CV F1: 0.981
  │
  ▼
tier + confidence + estimated cost
```

---

## Classifiers

Two classifiers trained and compared.

**Experiment 1 (with source flags):** included dataset-of-origin as features. F1: 0.996 -- artificially inflated, classifier learns dataset watermarks not prompt content.

**Experiment 2 (no source flags):** prompt features only. XGBoost F1: 0.991, LR F1: 0.996 -- honest generalization signal.

DistilBERT fine-tuned for 3 epochs matches XGBoost at 99.56% val F1, but XGBoost is preferred for production: faster inference, no GPU dependency, smaller artifact.

---

## API

Start the server:

```bash
uvicorn api.main:app --reload
```

Route a prompt:

```bash
POST /route
Content-Type: application/json

{
  "prompt": "Integrate x^2 from 0 to 1",
  "router": "xgboost",
  "threshold": 0.5
}
```

Response:

```json
{
  "tier": "expensive",
  "model": "llama-3.3-70b-versatile",
  "confidence": 0.9991,
  "estimated_cost_usd": 0.00008,
  "savings_vs_expensive_usd": 0.0,
  "router_used": "xgboost"
}
```

Available endpoints:

- `POST /route` -- route a prompt
- `GET /health` -- health check
- `GET /models` -- list available models and routers

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1    # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Add your Groq API key to `.env`:

```
GROQ_API_KEY=your_key_here
```

### Reproduce from scratch

```bash
python data/collect.py        # ~2 hours, requires Groq key
python data/relabel.py
python data/features.py
python models/baseline.py
python models/bert_router.py
python eval/cost_quality.py
uvicorn api.main:app --reload
```

---

## Testing

```bash
python -m pytest
```

Covers the pure feature-extraction and cost-metric logic (`data/features.py`, `data/relabel.py`, `models/baseline.py`) plus the FastAPI endpoints and a train/serve parity check confirming `api/main.py`'s `extract_features()` matches `data/features.py`'s `extract()` exactly, including on prompts over 1000 characters.

The API and parity tests need the trained XGBoost/DistilBERT artifacts under `models/saved/` (see "Reproduce from scratch" above). If they're not present yet, `tests/test_api.py` skips cleanly instead of failing.

---

## Project Structure

```
llm-query-router/
├── data/
│   ├── collect.py            # pull benchmarks, query Groq models, write labels.jsonl
│   ├── relabel.py            # source-based relabeling for balanced classes
│   └── features.py           # extract 15 hand-crafted features to features.csv
├── models/
│   ├── baseline.py           # logistic regression + XGBoost, two experiments
│   ├── bert_router.py        # DistilBERT fine-tune, 3 epochs
│   └── saved/                # serialized models and tokenizer
├── eval/
│   ├── cost_quality.py       # Pareto frontier: cost savings vs quality
│   └── plots/                # pareto_frontier.png, confusion matrices, training curve
├── api/
│   └── main.py               # FastAPI endpoint
├── tests/
│   ├── test_features.py      # unit tests for data/features.py
│   ├── test_relabel.py       # unit tests for data/relabel.py
│   ├── test_baseline_metrics.py  # unit tests for cost/savings functions
│   └── test_api.py           # FastAPI endpoint tests + train/serve parity check
├── .env                      # GROQ_API_KEY (not committed)
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Stack

Python, Groq API, HuggingFace Datasets, scikit-learn, XGBoost, DistilBERT (HuggingFace Transformers), FastAPI, spaCy, MLflow

---

## Reference

FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. Chen et al., Stanford 2023. https://arxiv.org/abs/2305.05176