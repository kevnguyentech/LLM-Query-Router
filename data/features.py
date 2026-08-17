import re
import json
import spacy
import textstat
import tiktoken
import numpy as np
import pandas as pd
from pathlib import Path

INPUT_PATH  = Path("data/labels.jsonl")
OUTPUT_PATH = Path("data/features.csv")

nlp      = spacy.load("en_core_web_sm")
enc      = tiktoken.get_encoding("cl100k_base")

# ── feature extractors ────────────────────────────────────────────────────────

def token_count(text: str) -> int:
    return len(enc.encode(text))

def sentence_count(text: str) -> int:
    doc = nlp(text)
    return len(list(doc.sents))

def avg_word_length(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w.strip(".,!?;:")) for w in words) / len(words)

def has_math(text: str) -> int:
    pattern = r"[\+\*\/\=\^\%\$]|\d-\d|\d+\.\d+|\bsin\b|\bcos\b|\blog\b|\bsqrt\b"
    return int(bool(re.search(pattern, text)))

def has_code(text: str) -> int:
    pattern = r"```|def |import |class |for |while |if |return |print\("
    return int(bool(re.search(pattern, text)))

def digit_count(text: str) -> int:
    return len(re.findall(r"\d", text))

def unique_word_ratio(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def named_entity_count(text: str) -> int:
    doc = nlp(text[:1000])   # cap at 1000 chars to keep spacy fast
    return len(doc.ents)

def flesch_reading_ease(text: str) -> float:
    score = textstat.flesch_reading_ease(text)
    # clamp to [-50, 120] to avoid extreme outliers
    return max(-50.0, min(120.0, score))

def question_word_flags(text: str) -> dict:
    lower = text.lower()
    return {
        "starts_with_what":  int(bool(re.search(r"\bwhat\b",  lower))),
        "starts_with_why":   int(bool(re.search(r"\bwhy\b",   lower))),
        "starts_with_how":   int(bool(re.search(r"\bhow\b",   lower))),
        "starts_with_which": int(bool(re.search(r"\bwhich\b", lower))),
        "starts_with_calc":  int(bool(re.search(r"\b(calculate|compute|solve|find)\b", lower))),
    }

def choice_count(text: str) -> int:
    """Number of MCQ options present (A. B. C. D.)"""
    return len(re.findall(r"\b[A-D]\.", text))

# ── main ──────────────────────────────────────────────────────────────────────

def extract(prompt: str) -> dict:
    feats = {
        "token_count":         token_count(prompt),
        "sentence_count":      sentence_count(prompt),
        "avg_word_length":     avg_word_length(prompt),
        "has_math":            has_math(prompt),
        "has_code":            has_code(prompt),
        "digit_count":         digit_count(prompt),
        "unique_word_ratio":   unique_word_ratio(prompt),
        "named_entity_count":  named_entity_count(prompt),
        "flesch_reading_ease": flesch_reading_ease(prompt),
        "choice_count":        choice_count(prompt),
    }
    feats.update(question_word_flags(prompt))
    return feats

def source_flags(source: str) -> dict:
    return {
        "src_mmlu":   int(source == "mmlu"),
        "src_arc":    int(source == "arc"),
        "src_gsm8k":  int(source == "gsm8k"),
    }

def main():
    rows = []
    with open(INPUT_PATH) as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Loaded {len(rows)} labeled rows.")

    records = []
    for i, row in enumerate(rows):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(rows)}...")

        feats = extract(row["prompt"])
        feats.update(source_flags(row["source"]))
        feats["label"] = row["tier"]          # "cheap" or "expensive"
        feats["source"] = row["source"]
        records.append(feats)

    df = pd.DataFrame(records)

    # sanity checks
    assert df.isnull().sum().sum() == 0, "NaN values found -- check extractors"
    print(f"\nFeature matrix shape: {df.shape}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    print(f"\nSource distribution:\n{df['source'].value_counts()}")
    print(f"\nFeature summary:\n{df.drop(columns=['label','source']).describe().T[['mean','std','min','max']]}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()