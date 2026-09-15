import os
import json
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

CHEAP_MODEL  = "llama-3.1-8b-instant"
EXP_MODEL    = "llama-3.3-70b-versatile"
OUTPUT_PATH  = Path("data/labels.jsonl")
SAMPLES_PER_DATASET = 400
SAMPLES_GSM8K       = 600   # oversample math to compensate for high discard   # 400 x 3 datasets = 1200 total, safe for free tier
SLEEP_BETWEEN_CALLS = 1.5   # seconds, avoids Groq rate limit

# ── helpers ──────────────────────────────────────────────────────────────────

def query_model(model: str, prompt: str, max_tokens: int = 64) -> str | None:
    """Returns the model's reply, or None if the API call failed.

    None is distinct from "": "" means the model replied with nothing,
    None means we never got a reply. Callers must discard the row rather
    than score it as a wrong answer.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [ERROR] {model}: {e}")
        return None

def extract_letter(text: str) -> str:
    """Pull the first A/B/C/D from a model response."""
    for char in text.upper():
        if char in "ABCD":
            return char
    return ""

def build_mcq_prompt(question: str, choices: list[str]) -> str:
    letters = ["A", "B", "C", "D"]
    choice_str = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))
    return (
        f"{question}\n\n{choice_str}\n\n"
        "Answer with a single letter (A, B, C, or D) and nothing else."
    )

# ── dataset loaders ──────────────────────────────────────────────────────────

def load_mmlu(n: int) -> list[dict]:
    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(n))
    rows = []
    for item in ds:
        choices = item["choices"]
        prompt  = build_mcq_prompt(item["question"], choices)
        answer  = ["A", "B", "C", "D"][item["answer"]]
        rows.append({"source": "mmlu", "prompt": prompt, "answer": answer})
    return rows

def load_arc(n: int) -> list[dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(n))
    rows = []
    for item in ds:
        choices = item["choices"]["text"]
        labels  = item["choices"]["label"]   # may be 1/2/3/4 or A/B/C/D
        # normalise to A/B/C/D
        label_map = {l: ["A","B","C","D"][i] for i, l in enumerate(labels)}
        prompt  = build_mcq_prompt(item["question"], choices)
        answer  = label_map.get(item["answerKey"], "")
        if not answer:
            continue
        rows.append({"source": "arc", "prompt": prompt, "answer": answer})
    return rows

def load_gsm8k(n: int) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(n))
    rows = []
    for item in ds:
        prompt = (
            f"{item['question']}\n\n"
            "Solve step by step, then write your final answer as a plain integer "
            "on the last line prefixed with 'Answer:'. Example: Answer: 42"
        )
        # ground truth is like "#### 42"
        raw_answer = item["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"source": "gsm8k", "prompt": prompt, "answer": raw_answer})
    return rows

def check_gsm8k_correct(response: str, answer: str) -> bool:
    """Extract the integer after 'Answer:' and compare."""
    import re
    # strip commas from both sides
    answer = answer.replace(",", "").strip()
    text   = response.replace(",", "")
    # look for "Answer: <number>" anywhere in the response
    match = re.search(r"answer[:\s]+(-?\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip() == answer
    # fallback: find all integers in response, check if any match
    numbers = re.findall(r"-?\d+", text)
    return answer in numbers

# ── main loop ────────────────────────────────────────────────────────────────

def label_row(row: dict) -> dict | None:
    prompt  = row["prompt"]
    answer  = row["answer"]
    source  = row["source"]

    # --- cheap model ---
    max_tok = 512 if source == "gsm8k" else 64
    print(f"  [cheap ] querying...")
    cheap_response = query_model(CHEAP_MODEL, prompt, max_tokens=max_tok)
    time.sleep(SLEEP_BETWEEN_CALLS)
    if cheap_response is None:
        return None   # call failed -- discard, don't score as wrong

    if source == "gsm8k":
        cheap_correct = check_gsm8k_correct(cheap_response, answer)
    else:
        cheap_correct = extract_letter(cheap_response) == answer

    # --- expensive model (always run to get full labels) ---
    print(f"  [exp   ] querying...")
    exp_response = query_model(EXP_MODEL, prompt, max_tokens=max_tok)
    time.sleep(SLEEP_BETWEEN_CALLS)
    if exp_response is None:
        return None   # call failed -- discard, don't score as wrong

    if source == "gsm8k":
        exp_correct = check_gsm8k_correct(exp_response, answer)
    else:
        exp_correct = extract_letter(exp_response) == answer

    # --- assign tier label ---
    if cheap_correct:
        tier = "cheap"
    elif exp_correct:
        tier = "expensive"
    else:
        return None   # both wrong, discard

    return {
        "source":         source,
        "prompt":         prompt,
        "answer":         answer,
        "cheap_correct":  cheap_correct,
        "exp_correct":    exp_correct,
        "tier":           tier,
    }

def main():
    print("Loading datasets...")
    rows = (
        load_mmlu(SAMPLES_PER_DATASET)
        + load_arc(SAMPLES_PER_DATASET)
        + load_gsm8k(SAMPLES_GSM8K)
    )
    random.Random(42).shuffle(rows)
    print(f"Total prompts to label: {len(rows)}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    # resume support: skip already-labeled prompts
    done_prompts = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                obj = json.loads(line)
                done_prompts.add(obj["prompt"][:80])
        print(f"Resuming -- {len(done_prompts)} already labeled, skipping.")

    labeled = 0
    discarded = 0

    with open(OUTPUT_PATH, "a") as out:
        for i, row in enumerate(rows):
            if row["prompt"][:80] in done_prompts:
                continue

            print(f"\n[{i+1}/{len(rows)}] source={row['source']}")
            result = label_row(row)

            if result is None:
                discarded += 1
                print("  -> discarded (both models wrong, or API call failed)")
            else:
                out.write(json.dumps(result) + "\n")
                out.flush()
                labeled += 1
                print(f"  -> tier={result['tier']}")

    print(f"\nDone. Labeled: {labeled}, Discarded: {discarded}")
    print(f"Output: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()