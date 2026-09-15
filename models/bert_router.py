import json
import pickle
import numpy as np
import pandas as pd
import torch
import mlflow
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

FEATURES_PATH = Path("data/features.csv")
LABELS_PATH   = Path("data/labels.jsonl")
MODELS_DIR    = Path("models/saved")
PLOTS_DIR     = Path("eval/plots")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = "distilbert-base-uncased"
MAX_LEN     = 256
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 2e-5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── dataset ───────────────────────────────────────────────────────────────────

class PromptDataset(Dataset):
    def __init__(self, prompts, labels, tokenizer):
        self.encodings = tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }

# ── train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader):
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1, all_preds, all_labels

def plot_training_curve(train_losses, val_losses, val_f1s, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(train_losses, label="train loss")
    ax1.plot(val_losses,   label="val loss")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(val_f1s, color="green", label="val F1 macro")
    ax2.set_title("Validation F1 per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    # load raw prompts + labels from jsonl (not features.csv)
    with open(LABELS_PATH) as _f:
        rows = [json.loads(l) for l in _f]
    prompts = [r["prompt"] for r in rows]
    raw_labels = [r["tier"] for r in rows]

    with open(MODELS_DIR / "label_encoder.pkl", "rb") as _f:
        le = pickle.load(_f)
    labels = le.transform(raw_labels)

    print(f"Total samples : {len(prompts)}")
    print(f"Classes       : {le.classes_}")

    # split
    (p_train, p_val,
     l_train, l_val) = train_test_split(
        prompts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    print(f"Train: {len(p_train)}  Val: {len(p_val)}")

    # tokenizer + datasets
    tokenizer   = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_ds    = PromptDataset(p_train, l_train, tokenizer)
    val_ds      = PromptDataset(p_val,   l_val,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # model -- freeze bottom 4 transformer layers
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(le.classes_)
    ).to(DEVICE)

    for i, layer in enumerate(model.distilbert.transformer.layer):
        if i < 4:
            for param in layer.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01,
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    train_losses, val_losses, val_f1s = [], [], []
    best_f1   = 0.0
    best_path = MODELS_DIR / "bert_router_hf"

    with mlflow.start_run(run_name="distilbert"):
        mlflow.log_params({
            "model":      MODEL_NAME,
            "epochs":     EPOCHS,
            "lr":         LR,
            "batch_size": BATCH_SIZE,
            "max_len":    MAX_LEN,
        })

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            train_loss = train_epoch(model, train_loader, optimizer, scheduler)
            val_loss, val_f1, val_preds, val_true = eval_epoch(model, val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1s.append(val_f1)

            print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}")
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_f1":     val_f1,
            }, step=epoch)

            if val_f1 > best_f1:
                best_f1 = val_f1
                model.save_pretrained(MODELS_DIR / "bert_router_hf")
                print(f"  ** New best F1: {best_f1:.4f} -- saved to {MODELS_DIR / 'bert_router_hf'}")

        mlflow.log_metric("best_val_f1", best_f1)

    # final eval with best checkpoint
    model = DistilBertForSequenceClassification.from_pretrained(
        str(MODELS_DIR / "bert_router_hf")
    ).to(DEVICE)
    _, final_f1, final_preds, final_true = eval_epoch(model, val_loader)

    y_true_labels = le.inverse_transform(final_true)
    y_pred_labels = le.inverse_transform(final_preds)

    print(f"\nFinal classification report (best checkpoint):")
    print(classification_report(y_true_labels, y_pred_labels))

    plot_training_curve(
        train_losses, val_losses, val_f1s,
        path=PLOTS_DIR / "bert_training_curve.png",
    )

    # save tokenizer alongside model for inference
    tokenizer.save_pretrained(MODELS_DIR / "bert_tokenizer")
    print(f"\nBest val F1: {best_f1:.4f}")
    print(f"Model saved : {best_path}")
    print(f"Tokenizer   : {MODELS_DIR / 'bert_tokenizer'}")

if __name__ == "__main__":
    main()