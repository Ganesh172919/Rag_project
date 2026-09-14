"""Train Evaluator — Fine-tune the retrieval quality evaluator."""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, f1_score


class RetrievalPairDataset(Dataset):
    """Dataset for retrieval quality evaluation."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["query"],
            item["document"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def train_evaluator(
    train_path: str = "data/processed/evaluator_train.json",
    val_path: str = "data/processed/evaluator_val.json",
    model_name: str = "microsoft/deberta-v3-small",
    output_dir: str = "models/evaluator",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
):
    """Train the retrieval quality evaluator."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    ).to(device)

    # Create datasets
    train_dataset = RetrievalPairDataset(train_data, tokenizer)
    val_dataset = RetrievalPairDataset(val_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_f1 = 0.0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1).cpu()

                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        f1 = f1_score(all_labels, all_preds, average="weighted")
        print(f"Epoch {epoch + 1}/{epochs} — Loss: {total_loss / len(train_loader):.4f}, Val F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))
            print(f"  Saved best model (F1: {f1:.4f})")

    print("\nFinal Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["irrelevant", "relevant"]))
    print(f"\nTraining complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/processed/evaluator_train.json")
    parser.add_argument("--val-data", default="data/processed/evaluator_val.json")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-small")
    parser.add_argument("--output-dir", default="models/evaluator")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train_evaluator(
        train_path=args.train_data,
        val_path=args.val_data,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
