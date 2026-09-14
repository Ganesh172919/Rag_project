"""Train Router — Fine-tune the query complexity classifier."""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, accuracy_score


class QueryDataset(Dataset):
    """Dataset for query complexity classification."""

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


def train_router(
    train_path: str = "data/processed/router_train.json",
    val_path: str = "data/processed/router_val.json",
    model_name: str = "microsoft/deberta-v3-base",
    output_dir: str = "models/router",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
):
    """Train the query complexity classifier."""

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
        num_labels=4,
        problem_type="single_label_classification",
    ).to(device)

    # Create datasets
    train_dataset = QueryDataset(train_data, tokenizer)
    val_dataset = QueryDataset(val_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    best_accuracy = 0.0
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

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1).cpu()

                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.4f}")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))
            print(f"  Saved best model (accuracy: {accuracy:.4f})")

    # Final evaluation
    print("\nFinal Classification Report:")
    label_names = ["no_retrieval", "single_step", "multi_step", "graph_global"]
    print(classification_report(all_labels, all_preds, target_names=label_names))

    # Save training metadata
    metadata = {
        "model_name": model_name,
        "best_accuracy": best_accuracy,
        "epochs": epochs,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
    }
    with open(output_path / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTraining complete. Best accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AARAG query complexity classifier")
    parser.add_argument("--train-data", default="data/processed/router_train.json")
    parser.add_argument("--val-data", default="data/processed/router_val.json")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--output-dir", default="models/router")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_router(
        train_path=args.train_data,
        val_path=args.val_data,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
