"""Train Reflection — Fine-tune a model for reflection token generation."""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


REFLECTION_TOKENS = [
    "[Retrieve]", "[No Retrieve]",
    "[Relevant]", "[Irrelevant]",
    "[Fully Supported]", "[Partially Supported]", "[No Support]",
    "[Very Useful]", "[Useful]", "[Not Useful]",
]


class ReflectionDataset(Dataset):
    """Dataset for reflection token training."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Question: {item['query']}\nContext: {item.get('context', '')}\nAnswer: {item.get('answer', '')}\nReflection: {item['reflection_token']}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


def generate_reflection_training_data(output_dir: str = "data/processed", num_samples: int = 2000):
    """Generate synthetic training data for reflection tokens."""
    import random

    data = []
    topics = ["science", "history", "geography", "technology", "literature", "mathematics"]

    for i in range(num_samples):
        topic = random.choice(topics)
        query = f"What is the importance of {topic} in modern education?"

        # Simulate different retrieval quality scenarios
        scenario = random.choice(["relevant_supported", "relevant_partial", "irrelevant", "no_context"])

        if scenario == "relevant_supported":
            context = f"{topic.capitalize()} is fundamental to modern education because it develops critical thinking skills."
            answer = f"{topic.capitalize()} is important in education because it develops critical thinking."
            token = "[Relevant]"
        elif scenario == "relevant_partial":
            context = f"{topic.capitalize()} has many applications in various fields."
            answer = f"{topic.capitalize()} is important because it helps with problem-solving and innovation."
            token = "[Partially Supported]"
        elif scenario == "irrelevant":
            context = "The weather forecast predicts rain tomorrow with temperatures dropping."
            answer = f"{topic.capitalize()} is important for education."
            token = "[Irrelevant]"
        else:
            context = ""
            answer = f"{topic.capitalize()} is a fundamental subject."
            token = "[No Retrieve]"

        data.append({
            "query": query,
            "context": context,
            "answer": answer,
            "reflection_token": token,
        })

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "reflection_train.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


def train_reflection(
    train_path: str = "data/processed/reflection_train.json",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    output_dir: str = "models/reflection",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    lora_rank: int = 16,
):
    """Train reflection token generator using LoRA."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate training data if not exists
    if not os.path.exists(train_path):
        print("Generating reflection training data...")
        generate_reflection_training_data()

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print(f"Training samples: {len(train_data)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create dataset
    dataset = ReflectionDataset(train_data, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in loader:
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

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    # Save LoRA weights
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"\nTraining complete. Model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/processed/reflection_train.json")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", default="models/reflection")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    train_reflection(
        train_path=args.train_data,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
    )
