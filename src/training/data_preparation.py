"""Data Preparation — Prepare training datasets for AARAG components."""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

from datasets import load_dataset


def prepare_router_data(output_dir: str = "data/processed", max_samples: int = 5000):
    """Prepare training data for the query complexity classifier.

    Labels:
    0: No-retrieval (simple factual)
    1: Single-step RAG
    2: Multi-step RAG
    3: Graph-global
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_data = []

    # Level 0: Simple factual — TriviaQA
    print("Loading TriviaQA (simple factual)...")
    try:
        trivia = load_dataset("trivia_qa", "rc", split="train[:2000]", trust_remote_code=True)
        for item in trivia:
            all_data.append({
                "query": item["question"],
                "label": 0,
                "source": "triviaqa",
            })
    except Exception as e:
        print(f"Warning: Could not load TriviaQA: {e}")
        # Generate synthetic simple questions
        simple_templates = [
            "Who is {person}?",
            "What is {thing}?",
            "When was {event}?",
            "Where is {place}?",
            "How many {noun} are in {group}?",
        ]
        for i in range(500):
            template = random.choice(simple_templates)
            all_data.append({"query": template, "label": 0, "source": "synthetic"})

    # Level 1: Single-hop — Natural Questions
    print("Loading Natural Questions (single-hop)...")
    try:
        nq = load_dataset("natural_questions", split="train[:2000]", trust_remote_code=True)
        for item in nq:
            question = item.get("question", "")
            if isinstance(question, dict):
                question = question.get("text", str(question))
            if question:
                all_data.append({
                    "query": question,
                    "label": 1,
                    "source": "natural_questions",
                })
    except Exception as e:
        print(f"Warning: Could not load NQ: {e}")

    # Level 2: Multi-hop — HotpotQA
    print("Loading HotpotQA (multi-hop)...")
    try:
        hotpot = load_dataset("hotpot_qa", "distractor", split="train[:2000]", trust_remote_code=True)
        for item in hotpot:
            all_data.append({
                "query": item["question"],
                "label": 2,
                "source": "hotpotqa",
            })
    except Exception as e:
        print(f"Warning: Could not load HotpotQA: {e}")

    # Level 3: Global — Synthetic corpus-level questions
    print("Generating global-level questions...")
    global_templates = [
        "Summarize the main themes across all documents.",
        "What are the common patterns in {topic}?",
        "Give an overview of all the entities mentioned.",
        "What are the overall trends described in the corpus?",
        "Describe the general relationship between all topics discussed.",
    ]
    for i in range(500):
        all_data.append({
            "query": random.choice(global_templates),
            "label": 3,
            "source": "synthetic",
        })

    # Shuffle and split
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # Save
    with open(output_path / "router_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(output_path / "router_val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)

    print(f"Router data: {len(train_data)} train, {len(val_data)} val")
    return train_data, val_data


def prepare_evaluator_data(output_dir: str = "data/processed", max_samples: int = 3000):
    """Prepare training data for the retrieval evaluator.

    Creates (query, document, relevance_label) triples.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = []

    # Positive pairs: query with relevant document
    print("Generating evaluator training data...")
    try:
        hotpot = load_dataset("hotpot_qa", "distractor", split="train[:1000]", trust_remote_code=True)
        for item in hotpot:
            question = item["question"]
            # Use supporting facts as positive evidence
            context_parts = item.get("context", [])
            if isinstance(context_parts, dict):
                titles = context_parts.get("title", [])
                sentences = context_parts.get("sentences", [])
                if titles and sentences:
                    positive_doc = " ".join(sentences[0]) if isinstance(sentences[0], list) else str(sentences[0])
                    data.append({"query": question, "document": positive_doc, "label": 1})
            elif isinstance(context_parts, list) and context_parts:
                data.append({"query": question, "document": str(context_parts[0]), "label": 1})
    except Exception as e:
        print(f"Warning: Could not generate evaluator data: {e}")

    # Generate synthetic data if needed
    if len(data) < 500:
        topics = ["science", "history", "geography", "technology", "literature"]
        for i in range(500):
            topic = random.choice(topics)
            query = f"What is the significance of {topic} in modern society?"
            relevant_doc = f"{topic.capitalize()} plays a crucial role in shaping modern society through various contributions."
            irrelevant_doc = "The weather today is sunny with clear skies expected throughout the afternoon."
            data.append({"query": query, "document": relevant_doc, "label": 1})
            data.append({"query": query, "document": irrelevant_doc, "label": 0})

    random.shuffle(data)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    with open(output_path / "evaluator_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(output_path / "evaluator_val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)

    print(f"Evaluator data: {len(train_data)} train, {len(val_data)} val")
    return train_data, val_data


def prepare_evaluation_data(output_dir: str = "data/processed"):
    """Prepare evaluation datasets for benchmarking."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    eval_data = {}

    # HotpotQA dev set
    print("Loading HotpotQA dev set...")
    try:
        hotpot_dev = load_dataset("hotpot_qa", "distractor", split="validation[:500]", trust_remote_code=True)
        eval_data["hotpotqa"] = [
            {"question": item["question"], "answer": item["answer"], "type": item.get("type", "bridge")}
            for item in hotpot_dev
        ]
    except Exception as e:
        print(f"Warning: Could not load HotpotQA dev: {e}")

    # TriviaQA dev set
    print("Loading TriviaQA dev set...")
    try:
        trivia_dev = load_dataset("trivia_qa", "rc", split="validation[:500]", trust_remote_code=True)
        eval_data["triviaqa"] = [
            {"question": item["question"], "answer": item["answer"]["value"] if isinstance(item["answer"], dict) else item["answer"]}
            for item in trivia_dev
        ]
    except Exception as e:
        print(f"Warning: Could not load TriviaQA dev: {e}")

    # Save each dataset
    for name, items in eval_data.items():
        with open(output_path / f"eval_{name}.json", "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        print(f"Saved {len(items)} evaluation items for {name}")

    return eval_data


if __name__ == "__main__":
    prepare_router_data()
    prepare_evaluator_data()
    prepare_evaluation_data()
