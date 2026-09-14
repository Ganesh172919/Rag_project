"""Metrics — Custom evaluation metrics for AARAG."""

import re
import string
from collections import Counter
from typing import List, Dict, Optional

import numpy as np


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    # Lowercase
    s = s.lower()
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove extra whitespace
    s = " ".join(s.split())
    return s.strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def exact_match_score(predictions: List[str], references: List[str]) -> float:
    """Compute exact match score over a batch."""
    assert len(predictions) == len(references)
    return np.mean([exact_match(p, r) for p, r in zip(predictions, references)])


def f1_score_single(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score for a single prediction."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def f1_score_batch(predictions: List[str], references: List[str]) -> float:
    """Compute average F1 score over a batch."""
    assert len(predictions) == len(references)
    return np.mean([f1_score_single(p, r) for p, r in zip(predictions, references)])


def recall_score_single(prediction: str, ground_truth: str) -> float:
    """Compute token-level recall for a single prediction."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    return num_common / len(gt_tokens)


def precision_score_single(prediction: str, ground_truth: str) -> float:
    """Compute token-level precision for a single prediction."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    return num_common / len(pred_tokens)


def rouge_l_score(prediction: str, reference: str) -> float:
    """Compute ROUGE-L score."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Compute LCS
    lcs_length = _lcs_length(pred_tokens, ref_tokens)

    if lcs_length == 0:
        return 0.0

    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def answer_relevancy(prediction: str, question: str) -> float:
    """Estimate answer relevancy using keyword overlap."""
    q_words = set(normalize_answer(question).split()) - {"what", "who", "when", "where", "how", "why", "which", "is", "are", "was", "were"}
    a_words = set(normalize_answer(prediction).split())

    if not q_words:
        return 0.5

    overlap = len(q_words & a_words)
    return min(overlap / len(q_words), 1.0)


def faithfulness_score(answer: str, context: str) -> float:
    """Estimate faithfulness of answer to context."""
    answer_words = set(normalize_answer(answer).split()) - {"the", "a", "an", "is", "are", "was", "were", "and", "or", "but"}
    context_lower = normalize_answer(context)

    if not answer_words:
        return 0.5

    found = sum(1 for w in answer_words if w in context_lower)
    return min(found / len(answer_words), 1.0)


def bert_score_single(prediction: str, reference: str) -> float:
    """Compute BERTScore for a single prediction-reference pair.

    Uses token-level cosine similarity with BERT embeddings.
    Falls back to word overlap if sentence-transformers is not available.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        pred_emb = model.encode(prediction, convert_to_numpy=True)
        ref_emb = model.encode(reference, convert_to_numpy=True)
        cosine_sim = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
        return float(max(0.0, min(1.0, cosine_sim)))
    except ImportError:
        # Fallback: word overlap approximation
        pred_words = set(normalize_answer(prediction).split())
        ref_words = set(normalize_answer(reference).split())
        if not pred_words or not ref_words:
            return 0.0
        overlap = len(pred_words & ref_words)
        return overlap / max(len(pred_words), len(ref_words))


def bert_score_batch(predictions: List[str], references: List[str]) -> float:
    """Compute average BERTScore over a batch."""
    assert len(predictions) == len(references)
    return float(np.mean([bert_score_single(p, r) for p, r in zip(predictions, references)]))


def mrr_score(predictions: List[str], references: List[str]) -> float:
    """Compute Mean Reciprocal Rank.

    For each reference, find its rank in the predictions list
    (useful for retrieval evaluation).
    """
    if not predictions or not references:
        return 0.0

    reciprocal_ranks = []
    for ref in references:
        ref_norm = normalize_answer(ref)
        for rank, pred in enumerate(predictions, 1):
            if normalize_answer(pred) == ref_norm:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks))


def answer_correctness(prediction: str, reference: str) -> float:
    """Compute answer correctness as a weighted combination of EM and F1.

    Correctness = 0.5 * EM + 0.5 * F1
    """
    em = exact_match(prediction, reference)
    f1 = f1_score_single(prediction, reference)
    return 0.5 * em + 0.5 * f1


def context_relevance(context: str, question: str) -> float:
    """Compute how relevant the context is to the question.

    Uses keyword overlap between context and question.
    """
    q_words = set(normalize_answer(question).split()) - {
        "what", "who", "when", "where", "how", "why", "which",
        "is", "are", "was", "were", "the", "a", "an",
    }
    c_words = set(normalize_answer(context).split())

    if not q_words:
        return 0.5

    overlap = len(q_words & c_words)
    return min(overlap / len(q_words), 1.0)


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    questions: Optional[List[str]] = None,
    contexts: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        predictions: Model predictions
        references: Ground truth answers
        questions: Original questions (for relevancy)
        contexts: Retrieved contexts (for faithfulness)

    Returns:
        Dict of metric_name -> score
    """
    metrics = {
        "exact_match": exact_match_score(predictions, references),
        "f1": f1_score_batch(predictions, references),
        "rouge_l": np.mean([rouge_l_score(p, r) for p, r in zip(predictions, references)]),
        "bert_score": bert_score_batch(predictions, references),
        "answer_correctness": float(np.mean([
            answer_correctness(p, r) for p, r in zip(predictions, references)
        ])),
    }

    if questions:
        metrics["answer_relevancy"] = np.mean([
            answer_relevancy(p, q) for p, q in zip(predictions, questions)
        ])

    if contexts:
        metrics["faithfulness"] = np.mean([
            faithfulness_score(p, c) for p, c in zip(predictions, contexts)
        ])
        metrics["context_relevance"] = np.mean([
            context_relevance(c, q) for c, q in zip(contexts, questions or [""] * len(contexts))
        ])

    return metrics
