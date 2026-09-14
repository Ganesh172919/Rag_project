"""RAGAS Evaluation — Integration with RAGAS framework."""

from typing import List, Dict, Optional


def evaluate_with_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
) -> Dict[str, float]:
    """Evaluate using RAGAS framework.

    Args:
        questions: List of questions
        answers: List of generated answers
        contexts: List of retrieved contexts (each is a list of strings)
        ground_truths: List of ground truth answers

    Returns:
        Dict with RAGAS metrics
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        # Create dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(data)

        # Evaluate
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        return {
            "faithfulness": result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"],
            "context_precision": result["context_precision"],
            "context_recall": result["context_recall"],
        }

    except ImportError:
        print("RAGAS not installed. Install with: pip install ragas")
        return _fallback_ragas_evaluation(questions, answers, contexts, ground_truths)
    except Exception as e:
        print(f"RAGAS evaluation error: {e}")
        return _fallback_ragas_evaluation(questions, answers, contexts, ground_truths)


def _fallback_ragas_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
) -> Dict[str, float]:
    """Fallback RAGAS-style evaluation without the RAGAS library."""
    import numpy as np

    def normalize(text):
        return set(text.lower().split()) - {"the", "a", "an", "is", "are", "was", "were", "in", "of", "to"}

    # Faithfulness: answer claims supported by context
    faithfulness_scores = []
    for answer, ctx_list in zip(answers, contexts):
        ctx_text = " ".join(ctx_list).lower()
        answer_words = normalize(answer)
        if answer_words:
            found = sum(1 for w in answer_words if w in ctx_text)
            faithfulness_scores.append(found / len(answer_words))
        else:
            faithfulness_scores.append(0.5)

    # Answer relevancy: answer relevance to question
    relevancy_scores = []
    for question, answer in zip(questions, answers):
        q_words = normalize(question)
        a_words = normalize(answer)
        if q_words:
            overlap = len(q_words & a_words)
            relevancy_scores.append(min(overlap / len(q_words), 1.0))
        else:
            relevancy_scores.append(0.5)

    # Context precision: relevant docs ranked high
    precision_scores = []
    for question, ctx_list in zip(questions, contexts):
        q_words = normalize(question)
        if ctx_list and q_words:
            scores = []
            for ctx in ctx_list:
                c_words = normalize(ctx)
                overlap = len(q_words & c_words)
                scores.append(overlap / len(q_words))
            precision_scores.append(np.mean(scores) if scores else 0.5)
        else:
            precision_scores.append(0.5)

    # Context recall: ground truth covered by context
    recall_scores = []
    for gt, ctx_list in zip(ground_truths, contexts):
        gt_words = normalize(gt)
        ctx_text = " ".join(ctx_list).lower()
        if gt_words:
            found = sum(1 for w in gt_words if w in ctx_text)
            recall_scores.append(found / len(gt_words))
        else:
            recall_scores.append(0.5)

    return {
        "faithfulness": np.mean(faithfulness_scores),
        "answer_relevancy": np.mean(relevancy_scores),
        "context_precision": np.mean(precision_scores),
        "context_recall": np.mean(recall_scores),
    }
