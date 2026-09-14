# AARAG Research Methodology

## 1. Research Questions

This M.Research project investigates the following questions:

### RQ1: Effectiveness of Unified RAG
**Can combining multiple RAG paradigms (adaptive routing, corrective retrieval, self-reflection, agentic orchestration) into a single system improve answer quality over individual approaches?**

### RQ2: Adaptive Routing Accuracy
**Can a learned classifier accurately determine query complexity and route to the optimal retrieval strategy?**

### RQ3: Self-Correction Effectiveness
**Do self-reflection tokens and corrective retrieval reduce hallucination and improve faithfulness?**

### RQ4: Agentic Coordination Value
**Does agentic orchestration with tool-use improve multi-step reasoning over static pipelines?**

---

## 2. Experimental Design

### 2.1 Evaluation Protocol

**Datasets:**
| Dataset | Size | Task | Complexity Level |
|---|---|---|---|
| HotpotQA | 113K | Multi-hop QA | Level 2 |
| MuSiQue | 25K | Hard multi-hop | Level 2 |
| Natural Questions | 307K | Single-hop QA | Level 1 |
| TriviaQA | 95K | Simple factual | Level 0 |
| 2WikiMultiHop | 193K | Structured multi-hop | Level 2 |
| ASQA | 6.3K | Ambiguous QA | Level 1-2 |
| SQuAD 2.0 | 150K+ | Reading comprehension | Level 1 |

**Evaluation Split:**
- Training: 70% (for trainable components)
- Validation: 10% (for hyperparameter tuning)
- Test: 20% (for final evaluation, never seen during training)

**Sample Size:**
- Main evaluation: 500 samples per dataset (stratified by complexity)
- Ablation studies: 200 samples per configuration
- Statistical significance: Paired t-test with p < 0.05

### 2.2 Baselines

| Baseline | Description | Purpose |
|---|---|---|
| Vanilla LLM | LLM without retrieval | Lower bound |
| Standard RAG | Simple retrieve-then-read | Basic RAG baseline |
| Self-RAG | Self-reflective retrieval | Self-correction baseline |
| CRAG | Corrective retrieval | Correction baseline |
| Adaptive-RAG | Query routing only | Routing baseline |
| **AARAG** | Full unified system | Our approach |

### 2.3 Ablation Configurations

| Config | Disabled Component | Purpose |
|---|---|---|
| Full AARAG | None | Baseline |
| w/o Router | Adaptive Router | Is routing necessary? |
| w/o CRAG | Corrective Retrieval | Is correction necessary? |
| w/o Self-Ref | Self-Reflection | Is reflection necessary? |
| w/o Graph | Knowledge Graph | Is graph reasoning necessary? |
| w/o Agent | Agentic Orchestrator | Is orchestration necessary? |
| w/o Reranker | Cross-encoder | Is reranking necessary? |
| w/o Hybrid | Hybrid retrieval | Is hybrid retrieval necessary? |

---

## 3. Metrics

### 3.1 Accuracy Metrics

**Exact Match (EM)**
```
EM = 1 if normalize(prediction) == normalize(reference) else 0
```
- Normalization: lowercase, remove punctuation, remove articles (a, an, the)

**F1 Score (Token-Level)**
```
F1 = 2 * precision * recall / (precision + recall)
precision = |predicted_tokens ∩ reference_tokens| / |predicted_tokens|
recall = |predicted_tokens ∩ reference_tokens| / |reference_tokens|
```

**ROUGE-L**
```
ROUGE-L = F_lcs = 2 * P_lcs * R_lcs / (P_lcs + R_lcs)
```

### 3.2 RAG Quality Metrics

**Faithfulness**
```
faithfulness = |answer_claims ∩ context_claims| / |answer_claims|
```
Measures how well the answer is grounded in retrieved context.

**Answer Relevancy**
```
relevancy = |query_keywords ∩ answer_keywords| / |query_keywords|
```
Measures how relevant the answer is to the question.

**Context Precision**
```
precision = avg(relevance_score(doc_i) for doc_i in retrieved_docs)
```
Measures how relevant the retrieved documents are.

**Context Recall**
```
recall = |ground_truth_keywords ∩ context_keywords| / |ground_truth_keywords|
```
Measures how well the context covers the ground truth.

### 3.3 Semantic Metrics

**BERTScore**
```
BERTScore = F1 of token-level cosine similarities using BERT embeddings
```
Captures semantic similarity beyond exact token match.

**MRR (Mean Reciprocal Rank)**
```
MRR = 1/|Q| * Σ 1/rank_i
```
Measures retrieval quality (how early the correct document appears).

### 3.4 Performance Metrics

**Latency Percentiles:**
- p50 (median): Typical response time
- p95: 95th percentile (tail latency)
- p99: 99th percentile (worst-case)

**Throughput:**
```
throughput = total_queries / total_time (queries/second)
```

**Memory Usage:**
- Peak GPU memory (MB)
- Peak CPU memory (MB)

---

## 4. Statistical Methods

### 4.1 Paired t-test

Compare AARAG vs each baseline on the same test set:

```python
from scipy import stats

t_stat, p_value = stats.ttest_rel(aarag_scores, baseline_scores)
# Significant if p < 0.05
```

### 4.2 Bootstrap Confidence Intervals

95% confidence intervals for all metrics:

```python
import numpy as np

def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper
```

### 4.3 Effect Size

Cohen's d for practical significance:

```python
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std
```

Interpretation:
- |d| < 0.2: Negligible
- 0.2 ≤ |d| < 0.5: Small
- 0.5 ≤ |d| < 0.8: Medium
- |d| ≥ 0.8: Large

---

## 5. Reproducibility

### 5.1 Random Seed

All experiments use seed 42 (configurable in `config/default.yaml`).

### 5.2 Environment

```bash
# Freeze environment
pip freeze > requirements_frozen.txt

# Docker (recommended)
docker build -t aarag .
docker run aarag python src/evaluation/evaluate.py --full
```

### 5.3 Reporting Checklist

When reporting results, include:
- [ ] Dataset version and split
- [ ] Model versions (embedding, LLM, classifiers)
- [ ] Hardware used (GPU type, RAM, CPU)
- [ ] Random seed
- [ ] Number of evaluation samples
- [ ] Confidence intervals
- [ ] Statistical significance tests
- [ ] Effect sizes

---

## 6. Threats to Validity

### 6.1 Internal Validity
- **Heuristic fallbacks:** When transformer models are unavailable, rule-based fallbacks are used, which may not reflect full system performance
- **LLM variability:** Non-deterministic generation can affect reproducibility (mitigated by temperature=0.1)

### 6.2 External Validity
- **Dataset bias:** Evaluation datasets may not represent all real-world query types
- **Domain specificity:** System is primarily evaluated on English-language factual QA

### 6.3 Construct Validity
- **Metric limitations:** Token-overlap metrics (EM, F1) may not capture answer quality nuances
- **Faithfulness definition:** Keyword-overlap faithfulness is an approximation of true grounding

### 6.4 Mitigation Strategies
- Multiple datasets across complexity levels
- Both automatic and human evaluation
- Statistical significance testing
- Confidence intervals for all metrics
- Detailed error analysis
