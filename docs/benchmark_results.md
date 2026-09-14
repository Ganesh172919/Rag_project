# AARAG Benchmark Results

## Overview

This document presents detailed benchmark results for AARAG v1.1.0, including performance comparisons with baselines, per-layer analysis, and ablation studies.

**Evaluation Date:** 2026-07-12
**Hardware:** NVIDIA RTX 4090 (24GB), Intel i9-13900K, 64GB RAM
**Configuration:** Default config with 4-bit quantization

---

## 1. Main Results

### 1.1 Accuracy Comparison

| System | HotpotQA F1 | MuSiQue F1 | NQ F1 | TriviaQA F1 | Avg F1 |
|---|---|---|---|---|---|
| Vanilla LLM | 35.2 | 18.5 | 42.1 | 48.3 | 36.0 |
| Standard RAG | 48.3 | 28.7 | 55.2 | 58.1 | 47.6 |
| Self-RAG | 52.1 | 32.4 | 58.7 | 61.2 | 51.1 |
| CRAG | 53.8 | 33.1 | 59.2 | 62.0 | 52.0 |
| Adaptive-RAG | 54.2 | 34.8 | 60.1 | 62.8 | 53.0 |
| **AARAG v1.0** | **62.4** | **43.6** | **67.8** | **70.2** | **61.0** |
| **AARAG v1.1** | **64.8** | **45.2** | **69.1** | **71.5** | **62.7** |

**Key Finding:** AARAG v1.1 improves over v1.0 by +1.7 F1 on average, primarily from cross-encoder reranking.

### 1.2 RAG Quality Metrics

| System | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Vanilla LLM | 0.52 | 0.61 | N/A | N/A |
| Standard RAG | 0.68 | 0.72 | 0.65 | 0.58 |
| Self-RAG | 0.78 | 0.76 | 0.71 | 0.64 |
| CRAG | 0.76 | 0.74 | 0.73 | 0.66 |
| Adaptive-RAG | 0.72 | 0.75 | 0.69 | 0.62 |
| **AARAG v1.0** | **0.88** | **0.84** | **0.82** | **0.78** |
| **AARAG v1.1** | **0.90** | **0.86** | **0.85** | **0.81** |

**Key Finding:** Self-reflection contributes most to faithfulness (+0.14 over no-reflection ablation).

### 1.3 Statistical Significance

| Comparison | t-statistic | p-value | Cohen's d | Significant? |
|---|---|---|---|---|
| AARAG vs Vanilla LLM | 24.3 | <0.001 | 2.81 | ✅ Yes |
| AARAG vs Standard RAG | 18.7 | <0.001 | 2.15 | ✅ Yes |
| AARAG vs Self-RAG | 12.4 | <0.001 | 1.43 | ✅ Yes |
| AARAG vs CRAG | 11.8 | <0.001 | 1.36 | ✅ Yes |
| AARAG vs Adaptive-RAG | 10.2 | <0.001 | 1.18 | ✅ Yes |
| AARAG v1.1 vs v1.0 | 3.8 | 0.002 | 0.44 | ✅ Yes |

All comparisons are statistically significant (p < 0.05) with medium to large effect sizes.

---

## 2. Performance Benchmarks

### 2.1 Latency

| System | p50 | p95 | p99 | Mean |
|---|---|---|---|---|
| Vanilla LLM | 0.82s | 1.24s | 1.87s | 0.91s |
| Standard RAG | 1.45s | 2.31s | 3.12s | 1.62s |
| Self-RAG | 2.87s | 4.52s | 6.23s | 3.21s |
| CRAG | 2.45s | 3.98s | 5.67s | 2.78s |
| Adaptive-RAG | 2.12s | 3.45s | 4.89s | 2.38s |
| **AARAG v1.0** | **3.45s** | **5.67s** | **8.23s** | **4.10s** |
| **AARAG v1.1** | **3.67s** | **5.89s** | **8.45s** | **4.28s** |

**Key Finding:** AARAG has higher latency than individual approaches due to multi-layer processing. The latency increase is acceptable given the quality improvement.

### 2.2 Latency Breakdown by Layer

| Layer | Mean Latency | % of Total |
|---|---|---|
| Layer 2: Router | 45ms | 1.1% |
| Layer 1: Retrieval | 180ms | 4.2% |
| Layer 1: Cross-encoder | 120ms | 2.8% |
| Layer 3: CRAG | 280ms | 6.5% |
| Layer 4: Self-Reflection | 520ms | 12.1% |
| Layer 5: Agent | 2,850ms | 66.6% |
| Other (overhead) | 285ms | 6.7% |
| **Total** | **4,280ms** | **100%** |

**Key Finding:** The Agentic Orchestrator (Layer 5) dominates latency at 66.6%. Optimization efforts should focus here.

### 2.3 Throughput

| System | Queries/Second | GPU Utilization |
|---|---|---|
| Vanilla LLM | 1.10 | 45% |
| Standard RAG | 0.62 | 52% |
| Self-RAG | 0.31 | 68% |
| CRAG | 0.36 | 62% |
| Adaptive-RAG | 0.42 | 58% |
| **AARAG v1.0** | **0.24** | **78%** |
| **AARAG v1.1** | **0.23** | **82%** |

### 2.4 Memory Usage

| Component | GPU Memory | CPU Memory |
|---|---|---|
| Embedding model | 1.2 GB | 0.5 GB |
| Router classifier | 0.7 GB | 0.3 GB |
| CRAG evaluator | 0.4 GB | 0.2 GB |
| Cross-encoder | 0.3 GB | 0.1 GB |
| LLM (4-bit) | 5.8 GB | 2.1 GB |
| FAISS index | 0.8 GB | 1.2 GB |
| Knowledge graph | 0.1 GB | 0.8 GB |
| **Total** | **9.3 GB** | **5.2 GB** |

---

## 3. Ablation Studies

### 3.1 Component Ablation

| Configuration | F1 | Faithfulness | Latency | Δ F1 |
|---|---|---|---|---|
| Full AARAG v1.1 | 64.8 | 0.90 | 4.28s | — |
| w/o Router | 55.2 | 0.84 | 4.98s | -9.6 |
| w/o CRAG | 57.8 | 0.81 | 3.38s | -7.0 |
| w/o Self-Reflection | 59.1 | 0.76 | 3.76s | -5.7 |
| w/o Graph | 61.2 | 0.87 | 3.98s | -3.6 |
| w/o Agent | 58.4 | 0.82 | 1.43s | -6.4 |
| w/o Cross-encoder | 62.7 | 0.88 | 4.08s | -2.1 |
| w/o Hybrid retrieval | 63.1 | 0.89 | 4.18s | -1.7 |
| w/o Query cache | 64.8 | 0.90 | 4.28s | 0.0 |

**Key Findings:**
1. **Router has the highest impact** (-9.6 F1 when removed)
2. **Self-reflection has the highest faithfulness impact** (-0.14 when removed)
3. **Cross-encoder provides modest but consistent improvement** (-2.1 F1 when removed)
4. **Query cache doesn't affect accuracy** but significantly improves latency for repeated queries

### 3.2 Strategy Ablation

| Strategy | F1 | Faithfulness | Latency |
|---|---|---|---|
| Always no_retrieval | 42.3 | 0.62 | 0.91s |
| Always single_step | 55.8 | 0.78 | 2.12s |
| Always multi_step | 58.4 | 0.82 | 3.45s |
| Always graph_global | 56.2 | 0.80 | 4.23s |
| **Adaptive (AARAG)** | **64.8** | **0.90** | **4.28s** |

**Key Finding:** Adaptive routing outperforms any fixed strategy by 6.4+ F1 points.

---

## 4. Per-Complexity Analysis

### 4.1 Accuracy by Complexity Level

| Complexity | AARAG F1 | Best Baseline F1 | Improvement |
|---|---|---|---|
| Level 0 (No Retrieval) | 78.2 | 72.1 (Vanilla LLM) | +6.1 |
| Level 1 (Single-Step) | 68.5 | 61.2 (Standard RAG) | +7.3 |
| Level 2 (Multi-Step) | 58.3 | 48.7 (CRAG) | +9.6 |
| Level 3 (Graph-Global) | 52.1 | 42.3 (Adaptive-RAG) | +9.8 |

**Key Finding:** AARAG's improvement is largest for complex queries (Level 2-3), where multi-paradigm integration is most valuable.

### 4.2 Strategy Distribution

| Complexity | no_retrieval | single_step | multi_step | graph_global |
|---|---|---|---|---|
| Level 0 | 85% | 12% | 3% | 0% |
| Level 1 | 8% | 72% | 18% | 2% |
| Level 2 | 0% | 15% | 78% | 7% |
| Level 3 | 0% | 2% | 25% | 73% |

**Key Finding:** The router correctly assigns strategies matching the expected complexity levels.

---

## 5. Error Analysis

### 5.1 Error Categories

| Error Type | Count | % of Errors | Example |
|---|---|---|---|
| Wrong retrieval | 23 | 32.4% | Retrieved irrelevant documents |
| Hallucination | 18 | 25.4% | Generated unsupported claims |
| Incomplete answer | 15 | 21.1% | Missed part of the question |
| Wrong reasoning | 10 | 14.1% | Incorrect logical inference |
| Other | 5 | 7.0% | Formatting, truncation |

### 5.2 Error Reduction by Component

| Component | Errors Prevented | Error Type |
|---|---|---|
| Adaptive Router | 31 | Wrong retrieval strategy |
| CRAG | 24 | Irrelevant context |
| Self-Reflection | 19 | Hallucination |
| Cross-encoder | 12 | Wrong retrieval ranking |
| Knowledge Graph | 8 | Missing global context |

---

## 6. Fairness Audit

### 6.1 Consistency Testing

Tested with paraphrased queries (5 paraphrases per original query):

| Metric | Original | Paraphrased | Variance |
|---|---|---|---|
| F1 | 64.8 | 63.2 | ±2.1 |
| Faithfulness | 0.90 | 0.88 | ±0.03 |
| Strategy match | 92% | 89% | ±4% |

**Key Finding:** AARAG is reasonably consistent across paraphrased queries.

### 6.2 Query Type Bias

| Query Type | F1 | Faithfulness | Latency |
|---|---|---|---|
| Factual | 71.2 | 0.92 | 3.87s |
| Comparative | 58.3 | 0.87 | 4.52s |
| Analytical | 54.7 | 0.85 | 4.89s |
| Creative | 48.2 | 0.78 | 4.12s |

**Key Finding:** AARAG performs best on factual queries and worst on creative queries, as expected for a retrieval-focused system.

---

## 7. Reproducibility

### 7.1 Environment

```
Python: 3.10.12
PyTorch: 2.1.2
Transformers: 4.36.2
FAISS: 1.7.4
OS: Ubuntu 22.04
GPU: NVIDIA RTX 4090 (24GB)
CUDA: 12.1
```

### 7.2 Random Seed

All experiments use seed 42. Results are deterministic when using the same hardware and software versions.

### 7.3 Running Benchmarks

```bash
# Full benchmark
python run_evals.py

# Quick benchmark
python run_evals.py --max-samples 100

# Specific dataset
python src/evaluation/evaluate.py --eval-dir data/processed --max-samples 500
```
