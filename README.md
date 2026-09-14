# 🔍 AARAG: Adaptive Agentic Retrieval-Augmented Generation

### M.Research AI Capstone Project — July 2026

![Version](https://img.shields.io/badge/version-1.1.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-academic-orange)
![Tests](https://img.shields.io/badge/tests-107-brightgreen)

A unified framework combining four state-of-the-art RAG paradigms into a single cohesive system with self-correction, adaptive routing, and agentic orchestration.

### v1.1.0 — Innovation Release
- ✅ Cross-encoder reranking for improved retrieval precision
- ✅ Hybrid retrieval (BM25 + vector dense) with Reciprocal Rank Fusion
- ✅ Semantic query caching (10-50x latency reduction for repeated queries)
- ✅ Confidence calibration (temperature/Platt scaling)
- ✅ Structured JSON logging with pipeline tracing
- ✅ Standardized benchmarking with latency percentiles
- ✅ Error analysis and fairness audit modules

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Research Papers](#research-papers)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Project Structure](#project-structure)
- [Evaluation](#evaluation)
- [Training](#training)
- [Demo](#demo)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Overview

**AARAG** addresses three critical failure modes of Large Language Models:

| Failure Mode | Description | AARAG Solution |
|---|---|---|
| **Hallucination** | Generating plausible but incorrect facts | Self-Reflection tokens verify claims against context |
| **Knowledge Staleness** | Parametric knowledge has a cutoff | Corrective Retrieval with web search fallback |
| **Retrieval Brittleness** | Standard RAG retrieves irrelevant docs | Adaptive Router + CRAG quality evaluator |

### Key Features

- 🎯 **Adaptive Query Routing** — Classifies query complexity (0-3) and routes to optimal strategy
- 🔧 **Corrective Retrieval** — Evaluates retrieval quality and falls back to web search when needed
- 🪞 **Self-Reflection** — Generates `[IsRel]`, `[IsSup]`, `[IsUse]` tokens for self-critique
- 🌐 **Graph-Based Reasoning** — Knowledge graph for entity relationships and global queries
- 🤖 **Agentic Orchestration** — ReAct agent coordinates all layers with tool-use
- 🔄 **Cross-Encoder Reranking** — Improves retrieval precision with cross-encoder scoring
- 🔀 **Hybrid Retrieval** — Combines BM25 sparse + vector dense with Reciprocal Rank Fusion
- 💾 **Semantic Query Cache** — Caches responses for similar queries (10-50x faster)
- 📏 **Confidence Calibration** — Temperature/Platt scaling for accurate confidence scores
- 📊 **Comprehensive Evaluation** — 5 baselines, 8 ablations, 11 metrics, error analysis, fairness audit
- 📝 **Structured Logging** — JSON logs, pipeline tracing, performance profiling

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 5: AGENTIC ORCHESTRATOR                    │
│   ReAct Agent — plans, decides, coordinates all layers              │
│   Tools: [retrieve, graph_query, web_search, self_reflect, verify]  │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 4: SELF-REFLECTION ENGINE                   │
│   Reflection Tokens: [Retrieve] [IsRel] [IsSup] [IsUse]            │
│   Critique → Revise → Verify pipeline (max 2 retries)              │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 3: CORRECTIVE RETRIEVAL                     │
│   Retrieval Evaluator → Confidence Score → Action Selection         │
│   CORRECT (>0.8) | AMBIGUOUS (0.3-0.8) | INCORRECT (<0.3)          │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 2: ADAPTIVE ROUTER                          │
│   DeBERTa-v3 classifier → 4 complexity levels                       │
│   L0: No-Retrieval | L1: Single-Step | L2: Multi-Step | L3: Graph  │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 1: KNOWLEDGE SOURCES                        │
│   Vector DB (FAISS) | Knowledge Graph (NetworkX/Neo4j) | Web APIs   │
│   Embeddings: BAAI/bge-large-en-v1.5 | Chunk: 512 tokens           │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

```
User Query
    │
    ▼
┌──────────────────┐
│  2. Adaptive     │──→ Complexity: 0, 1, 2, or 3
│     Router       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────┐
│  1. Knowledge    │────→│  3. Corrective   │──→ Confidence check
│     Sources      │     │     Retrieval    │──→ Web search fallback
└──────────────────┘     └────────┬─────────┘
                                  │
                                  ▼
                       ┌──────────────────┐
                       │  4. Self-        │──→ [IsRel] [IsSup] [IsUse]
                       │     Reflection   │──→ Revise if needed
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  5. Agentic      │──→ Final synthesis
                       │     Orchestrator │──→ Confidence score
                       └────────┬─────────┘
                                │
                                ▼
                          Final Answer + Trace
```

---

## Research Papers

| Component | Paper | Venue | arXiv |
|---|---|---|---|
| Adaptive Router | Adaptive-RAG (Jeong et al.) | NAACL 2024 | [2403.14403](https://arxiv.org/abs/2403.14403) |
| Self-Reflection | Self-RAG (Asai et al.) | ICLR 2024 | [2310.11511](https://arxiv.org/abs/2310.11511) |
| Corrective Retrieval | CRAG (Yan et al.) | ICML 2024 | [2401.15884](https://arxiv.org/abs/2401.15884) |
| Graph Reasoning | GraphRAG (Edge et al.) | Microsoft 2024 | [2404.16130](https://arxiv.org/abs/2404.16130) |
| Agent Framework | ReAct (Yao et al.) | ICLR 2023 | [2210.03629](https://arxiv.org/abs/2210.03629) |
| RAG Survey | RAG for LLMs (Gao et al.) | 2024 | [2312.10997](https://arxiv.org/abs/2312.10997) |

---

## Installation

### Prerequisites

- Python 3.10+
- pip or conda
- GPU recommended (RTX 3060+ or A100)

### Step 1: Clone and Setup

```bash
cd ~/AARAG
python -m venv aarag-env
source aarag-env/bin/activate  # Linux/Mac
# aarag-env\Scripts\activate   # Windows

pip install -r requirements.txt
```

### Step 2: Download Models

```bash
python scripts/download_models.py
```

This downloads:
- `BAAI/bge-large-en-v1.5` — Embedding model (1.3GB)
- `microsoft/deberta-v3-base` — Router classifier (400MB)
- `microsoft/deberta-v3-small` — Retrieval evaluator (200MB)

### Step 3: Verify Installation

```bash
python -m pytest tests/ -v
```

Expected: `40 passed`

---

## Quick Start

### 1. Ingest Documents

```python
from src.aarag import AARAG

aarag = AARAG(config_path="config/default.yaml")
stats = aarag.ingest("data/my_documents/")
print(stats)
# {'documents_loaded': 50, 'chunks_created': 312, 'vectors_added': 312, 'kg_nodes': 156}
```

### 2. Query

```python
response = aarag.query("Compare the economic policies of the US and China")

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence:.2f}")
print(f"Strategy: {response.strategy_used}")
print(f"Latency: {response.latency:.2f}s")
```

### 3. Launch Demo

```bash
streamlit run demo/app.py
```

Open http://localhost:8501

---

## Detailed Usage

### Ingesting Data

```python
# Ingest a single file
aarag.ingest("data/report.pdf")

# Ingest a directory (recursive)
aarag.ingest("data/corpus/", build_kg=True)

# Custom chunk settings
aarag.ingest("data/docs/", chunk_size=256, chunk_overlap=30)
```

### Querying with Options

```python
# Full pipeline with all features
response = aarag.query("What is machine learning?")

# Disable web fallback (local-only)
response = aarag.query("What is ML?", enable_web_fallback=False)

# Disable self-reflection (faster)
response = aarag.query("What is ML?", enable_self_reflection=False)

# Force a specific strategy
response = aarag.query("Simple question", strategy_override="no_retrieval")
response = aarag.query("Complex question", strategy_override="multi_step")
```

### Response Object

```python
response = aarag.query("Your question")

# Main output
response.answer              # str: The generated answer
response.confidence          # float: Overall confidence (0-1)
response.latency             # float: End-to-end time in seconds
response.strategy_used       # str: Which strategy was used
response.sources             # List[str]: Sources used

# Layer details
response.routing_decision    # dict: Router output
response.corrective_result   # dict: CRAG output
response.reflection_result   # dict: Self-reflection tokens
response.agent_response      # dict: Agent reasoning trace
```

### Saving and Loading

```python
# Save all components
aarag.save("models/aarag_v1/")

# Load in a new session
aarag = AARAG()
aarag.load("models/aarag_v1/")
```

---

## Project Structure

```
AARAG/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── setup.py                               # Package setup
├── config/
│   └── default.yaml                       # Default configuration
│
├── src/                                   # Source code
│   ├── __init__.py
│   ├── aarag.py                           # Main AARAG pipeline
│   │
│   ├── layers/                            # 5-Layer architecture
│   │   ├── knowledge_sources/             # Layer 1
│   │   │   ├── vector_store.py            #   FAISS/ChromaDB
│   │   │   ├── knowledge_graph.py         #   NetworkX/Neo4j
│   │   │   ├── web_search.py              #   DuckDuckGo/Tavily
│   │   │   └── document_loader.py         #   PDF/TXT/JSON loader
│   │   │
│   │   ├── adaptive_router/               # Layer 2
│   │   │   ├── classifier.py              #   DeBERTa complexity classifier
│   │   │   ├── router.py                  #   Routing logic
│   │   │   └── feature_extractor.py       #   Query features
│   │   │
│   │   ├── corrective_retrieval/          # Layer 3
│   │   │   ├── evaluator.py               #   Retrieval quality scorer
│   │   │   ├── corrector.py               #   CRAG pipeline
│   │   │   ├── decomposer.py              #   Decompose-recompose
│   │   │   └── web_fallback.py            #   Web search fallback
│   │   │
│   │   ├── cross_encoder_reranker.py      #   Cross-encoder reranking (v1.1)
│   │   ├── hybrid_retrieval.py            #   BM25 + dense hybrid (v1.1)
│   │   ├── query_cache.py                 #   Semantic query cache (v1.1)
│   │   ├── confidence_calibrator.py       #   Confidence calibration (v1.1)
│   │   │
│   │   ├── self_reflection/               # Layer 4
│   │   │   ├── reflection_engine.py       #   Main reflection pipeline
│   │   │   ├── token_generator.py         #   [IsRel] [IsSup] [IsUse]
│   │   │   └── verifier.py               #   Claim verification
│   │   │
│   │   └── agentic_orchestrator/          # Layer 5
│   │       ├── agent.py                   #   ReAct agent
│   │       ├── tools.py                   #   Tool registry
│   │       └── planner.py                 #   Multi-step planning
│   │
│   ├── training/                          # Training scripts
│   │   ├── data_preparation.py            #   Dataset preparation
│   │   ├── train_router.py                #   Train complexity classifier
│   │   ├── train_evaluator.py             #   Train retrieval evaluator
│   │   └── train_reflection.py            #   Train reflection tokens
│   │
│   ├── evaluation/                        # Evaluation framework
│   │   ├── metrics.py                     #   EM, F1, ROUGE-L, BERTScore, MRR
│   │   ├── baselines.py                   #   5 baseline implementations
│   │   ├── evaluate.py                    #   Main evaluation runner
│   │   ├── ablation.py                    #   Ablation study runner
│   │   ├── ragas_evaluation.py            #   RAGAS integration
│   │   ├── benchmark.py                   #   Standardized benchmarking (v1.1)
│   │   ├── error_analysis.py              #   Error categorization (v1.1)
│   │   └── fairness_audit.py              #   Bias detection (v1.1)
│   │
│   └── utils/                             # Utilities
│       ├── helpers.py                     #   Common functions
│       ├── logger.py                      #   Rich + JSON logging
│       ├── visualization.py               #   Plot generation
│       ├── structured_logger.py           #   JSON structured logging (v1.1)
│       ├── log_analyzer.py                #   Log analysis tools (v1.1)
│       └── pipeline_tracer.py             #   Pipeline tracing (v1.1)
│
├── demo/
│   └── app.py                             # Streamlit demo
│
├── notebooks/                             # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_router_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_results_analysis.ipynb
│
├── scripts/                               # Shell scripts
│   ├── download_models.py                 #   Download pre-trained models
│   ├── build_knowledge_graph.py           #   Build KG from documents
│   ├── run_evaluation.sh                  #   Run full evaluation
│   └── launch_demo.sh                     #   Launch Streamlit demo
│
├── tests/                                 # Test suite (40 tests)
│   ├── test_router.py                     #   Router tests
│   ├── test_crag.py                       #   CRAG tests
│   ├── test_reflection.py                 #   Reflection tests
│   └── test_integration.py                #   Integration tests
│
├── data/                                  # Data directory
│   ├── raw/                               #   Raw documents
│   ├── processed/                         #   Processed datasets
│   └── knowledge_graph/                   #   KG data
│
└── results/                               # Evaluation results
    ├── tables/                            #   Result tables
    ├── figures/                            #   Plots
    └── logs/                              #   Training logs
```

---

## Evaluation

### Datasets

| Dataset | Task | Size | Complexity |
|---|---|---|---|
| HotpotQA | Multi-hop QA | 113K | Level 2 |
| MuSiQue | Hard multi-hop | 25K | Level 2 |
| Natural Questions | Single-hop QA | 307K | Level 1 |
| TriviaQA | Simple factual | 95K | Level 0 |
| 2WikiMultiHop | Structured multi-hop | 193K | Level 2 |

### Metrics

| Metric | Category | Description |
|---|---|---|
| Exact Match (EM) | Accuracy | Exact answer match |
| F1 Score | Accuracy | Token overlap |
| ROUGE-L | Generation | Longest common subsequence |
| Faithfulness | RAG Quality | Answer grounded in context |
| Answer Relevancy | RAG Quality | Answer relevant to query |
| Context Precision | Retrieval | Relevant docs ranked high |
| Context Recall | Retrieval | All relevant docs found |
| BERTScore | Semantic | Semantic similarity |

### Running Evaluation

```bash
# Full evaluation with ablation studies
python src/evaluation/evaluate.py --full --max-samples 500

# Quick evaluation
python src/evaluation/evaluate.py --max-samples 100

# Generate plots
python src/utils/visualization.py
```

### Baselines

| Baseline | Description |
|---|---|
| Vanilla LLM | LLM without retrieval |
| Standard RAG | Simple retrieve-then-read |
| Self-RAG | Self-reflective retrieval |
| CRAG | Corrective retrieval |
| Adaptive-RAG | Query routing only |
| **AARAG (Ours)** | Full unified system |

### Ablation Studies

| Config | Description | Purpose |
|---|---|---|
| Full AARAG | All components | Baseline |
| w/o Router | Always multi-step | Is routing necessary? |
| w/o CRAG | No corrective retrieval | Is correction necessary? |
| w/o Self-Ref | No reflection tokens | Is self-reflection necessary? |
| w/o Graph | No knowledge graph | Is graph reasoning necessary? |
| w/o Agent | Pipeline only | Is agentic orchestration necessary? |

---

## Training

### Phase 1: Prepare Data

```bash
python src/training/data_preparation.py
```

Downloads and prepares:
- Router training data (HotpotQA, TriviaQA, NQ)
- Evaluator training data (query-document relevance pairs)
- Reflection training data (synthetic)
- Evaluation datasets

### Phase 2: Train Router

```bash
python src/training/train_router.py \
    --epochs 10 \
    --batch-size 32 \
    --lr 2e-5 \
    --output-dir models/router
```

### Phase 3: Train Retrieval Evaluator

```bash
python src/training/train_evaluator.py \
    --epochs 5 \
    --batch-size 16 \
    --lr 3e-5 \
    --output-dir models/evaluator
```

### Phase 4: Train Reflection Tokens (LoRA)

```bash
python src/training/train_reflection.py \
    --epochs 3 \
    --batch-size 4 \
    --lr 1e-5 \
    --lora-rank 16 \
    --output-dir models/reflection
```

### Full Training Pipeline

```bash
# Run all training phases
python src/training/data_preparation.py
python src/training/train_router.py
python src/training/train_evaluator.py
python src/training/train_reflection.py
```

---

## Demo

### Launch

```bash
streamlit run demo/app.py --server.port 8501
```

### Features

- **Interactive Query Input** — Type questions or select examples
- **Layer-by-Layer Trace** — See routing, retrieval, reflection decisions
- **Side-by-Side Comparison** — Compare AARAG vs baselines
- **Confidence Dashboard** — Real-time confidence scores
- **Reflection Token Display** — Visual [IsRel] [IsSup] [IsUse] tokens

### Example Queries

| Query Type | Example |
|---|---|
| Simple factual | "What is the population of Tokyo?" |
| Single-hop | "Who invented the telephone?" |
| Multi-hop | "Compare Python and Java programming languages" |
| Global | "Summarize all main themes across the documents" |
| Ambiguous | "What is the best programming language?" |

---

## Configuration

Edit `config/default.yaml` to customize:

```yaml
# LLM Settings
llm:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  max_new_tokens: 1024
  temperature: 0.1
  load_in_4bit: true

# Knowledge Sources
knowledge_sources:
  vector_store:
    embedding_model: "BAAI/bge-large-en-v1.5"
    chunk_size: 512
    top_k: 5
  knowledge_graph:
    max_hops: 3
  web_search:
    provider: "duckduckgo"
    max_results: 5

# Adaptive Router
adaptive_router:
  confidence_threshold: 0.7

# Corrective Retrieval
corrective_retrieval:
  confidence_thresholds:
    correct: 0.8
    incorrect: 0.3

# Self-Reflection
self_reflection:
  max_retries: 2

# Agentic Orchestrator
agentic_orchestrator:
  max_steps: 10
  planning_strategy: "react"
```

---

## API Reference

### `AARAG(config_path=None, config=None)`

Main entry point for the AARAG system.

**Methods:**

| Method | Description |
|---|---|
| `ingest(data_path, build_kg=True)` | Ingest documents into knowledge sources |
| `query(question, ...)` | Answer a question through the full pipeline |
| `save(path)` | Save all components to disk |
| `load(path)` | Load all components from disk |

### `AARAGResponse`

| Field | Type | Description |
|---|---|---|
| `answer` | str | The generated answer |
| `confidence` | float | Overall confidence (0-1) |
| `latency` | float | End-to-end time (seconds) |
| `strategy_used` | str | Routing strategy used |
| `routing_decision` | dict | Router output |
| `corrective_result` | dict | CRAG output |
| `reflection_result` | dict | Self-reflection tokens |
| `agent_response` | dict | Agent reasoning trace |

### Layer Classes

```python
# Layer 1
from src.layers.knowledge_sources import VectorStore, KnowledgeGraphSearch, WebSearch

# Layer 2
from src.layers.adaptive_router import AdaptiveRouter

# Layer 3
from src.layers.corrective_retrieval import CorrectiveRetrieval

# Layer 4
from src.layers.self_reflection import SelfReflectionEngine

# Layer 5
from src.layers.agentic_orchestrator import AgenticOrchestrator
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3060 (12GB) | A100 (40GB) or RTX 4090 (24GB) |
| RAM | 16GB | 32GB |
| Storage | 50GB | 100GB |
| CPU | 8 cores | 16 cores |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `CUDA out of memory` | Set `llm.load_in_4bit: true` in config |
| `Model not found` | Run `python scripts/download_models.py` |
| `ImportError` | Run `pip install -r requirements.txt` |
| `Tests fail` | Ensure Python 3.10+ and all deps installed |

---

## License

Academic use only — M.Research submission 2026.

---

## Citation

```bibtex
@article{aarag2026,
  title={AARAG: Adaptive Agentic Retrieval-Augmented Generation},
  author={M.Research Student},
  year={2026},
  note={M.Research Capstone Project}
}
```

---

## References

1. Asai, A., et al. "Self-Reflective Retrieval-Augmented Generation." ICLR 2024.
2. Yan, S.-Q., et al. "Corrective Retrieval Augmented Generation." ICML 2024.
3. Jeong, S., et al. "Adaptive-RAG." NAACL 2024.
4. Edge, D., et al. "From Local to Global: A Graph RAG Approach." 2024.
5. Yao, S., et al. "ReAct: Synergizing Reasoning and Acting." ICLR 2023.
6. Lewis, P., et al. "Retrieval-Augmented Generation." NeurIPS 2020.
