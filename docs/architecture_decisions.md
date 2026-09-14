# Architecture Decision Records (ADR)

This document records the key architectural decisions made during the development of AARAG.

---

## ADR-001: Five-Layer Architecture

**Date:** 2026-06-15
**Status:** Accepted

### Context
We need an architecture that can handle diverse query types (simple factual, multi-hop, global) while maintaining clean separation of concerns.

### Decision
Implement a 5-layer architecture:
1. Knowledge Sources (data layer)
2. Adaptive Router (decision layer)
3. Corrective Retrieval (quality layer)
4. Self-Reflection (verification layer)
5. Agentic Orchestrator (coordination layer)

### Consequences
- **Positive:** Clear separation of concerns, each layer can be developed/tested independently
- **Positive:** Layers can be disabled for ablation studies
- **Negative:** Increased complexity, more components to maintain
- **Negative:** Latency accumulates across layers

---

## ADR-002: DeBERTa for Query Classification

**Date:** 2026-06-15
**Status:** Accepted

### Context
We need a classifier to determine query complexity (0-3) for adaptive routing.

### Decision
Use DeBERTa-v3-base as the query complexity classifier.

### Alternatives Considered
| Model | Accuracy | Speed | Parameters |
|---|---|---|---|
| BERT-base | 84.2% | Fast | 110M |
| RoBERTa-base | 86.1% | Fast | 125M |
| **DeBERTa-v3-base** | **88.3%** | **Medium** | **184M** |
| T5-base | 85.7% | Slow | 220M |

### Consequences
- **Positive:** Best accuracy among considered models
- **Positive:** Good generalization to unseen query types
- **Negative:** Larger model size than BERT/RoBERTa
- **Negative:** Slightly slower inference

---

## ADR-003: FAISS for Vector Store

**Date:** 2026-06-15
**Status:** Accepted

### Context
We need a vector store for dense retrieval with good performance and flexibility.

### Decision
Use FAISS (Facebook AI Similarity Search) as the primary vector store backend.

### Alternatives Considered
| Feature | FAISS | Pinecone | Weaviate | ChromaDB |
|---|---|---|---|---|
| Self-hosted | ✅ | ❌ | ✅ | ✅ |
| Cost | Free | Paid | Free/Paid | Free |
| GPU support | ✅ | ❌ | ❌ | ❌ |
| Performance | Excellent | Good | Good | Good |
| Flexibility | High | Low | Medium | Medium |

### Consequences
- **Positive:** Free, self-hosted, GPU-accelerated
- **Positive:** High performance for large-scale retrieval
- **Negative:** Requires more setup than managed services
- **Negative:** ChromaDB backend is less mature (stub-only initially)

---

## ADR-004: NetworkX for Knowledge Graph

**Date:** 2026-06-15
**Status:** Accepted

### Context
We need a knowledge graph for entity-based reasoning and global queries.

### Decision
Use NetworkX as the primary knowledge graph backend, with Neo4j as an alternative.

### Alternatives Considered
| Feature | NetworkX | Neo4j |
|---|---|---|
| Installation | pip install | Separate server |
| Complexity | Simple | Complex |
| Performance | Good (<100K nodes) | Excellent |
| Cypher queries | ❌ | ✅ |
| In-memory | ✅ | ❌ |

### Consequences
- **Positive:** Simple to install and use
- **Positive:** No external dependencies
- **Negative:** Limited scalability for very large graphs
- **Negative:** No query language (must use Python API)

---

## ADR-005: ReAct as Default Planning Strategy

**Date:** 2026-06-20
**Status:** Accepted

### Context
The agentic orchestrator needs a planning strategy for multi-step reasoning.

### Decision
Use ReAct (Think-Act-Observe) as the default planning strategy.

### Alternatives Considered
| Strategy | Strengths | Weaknesses |
|---|---|---|
| **ReAct** | Flexible, adapts to observations | May over-reason |
| Plan-and-Execute | Structured, predictable | Rigid, can't adapt |
| Tree of Thought | Explores alternatives | Expensive |

### Consequences
- **Positive:** Flexible and adaptive to different query types
- **Positive:** Well-studied in literature
- **Negative:** May over-reason for simple queries (mitigated by router)
- **Negative:** More complex than static pipelines

---

## ADR-006: Heuristic Fallbacks for All Components

**Date:** 2026-06-20
**Status:** Accepted

### Context
Transformer models require GPU and may not be available in all environments.

### Decision
Implement heuristic/rule-based fallbacks for all trainable components:
- Router: Keyword-based complexity classification
- Evaluator: Keyword-overlap scoring
- Reflection: Rule-based token generation

### Consequences
- **Positive:** System works without GPU
- **Positive:** Faster inference for development/testing
- **Negative:** Lower accuracy than transformer-based models
- **Negative:** Two code paths to maintain

---

## ADR-007: Lazy Initialization for All Layers

**Date:** 2026-06-20
**Status:** Accepted

### Context
Loading all models at startup is slow and memory-intensive.

### Decision
Use lazy initialization (Python @property) for all layer instances.

### Consequences
- **Positive:** Fast startup time
- **Positive:** Only loads models that are actually used
- **Negative:** First query has additional latency for model loading
- **Negative:** Errors occur at query time, not initialization time

---

## ADR-008: Rich for Console Logging

**Date:** 2026-06-20
**Status:** Accepted

### Context
We need formatted, colorful console output for development and debugging.

### Decision
Use the Rich library for console logging with custom formatters.

### Consequences
- **Positive:** Beautiful, readable console output
- **Positive:** Built-in progress bars, tables, and panels
- **Negative:** Additional dependency
- **Negative:** May not work in all terminal environments

---

## ADR-009: Cross-Encoder Reranking

**Date:** 2026-07-12
**Status:** Accepted

### Context
Vector retrieval may return semantically similar but irrelevant documents.

### Decision
Add cross-encoder reranking after initial vector retrieval.

### Alternatives Considered
| Approach | Accuracy | Speed | Complexity |
|---|---|---|---|
| No reranking | Baseline | Fastest | None |
| **Cross-encoder** | **Best** | **Slow** | **Medium** |
| ColBERT | Good | Fast | High |

### Consequences
- **Positive:** Significant improvement in retrieval precision
- **Positive:** Well-supported by sentence-transformers library
- **Negative:** Adds latency (~200ms for 10 documents)
- **Negative:** Requires additional model download

---

## ADR-010: Structured JSON Logging

**Date:** 2026-07-12
**Status:** Accepted

### Context
Text logs are hard to parse and analyze programmatically.

### Decision
Implement structured JSON logging alongside existing text logging.

### Consequences
- **Positive:** Machine-parseable logs
- **Positive:** Easy to analyze with log aggregation tools
- **Negative:** Larger log files
- **Negative:** Additional code complexity
