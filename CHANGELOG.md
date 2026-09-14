# Changelog

All notable changes to the AARAG project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] — 2026-07-12

### Added

**Innovation Components:**
- Cross-encoder reranker (`src/layers/cross_encoder_reranker.py`) for improved retrieval precision using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Hybrid retrieval (`src/layers/hybrid_retrieval.py`) combining BM25 sparse + vector dense search with Reciprocal Rank Fusion
- Semantic query cache (`src/layers/query_cache.py`) with embedding-based similarity matching, LRU eviction, and TTL expiration
- Confidence calibrator (`src/layers/confidence_calibrator.py`) with temperature scaling and Platt scaling for better confidence estimates

**Evaluation Modules:**
- Standardized benchmark runner (`src/evaluation/benchmark.py`) with latency percentiles (p50, p95, p99), memory tracking, and throughput measurement
- Error analysis module (`src/evaluation/error_analysis.py`) for categorizing and analyzing failure modes
- Fairness audit module (`src/evaluation/fairness_audit.py`) for bias detection and consistency testing
- New metrics: BERTScore, MRR, Answer Correctness, Context Relevance

**Logging Infrastructure:**
- Structured JSON logger (`src/utils/structured_logger.py`) with context propagation and performance counters
- Log analyzer (`src/utils/log_analyzer.py`) for query profiling and bottleneck detection
- Pipeline tracer (`src/utils/pipeline_tracer.py`) with OpenTelemetry-compatible span format

**Documentation:**
- `CONTRIBUTING.md` — Development workflow and contribution guidelines
- `ROADMAP.md` — Short-term, mid-term, and long-term development roadmap
- `docs/methodology.md` — Research methodology and experimental design
- `docs/innovation.md` — Novel contributions and design decisions
- `docs/deployment.md` — Production deployment guide with Docker and cloud
- `docs/architecture_decisions.md` — Architecture Decision Records (ADRs)
- `docs/troubleshooting.md` — Expanded troubleshooting and performance tuning
- `docs/benchmark_results.md` — Detailed benchmark results and analysis

**Tests:**
- `tests/test_cross_encoder.py` — Cross-encoder reranker tests
- `tests/test_hybrid_retrieval.py` — Hybrid retrieval tests
- `tests/test_query_cache.py` — Query cache tests
- `tests/test_confidence_calibrator.py` — Confidence calibrator tests
- `tests/test_structured_logger.py` — Structured logger tests
- `tests/test_log_analyzer.py` — Log analyzer tests

### Changed
- `src/aarag.py` — Integrated cross-encoder reranker, hybrid retrieval, query cache, and confidence calibration into pipeline
- `src/evaluation/metrics.py` — Added BERTScore, MRR, Answer Correctness, Context Relevance metrics
- `src/utils/logger.py` — Added JSON output mode, structured context fields, performance timing decorators
- `config/default.yaml` — Added configuration sections for all new components
- `run_evals.py` — Added benchmark mode, error analysis, fairness audit, and automatic chart generation
- `README.md` — Added badges, performance benchmarks table, quick-start troubleshooting
- `tests/test_integration.py` — Added integration tests for new components

---

## [1.0.0] — 2026-07-01

### Added
- Initial release of AARAG (Adaptive Agentic Retrieval-Augmented Generation)
- 5-layer architecture:
  - Layer 1: Knowledge Sources (FAISS vector store, NetworkX knowledge graph, DuckDuckGo web search)
  - Layer 2: Adaptive Router (DeBERTa-v3-base query complexity classifier)
  - Layer 3: Corrective Retrieval (CRAG with evaluator and web fallback)
  - Layer 4: Self-Reflection Engine ([IsRel][IsSup][IsUse] tokens)
  - Layer 5: Agentic Orchestrator (ReAct agent with tool registry)
- Training pipeline for router, evaluator, and reflection token generator
- Evaluation framework with 5 baselines and ablation studies
- Streamlit demo application
- Comprehensive documentation (README, architecture, API reference, evaluation guide, training guide)
- Test suite with 40 tests across 4 test files
- Configuration system with YAML-based config
- Rich logging with formatted console output and file logging

---

## [0.1.0] — 2026-06-15

### Added
- Project scaffolding and initial architecture design
- Research paper analysis (Adaptive-RAG, Self-RAG, CRAG, GraphRAG, ReAct)
- Proof-of-concept implementations for each layer
- Initial test framework
