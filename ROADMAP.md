# AARAG Development Roadmap

This document outlines the development roadmap for AARAG, organized by release timeline.

---

## Vision

Build the most comprehensive, production-ready Adaptive RAG framework that unifies multiple retrieval paradigms into a single, self-correcting, agentic system.

---

## v1.1.0 — Innovation & Quality (July 2026) ✅

**Status:** Complete

### Innovation
- [x] Cross-encoder reranking for improved retrieval precision
- [x] Hybrid retrieval (BM25 + vector dense) with Reciprocal Rank Fusion
- [x] Semantic query caching with embedding-based similarity
- [x] Confidence calibration (temperature scaling, Platt scaling)

### Evaluation
- [x] Standardized benchmark runner with latency percentiles
- [x] Error analysis and categorization module
- [x] Fairness audit for bias detection
- [x] New metrics: BERTScore, MRR, Answer Correctness, Context Relevance

### Logging
- [x] Structured JSON logging with context propagation
- [x] Log analysis and profiling tools
- [x] Pipeline tracing with OpenTelemetry-compatible spans

### Documentation
- [x] Methodology, innovation, deployment, ADR docs
- [x] Expanded troubleshooting guide
- [x] Benchmark results documentation

---

## v1.2.0 — Performance & Scale (August 2026)

**Status:** Planned

### Performance
- [ ] Async pipeline execution (parallel layer processing)
- [ ] Batch query processing
- [ ] Streaming response generation
- [ ] GPU memory optimization (gradient checkpointing, model sharding)

### Scalability
- [ ] Distributed vector store (FAISS on multiple GPUs)
- [ ] Redis-backed query cache
- [ ] Horizontal scaling with load balancing
- [ ] Kubernetes deployment manifests

### Monitoring
- [ ] Prometheus metrics export
- [ ] Grafana dashboard templates
- [ ] Alerting rules for error rates, latency spikes
- [ ] Health check endpoints

---

## v2.0.0 — Advanced RAG (September 2026)

**Status:** Planned

### New Retrieval Strategies
- [ ] ColBERTv2 late interaction retrieval
- [ ] SPLADE sparse learned retrieval
- [ ] Multi-vector retrieval (ColBERT-style)
- [ ] Adaptive chunk sizing (small-to-big, big-to-small)

### Advanced Reasoning
- [ ] Tree of Thought (ToT) planning strategy
- [ ] Graph of Thought (GoT) for complex reasoning
- [ ] Self-consistency with multiple reasoning paths
- [ ] Chain-of-Verification (CoVe) for fact-checking

### Knowledge Graph Enhancements
- [ ] LLM-based entity extraction (replace rule-based)
- [ ] Relation extraction with LLM
- [ ] Graph neural network for entity embeddings
- [ ] Temporal knowledge graph support

### Training
- [ ] DPO (Direct Preference Optimization) for reflection tokens
- [ ] RLHF for answer quality optimization
- [ ] Curriculum learning for router training
- [ ] Active learning for data-efficient training

---

## v2.1.0 — Multi-Modal RAG (October 2026)

**Status:** Planned

### Multi-Modal Support
- [ ] Image retrieval (CLIP embeddings)
- [ ] Table extraction and reasoning
- [ ] Chart/graph understanding
- [ ] PDF layout analysis

### Multi-Modal Generation
- [ ] Image-grounded answers
- [ ] Table-aware generation
- [ ] Citation with page numbers and figures

---

## v3.0.0 — Production Platform (December 2026)

**Status:** Vision

### Platform Features
- [ ] REST API with FastAPI
- [ ] WebSocket support for streaming
- [ ] User authentication and authorization
- [ ] Multi-tenant support
- [ ] Rate limiting and quota management

### Enterprise Features
- [ ] Audit logging and compliance
- [ ] Data encryption at rest and in transit
- [ ] SSO integration (OAuth2, SAML)
- [ ] Role-based access control (RBAC)

### Developer Experience
- [ ] CLI tool for common operations
- [ ] SDK for Python, JavaScript, Go
- [ ] Plugin system for custom components
- [ ] Visual pipeline builder (no-code)

---

## Long-Term Vision (2027+)

### Research Directions
- [ ] Federated RAG across organizations
- [ ] Privacy-preserving retrieval (differential privacy)
- [ ] Continuous learning from user feedback
- [ ] Self-improving retrieval (meta-learning)
- [ ] Agentic RAG with autonomous tool creation

### Community
- [ ] Open-source release with Apache 2.0 license
- [ ] Community plugins and extensions
- [ ] Annual benchmark competition
- [ ] Academic collaboration program

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to any of these milestones.

---

## Feedback

Have ideas for the roadmap? Open an issue with the `roadmap` label!
