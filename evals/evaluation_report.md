# AARAG Evaluation Report

**Generated:** 2026-07-11 14:03:46
**Total Evaluation Time:** 37.8s
**Queries Evaluated:** 16

## Overall Metrics

| Metric | Value |
|---|---|
| Total Queries | 16 |
| Successful | 16 |
| Errors | 0 |
| Avg Confidence | 0.3469 |
| Min Confidence | 0.3125 |
| Max Confidence | 0.3917 |
| Avg Latency | 2.363s |
| Min Latency | 1.338s |
| Max Latency | 15.377s |
| Total Latency | 37.801s |

## Per-Layer Average Confidence

| Layer | Avg Confidence |
|---|---|
| router | 0.7125 |
| crag | 0.1000 |
| reflection | 0.0000 |
| agent | 0.5750 |

## Complexity-Level Breakdown

### no_retrieval
- Queries: 4
- Avg Confidence: 0.3250
- Avg Latency: 4.968s
- Strategy Distribution: {"no_retrieval": 1, "multi_step": 3}

### single_step
- Queries: 4
- Avg Confidence: 0.3167
- Avg Latency: 1.488s
- Strategy Distribution: {"no_retrieval": 3, "multi_step": 1}

### multi_step
- Queries: 4
- Avg Confidence: 0.3730
- Avg Latency: 1.532s
- Strategy Distribution: {"multi_step": 4}

### graph_global
- Queries: 4
- Avg Confidence: 0.3730
- Avg Latency: 1.461s
- Strategy Distribution: {"multi_step": 2, "graph_global": 2}

## Ablation Study

| Config | Confidence | Latency | Strategy | Answer Len |
|---|---|---|---|---|
| full | 0.3125 | 0.124s | no_retrieval | 62 |
| no_reflection | 0.4167 | 0.088s | no_retrieval | 62 |
| no_web_fallback | 0.4125 | 0.126s | no_retrieval | 66 |
| forced_single_step | 0.4042 | 0.188s | single_step | 62 |
| forced_multi_step | 0.4042 | 0.170s | multi_step | 62 |

## Per-Query Results

| # | Complexity | Strategy | Confidence | Latency | Status |
|---|---|---|---|---|---|
| 001 | no_retrieval | no_retrieval | 0.3125 | 15.377s | ✓ |
| 002 | no_retrieval | multi_step | 0.3292 | 1.338s | ✓ |
| 003 | no_retrieval | multi_step | 0.3292 | 1.589s | ✓ |
| 004 | no_retrieval | multi_step | 0.3292 | 1.569s | ✓ |
| 005 | single_step | no_retrieval | 0.3125 | 1.666s | ✓ |
| 006 | single_step | multi_step | 0.3292 | 1.467s | ✓ |
| 007 | single_step | no_retrieval | 0.3125 | 1.452s | ✓ |
| 008 | single_step | no_retrieval | 0.3125 | 1.369s | ✓ |
| 009 | multi_step | multi_step | 0.3792 | 1.517s | ✓ |
| 010 | multi_step | multi_step | 0.3542 | 1.544s | ✓ |
| 011 | multi_step | multi_step | 0.3792 | 1.535s | ✓ |
| 012 | multi_step | multi_step | 0.3792 | 1.534s | ✓ |
| 013 | graph_global | multi_step | 0.3542 | 1.562s | ✓ |
| 014 | graph_global | graph_global | 0.3917 | 1.439s | ✓ |
| 015 | graph_global | multi_step | 0.3542 | 1.399s | ✓ |
| 016 | graph_global | graph_global | 0.3917 | 1.444s | ✓ |

## Sample Answers

### Q: What is 2+2?
- Strategy: no_retrieval, Confidence: 0.3125, Latency: 15.377s
> Based on the available information:  what is 2+2=explain - Brainly.in.

### Q: Hello, how are you?
- Strategy: multi_step, Confidence: 0.3292, Latency: 1.338s
> Based on the available information:  | Kids Greeting Song and Feelings Song | Super Simple Songs.

### Q: Thank you!
- Strategy: multi_step, Confidence: 0.3292, Latency: 1.589s
> Based on the available information:  110 Best Thank You Messages to Express Your Gratitude.

### Q: What color is the sky?
- Strategy: multi_step, Confidence: 0.3292, Latency: 1.569s
> Based on the available information:  The Sky Isn't Blue: Here's What Color It Actually Is.

### Q: What is machine learning?
- Strategy: no_retrieval, Confidence: 0.3125, Latency: 1.666s
> Based on the available information:  What is machine learning?

### Q: Who invented the telephone?
- Strategy: multi_step, Confidence: 0.3292, Latency: 1.467s
> Based on the available information:  Alexander Graham Bell - Wikipedia.

### Q: What is the capital of France?
- Strategy: no_retrieval, Confidence: 0.3125, Latency: 1.452s
> Based on the available information:  Paris - Wikipedia.

### Q: What is Python programming language?
- Strategy: no_retrieval, Confidence: 0.3125, Latency: 1.369s
> Based on the available information:  Python (programming language) - Wikipedia.

### Q: Compare supervised and unsupervised learning approaches in modern ML.
- Strategy: multi_step, Confidence: 0.3792, Latency: 1.517s
> Based on the available information:  Difference between Supervised and Unsupervised Learning.

### Q: How does the transformer architecture differ from RNNs in NLP?
- Strategy: multi_step, Confidence: 0.3542, Latency: 1.544s
> Based on the available information:  Why Transformer Models Replaced RNN in NLP - ML Journey.

### Q: Explain the relationship between gradient descent and backpropagation.
- Strategy: multi_step, Confidence: 0.3792, Latency: 1.535s
> Based on the available information:  How Does Gradient Descent and Backpropagation Work Together?.

### Q: What are the differences between FAISS and ChromaDB vector stores?
- Strategy: multi_step, Confidence: 0.3792, Latency: 1.534s
> Based on the available information:  Faiss Vector Database vs ChromaDB: Comparison for Modern AI ....

### Q: What are all the major breakthroughs in AI from 2017 to 2025?
- Strategy: multi_step, Confidence: 0.3542, Latency: 1.562s
> Based on the available information:  AI Breakthrough Timeline: Key Milestones from 1997 to 2026 | AI Flash ....

### Q: Summarize the entire field of retrieval-augmented generation.
- Strategy: graph_global, Confidence: 0.3917, Latency: 1.439s
> Based on the available information:  Retrieval-augmented generation - Wikipedia.

### Q: How do all the components of a modern RAG pipeline work together?
- Strategy: multi_step, Confidence: 0.3542, Latency: 1.399s
> Based on the available information:  RAG Pipeline Diagram: A Beginner's Guide to Understanding ...

### Q: Give a comprehensive overview of knowledge graph construction methods.
- Strategy: graph_global, Confidence: 0.3917, Latency: 1.444s
> Based on the available information:  A systematic literature review of knowledge graph construction and ....
