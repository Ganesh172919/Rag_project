# AARAG API Reference

## 1. Main Entry Point

### `AARAG(config_path=None, config=None)`

The main class for the AARAG system.

**Parameters:**
- `config_path` (str, optional): Path to YAML configuration file
- `config` (dict, optional): Configuration dictionary (overrides config_path)

**Properties:**
- `vector_store` — VectorStore instance (Layer 1)
- `knowledge_graph` — KnowledgeGraphSearch instance (Layer 1)
- `web_search` — WebSearch instance (Layer 1)
- `router` — AdaptiveRouter instance (Layer 2)
- `crag` — CorrectiveRetrieval instance (Layer 3)
- `self_reflection` — SelfReflectionEngine instance (Layer 4)
- `orchestrator` — AgenticOrchestrator instance (Layer 5)

**Methods:**

#### `ingest(data_path, build_kg=True, chunk_size=None, chunk_overlap=None)`

Ingest documents into the knowledge sources.

**Parameters:**
- `data_path` (str): Path to file or directory
- `build_kg` (bool): Whether to build knowledge graph (default: True)
- `chunk_size` (int, optional): Override default chunk size
- `chunk_overlap` (int, optional): Override default chunk overlap

**Returns:** dict with statistics

```python
stats = aarag.ingest("data/corpus/")
# {'documents_loaded': 50, 'chunks_created': 312, 'vectors_added': 312, 'kg_nodes': 156}
```

#### `query(question, enable_web_fallback=True, enable_self_reflection=True, strategy_override=None)`

Answer a question through the full AARAG pipeline.

**Parameters:**
- `question` (str): The user's question
- `enable_web_fallback` (bool): Use web search as fallback (default: True)
- `enable_self_reflection` (bool): Use self-reflection (default: True)
- `strategy_override` (str, optional): Force a specific strategy

**Returns:** `AARAGResponse`

```python
response = aarag.query("What is machine learning?")
print(response.answer)
print(response.confidence)
```

#### `save(path)`

Save all AARAG components to disk.

#### `load(path)`

Load all AARAG components from disk.

---

## 2. Response Object

### `AARAGResponse`

| Field | Type | Description |
|---|---|---|
| `answer` | str | The generated answer |
| `confidence` | float | Overall confidence (0-1) |
| `latency` | float | End-to-end time in seconds |
| `strategy_used` | str | Which strategy was used |
| `sources` | List[str] | Sources used for the answer |
| `routing_decision` | dict | Router output details |
| `corrective_result` | dict | CRAG output details |
| `reflection_result` | dict | Self-reflection tokens |
| `agent_response` | dict | Agent reasoning trace |

---

## 3. Layer 1: Knowledge Sources

### `VectorStore(embedding_model, backend, persist_dir)`

Dense vector store for semantic search.

**Parameters:**
- `embedding_model` (str): HuggingFace model name (default: "BAAI/bge-large-en-v1.5")
- `backend` (str): "faiss" or "chroma"
- `persist_dir` (str, optional): Directory to persist the store

**Methods:**

#### `add_documents(chunks, batch_size=64) -> int`

Add chunks to the vector store. Returns number added.

#### `search(query, top_k=5) -> List[RetrievalResult]`

Search for most similar chunks.

#### `save(path)` / `load(path)`

Persist/restore the vector store.

### `KnowledgeGraphSearch(max_hops=3)`

Knowledge graph for entity-based reasoning.

**Methods:**

#### `build_from_documents(documents, llm_fn=None)`

Build the knowledge graph from documents.

#### `search(query, top_k=5) -> GraphSearchResult`

Search the graph for query-relevant information.

### `WebSearch(provider, max_results, timeout)`

Web search interface.

**Methods:**

#### `search(query, max_results=None) -> List[WebSearchResult]`

Search the web.

---

## 4. Layer 2: Adaptive Router

### `AdaptiveRouter(classifier_model, confidence_threshold, device)`

Route queries to appropriate strategies.

**Methods:**

#### `route(query) -> RoutingDecision`

Route a single query.

**Returns:** `RoutingDecision` with fields:
- `level` (int): 0-3
- `strategy` (str): "no_retrieval", "single_step", "multi_step", "graph_global"
- `confidence` (float): 0-1
- `query` (str): Original query
- `fallback_used` (bool): Whether fallback was used

#### `route_batch(queries) -> List[RoutingDecision]`

Route a batch of queries.

### `QueryComplexityClassifier(model_name, device)`

Classify query complexity.

**Methods:**

#### `predict(query) -> Tuple[int, float]`

Returns (level, confidence).

### `FeatureExtractor`

Extract features from queries.

**Methods:**

#### `extract(query) -> Dict[str, float]`

Returns dict of feature_name -> feature_value.

---

## 5. Layer 3: Corrective Retrieval

### `CorrectiveRetrieval(evaluator_model, web_provider, thresholds, device)`

CRAG pipeline.

**Methods:**

#### `process(query, retrieved_docs, enable_web_fallback=True) -> CorrectiveResult`

Process retrieved documents through the CRAG pipeline.

**Returns:** `CorrectiveResult` with fields:
- `context` (str): Filtered/expanded context
- `action` (str): "CORRECT", "INCORRECT", "AMBIGUOUS"
- `confidence` (float): 0-1
- `sources` (List[str]): Sources used
- `web_used` (bool): Whether web search was used

### `RetrievalEvaluator(model_name, device)`

Evaluate retrieval quality.

**Methods:**

#### `score(query, documents) -> float`

Returns confidence score [0, 1].

#### `score_per_document(query, documents) -> List[float]`

Score each document individually.

### `DocumentDecomposer(strategy, min_relevance)`

Decompose-recompose algorithm.

**Methods:**

#### `decompose_recompose(query, documents, max_length=2000) -> str`

Full decompose-recompose pipeline.

#### `keypoint_extraction(documents, max_length=2000) -> str`

Extract key points from documents.

---

## 6. Layer 4: Self-Reflection Engine

### `SelfReflectionEngine(llm_fn, max_retries)`

Self-reflection pipeline.

**Methods:**

#### `reflect_and_generate(query, context, generate_fn=None) -> ReflectionResult`

Full self-reflection pipeline.

**Returns:** `ReflectionResult` with fields:
- `answer` (str): Generated (possibly revised) answer
- `tokens` (ReflectionTokens): Reflection token values
- `verification` (dict): Claim verification results
- `revised` (bool): Whether answer was revised
- `revision_count` (int): Number of revisions
- `final_confidence` (float): 0-1

### `ReflectionTokenGenerator(llm_fn)`

Generate reflection tokens.

**Methods:**

#### `generate_all(query, context, answer) -> ReflectionTokens`

Generate all reflection tokens.

**Returns:** `ReflectionTokens` with properties:
- `is_retrieval_needed` (bool)
- `is_relevant` (bool)
- `is_supported` (bool)
- `is_useful` (bool)
- `needs_revision` (bool)

### `ClaimVerifier(llm_fn)`

Verify factual claims.

**Methods:**

#### `verify_answer(answer, context) -> dict`

Returns dict with `claims`, `verified_count`, `contradicted_count`, `overall_confidence`.

---

## 7. Layer 5: Agentic Orchestrator

### `AgenticOrchestrator(llm_fn, tool_registry, max_steps, planning_strategy)`

ReAct agent for pipeline coordination.

**Methods:**

#### `answer(query, routing_strategy, context=None) -> AgentResponse`

Main entry point.

**Returns:** `AgentResponse` with fields:
- `answer` (str): Final answer
- `reasoning_trace` (List[AgentStep]): Step-by-step reasoning
- `confidence` (float): 0-1
- `tools_used` (List[str]): Tools used
- `total_steps` (int): Number of steps
- `strategy_used` (str): Strategy used

### `ToolRegistry`

Registry of tools available to the agent.

**Methods:**

#### `register(tool)`

Register a tool.

#### `execute(name, **kwargs)`

Execute a tool by name.

### `MultiStepPlanner(llm_fn, max_steps)`

Plan multi-step retrieval.

**Methods:**

#### `create_plan(query, routing_strategy) -> Plan`

Create a multi-step plan.

---

## 8. Evaluation

### `evaluate_system(system, eval_data, system_name, max_samples)`

Evaluate a system on a dataset.

### `run_full_evaluation(aarag, eval_data, ...)`

Run full evaluation on all datasets and baselines.

### `compute_all_metrics(predictions, references, questions, contexts)`

Compute all evaluation metrics.

**Returns:** dict with `exact_match`, `f1`, `rouge_l`, `answer_relevancy`, `faithfulness`.

---

## 9. Utilities

### `get_logger(name, level, log_file)`

Get a configured logger with Rich formatting.

### `timer(func)`

Decorator to time function execution.

### `load_yaml(path)` / `save_json(data, path)`

File I/O utilities.

### `get_device()`

Get the best available device (cuda/mps/cpu).
