# AARAG Architecture — Detailed Technical Documentation

## 1. System Overview

AARAG is a 5-layer architecture that unifies four RAG paradigms:

```
User Query → Layer 2 (Router) → Layer 1 (Knowledge) → Layer 3 (CRAG)
             → Layer 4 (Self-Reflection) → Layer 5 (Agent) → Answer
```

Each layer has a specific responsibility and can operate independently or as part of the full pipeline.

---

## 2. Layer 1: Knowledge Sources

### 2.1 Vector Store (FAISS/ChromaDB)

**Purpose:** Dense retrieval using semantic similarity.

**Architecture:**
- Embedding model: `BAAI/bge-large-en-v1.5` (1024-dim)
- Index: FAISS IndexFlatIP (inner product) for <1K docs, IndexIVFFlat for larger collections
- Chunking: 512 tokens with 50-token overlap

**Algorithm:**
```
1. Load documents → split into chunks (512 tokens, 50 overlap)
2. Encode chunks → 1024-dim vectors (batch processing)
3. Build FAISS index (FlatIP or IVFFlat)
4. Query: encode query → search index → return top-k results
```

### 2.2 Knowledge Graph (NetworkX/Neo4j)

**Purpose:** Entity-based reasoning for global/corpus-level queries.

**Architecture:**
- Entity extraction: Rule-based (capitalized words, abbreviations, quoted terms)
- Relations: Co-occurrence based (entities in same document)
- Community detection: Louvain method (or connected components fallback)

**Graph Structure:**
- Nodes: Entities (PERSON, ORG, LOCATION, CONCEPT)
- Edges: Relations (CO_OCCURS, weight = co-occurrence count)
- Communities: Groups of closely related entities

### 2.3 Web Search (DuckDuckGo/Tavily)

**Purpose:** Fallback when local retrieval quality is insufficient.

**Providers:**
- DuckDuckGo: Free, no API key required
- Tavily: Better quality, requires API key
- Fallback: BeautifulSoup scraping of DuckDuckGo HTML

---

## 3. Layer 2: Adaptive Router

### 3.1 Query Complexity Classifier

**Model:** DeBERTa-v3-base fine-tuned on 4-class classification

**Complexity Levels:**
| Level | Strategy | Example |
|---|---|---|
| 0 | No Retrieval | "Who is Albert Einstein?" |
| 1 | Single-Step RAG | "What is the capital of France?" |
| 2 | Multi-Step RAG | "Compare Python and Java" |
| 3 | Graph-Global | "Summarize all main themes" |

**Features Extracted:**
- Word count, character count, average word length
- Question type (who, what, when, where, why, how, which)
- Multi-hop indicators (both, and, compare, difference, versus)
- Global indicators (all, every, summary, overview, trends)
- Simple patterns (regex matching question starters)

**Fallback:** Rule-based classifier using keyword heuristics

### 3.2 Routing Logic

```python
if confidence < threshold:
    strategy = "multi_step"  # Safe default
else:
    strategy = COMPLEXITY_MAP[level]
```

---

## 4. Layer 3: Corrective Retrieval (CRAG)

### 4.1 Retrieval Evaluator

**Model:** DeBERTa-v3-small fine-tuned on NLI-style data

**Input:** (query, document) pair
**Output:** Confidence score [0, 1]

**Decision Logic:**
```
confidence > 0.8  → CORRECT: Use docs as-is (decompose-recompose)
confidence < 0.3  → INCORRECT: Fall back to web search
0.3 ≤ confidence ≤ 0.8 → AMBIGUOUS: Combine local + web
```

### 4.2 Decompose-Recompose Algorithm

```
1. DECOMPOSE: Split each document into atomic knowledge snippets
   - Strategy: keypoint (sentences > 10 chars), sentence, or paragraph
2. SCORE: Each snippet for relevance to query
   - Word overlap + substring boost
3. FILTER: Remove snippets below min_relevance threshold (0.3)
4. RECOMPOSE: Sort by relevance, concatenate up to max_length
```

### 4.3 Web Fallback

When retrieval is INCORRECT or AMBIGUOUS:
1. Reformulate query for web search
2. Search web (DuckDuckGo/Tavily)
3. Extract key points from web results
4. Combine with local results (if AMBIGUOUS)

---

## 5. Layer 4: Self-Reflection Engine

### 5.1 Reflection Tokens

Based on Self-RAG (Asai et al., ICLR 2024):

| Token | Options | Purpose |
|---|---|---|
| `[Retrieve]` / `[No Retrieve]` | Should we retrieve? | Skip retrieval for simple queries |
| `[Relevant]` / `[Irrelevant]` | Is context relevant? | Trigger corrective retrieval |
| `[Fully Supported]` / `[Partially]` / `[No Support]` | Is answer grounded? | Detect hallucination |
| `[Very Useful]` / `[Useful]` / `[Not Useful]` | Is answer helpful? | Quality assessment |

### 5.2 Reflection Pipeline

```
1. Generate [Retrieve] token → decide if retrieval needed
2. Generate [IsRel] token → evaluate context relevance
3. Generate answer
4. Generate [IsSup] token → check answer support
5. Generate [IsUse] token → evaluate utility
6. If needs_revision → regenerate with instructions (max 2 retries)
```

### 5.3 Claim Verification

```
1. Extract claims from answer (sentence splitting)
2. Verify each claim against context
3. Status: verified / contradicted / unverified
4. Compute overall confidence = verified_count / total_claims
```

---

## 6. Layer 5: Agentic Orchestrator

### 6.1 ReAct Agent

Based on ReAct (Yao et al., ICLR 2023):

```
LOOP (max 10 steps):
  THINK: Reason about what to do next
  ACT: Choose a tool and execute
  OBSERVE: Record the result
  IF enough information → FINAL_ANSWER
```

### 6.2 Available Tools

| Tool | Description | When Used |
|---|---|---|
| `retrieve` | Search local vector store | Always for non-trivial queries |
| `graph_query` | Query knowledge graph | Global/corpus-level queries |
| `web_search` | Search the web | When local retrieval fails |
| `self_reflect` | Reflect on answer quality | After generating answer |
| `decompose` | Break query into sub-queries | Multi-hop queries |
| `verify` | Verify factual claims | Before finalizing answer |

### 6.3 Planning Strategies

- **ReAct:** Think-Act-Observe loop (default)
- **Plan-and-Execute:** Create plan first, then execute steps
- **Tree of Thought:** Explore multiple reasoning paths (future)

---

## 7. Data Flow

```
Input: "Compare the economic policies of the US and China"

Layer 2 (Router):
  → Features: word_count=9, has_comparison=1, multi_hop_keywords=1
  → Classification: Level 2 (Multi-Step)
  → Strategy: multi_step

Layer 1 (Knowledge Sources):
  → Vector search: top-5 documents about US/China economics
  → Graph query: entities [US, China, economic_policy]

Layer 3 (CRAG):
  → Evaluator score: 0.72 (AMBIGUOUS)
  → Web search: additional context from web
  → Decompose-recompose: 8 key snippets extracted

Layer 4 (Self-Reflection):
  → [Retrieve]: [Retrieve]
  → [IsRel]: [Relevant] (0.89)
  → [IsSup]: [Fully Supported] (0.92)
  → [IsUse]: [Very Useful] (0.87)
  → No revision needed

Layer 5 (Agent):
  → Step 1: retrieve(query) → got context
  → Step 2: verify(answer, context) → all claims verified
  → FINAL_ANSWER

Output: Comprehensive comparison with citations
```

---

## 8. Performance Characteristics

| Metric | Value |
|---|---|
| Average latency | 4.1s (full pipeline) |
| Router latency | 50ms |
| Retrieval latency | 200ms |
| CRAG evaluation | 300ms |
| Self-reflection | 500ms |
| Agent orchestration | 2-3s |
| Memory (7B model) | ~8GB GPU |
| Memory (no LLM) | ~2GB GPU |

---

## 9. Extension Points

### Adding a New Knowledge Source

```python
class MyKnowledgeSource:
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        # Your implementation
        pass

# Register in tools.py
registry.register(Tool(
    name="my_source",
    description="Search my custom knowledge source",
    fn=my_source.search,
))
```

### Adding a New Reflection Token

```python
# In token_generator.py
def generate_my_token(self, query: str, context: str) -> str:
    # Your implementation
    return "[MyToken]"

# In reflection_engine.py
my_token = self.token_generator.generate_my_token(query, context)
```

### Adding a New Planning Strategy

```python
# In planner.py
def _plan_my_strategy(self, query: str) -> Plan:
    # Your implementation
    return Plan(query=query, steps=[...])

# In agent.py
elif self.planning_strategy == "my_strategy":
    return self._plan_and_execute(query, routing_strategy, context)
```
