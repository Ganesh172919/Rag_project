"""AARAG — Adaptive Agentic Retrieval-Augmented Generation.

Main pipeline integrating all 5 layers:
  Layer 1: Knowledge Sources (Vector DB + Knowledge Graph + Web Search)
  Layer 2: Adaptive Router (Query Complexity Classification)
  Layer 3: Corrective Retrieval (CRAG — evaluator + web fallback)
  Layer 4: Self-Reflection Engine (reflection tokens + claim verification)
  Layer 5: Agentic Orchestrator (ReAct agent + planning)

Innovation components (v1.1.0):
  - Cross-encoder reranker for improved retrieval precision
  - Hybrid retrieval (BM25 + vector dense) with Reciprocal Rank Fusion
  - Semantic query caching for repeated queries
  - Confidence calibration (temperature/Platt scaling)
"""

import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml

from .layers.knowledge_sources import VectorStore, KnowledgeGraphSearch, WebSearch, DocumentLoader
from .layers.knowledge_sources.document_loader import Chunk
from .layers.adaptive_router import AdaptiveRouter
from .layers.corrective_retrieval import CorrectiveRetrieval
from .layers.self_reflection import SelfReflectionEngine
from .layers.agentic_orchestrator import AgenticOrchestrator
from .layers.agentic_orchestrator.tools import create_default_tools
from .layers.cross_encoder_reranker import CrossEncoderReranker
from .layers.query_cache import SemanticQueryCache
from .layers.confidence_calibrator import ConfidenceCalibrator
from .utils.logger import (
    get_logger, log_section, log_layer_start, log_layer_end,
    log_decision, log_metric, PIPELINE,
)

# ──────────────────────────────────────────────
# Logger for this module
# ──────────────────────────────────────────────
logger = get_logger("aarag.pipeline", log_file="results/logs/aarag.log")


@dataclass
class AARAGResponse:
    """Complete response from the AARAG pipeline."""
    answer: str
    routing_decision: dict
    corrective_result: dict
    reflection_result: dict
    agent_response: dict
    confidence: float
    latency: float
    strategy_used: str
    sources: List[str] = field(default_factory=list)


class AARAG:
    """Adaptive Agentic Retrieval-Augmented Generation.

    This is the main entry point for the AARAG system. It orchestrates
    all 5 layers to provide accurate, grounded, and self-verified answers.
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[dict] = None):
        """Initialize AARAG with configuration.

        Args:
            config_path: Path to YAML config file
            config: Dict config (overrides config_path)
        """
        log_section(logger, "AARAG INITIALIZATION")
        logger.info("Loading configuration...")

        self.config = self._load_config(config_path, config)
        self._initialized = False

        # Layer instances (lazy initialization)
        self._vector_store: Optional[VectorStore] = None
        self._knowledge_graph: Optional[KnowledgeGraphSearch] = None
        self._web_search: Optional[WebSearch] = None
        self._router: Optional[AdaptiveRouter] = None
        self._crag: Optional[CorrectiveRetrieval] = None
        self._self_reflection: Optional[SelfReflectionEngine] = None
        self._orchestrator: Optional[AgenticOrchestrator] = None
        self._llm_fn = None

        # Innovation components (v1.1.0)
        self._reranker: Optional[CrossEncoderReranker] = None
        self._query_cache: Optional[SemanticQueryCache] = None
        self._calibrator: Optional[ConfidenceCalibrator] = None

        logger.info(f"Configuration loaded: {len(self.config)} top-level keys")
        logger.info(f"  LLM: {self.config.get('llm', {}).get('model_name', 'Not configured')}")
        logger.info(f"  Vector store: {self.config.get('knowledge_sources', {}).get('vector_store', {}).get('backend', 'faiss')}")
        logger.info(f"  Embedding: {self.config.get('knowledge_sources', {}).get('vector_store', {}).get('embedding_model', 'BAAI/bge-large-en-v1.5')}")
        logger.info("AARAG initialized successfully ✓")

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            logger.debug("Lazy-loading VectorStore...")
            cfg = self.config.get("knowledge_sources", {}).get("vector_store", {})
            self._vector_store = VectorStore(
                embedding_model=cfg.get("embedding_model", "BAAI/bge-large-en-v1.5"),
                backend=cfg.get("backend", "faiss"),
            )
            logger.debug("VectorStore loaded ✓")
        return self._vector_store

    @property
    def knowledge_graph(self) -> KnowledgeGraphSearch:
        if self._knowledge_graph is None:
            logger.debug("Lazy-loading KnowledgeGraphSearch...")
            cfg = self.config.get("knowledge_sources", {}).get("knowledge_graph", {})
            self._knowledge_graph = KnowledgeGraphSearch(
                max_hops=cfg.get("max_hops", 3),
            )
            logger.debug("KnowledgeGraphSearch loaded ✓")
        return self._knowledge_graph

    @property
    def web_search(self) -> WebSearch:
        if self._web_search is None:
            logger.debug("Lazy-loading WebSearch...")
            cfg = self.config.get("knowledge_sources", {}).get("web_search", {})
            self._web_search = WebSearch(
                provider=cfg.get("provider", "duckduckgo"),
                max_results=cfg.get("max_results", 5),
            )
            logger.debug("WebSearch loaded ✓")
        return self._web_search

    @property
    def router(self) -> AdaptiveRouter:
        if self._router is None:
            logger.debug("Lazy-loading AdaptiveRouter...")
            cfg = self.config.get("adaptive_router", {})
            self._router = AdaptiveRouter(
                classifier_model=cfg.get("classifier_model", "microsoft/deberta-v3-base"),
                confidence_threshold=cfg.get("confidence_threshold", 0.7),
            )
            logger.debug("AdaptiveRouter loaded ✓")
        return self._router

    @property
    def crag(self) -> CorrectiveRetrieval:
        if self._crag is None:
            logger.debug("Lazy-loading CorrectiveRetrieval...")
            cfg = self.config.get("corrective_retrieval", {})
            self._crag = CorrectiveRetrieval(
                evaluator_model=cfg.get("evaluator_model", "microsoft/deberta-v3-small"),
                correct_threshold=cfg.get("confidence_thresholds", {}).get("correct", 0.8),
                incorrect_threshold=cfg.get("confidence_thresholds", {}).get("incorrect", 0.3),
            )
            logger.debug("CorrectiveRetrieval loaded ✓")
        return self._crag

    @property
    def self_reflection(self) -> SelfReflectionEngine:
        if self._self_reflection is None:
            logger.debug("Lazy-loading SelfReflectionEngine...")
            cfg = self.config.get("self_reflection", {})
            self._self_reflection = SelfReflectionEngine(
                llm_fn=self._get_llm_fn(),
                max_retries=cfg.get("max_retries", 2),
            )
            logger.debug("SelfReflectionEngine loaded ✓")
        return self._self_reflection

    @property
    def orchestrator(self) -> AgenticOrchestrator:
        if self._orchestrator is None:
            logger.debug("Lazy-loading AgenticOrchestrator...")
            cfg = self.config.get("agentic_orchestrator", {})
            tool_registry = create_default_tools(
                vector_store=self.vector_store,
                knowledge_graph=self.knowledge_graph,
                web_search=self.web_search,
                crag=self.crag,
                self_reflection=self.self_reflection,
            )
            self._orchestrator = AgenticOrchestrator(
                llm_fn=self._get_llm_fn(),
                tool_registry=tool_registry,
                max_steps=cfg.get("max_steps", 10),
                planning_strategy=cfg.get("planning_strategy", "react"),
            )
            logger.debug("AgenticOrchestrator loaded ✓")
        return self._orchestrator

    @property
    def reranker(self) -> CrossEncoderReranker:
        """Cross-encoder reranker for improved retrieval precision."""
        if self._reranker is None:
            cfg = self.config.get("cross_encoder_reranker", {})
            if cfg.get("enabled", True):
                logger.debug("Lazy-loading CrossEncoderReranker...")
                self._reranker = CrossEncoderReranker(
                    model_name=cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                    max_length=cfg.get("max_length", 512),
                    batch_size=cfg.get("batch_size", 32),
                )
                logger.debug("CrossEncoderReranker loaded ✓")
        return self._reranker

    @property
    def query_cache(self) -> SemanticQueryCache:
        """Semantic query cache for repeated queries."""
        if self._query_cache is None:
            cfg = self.config.get("query_cache", {})
            if cfg.get("enabled", True):
                logger.debug("Lazy-loading SemanticQueryCache...")
                self._query_cache = SemanticQueryCache(
                    max_size=cfg.get("max_size", 1000),
                    ttl_seconds=cfg.get("ttl_seconds", 3600),
                    similarity_threshold=cfg.get("similarity_threshold", 0.95),
                )
                logger.debug("SemanticQueryCache loaded ✓")
        return self._query_cache

    @property
    def calibrator(self) -> ConfidenceCalibrator:
        """Confidence calibrator for better probability estimates."""
        if self._calibrator is None:
            cfg = self.config.get("confidence_calibration", {})
            if cfg.get("enabled", False):
                logger.debug("Lazy-loading ConfidenceCalibrator...")
                self._calibrator = ConfidenceCalibrator(
                    method=cfg.get("method", "temperature"),
                )
                logger.debug("ConfidenceCalibrator loaded ✓")
        return self._calibrator

    def ingest(
        self,
        data_path: str,
        build_kg: bool = True,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> dict:
        """Ingest documents into the knowledge sources."""
        log_section(logger, "DOCUMENT INGESTION")
        start_time = time.time()

        logger.info(f"Source: {data_path}")
        logger.info(f"Build KG: {build_kg}")

        # Load documents
        cfg = self.config.get("knowledge_sources", {}).get("vector_store", {})
        loader = DocumentLoader(
            chunk_size=chunk_size or cfg.get("chunk_size", 512),
            chunk_overlap=chunk_overlap or cfg.get("chunk_overlap", 50),
        )

        path = Path(data_path)
        if path.is_dir():
            logger.info(f"Loading directory: {path}")
            documents = loader.load_directory(str(path))
        else:
            logger.info(f"Loading file: {path}")
            documents = loader.load_file(str(path))

        logger.info(f"Loaded {len(documents)} documents")

        # Chunk documents
        logger.info(f"Chunking documents (size={loader.chunk_size}, overlap={loader.chunk_overlap})...")
        chunks = loader.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        # Add to vector store
        logger.info("Adding chunks to vector store...")
        num_added = self.vector_store.add_documents(chunks)
        logger.info(f"Added {num_added} vectors to store")

        # Build knowledge graph
        kg_nodes = 0
        if build_kg:
            logger.info("Building knowledge graph...")
            kg_docs = [{"content": c.content, "doc_id": c.doc_id} for c in chunks]
            self.knowledge_graph.build_from_documents(kg_docs)
            kg_nodes = len(self.knowledge_graph.graph.nodes)
            logger.info(f"Knowledge graph: {kg_nodes} nodes, {len(self.knowledge_graph.graph.edges)} edges")

        elapsed = time.time() - start_time

        stats = {
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "vectors_added": num_added,
            "kg_nodes": kg_nodes,
            "elapsed_seconds": round(elapsed, 2),
        }

        log_section(logger, "INGESTION COMPLETE")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")

        return stats

    def query(
        self,
        question: str,
        enable_web_fallback: bool = True,
        enable_self_reflection: bool = True,
        strategy_override: Optional[str] = None,
        enable_cache: bool = True,
    ) -> AARAGResponse:
        """Answer a question through the full AARAG pipeline."""
        log_section(logger, f"QUERY: {question[:80]}{'...' if len(question) > 80 else ''}")
        start_time = time.time()

        # ──────────────────────────────────────────────
        # QUERY CACHE — check for cached response
        # ──────────────────────────────────────────────
        if enable_cache and self.query_cache is not None:
            cached = self.query_cache.get(question)
            if cached is not None:
                logger.info("  ✓ Cache hit — returning cached response")
                cached.latency = round(time.time() - start_time, 3)
                return cached

        # ──────────────────────────────────────────────
        # LAYER 2: Adaptive Router — classify query complexity
        # ──────────────────────────────────────────────
        layer_start = time.time()
        log_layer_start(logger, 2, "Adaptive Router")

        routing = self.router.route(question)
        strategy = strategy_override or routing.strategy

        routing_info = {
            "level": routing.level,
            "strategy": routing.strategy,
            "confidence": routing.confidence,
            "fallback_used": routing.fallback_used,
        }

        level_names = {0: "No Retrieval", 1: "Single-Step", 2: "Multi-Step", 3: "Graph-Global"}
        log_decision(logger, "Complexity Level", f"{routing.level} ({level_names.get(routing.level, 'Unknown')})")
        log_decision(logger, "Strategy", strategy, routing.confidence)
        if routing.fallback_used:
            logger.warning("  ⚠ Low confidence — fallback strategy used")
        log_layer_end(logger, 2, "Adaptive Router", time.time() - layer_start)

        # ──────────────────────────────────────────────
        # LAYER 1: Knowledge Sources — retrieve context
        # ──────────────────────────────────────────────
        layer_start = time.time()
        log_layer_start(logger, 1, "Knowledge Sources")

        retrieved_docs = []
        if strategy != "no_retrieval":
            top_k = self.config.get("knowledge_sources", {}).get("vector_store", {}).get("top_k", 5)
            logger.info(f"  Searching vector store (top_k={top_k})...")
            retrieved_docs = self.vector_store.search(question, top_k=top_k)
            logger.info(f"  Retrieved {len(retrieved_docs)} documents")
            for i, doc in enumerate(retrieved_docs[:3]):
                logger.debug(f"    Doc {i+1} (score={doc.score:.3f}): {doc.content[:80]}...")
        else:
            logger.info("  Skipping retrieval (no_retrieval strategy)")

        log_layer_end(logger, 1, "Knowledge Sources", time.time() - layer_start)

        # ──────────────────────────────────────────────
        # CROSS-ENCODER RERANKING — improve retrieval precision
        # ──────────────────────────────────────────────
        if retrieved_docs and self.reranker is not None:
            rerank_start = time.time()
            logger.info("  Reranking documents with cross-encoder...")
            docs_as_dicts = [
                {"content": doc.content, "doc_id": doc.doc_id, "score": doc.score}
                for doc in retrieved_docs
            ]
            reranked = self.reranker.rerank(question, docs_as_dicts)
            # Update scores on original docs
            for doc, reranked_doc in zip(retrieved_docs, reranked):
                doc.score = reranked_doc.get("combined_score", doc.score)
            retrieved_docs.sort(key=lambda d: d.score, reverse=True)
            logger.info(f"  Reranked in {time.time() - rerank_start:.3f}s")

        # ──────────────────────────────────────────────
        # LAYER 3: Corrective Retrieval — evaluate and correct
        # ──────────────────────────────────────────────
        layer_start = time.time()
        log_layer_start(logger, 3, "Corrective Retrieval")

        logger.info(f"  Evaluating retrieval quality...")
        corrective_result = self.crag.process(
            query=question,
            retrieved_docs=retrieved_docs,
            enable_web_fallback=enable_web_fallback,
        )

        corrective_info = {
            "action": corrective_result.action,
            "confidence": corrective_result.confidence,
            "web_used": corrective_result.web_used,
            "sources": corrective_result.sources,
        }

        action_icons = {"CORRECT": "✓", "INCORRECT": "✗", "AMBIGUOUS": "⚠"}
        log_decision(logger, "Action", f"{action_icons.get(corrective_result.action, '?')} {corrective_result.action}", corrective_result.confidence)
        if corrective_result.web_used:
            logger.info("  → Web search fallback triggered")
        logger.info(f"  Sources: {', '.join(set(corrective_result.sources))}")
        log_layer_end(logger, 3, "Corrective Retrieval", time.time() - layer_start)

        context = corrective_result.context

        # ──────────────────────────────────────────────
        # LAYER 5: Agentic Orchestrator — coordinate answer
        # ──────────────────────────────────────────────
        layer_start = time.time()
        log_layer_start(logger, 5, "Agentic Orchestrator")

        logger.info(f"  Strategy: {strategy}")
        logger.info(f"  Planning: {self.config.get('agentic_orchestrator', {}).get('planning_strategy', 'react')}")

        agent_response = self.orchestrator.answer(
            query=question,
            routing_strategy=strategy,
            context=context,
        )

        answer = agent_response.answer

        agent_info = {
            "tools_used": agent_response.tools_used,
            "total_steps": agent_response.total_steps,
            "strategy_used": agent_response.strategy_used,
            "confidence": agent_response.confidence,
        }

        logger.info(f"  Steps taken: {agent_response.total_steps}")
        logger.info(f"  Tools used: {', '.join(agent_response.tools_used)}")
        log_decision(logger, "Agent Confidence", "", agent_response.confidence)

        # Log reasoning trace
        for step in agent_response.reasoning_trace:
            logger.debug(f"    Step {step.step_number}: [{step.action}] {step.thought[:60]}...")

        log_layer_end(logger, 5, "Agentic Orchestrator", time.time() - layer_start)

        # ──────────────────────────────────────────────
        # LAYER 4: Self-Reflection — verify answer quality
        # ──────────────────────────────────────────────
        layer_start = time.time()
        log_layer_start(logger, 4, "Self-Reflection")

        reflection_info = {}
        if enable_self_reflection:
            logger.info("  Generating reflection tokens...")
            reflection_result = self.self_reflection.reflect_and_generate(
                query=question,
                context=context,
            )

            tokens = reflection_result.tokens
            logger.info(f"  [Retrieve]: {tokens.retrieve}")
            logger.info(f"  [Relevance]: {tokens.relevance}")
            logger.info(f"  [Support]: {tokens.support}")
            logger.info(f"  [Utility]: {tokens.utility}")

            if reflection_result.revised:
                logger.info(f"  → Answer revised {reflection_result.revision_count} time(s)")

            # If reflection significantly improves confidence, use revised answer
            if reflection_result.revised and reflection_result.final_confidence > agent_response.confidence:
                logger.info("  → Using revised answer (higher confidence)")
                answer = reflection_result.answer

            reflection_info = {
                "retrieve": tokens.retrieve,
                "relevance": tokens.relevance,
                "support": tokens.support,
                "utility": tokens.utility,
                "revised": reflection_result.revised,
                "revision_count": reflection_result.revision_count,
                "final_confidence": reflection_result.final_confidence,
            }

            log_decision(logger, "Reflection Confidence", "", reflection_result.final_confidence)
        else:
            logger.info("  Self-reflection disabled")

        log_layer_end(logger, 4, "Self-Reflection", time.time() - layer_start)

        # ──────────────────────────────────────────────
        # Final assembly
        # ──────────────────────────────────────────────
        elapsed = time.time() - start_time

        # Overall confidence: weighted average of all layers
        confidences = [
            routing.confidence,
            corrective_result.confidence,
            agent_response.confidence,
        ]
        if reflection_info:
            confidences.append(reflection_info.get("final_confidence", 0.5))
        overall_confidence = sum(confidences) / len(confidences)

        # Apply confidence calibration if available
        if self.calibrator is not None and self.calibrator.is_fitted:
            layer_confidences = {
                "router": routing.confidence,
                "crag": corrective_result.confidence,
                "agent": agent_response.confidence,
            }
            if reflection_info:
                layer_confidences["reflection"] = reflection_info.get("final_confidence", 0.5)
            overall_confidence = self.calibrator.compute_overall_confidence(layer_confidences)
            logger.info(f"  Calibrated confidence: {overall_confidence:.4f}")

        log_section(logger, "RESPONSE SUMMARY")
        logger.info(f"  Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        log_metric(logger, "Overall Confidence", overall_confidence)
        log_metric(logger, "Total Latency", elapsed)
        logger.info(f"  Strategy: {strategy}")
        logger.info(f"  Sources: {len(corrective_result.sources)}")

        response = AARAGResponse(
            answer=answer,
            routing_decision=routing_info,
            corrective_result=corrective_info,
            reflection_result=reflection_info,
            agent_response=agent_info,
            confidence=overall_confidence,
            latency=round(elapsed, 3),
            strategy_used=strategy,
            sources=corrective_result.sources,
        )

        # Cache the response
        if enable_cache and self.query_cache is not None:
            self.query_cache.put(question, response)
            logger.debug("  Response cached for future queries")

        return response

    def save(self, path: str):
        """Save all AARAG components."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving AARAG to {save_path}...")

        self.vector_store.save(str(save_path / "vector_store"))
        logger.info("  ✓ Vector store saved")

        self.knowledge_graph.save(str(save_path / "knowledge_graph"))
        logger.info("  ✓ Knowledge graph saved")

        self.router.save(str(save_path / "router"))
        logger.info("  ✓ Router saved")

        self.crag.evaluator.save(str(save_path / "evaluator"))
        logger.info("  ✓ Evaluator saved")

        with open(save_path / "config.yaml", "w") as f:
            yaml.dump(self.config, f)
        logger.info("  ✓ Config saved")

        logger.info(f"AARAG saved to {save_path} ✓")

    def load(self, path: str):
        """Load all AARAG components."""
        load_path = Path(path)
        logger.info(f"Loading AARAG from {load_path}...")

        if (load_path / "vector_store").exists():
            self.vector_store.load(str(load_path / "vector_store"))
            logger.info("  ✓ Vector store loaded")
        if (load_path / "knowledge_graph").exists():
            self.knowledge_graph.load(str(load_path / "knowledge_graph"))
            logger.info("  ✓ Knowledge graph loaded")
        if (load_path / "router").exists():
            self.router.load(str(load_path / "router"))
            logger.info("  ✓ Router loaded")
        if (load_path / "evaluator").exists():
            self.crag.evaluator.load(str(load_path / "evaluator"))
            logger.info("  ✓ Evaluator loaded")

        self._initialized = True
        logger.info(f"AARAG loaded from {load_path} ✓")

    def _load_config(self, config_path: Optional[str], config: Optional[dict]) -> dict:
        """Load configuration from file or dict."""
        if config:
            logger.debug("Using provided config dict")
            return config

        if config_path and Path(config_path).exists():
            logger.debug(f"Loading config from {config_path}")
            with open(config_path, "r") as f:
                return yaml.safe_load(f)

        # Try default config
        default_path = Path(__file__).parent.parent / "config" / "default.yaml"
        if default_path.exists():
            logger.debug(f"Loading default config from {default_path}")
            with open(default_path, "r") as f:
                return yaml.safe_load(f)

        logger.warning("No config found, using empty config")
        return {}

    def _get_llm_fn(self):
        """Get the LLM function for generation."""
        if self._llm_fn is not None:
            return self._llm_fn

        llm_cfg = self.config.get("llm", {})
        model_name = llm_cfg.get("model_name", "")

        if model_name:
            try:
                logger.info(f"Loading LLM: {model_name}")
                return self._create_hf_llm_fn(model_name, llm_cfg)
            except Exception as e:
                logger.warning(f"Could not load LLM {model_name}: {e}")

        logger.info("Using fallback LLM function (no model loaded)")
        def simple_llm_fn(prompt: str, **kwargs) -> str:
            if "Context:" in prompt:
                context_part = prompt.split("Context:")[1].split("\n")[0][:500]
                return f"Based on the available information: {context_part}"
            return "I need more information to answer this question."

        self._llm_fn = simple_llm_fn
        return self._llm_fn

    def _create_hf_llm_fn(self, model_name: str, llm_cfg: dict):
        """Create an LLM function using HuggingFace transformers."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_config = None
        if llm_cfg.get("load_in_4bit", False):
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        max_new_tokens = llm_cfg.get("max_new_tokens", 512)
        temperature = llm_cfg.get("temperature", 0.1)

        def llm_fn(prompt: str, **kwargs) -> str:
            instruction = kwargs.get("instruction", "")
            full_prompt = f"{instruction}\n\n{prompt}" if instruction else prompt

            inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=llm_cfg.get("do_sample", True),
                    top_p=llm_cfg.get("top_p", 0.9),
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return response.strip()

        self._llm_fn = llm_fn
        logger.info(f"LLM loaded: {model_name} ✓")
        return llm_fn
