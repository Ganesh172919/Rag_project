"""Knowledge Graph — Entity extraction, graph construction, and graph-based retrieval."""

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

import networkx as nx


@dataclass
class Entity:
    """An extracted entity."""
    name: str
    entity_type: str  # PERSON, ORG, LOCATION, CONCEPT, etc.
    mentions: List[str] = field(default_factory=list)
    doc_ids: List[str] = field(default_factory=list)


@dataclass
class Relation:
    """A relation between two entities."""
    source: str
    target: str
    relation_type: str
    evidence: str
    doc_id: str = ""


@dataclass
class GraphSearchResult:
    """Result from graph-based search."""
    entities: List[Entity]
    relations: List[Relation]
    community_summaries: List[str]
    relevant_subgraph: dict
    score: float


class KnowledgeGraphSearch:
    """Build and query a knowledge graph for entity-based reasoning."""

    def __init__(self, max_hops: int = 3):
        self.max_hops = max_hops
        self.graph = nx.Graph()
        self.entities: Dict[str, Entity] = {}
        self.communities: Dict[int, List[str]] = {}
        self._entity_extractor = None

    @property
    def entity_extractor(self):
        if self._entity_extractor is None:
            self._entity_extractor = _SimpleEntityExtractor()
        return self._entity_extractor

    def build_from_documents(self, documents: List[dict], llm_fn=None):
        """Build knowledge graph from documents.

        Args:
            documents: List of dicts with 'content' and 'doc_id' keys
            llm_fn: Optional LLM function for entity/relation extraction
        """
        for doc in documents:
            content = doc.get("content", "")
            doc_id = doc.get("doc_id", "")

            # Extract entities
            entities = self.entity_extractor.extract(content)
            for entity in entities:
                entity.doc_ids.append(doc_id)
                if entity.name not in self.entities:
                    self.entities[entity.name] = entity
                    self.graph.add_node(entity.name, type=entity.entity_type)
                else:
                    self.entities[entity.name].mentions.extend(entity.mentions)

            # Extract relations between co-occurring entities
            entity_names = [e.name for e in entities]
            for i, e1 in enumerate(entity_names):
                for e2 in entity_names[i + 1:]:
                    if not self.graph.has_edge(e1, e2):
                        self.graph.add_edge(e1, e2, relation="CO_OCCURS", weight=1.0, evidence=content[:200])
                    else:
                        self.graph[e1][e2]["weight"] += 1.0

        # Detect communities
        self._detect_communities()

    def search(self, query: str, top_k: int = 5) -> GraphSearchResult:
        """Search the knowledge graph for query-relevant information."""
        # Extract entities from query
        query_entities = self.entity_extractor.extract(query)
        query_entity_names = [e.name for e in query_entities if e.name in self.graph]

        if not query_entity_names:
            # Try fuzzy matching
            query_entity_names = self._fuzzy_match_entities(query)

        if not query_entity_names:
            return GraphSearchResult(
                entities=[], relations=[], community_summaries=[],
                relevant_subgraph={}, score=0.0,
            )

        # Expand neighborhood
        subgraph_nodes: Set[str] = set()
        for entity in query_entity_names:
            subgraph_nodes.add(entity)
            # BFS up to max_hops
            neighbors = self._get_neighbors(entity, self.max_hops)
            subgraph_nodes.update(neighbors)

        # Extract relevant entities and relations
        relevant_entities = [self.entities[n] for n in subgraph_nodes if n in self.entities]
        relevant_relations = []
        for n1 in subgraph_nodes:
            for n2 in subgraph_nodes:
                if n1 < n2 and self.graph.has_edge(n1, n2):
                    edge_data = self.graph[n1][n2]
                    relevant_relations.append(Relation(
                        source=n1, target=n2,
                        relation_type=edge_data.get("relation", "UNKNOWN"),
                        evidence=edge_data.get("evidence", ""),
                    ))

        # Get community summaries
        community_ids = set()
        for node in subgraph_nodes:
            if node in self.graph.nodes:
                cid = self.graph.nodes[node].get("community", -1)
                if cid >= 0:
                    community_ids.add(cid)

        community_summaries = []
        for cid in community_ids:
            members = self.communities.get(cid, [])
            summary = f"Community {cid}: {', '.join(members[:10])}"
            community_summaries.append(summary)

        # Build subgraph dict
        subgraph = {
            "nodes": list(subgraph_nodes),
            "edges": [(r.source, r.target, r.relation_type) for r in relevant_relations],
        }

        # Score based on entity overlap and graph connectivity
        score = len(query_entity_names) / max(len(query_entities), 1)
        if relevant_relations:
            score *= min(1.0, len(relevant_relations) / 5.0)

        return GraphSearchResult(
            entities=relevant_entities[:top_k],
            relations=relevant_relations[:top_k * 2],
            community_summaries=community_summaries[:3],
            relevant_subgraph=subgraph,
            score=min(score, 1.0),
        )

    def _get_neighbors(self, node: str, max_hops: int) -> Set[str]:
        """Get all neighbors within max_hops."""
        visited = set()
        current_level = {node}
        for _ in range(max_hops):
            next_level = set()
            for n in current_level:
                if n in self.graph:
                    for neighbor in self.graph.neighbors(n):
                        if neighbor not in visited:
                            next_level.add(neighbor)
            visited.update(next_level)
            current_level = next_level
        return visited

    def _fuzzy_match_entities(self, query: str) -> List[str]:
        """Fuzzy match query terms to entity names."""
        query_lower = query.lower()
        matches = []
        for name in self.entities:
            if name.lower() in query_lower or query_lower in name.lower():
                matches.append(name)
        return matches

    def _detect_communities(self):
        """Detect communities using Louvain method."""
        if len(self.graph.nodes) < 2:
            return
        try:
            from community import best_partition
            partition = best_partition(self.graph)
        except ImportError:
            # Fallback: connected components
            partition = {}
            for i, component in enumerate(nx.connected_components(self.graph)):
                for node in component:
                    partition[node] = i

        # Build community mapping
        self.communities = defaultdict(list)
        for node, cid in partition.items():
            self.communities[cid].append(node)
            if node in self.graph.nodes:
                self.graph.nodes[node]["community"] = cid

    def save(self, path: str):
        """Save the knowledge graph to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        nx.write_gml(self.graph, str(save_path / "graph.gml"))

        entities_data = {name: {"type": e.entity_type, "mentions": e.mentions, "doc_ids": e.doc_ids}
                        for name, e in self.entities.items()}
        with open(save_path / "entities.json", "w", encoding="utf-8") as f:
            json.dump(entities_data, f)

    def load(self, path: str):
        """Load the knowledge graph from disk."""
        load_path = Path(path)
        self.graph = nx.read_gml(str(load_path / "graph.gml"))

        with open(load_path / "entities.json", "r", encoding="utf-8") as f:
            entities_data = json.load(f)
        self.entities = {
            name: Entity(name=name, entity_type=d["type"], mentions=d["mentions"], doc_ids=d["doc_ids"])
            for name, d in entities_data.items()
        }
        self._detect_communities()


class _SimpleEntityExtractor:
    """Simple rule-based entity extraction (no external NER model required)."""

    # Common entity patterns
    PATTERNS = [
        # Capitalized multi-word names
        (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', "PERSON"),
        # All-caps abbreviations
        (r'\b([A-Z]{2,})\b', "ORG"),
        # Quoted terms
        (r'"([^"]+)"', "CONCEPT"),
        # Year references
        (r'\b((?:19|20)\d{2})\b', "DATE"),
    ]

    # Common stop words to filter
    STOP_WORDS = {
        "The", "This", "That", "These", "Those", "What", "When", "Where",
        "Which", "Who", "How", "Not", "But", "And", "For", "With", "From",
    }

    def extract(self, text: str) -> List[Entity]:
        """Extract entities from text using pattern matching."""
        entities = []
        seen = set()

        for pattern, entity_type in self.PATTERNS:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if name not in seen and name not in self.STOP_WORDS and len(name) > 2:
                    seen.add(name)
                    # Get context around the mention
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    entities.append(Entity(
                        name=name,
                        entity_type=entity_type,
                        mentions=[context],
                    ))

        return entities
