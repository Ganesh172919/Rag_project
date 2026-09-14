"""Document Decomposer — Decompose-recompose algorithm from CRAG."""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KnowledgeSnippet:
    """An atomic piece of knowledge extracted from a document."""
    text: str
    source_doc: str
    relevance_score: float = 0.0
    snippet_id: str = ""


class DocumentDecomposer:
    """Decompose documents into key snippets and recompose for relevance.

    Based on CRAG's decompose-recompose algorithm:
    1. Decompose each document into atomic knowledge snippets
    2. Score each snippet for relevance to the query
    3. Filter out irrelevant snippets
    4. Recompose remaining snippets in order of relevance
    """

    def __init__(self, strategy: str = "keypoint", min_relevance: float = 0.3):
        self.strategy = strategy
        self.min_relevance = min_relevance

    def decompose(self, document: str, doc_id: str = "") -> List[KnowledgeSnippet]:
        """Decompose a document into atomic knowledge snippets."""
        if self.strategy == "sentence":
            return self._decompose_sentences(document, doc_id)
        elif self.strategy == "paragraph":
            return self._decompose_paragraphs(document, doc_id)
        else:  # keypoint
            return self._decompose_keypoints(document, doc_id)

    def score_snippets(self, query: str, snippets: List[KnowledgeSnippet]) -> List[KnowledgeSnippet]:
        """Score each snippet for relevance to the query."""
        import string
        query_words = set(query.lower().translate(str.maketrans("", "", string.punctuation)).split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "who",
                      "when", "where", "how", "which", "that", "this", "it", "of", "in", "to", "and", "or", "be"}
        query_words -= stop_words

        if not query_words:
            for s in snippets:
                s.relevance_score = 0.5
            return snippets

        for snippet in snippets:
            snippet_lower = snippet.text.lower().translate(str.maketrans("", "", string.punctuation))
            snippet_words = set(snippet_lower.split())
            overlap = len(query_words & snippet_words)
            # Boost for substring matches (e.g., "python" in "python is a lang")
            substring_boost = sum(0.15 for w in query_words if w in snippet_lower)
            snippet.relevance_score = min((overlap / len(query_words)) + substring_boost, 1.0)

        return snippets

    def filter_snippets(self, snippets: List[KnowledgeSnippet]) -> List[KnowledgeSnippet]:
        """Filter out low-relevance snippets."""
        return [s for s in snippets if s.relevance_score >= self.min_relevance]

    def recompose(self, snippets: List[KnowledgeSnippet], max_length: int = 2000) -> str:
        """Recompose filtered snippets into a coherent context."""
        # Sort by relevance (highest first)
        sorted_snippets = sorted(snippets, key=lambda s: s.relevance_score, reverse=True)

        result = []
        current_length = 0
        for snippet in sorted_snippets:
            if current_length + len(snippet.text) > max_length:
                break
            result.append(snippet.text)
            current_length += len(snippet.text)

        return "\n\n".join(result)

    def decompose_recompose(self, query: str, documents: List[str], max_length: int = 2000) -> str:
        """Full decompose-recompose pipeline.

        Args:
            query: The user's question
            documents: List of document texts
            max_length: Maximum length of recomposed output

        Returns:
            Filtered and recomposed context string
        """
        all_snippets = []
        for i, doc in enumerate(documents):
            snippets = self.decompose(doc, doc_id=f"doc_{i}")
            all_snippets.extend(snippets)

        # Score and filter
        scored = self.score_snippets(query, all_snippets)
        filtered = self.filter_snippets(scored)

        # Recompose
        return self.recompose(filtered, max_length)

    def keypoint_extraction(self, documents: List[str], max_length: int = 2000) -> str:
        """Extract key points from documents (simplified version)."""
        all_snippets = []
        for i, doc in enumerate(documents):
            snippets = self._decompose_keypoints(doc, f"doc_{i}")
            all_snippets.extend(snippets)

        # Take top snippets by position (earlier = more important in keypoint extraction)
        for i, s in enumerate(all_snippets):
            s.relevance_score = 1.0 - (i * 0.05)

        return self.recompose(all_snippets, max_length)

    def _decompose_keypoints(self, document: str, doc_id: str) -> List[KnowledgeSnippet]:
        """Decompose into key points (sentences with key information)."""
        sentences = self._split_sentences(document)
        snippets = []
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) > 10:  # Skip very short sentences
                snippets.append(KnowledgeSnippet(
                    text=sent,
                    source_doc=doc_id,
                    snippet_id=f"{doc_id}_kp_{i}",
                ))
        return snippets

    def _decompose_sentences(self, document: str, doc_id: str) -> List[KnowledgeSnippet]:
        """Decompose into individual sentences."""
        sentences = self._split_sentences(document)
        return [
            KnowledgeSnippet(text=s.strip(), source_doc=doc_id, snippet_id=f"{doc_id}_s_{i}")
            for i, s in enumerate(sentences) if len(s.strip()) > 10
        ]

    def _decompose_paragraphs(self, document: str, doc_id: str) -> List[KnowledgeSnippet]:
        """Decompose into paragraphs."""
        paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            # Fall back to sentence decomposition for single paragraphs
            return self._decompose_sentences(document, doc_id)
        return [
            KnowledgeSnippet(text=p, source_doc=doc_id, snippet_id=f"{doc_id}_p_{i}")
            for i, p in enumerate(paragraphs)
        ]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Also split on newlines
        result = []
        for s in sentences:
            result.extend(s.split("\n"))
        return [s.strip() for s in result if s.strip()]
