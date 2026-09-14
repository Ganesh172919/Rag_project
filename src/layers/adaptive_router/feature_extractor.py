"""Feature Extractor — Extract features from queries for complexity classification."""

import re
from typing import List, Dict


class FeatureExtractor:
    """Extract features from queries to determine complexity level."""

    # Multi-hop indicators
    MULTI_HOP_KEYWORDS = [
        "both", "and", "compare", "contrast", "difference", "similar",
        "relationship", "versus", "vs", "which one", "which of",
        "first", "second", "then", "after", "before", "while",
    ]

    # Global/corpus-level indicators
    GLOBAL_KEYWORDS = [
        "all", "every", "each", "overall", "summary", "summarize",
        "main themes", "common", "trends", "across", "general",
        "overview", "total", "aggregate", "distribution",
    ]

    # Simple factual indicators
    SIMPLE_PATTERNS = [
        r"^who (is|was|are|were)",
        r"^what (is|was|are|were)",
        r"^when (did|was|is)",
        r"^where (is|was|are|were)",
        r"^how (old|many|much|long|far|tall)",
    ]

    def extract(self, query: str) -> Dict[str, float]:
        """Extract numerical features from a query.

        Returns dict of feature_name -> feature_value.
        """
        query_lower = query.lower().strip()
        words = query_lower.split()

        features = {
            # Length features
            "word_count": len(words),
            "char_count": len(query_lower),
            "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),

            # Question type features
            "has_who": float("who" in query_lower),
            "has_what": float("what" in query_lower),
            "has_when": float("when" in query_lower),
            "has_where": float("where" in query_lower),
            "has_why": float("why" in query_lower),
            "has_how": float("how" in query_lower),
            "has_which": float("which" in query_lower),

            # Multi-hop indicators
            "multi_hop_keyword_count": sum(
                1 for kw in self.MULTI_HOP_KEYWORDS if kw in query_lower
            ),
            "has_comparison": float(any(
                kw in query_lower for kw in ["compare", "difference", "versus", "vs", "similar"]
            )),
            "has_sequence": float(any(
                kw in query_lower for kw in ["first", "then", "after", "before", "finally"]
            )),
            "conjunction_count": query_lower.count(" and ") + query_lower.count(" or "),

            # Global indicators
            "global_keyword_count": sum(
                1 for kw in self.GLOBAL_KEYWORDS if kw in query_lower
            ),
            "has_all": float("all" in query_lower.split()),
            "has_summary": float(any(
                kw in query_lower for kw in ["summary", "summarize", "overview"]
            )),

            # Simple indicators
            "is_simple_pattern": float(any(
                re.match(p, query_lower) for p in self.SIMPLE_PATTERNS
            )),

            # Named entity indicators (simple heuristic)
            "capitalized_word_count": sum(
                1 for w in query.split() if w[0].isupper() and len(w) > 2
            ) if query else 0,

            # Punctuation features
            "question_mark_count": query.count("?"),
            "comma_count": query.count(","),
        }

        return features

    def extract_batch(self, queries: List[str]) -> List[Dict[str, float]]:
        """Extract features from a batch of queries."""
        return [self.extract(q) for q in queries]

    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names."""
        dummy = self.extract("test")
        return sorted(dummy.keys())
