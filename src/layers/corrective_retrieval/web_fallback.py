"""Web Fallback — Retrieve from web when local retrieval fails."""

from typing import List, Optional

from ..knowledge_sources.web_search import WebSearch, WebSearchResult


class WebFallback:
    """Web search fallback for when local retrieval quality is insufficient."""

    def __init__(self, provider: str = "duckduckgo", max_results: int = 5):
        self.web_search = WebSearch(provider=provider, max_results=max_results)
        self.max_results = max_results

    def search(self, query: str, reformulated_query: Optional[str] = None) -> List[str]:
        """Search the web for additional context.

        Args:
            query: Original query
            reformulated_query: Optional reformulated query for better web results

        Returns:
            List of text snippets from web search results
        """
        search_query = reformulated_query or query
        results = self.web_search.search(search_query, max_results=self.max_results)

        # Combine title and snippet for each result
        texts = []
        for r in results:
            text = f"{r.title}. {r.snippet}" if r.title else r.snippet
            if text.strip():
                texts.append(text.strip())

        return texts

    def reformulate_for_web(self, query: str) -> str:
        """Reformulate a query for better web search results.

        Removes question words and focuses on key terms.
        """
        # Remove common question prefixes
        reformulated = query.lower()
        prefixes = [
            "can you tell me", "could you explain", "please explain",
            "what is the", "what are the", "who is the", "who are the",
            "tell me about", "i want to know", "explain",
        ]
        for prefix in prefixes:
            if reformulated.startswith(prefix):
                reformulated = reformulated[len(prefix):].strip()
                break

        # Clean up
        reformulated = reformulated.strip("?.,!")

        return reformulated if reformulated else query
