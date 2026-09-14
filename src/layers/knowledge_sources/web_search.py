"""Web Search — Fallback web search for corrective retrieval."""

import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class WebSearchResult:
    """A single web search result."""
    title: str
    snippet: str
    url: str
    score: float = 0.0


class WebSearch:
    """Web search interface with multiple provider support."""

    def __init__(self, provider: str = "duckduckgo", max_results: int = 5, timeout: int = 10):
        self.provider = provider
        self.max_results = max_results
        self.timeout = timeout
        self._client = None
        self._cache: dict = {}

    def search(self, query: str, max_results: Optional[int] = None) -> List[WebSearchResult]:
        """Search the web for the given query.

        Args:
            query: Search query string
            max_results: Override default max results

        Returns:
            List of WebSearchResult objects
        """
        max_results = max_results or self.max_results

        # Check cache
        cache_key = f"{query}_{max_results}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        results = []
        try:
            if self.provider == "duckduckgo":
                results = self._search_duckduckgo(query, max_results)
            elif self.provider == "tavily":
                results = self._search_tavily(query, max_results)
            else:
                results = self._search_duckduckgo(query, max_results)
        except Exception as e:
            print(f"Web search error ({self.provider}): {e}")
            # Return empty results on failure
            results = []

        # Cache results
        self._cache[cache_key] = results
        return results

    def _search_duckduckgo(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Search using DuckDuckGo (free, no API key needed)."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))
                return [
                    WebSearchResult(
                        title=r.get("title", ""),
                        snippet=r.get("body", ""),
                        url=r.get("href", ""),
                        score=1.0 - (i * 0.1),  # Simple rank-based score
                    )
                    for i, r in enumerate(raw_results)
                ]
        except ImportError:
            print("Warning: duckduckgo-search not installed. Install with: pip install duckduckgo-search")
            return self._fallback_search(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Search using Tavily API (requires API key)."""
        import os
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            print("Warning: TAVILY_API_KEY not set, falling back to DuckDuckGo")
            return self._search_duckduckgo(query, max_results)

        try:
            import requests
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            return [
                WebSearchResult(
                    title=r.get("title", ""),
                    snippet=r.get("content", ""),
                    url=r.get("url", ""),
                    score=r.get("score", 0.5),
                )
                for r in data.get("results", [])
            ]
        except Exception as e:
            print(f"Tavily search error: {e}")
            return self._search_duckduckgo(query, max_results)

    def _fallback_search(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Fallback search using requests + BeautifulSoup."""
        try:
            import requests
            from bs4 import BeautifulSoup

            url = f"https://html.duckduckgo.com/html/?q={query}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=self.timeout)
            soup = BeautifulSoup(response.text, "html.parser")

            results = []
            for i, result in enumerate(soup.select(".result")):
                if i >= max_results:
                    break
                title_el = result.select_one(".result__title a")
                snippet_el = result.select_one(".result__snippet")
                if title_el:
                    results.append(WebSearchResult(
                        title=title_el.get_text(strip=True),
                        snippet=snippet_el.get_text(strip=True) if snippet_el else "",
                        url=title_el.get("href", ""),
                        score=1.0 - (i * 0.1),
                    ))
            return results
        except Exception as e:
            print(f"Fallback search error: {e}")
            return []

    def clear_cache(self):
        """Clear the search cache."""
        self._cache.clear()
