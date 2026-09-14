"""Tests for SemanticQueryCache."""

import time
import pytest
from src.layers.query_cache import SemanticQueryCache, CacheEntry, CacheStats


class TestSemanticQueryCache:
    """Tests for the semantic query cache."""

    @pytest.fixture
    def cache(self):
        """Create a small cache for testing."""
        return SemanticQueryCache(
            max_size=10,
            ttl_seconds=60,
            similarity_threshold=0.95,
        )

    def test_put_and_get(self, cache):
        """Should be able to put and get values."""
        cache.put("What is ML?", {"answer": "Machine learning is..."})
        result = cache.get("What is ML?")
        assert result is not None
        assert result["answer"] == "Machine learning is..."

    def test_exact_match(self, cache):
        """Exact query match should return cached value."""
        cache.put("test query", "test answer")
        result = cache.get("test query")
        assert result == "test answer"

    def test_case_insensitive_match(self, cache):
        """Case-insensitive match should work."""
        cache.put("What is ML?", "answer")
        result = cache.get("what is ml?")
        assert result == "answer"

    def test_cache_miss(self, cache):
        """Cache miss should return None."""
        result = cache.get("nonexistent query")
        assert result is None

    def test_ttl_expiration(self):
        """Entries should expire after TTL."""
        cache = SemanticQueryCache(max_size=10, ttl_seconds=0.1)
        cache.put("test", "answer")
        time.sleep(0.2)
        result = cache.get("test")
        assert result is None

    def test_lru_eviction(self):
        """Should evict least recently used when at capacity."""
        cache = SemanticQueryCache(max_size=3, ttl_seconds=3600)

        cache.put("query1", "answer1")
        cache.put("query2", "answer2")
        cache.put("query3", "answer3")

        # Adding 4th should evict query1
        cache.put("query4", "answer4")

        assert cache.get("query1") is None
        assert cache.get("query4") == "answer4"

    def test_invalidate(self, cache):
        """Should be able to invalidate specific entries."""
        cache.put("test", "answer")
        cache.invalidate("test")
        assert cache.get("test") is None

    def test_clear(self, cache):
        """Clear should remove all entries."""
        cache.put("q1", "a1")
        cache.put("q2", "a2")
        cache.clear()
        assert cache.get("q1") is None
        assert cache.get("q2") is None

    def test_stats(self, cache):
        """Stats should track hits and misses."""
        cache.put("test", "answer")
        cache.get("test")  # hit
        cache.get("missing")  # miss

        stats = cache.stats
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

    def test_stats_dict(self, cache):
        """get_stats_dict should return a dictionary."""
        stats = cache.get_stats_dict()
        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_save_load_config(self, cache, tmp_path):
        """Should be able to save and load configuration."""
        cache.save(str(tmp_path / "cache"))
        assert (tmp_path / "cache" / "config.json").exists()

        new_cache = SemanticQueryCache()
        new_cache.load(str(tmp_path / "cache"))
        assert new_cache.max_size == cache.max_size
        assert new_cache.ttl_seconds == cache.ttl_seconds

    def test_update_existing_entry(self, cache):
        """Updating an existing key should replace the value."""
        cache.put("test", "old_answer")
        cache.put("test", "new_answer")
        result = cache.get("test")
        assert result == "new_answer"
