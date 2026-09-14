"""Tests for Self-Reflection Engine."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from layers.self_reflection.token_generator import ReflectionTokenGenerator, ReflectionTokens
from layers.self_reflection.verifier import ClaimVerifier, Claim
from layers.self_reflection.reflection_engine import SelfReflectionEngine


class TestReflectionTokenGenerator:
    def setup_method(self):
        self.gen = ReflectionTokenGenerator()

    def test_retrieve_token_factual(self):
        token = self.gen.generate_retrieve_token("Who is the president of France?")
        assert token in ["[Retrieve]", "[No Retrieve]"]
        assert token == "[Retrieve]"  # Factual question should retrieve

    def test_relevance_token_relevant(self):
        token = self.gen.generate_relevance_token(
            "What is Python?",
            "Python is a popular programming language."
        )
        assert token == "[Relevant]"

    def test_relevance_token_irrelevant(self):
        token = self.gen.generate_relevance_token(
            "What is Python?",
            "The weather is sunny today."
        )
        assert token == "[Irrelevant]"

    def test_support_token(self):
        token = self.gen.generate_support_token(
            "Python is a programming language.",
            "Python is a popular programming language used for web development."
        )
        assert token in ["[Fully Supported]", "[Partially Supported]", "[No Support]"]

    def test_utility_token(self):
        token = self.gen.generate_utility_token(
            "What is Python?",
            "Python is a programming language."
        )
        assert token in ["[Very Useful]", "[Useful]", "[Not Useful]"]

    def test_generate_all(self):
        tokens = self.gen.generate_all(
            query="What is Python?",
            context="Python is a programming language.",
            answer="Python is a popular programming language.",
        )
        assert isinstance(tokens, ReflectionTokens)
        assert tokens.retrieve in ["[Retrieve]", "[No Retrieve]"]
        assert tokens.is_relevant in [True, False]


class TestReflectionTokens:
    def test_properties(self):
        tokens = ReflectionTokens(
            retrieve="[Retrieve]",
            relevance="[Relevant]",
            support="[Fully Supported]",
            utility="[Very Useful]",
        )
        assert tokens.is_retrieval_needed is True
        assert tokens.is_relevant is True
        assert tokens.is_supported is True
        assert tokens.is_useful is True
        assert tokens.needs_revision is False

    def test_needs_revision(self):
        tokens = ReflectionTokens(
            retrieve="[Retrieve]",
            relevance="[Irrelevant]",
            support="[No Support]",
            utility="[Not Useful]",
        )
        assert tokens.needs_revision is True


class TestClaimVerifier:
    def setup_method(self):
        self.verifier = ClaimVerifier()

    def test_extract_claims(self):
        text = "Python is a programming language. It was created by Guido van Rossum."
        claims = self.verifier.extract_claims(text)
        assert len(claims) >= 1

    def test_verify_claims(self):
        claims = [Claim(text="Python is a programming language")]
        verified = self.verifier.verify_claims(
            claims,
            "Python is a popular programming language used for web development."
        )
        assert len(verified) == 1
        assert verified[0].verification_status in ["verified", "contradicted", "unverified"]

    def test_verify_answer(self):
        result = self.verifier.verify_answer(
            "Python is a programming language.",
            "Python is a popular programming language."
        )
        assert "claims" in result
        assert "overall_confidence" in result
        assert 0 <= result["overall_confidence"] <= 1


class TestSelfReflectionEngine:
    def setup_method(self):
        self.engine = SelfReflectionEngine(max_retries=1)

    def test_reflect_and_generate(self):
        result = self.engine.reflect_and_generate(
            query="What is Python?",
            context="Python is a popular programming language.",
            generate_fn=lambda q, c, **kw: "Python is a programming language.",
        )
        assert result.answer is not None
        assert result.tokens is not None
        assert 0 <= result.final_confidence <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
