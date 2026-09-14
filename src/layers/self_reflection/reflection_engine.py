"""Self-Reflection Engine — Main integration of reflection tokens and verification."""

from dataclasses import dataclass
from typing import Optional, Dict, List

from .token_generator import ReflectionTokenGenerator, ReflectionTokens
from .verifier import ClaimVerifier, Claim


@dataclass
class ReflectionResult:
    """Result of the self-reflection process."""
    answer: str
    tokens: ReflectionTokens
    verification: Optional[dict] = None
    revised: bool = False
    revision_count: int = 0
    final_confidence: float = 0.0


class SelfReflectionEngine:
    """Self-Reflection Engine for RAG output quality control.

    Based on: Self-RAG (Asai et al., ICLR 2024)

    Pipeline:
    1. Decide if retrieval is needed
    2. Evaluate relevance of retrieved context
    3. Generate answer
    4. Check if answer is supported by context
    5. Evaluate utility of answer
    6. Revise if needed (up to max_retries)
    """

    def __init__(self, llm_fn=None, max_retries: int = 2):
        self.token_generator = ReflectionTokenGenerator(llm_fn=llm_fn)
        self.claim_verifier = ClaimVerifier(llm_fn=llm_fn)
        self.max_retries = max_retries
        self.llm_fn = llm_fn

    def reflect_and_generate(
        self,
        query: str,
        context: str,
        generate_fn=None,
    ) -> ReflectionResult:
        """Full self-reflection pipeline.

        Args:
            query: The user's question
            context: Retrieved context (from CRAG)
            generate_fn: Function to generate answers (prompt -> answer)

        Returns:
            ReflectionResult with answer and reflection metadata
        """
        if generate_fn is None:
            generate_fn = self._default_generate

        # Step 1: Should we retrieve?
        retrieve_token = self.token_generator.generate_retrieve_token(query)

        # Step 2: Evaluate relevance
        relevance_token = self.token_generator.generate_relevance_token(query, context)

        # Step 3: Generate answer
        if relevance_token == "[Relevant]":
            answer = generate_fn(query, context)
        else:
            # Context is not relevant — generate without context
            answer = generate_fn(query, "")

        # Step 4: Check support
        support_token = self.token_generator.generate_support_token(answer, context)

        # Step 5: Check utility
        utility_token = self.token_generator.generate_utility_token(query, answer)

        tokens = ReflectionTokens(
            retrieve=retrieve_token,
            relevance=relevance_token,
            support=support_token,
            utility=utility_token,
        )

        # Step 6: Revise if needed
        revised = False
        revision_count = 0

        if tokens.needs_revision and revision_count < self.max_retries:
            answer, tokens, revision_count = self._revise(
                query, context, answer, tokens, generate_fn
            )
            revised = revision_count > 0

        # Verify claims
        verification = self.claim_verifier.verify_answer(answer, context)

        # Compute final confidence
        final_confidence = self._compute_confidence(tokens, verification)

        return ReflectionResult(
            answer=answer,
            tokens=tokens,
            verification=verification,
            revised=revised,
            revision_count=revision_count,
            final_confidence=final_confidence,
        )

    def _revise(
        self,
        query: str,
        context: str,
        answer: str,
        tokens: ReflectionTokens,
        generate_fn,
    ) -> tuple:
        """Revise the answer based on reflection feedback."""
        current_answer = answer
        current_tokens = tokens
        count = 0

        while count < self.max_retries and current_tokens.needs_revision:
            # Build revision instruction
            instructions = []
            if not current_tokens.is_relevant:
                instructions.append("The retrieved context was not relevant. Generate from general knowledge.")
            if not current_tokens.is_supported:
                instructions.append(
                    f"The answer was only {current_tokens.support}. "
                    "Only include information that is directly supported by the context. "
                    "If unsure, say 'I don't have enough information to answer this.'"
                )
            if not current_tokens.is_useful:
                instructions.append(
                    f"The answer was rated as {current_tokens.utility}. "
                    "Provide a more direct and complete answer to the question."
                )

            revision_prompt = f"Previous answer: {current_answer}\n\nInstructions: {' '.join(instructions)}"

            # Regenerate
            if current_tokens.is_relevant:
                current_answer = generate_fn(query, context, instruction=revision_prompt)
            else:
                current_answer = generate_fn(query, "", instruction=revision_prompt)

            # Re-evaluate
            current_tokens = self.token_generator.generate_all(
                query, context, current_answer
            )
            count += 1

        return current_answer, current_tokens, count

    def _compute_confidence(self, tokens: ReflectionTokens, verification: dict) -> float:
        """Compute overall confidence from reflection tokens and verification."""
        scores = []

        # Relevance score
        scores.append(1.0 if tokens.is_relevant else 0.0)

        # Support score
        support_scores = {
            "[Fully Supported]": 1.0,
            "[Partially Supported]": 0.5,
            "[No Support]": 0.0,
        }
        scores.append(support_scores.get(tokens.support, 0.5))

        # Utility score
        utility_scores = {
            "[Very Useful]": 1.0,
            "[Useful]": 0.6,
            "[Not Useful]": 0.0,
        }
        scores.append(utility_scores.get(tokens.utility, 0.5))

        # Verification confidence
        if verification:
            scores.append(verification.get("overall_confidence", 0.5))

        return sum(scores) / len(scores) if scores else 0.5

    def _default_generate(self, query: str, context: str, instruction: str = "") -> str:
        """Default generation when no LLM function is provided."""
        if context:
            return f"Based on the context, here is information about: {query}\n\n{context[:500]}"
        return f"I don't have specific context to answer: {query}"
