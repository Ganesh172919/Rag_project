"""Claim Verifier — Verify factual claims in generated answers."""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Claim:
    """A factual claim extracted from text."""
    text: str
    is_verifiable: bool = True
    verification_status: str = "unverified"  # verified, contradicted, unverified
    supporting_evidence: str = ""
    confidence: float = 0.0


class ClaimVerifier:
    """Extract and verify factual claims from generated answers."""

    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn

    def extract_claims(self, text: str) -> List[Claim]:
        """Extract factual claims from text."""
        if self.llm_fn:
            return self._extract_claims_llm(text)
        return self._extract_claims_heuristic(text)

    def verify_claims(self, claims: List[Claim], context: str) -> List[Claim]:
        """Verify claims against the provided context."""
        verified = []
        for claim in claims:
            if self.llm_fn:
                verified_claim = self._verify_claim_llm(claim, context)
            else:
                verified_claim = self._verify_claim_heuristic(claim, context)
            verified.append(verified_claim)
        return verified

    def verify_answer(self, answer: str, context: str) -> dict:
        """Full verification pipeline: extract claims, verify, summarize.

        Returns:
            Dict with 'claims', 'verified_count', 'contradicted_count',
            'unverified_count', 'overall_confidence'
        """
        claims = self.extract_claims(answer)
        verified_claims = self.verify_claims(claims, context)

        verified_count = sum(1 for c in verified_claims if c.verification_status == "verified")
        contradicted_count = sum(1 for c in verified_claims if c.verification_status == "contradicted")
        unverified_count = sum(1 for c in verified_claims if c.verification_status == "unverified")

        total = len(verified_claims) if verified_claims else 1
        overall_confidence = verified_count / total

        return {
            "claims": verified_claims,
            "total_claims": len(verified_claims),
            "verified_count": verified_count,
            "contradicted_count": contradicted_count,
            "unverified_count": unverified_count,
            "overall_confidence": overall_confidence,
        }

    def _extract_claims_llm(self, text: str) -> List[Claim]:
        """Extract claims using LLM."""
        prompt = f"""Extract all factual claims from the following text. List each claim on a separate line.

Text: {text}

Claims:"""
        response = self.llm_fn(prompt).strip()
        claims = []
        for line in response.split("\n"):
            line = line.strip()
            if line and not line.startswith("Claims"):
                # Remove numbering
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
                if cleaned:
                    claims.append(Claim(text=cleaned))
        return claims if claims else self._extract_claims_heuristic(text)

    def _extract_claims_heuristic(self, text: str) -> List[Claim]:
        """Extract claims using sentence splitting (heuristic fallback)."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 15:  # Skip very short sentences
                # Sentences with factual indicators are more likely claims
                is_factual = any(
                    indicator in sent.lower()
                    for indicator in ["is", "was", "are", "were", "has", "have",
                                      "born", "founded", "created", "invented",
                                      "million", "billion", "percent", "%"]
                )
                claims.append(Claim(text=sent, is_verifiable=True))
        return claims

    def _verify_claim_llm(self, claim: Claim, context: str) -> Claim:
        """Verify a single claim using LLM."""
        prompt = f"""Verify if the following claim is supported by the context.

Claim: {claim.text}
Context: {context[:1500]}

Classify as:
- [VERIFIED]: The claim is directly supported by the context.
- [CONTRADICTED]: The claim contradicts information in the context.
- [UNVERIFIED]: The claim cannot be verified from the context.

Answer:"""
        response = self.llm_fn(prompt).strip()

        if "[VERIFIED]" in response:
            claim.verification_status = "verified"
            claim.confidence = 0.9
        elif "[CONTRADICTED]" in response:
            claim.verification_status = "contradicted"
            claim.confidence = 0.1
        else:
            claim.verification_status = "unverified"
            claim.confidence = 0.5

        return claim

    def _verify_claim_heuristic(self, claim: Claim, context: str) -> Claim:
        """Verify a single claim using keyword overlap (heuristic fallback)."""
        claim_words = set(claim.text.lower().split()) - {
            "the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
            "in", "of", "to", "for", "with", "on", "at", "by",
        }
        context_lower = context.lower()

        if not claim_words:
            claim.verification_status = "unverified"
            claim.confidence = 0.5
            return claim

        found = sum(1 for w in claim_words if w in context_lower)
        ratio = found / len(claim_words)

        if ratio > 0.6:
            claim.verification_status = "verified"
            claim.confidence = min(ratio, 0.95)
        elif ratio > 0.3:
            claim.verification_status = "unverified"
            claim.confidence = ratio
        else:
            claim.verification_status = "contradicted"
            claim.confidence = max(0.1, ratio)

        return claim
