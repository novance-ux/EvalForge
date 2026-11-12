"""Semantic similarity scorer (cosine similarity of embeddings)."""

from __future__ import annotations

from app.models.results import ScoreResult
from app.scorers.base import Scorer, ScorerInputError, ScoringContext, register_scorer
from app.scorers.embeddings import cosine, get_embedding_provider


@register_scorer("semantic_similarity")
class SemanticSimilarityScorer(Scorer):
    async def score(self, ctx: ScoringContext) -> ScoreResult:
        reference = self.expected_value(ctx, "reference")
        if reference is None:
            raise ScorerInputError("semantic_similarity needs 'reference' or 'value_from'")
        target = self.target(ctx)
        provider = get_embedding_provider(self.spec.embedding, ctx.settings)
        a, b = await provider.embed([str(target), str(reference)])
        sim = cosine(a, b)
        ok = sim >= self.spec.threshold
        return self.result(
            max(0.0, sim), ok,
            f"Cosine similarity {sim:.3f} {'>=' if ok else '<'} threshold {self.spec.threshold}"
            f" ({self.spec.embedding} embeddings)",
            similarity=sim,
        )
