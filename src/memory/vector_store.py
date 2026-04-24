"""Vector memory store using ChromaDB."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import chromadb
from chromadb.types import Collection

from config import settings
from models.schema import NewsArticle, RiskAssessment
from utils.embeddings import EmbeddingModel
from utils.logging import get_logger

logger = get_logger(__name__)


def _parse_datetime(value: object) -> datetime | None:
    """Convert stored ISO strings back into datetime objects."""

    if not value:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    return None


class VectorStore:
    """Local vector database for news and assessments."""

    def __init__(self):
        self._persistent = True
        try:
            settings.chromadb_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(settings.chromadb_path.absolute())
            )
        except Exception as exc:
            self._persistent = False
            logger.warning(
                "Persistent ChromaDB unavailable at %s, falling back to ephemeral storage: %s",
                settings.chromadb_path,
                exc,
            )
            self.client = chromadb.EphemeralClient()
        self.embedding_model = EmbeddingModel()
        self.news_collection = self._get_or_create_collection("news")
        self.assessments_collection = self._get_or_create_collection(
            "assessments", metadata={"hnsw:space": "cosine"}
        )
        logger.info("VectorStore initialized")

    def _get_or_create_collection(self, name: str, **kwargs) -> Collection:
        """Get existing collection or create new one."""

        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(name, **kwargs)

    def add(self, article: NewsArticle, embedding: List[float]) -> None:
        """Add a news article with its embedding."""

        self.news_collection.upsert(
            ids=[article.id],
            embeddings=[embedding],
            documents=[f"{article.title}\n\n{article.content}"],
            metadatas=[
                {
                    "title": article.title,
                    "url": article.url,
                    "source": article.source,
                    "published_at": article.published_at.isoformat()
                    if article.published_at
                    else None,
                }
            ],
        )

    def contains(self, news_id: str) -> bool:
        """Check if a news article ID exists in the store."""

        result = self.news_collection.get(ids=[news_id])
        return len(result["ids"]) > 0

    def add_assessment(self, news_id: str, assessment: RiskAssessment) -> None:
        """Store a risk assessment."""

        self.assessments_collection.upsert(
            ids=[news_id],
            embeddings=[[0.0] * 384],  # Placeholder - not used for retrieval
            documents=[
                f"Risk: {assessment.risk_level}\nReasoning: {assessment.reasoning}\nAction: {assessment.proposed_action or ''}"
            ],
            metadatas=[
                {
                    "news_id": news_id,
                    "risk_level": assessment.risk_level,
                    "reasoning": assessment.reasoning,
                    "proposed_action": assessment.proposed_action,
                    "confidence": assessment.confidence,
                    "created_at": assessment.created_at.isoformat(),
                }
            ],
        )

    def _assessment_from_metadata(self, metadata: dict) -> RiskAssessment:
        """Rebuild a RiskAssessment from stored metadata."""

        created_at = _parse_datetime(metadata.get("created_at")) or datetime.now()
        confidence_value = metadata.get("confidence")
        return RiskAssessment(
            risk_level=metadata.get("risk_level", "Low"),
            reasoning=metadata.get("reasoning", ""),
            proposed_action=metadata.get("proposed_action"),
            confidence=float(confidence_value) if confidence_value is not None else 0.0,
            created_at=created_at,
        )

    def _risk_sort_key(self, assessment: RiskAssessment):
        """Sort by severity first, then confidence, then recency."""

        risk_rank = {"High": 0, "Medium": 1, "Low": 2}
        return (
            risk_rank.get(assessment.risk_level, 3),
            -float(assessment.confidence),
            -assessment.created_at.timestamp(),
        )

    def get_assessment(self, news_id: str) -> Optional[RiskAssessment]:
        """Retrieve a risk assessment by news ID."""

        result = self.assessments_collection.get(ids=[news_id])
        if not result["ids"]:
            return None

        metadata = result["metadatas"][0]
        return self._assessment_from_metadata(metadata)

    def get_news_article(self, news_id: str) -> Optional[NewsArticle]:
        """Retrieve the stored news article for a given news ID."""

        result = self.news_collection.get(ids=[news_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None

        metadata = (result.get("metadatas") or [None])[0] or {}
        document = (result.get("documents") or [None])[0] or ""
        content = document
        if "\n\n" in content:
            _, content = content.split("\n\n", 1)

        return NewsArticle(
            id=news_id,
            title=metadata.get("title", "No Title"),
            content=content[:2000],
            url=metadata.get("url", ""),
            published_at=_parse_datetime(metadata.get("published_at")),
            source=metadata.get("source", "itln.in"),
        )

    def get_latest_assessments(self, limit: int = 10) -> List[tuple[str, RiskAssessment]]:
        """Get the most recent assessments with their associated news IDs."""

        results = self.assessments_collection.get(include=["metadatas"])
        assessments: List[tuple[str, RiskAssessment]] = []

        ids = results.get("ids", []) or []
        for news_id, metadata in zip(ids, results.get("metadatas", []) or []):
            if not metadata:
                continue
            assessments.append((news_id, self._assessment_from_metadata(metadata)))

        assessments.sort(key=lambda item: self._risk_sort_key(item[1]))
        return assessments[:limit]

    def search_similar(self, query: str, limit: int = 5) -> List[NewsArticle]:
        """
        Search for similar news articles using embeddings.
        Uses ChromaDB's semantic search.
        """

        if not query.strip():
            return []

        try:
            query_embedding = self.embedding_model.encode_single(query)
            results = self.news_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning("Similarity search failed: %s", exc)
            return []

        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        articles: List[NewsArticle] = []
        for news_id, document, metadata in zip(ids, documents, metadatas):
            if not metadata:
                continue

            title = metadata.get("title") or "No Title"
            content = document or ""
            if "\n\n" in content:
                _, content = content.split("\n\n", 1)

            articles.append(
                NewsArticle(
                    id=news_id,
                    title=title,
                    content=content[:2000],
                    url=metadata.get("url", ""),
                    published_at=_parse_datetime(metadata.get("published_at")),
                    source=metadata.get("source", "itln.in"),
                )
            )

        return articles

    def clear(self) -> None:
        """Clear all data from the store."""

        self.client.reset()
        self.news_collection = self._get_or_create_collection("news")
        self.assessments_collection = self._get_or_create_collection("assessments")
        logger.info("Vector store cleared")
