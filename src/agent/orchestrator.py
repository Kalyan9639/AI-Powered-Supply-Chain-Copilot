"""Main agent orchestrator."""

from __future__ import annotations

import asyncio
from typing import List

from agent.llm_client import GemmaRiskAnalyzer, load_business_profile
from memory.vector_store import VectorStore
from models.schema import (
    ApprovalDecision,
    NewsArticle,
    RiskAssessment,
    StakeholderEntry,
    UserConfig,
)
from scraper.news import NewsScraper
from utils.embeddings import EmbeddingModel
from utils.logging import get_logger

logger = get_logger(__name__)


class AgenticCopilot:
    """
    Main agentic AI copilot that orchestrates:
    1. News scraping
    2. Embedding generation
    3. Risk analysis
    4. Action proposal
    """

    def __init__(self):
        self.scraper = NewsScraper()
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.business_profile: UserConfig = load_business_profile()
        self.llm = GemmaRiskAnalyzer()
        self.logger = get_logger(__name__)
        self._cycle_lock = asyncio.Lock()

    async def run_cycle(self) -> List[tuple[str, RiskAssessment]]:
        """
        Run one complete cycle:
        - Scrape news
        - Check for duplicates
        - Analyze new news
        - Return risk assessments
        """
        async with self._cycle_lock:
            self.logger.info("Starting agent cycle")

            # 1. Scrape news
            articles = await self.scraper.fetch_all()
            self.logger.info("Scraped %s articles", len(articles))

            # 2. Filter new articles (check against vector store)
            new_articles = []
            for article in articles:
                if not self.vector_store.contains(article.id):
                    new_articles.append(article)

            self.logger.info("Found %s new articles", len(new_articles))

            if not new_articles:
                return []

            # 3. Process each new article
            assessments: List[tuple[str, RiskAssessment]] = []
            for article in new_articles:
                embedding_text = f"{article.title}\n{article.content[:500]}"
                embedding = self.embedding_model.encode_single(embedding_text)

                # Persist the scraped article immediately so it will not be reprocessed
                # on the next interval even if the reasoning step fails later.
                self.vector_store.add(article, embedding)

                try:
                    similar_articles = await asyncio.to_thread(
                        self.vector_store.search_similar,
                        embedding_text,
                        3,
                    )
                    assessment = await self._analyze_article(article, similar_articles)
                except Exception as exc:
                    self.logger.exception(
                        "Failed to analyze article %s: %s", article.id, exc
                    )
                    continue

                self.vector_store.add_assessment(article.id, assessment)
                assessments.append((article.id, assessment))

            self.logger.info("Completed cycle with %s assessments", len(assessments))
            return assessments

    async def _analyze_article(
        self,
        article: NewsArticle,
        similar_articles: list[NewsArticle] | None = None,
    ) -> RiskAssessment:
        """
        Analyze a news article using the LLM.
        Returns a risk assessment with reasoning and proposed action.
        """

        return await self.llm.analyze_article(
            article=article,
            business_profile=self.business_profile,
            similar_articles=similar_articles or [],
        )

    async def get_latest_assessments(self, limit: int = 10) -> List[tuple[str, RiskAssessment]]:
        """Get the most recent risk assessments."""
        return self.vector_store.get_latest_assessments(limit)

    async def get_assessment_for_news(self, news_id: str) -> RiskAssessment | None:
        """Get risk assessment for a specific news article."""
        return self.vector_store.get_assessment(news_id)

    async def decide_approval_action(
        self,
        news_id: str,
        stakeholders: list[StakeholderEntry],
    ) -> ApprovalDecision:
        """Run Gemma again to decide whether a stakeholder should be messaged."""

        assessment = self.vector_store.get_assessment(news_id)
        article = self.vector_store.get_news_article(news_id)

        if assessment is None:
            raise ValueError("Assessment not found")
        if article is None:
            raise ValueError("News article not found")

        return await self.llm.decide_approval_action(
            article=article,
            assessment=assessment,
            business_profile=self.business_profile,
            stakeholders=stakeholders,
        )
