"""Scraper module using Crawl4AI."""

import hashlib
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_STORAGE_DIR = Path(os.getenv("APP_STORAGE_DIR", str(PROJECT_ROOT / "data"))).expanduser()
os.environ.setdefault(
    "CRAWL4_AI_BASE_DIRECTORY", str(APP_STORAGE_DIR / "cache")
)
from models.schema import NewsArticle
from config import settings
from utils.logging import get_logger

logger = get_logger(__name__)


def _stable_article_id(source_url: str, title: str, content: str) -> str:
    """Create a deterministic article ID so duplicate detection survives restarts."""

    payload = f"{source_url}|{title}|{content}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"news_{digest[:16]}"


class NewsScraper:
    """Scrapes news articles from ITLN websites."""

    def __init__(self):
        self.urls = settings.target_urls
        self.logger = logger
        self.extraction_schemas = self.extraction_schemas = [
            {
                "name": "ITLN Primary Listings",
                # This matches exactly what is shown in your DOM inspector image
                "baseSelector": "div.listing_item.news_listing", 
                "fields": [
                    {
                        "name": "title",
                        # On ITLN, the headline is usually an <a> tag or inside h2/h3
                        "selector": "h2 a, h3 a, a.title, .news-title",
                        "type": "text",
                    },
                    {
                        "name": "snippet",
                        "selector": "p, .excerpt, .summary",
                        "type": "text",
                        "default": "",
                    },
                    {
                        "name": "link",
                        "selector": "a",
                        "type": "attribute",
                        "attribute": "href",
                        "default": "",
                    },
                    {
                        "name": "date",
                        "selector": ".date, time, .published",
                        "type": "text",
                        "default": "",
                    },
                ],
            }
        ]

    async def fetch_all(self) -> List[NewsArticle]:
        """Fetch news from all configured URLs."""
        all_articles = []

        from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, JsonCssExtractionStrategy

        async with AsyncWebCrawler(verbose=False) as crawler:
            for url in self.urls:
                try:
                    self.logger.info(f"Scraping: {url}")
                    articles = await self._fetch_articles_from_url(
                        crawler,
                        url,
                        CacheMode,
                        CrawlerRunConfig,
                        JsonCssExtractionStrategy,
                    )
                    all_articles.extend(articles)
                    self.logger.info(f"Parsed {len(articles)} articles from {url}")

                except Exception as e:
                    self.logger.error(f"Error scraping {url}: {e}")

        return all_articles

    async def _fetch_articles_from_url(
        self,
        crawler,
        url: str,
        cache_mode_cls,
        crawler_run_config_cls,
        extraction_strategy_cls,
    ) -> List[NewsArticle]:
        """Fetch and extract article cards from one page."""

        for schema in self.extraction_schemas:
            try:
                result = await crawler.arun(
                    url,
                    config=crawler_run_config_cls(
                        cache_mode=cache_mode_cls.BYPASS,
                        extraction_strategy=extraction_strategy_cls(schema, verbose=False),
                        # Keep the crawl focused on the page itself and avoid noisy markdown
                        prettiify=False,
                        verbose=False,
                    ),
                )

                articles = self._parse_extracted_content(result.extracted_content, url)
                if articles:
                    return articles
            except Exception as exc:
                self.logger.warning(
                    "Structured extraction failed for %s with schema '%s': %s",
                    url,
                    schema["name"],
                    exc,
                )

        self.logger.warning(
            "Structured extraction returned no articles for %s; skipping markdown fallback to avoid noisy results.",
            url,
        )
        return []

    def _parse_extracted_content(self, extracted_content: str | None, source_url: str) -> List[NewsArticle]:
        """Parse JSON output from JsonCssExtractionStrategy into NewsArticle objects."""

        if not extracted_content:
            return []

        try:
            payload = json.loads(extracted_content)
        except json.JSONDecodeError:
            self.logger.warning("Structured extraction output was not valid JSON for %s", source_url)
            return []

        if isinstance(payload, dict):
            items = payload.get("items") or payload.get("data") or []
        elif isinstance(payload, list):
            items = payload
        else:
            items = []

        articles: List[NewsArticle] = []
        seen_keys: set[str] = set()

        for item in items:
            if not isinstance(item, dict):
                continue

            title = self._clean_text(item.get("title"))
            snippet = self._clean_text(item.get("snippet"))
            link = self._normalize_url(source_url, item.get("link"))
            date_text = self._clean_text(item.get("date"))

            if not title or not link:
                continue

            content_parts = [title]
            if snippet and snippet != title:
                content_parts.append(snippet)
            content = "\n\n".join(content_parts).strip()

            dedupe_key = f"{title}|{link}"
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            articles.append(
                NewsArticle(
                    id=_stable_article_id(source_url, title, content),
                    title=title[:200],
                    content=content[:2000],
                    url=link,
                    published_at=self._parse_date(date_text),
                    source="itln.in",
                )
            )

        return articles

    def _parse_markdown(self, markdown: str, source_url: str) -> List[NewsArticle]:
        """
        Parse markdown content into NewsArticle objects.
        This is a basic parser - may need adjustment based on actual page structure.
        """
        articles = []

        # Split by potential article separators (heuristic)
        # In production, this would use DOM structure or better parsing
        sections = markdown.split("\n\n")

        for i, section in enumerate(sections):
            section = section.strip()
            if not section or len(section) < 20:
                continue

            # Extract title (first line) and content
            lines = section.split("\n", 1)
            title = lines[0].strip() if lines else "No Title"
            content = lines[1].strip() if len(lines) > 1 else ""

            if len(content) < 10:
                continue  # Skip too-short entries

            article = NewsArticle(
                id=_stable_article_id(source_url, title, content),
                title=title[:200],
                content=content[:2000],
                url=source_url,
                published_at=None,  # Could parse from content
                source="itln.in",
            )
            articles.append(article)

        return articles

    def _clean_text(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            value = " ".join(str(item) for item in value)
        return " ".join(str(value).split()).strip()

    def _normalize_url(self, base_url: str, href) -> str:
        from urllib.parse import urljoin

        href_text = self._clean_text(href)
        if not href_text:
            return base_url
        return urljoin(base_url, href_text)

    def _parse_date(self, date_text: str):
        if not date_text:
            return None

        date_text = date_text.strip()
        for fmt in (
            "%d %b %Y %I:%M %p IST",
            "%d %b %Y %H:%M %p IST",
            "%d %b %Y %I:%M %p",
            "%d %b %Y",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(date_text, fmt)
            except ValueError:
                continue
        return None

    async def fetch_single(self, url: str) -> NewsArticle | None:
        """Fetch a single page and return as article."""
        from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, JsonCssExtractionStrategy

        async with AsyncWebCrawler(verbose=False) as crawler:
            try:
                result = await crawler.arun(
                    url,
                    config=CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,
                        extraction_strategy=JsonCssExtractionStrategy(
                            self.extraction_schemas[0], verbose=False
                        ),
                        verbose=False,
                    ),
                )
                articles = self._parse_extracted_content(result.extracted_content, url)
                if articles:
                    return articles[0] if articles else None
                self.logger.warning(
                    "Structured extraction returned no article cards for %s",
                    url,
                )
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {e}")
        return None
