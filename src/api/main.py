"""API - FastAPI Backend."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.orchestrator import AgenticCopilot
from config import get_runtime_config_payload, save_runtime_overrides, settings
from models.schema import ApprovalRequest, NewsArticle, RiskAssessment, UserConfig
from utils.logging import get_logger

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logger = get_logger(__name__)

app = FastAPI(
    title="Agentic AI Supply Chain Copilot API",
    description="API for the agentic AI copilot that monitors logistics news and proposes actions",
    version="0.1.0",
)

copilot: AgenticCopilot | None = None
business_profile_cache: UserConfig | None = None
background_task: asyncio.Task | None = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
BUSINESS_PROFILE_PATH = DATA_DIR / "config" / "business_profile.json"
LEGACY_CONFIG_PATH = PROJECT_ROOT / "config.json"
MANAGER_PAGE_PATH = PROJECT_ROOT / "manager.html"
DEFAULT_ENABLE_BACKGROUND_SCRAPER = "false" if os.getenv("K_SERVICE") else "true"
ENABLE_BACKGROUND_SCRAPER = os.getenv("ENABLE_BACKGROUND_SCRAPER", DEFAULT_ENABLE_BACKGROUND_SCRAPER).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


class RuntimeConfigUpdate(BaseModel):
    """Runtime env values managed from the website."""

    hf_api_key: str | None = None
    hf_model: str | None = None
    scrape_interval_minutes: int | None = Field(default=None, ge=1)
    target_urls: list[str] | None = None
    log_level: str | None = None
    debug: bool | None = None


def _load_business_profile() -> UserConfig:
    """Load the business profile from disk."""

    global business_profile_cache
    if business_profile_cache is not None:
        return business_profile_cache

    for path in (BUSINESS_PROFILE_PATH, LEGACY_CONFIG_PATH):
        if path.exists():
            business_profile_cache = UserConfig.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            return business_profile_cache

    raise HTTPException(status_code=404, detail="Business profile not found")


def _save_business_profile(profile: UserConfig) -> UserConfig:
    """Persist the business profile to both supported config files."""

    global business_profile_cache
    business_profile_cache = profile
    payload = profile.model_dump_json(indent=2)
    write_errors: list[str] = []

    try:
        BUSINESS_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        write_errors.append(f"{BUSINESS_PROFILE_PATH.parent}: {exc}")

    for path, writer in (
        (LEGACY_CONFIG_PATH, lambda p: p.write_text(json.dumps(profile.model_dump(mode="json"), indent=2), encoding="utf-8")),
        (BUSINESS_PROFILE_PATH, lambda p: p.write_text(payload, encoding="utf-8")),
    ):
        try:
            writer(path)
        except Exception as exc:
            write_errors.append(f"{path}: {exc}")

    if len(write_errors) == 2:
        raise HTTPException(
            status_code=500,
            detail="Could not save business profile to disk: " + "; ".join(write_errors),
        )

    return profile


def _get_copilot() -> AgenticCopilot:
    """Create the copilot lazily so config endpoints can run without the vector store."""

    global copilot
    if copilot is None:
        copilot = AgenticCopilot()
    return copilot


def _assessment_payload(news_id: str, assessment: RiskAssessment) -> dict:
    """Combine an assessment with its source article metadata."""

    article = _get_copilot().vector_store.get_news_article(news_id)
    payload = assessment.model_dump(mode="json")
    payload["news_id"] = news_id
    payload["news_title"] = article.title if article else None
    payload["news_url"] = article.url if article else None
    payload["news_source"] = article.source if article else None
    return payload


async def _background_scrape_loop():
    """Continuously scrape and store new articles on the configured interval."""
    logger.info(
        "Background scraper started with %s minute interval",
        settings.scrape_interval_minutes,
    )

    while True:
        try:
            settings.reload()
            logger.info("Background scrape cycle starting")
            await _get_copilot().run_cycle()
            logger.info("Background scrape cycle completed")
        except asyncio.CancelledError:
            logger.info("Background scraper task cancelled")
            raise
        except Exception as exc:
            logger.exception("Background scrape cycle failed: %s", exc)

        interval_seconds = max(settings.scrape_interval_minutes, 1) * 60
        await asyncio.sleep(interval_seconds)


@app.on_event("startup")
async def _startup_background_scraper():
    """Start the silent scraper loop when the website/API boots."""

    global background_task
    if not ENABLE_BACKGROUND_SCRAPER:
        logger.info("Background scraper disabled; Cloud Run-safe request-driven mode is active")
        return

    if background_task is None or background_task.done():
        background_task = asyncio.create_task(_background_scrape_loop())


@app.on_event("shutdown")
async def _shutdown_background_scraper():
    """Stop the silent scraper loop cleanly."""

    global background_task
    if background_task is not None:
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass
        background_task = None



@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/runtime-config")
@app.get("/runtime-config")
async def get_runtime_config():
    """Return runtime-configurable environment values."""

    return get_runtime_config_payload()


@app.put("/api/runtime-config")
@app.put("/runtime-config")
async def update_runtime_config(payload: RuntimeConfigUpdate):
    """Persist runtime env values from the manager page."""

    current = settings.as_settings().model_dump(mode="json")
    updates = payload.model_dump(mode="json", exclude_none=True)
    current.update(updates)
    save_runtime_overrides(current)
    settings.reload()

    if copilot is not None:
        copilot.llm._client = None

    return get_runtime_config_payload()


@app.post("/api/run-cycle")
async def run_agent_cycle():
    """Manually trigger one agent cycle."""
    try:
        assessments = await _get_copilot().run_cycle()
        return {
            "status": "success",
            "assessments_count": len(assessments),
            "assessments": [_assessment_payload(news_id, assessment) for news_id, assessment in assessments],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/assessments")
@app.get("/assessments")
async def get_assessments(limit: int = 10):
    """Get the most recent risk assessments."""
    assessments = await _get_copilot().get_latest_assessments(limit)
    return [
        _assessment_payload(news_id, assessment)
        for news_id, assessment in assessments
    ]


@app.get("/api/assessments/{news_id}")
@app.get("/assessments/{news_id}")
async def get_assessment(news_id: str):
    """Get risk assessment for a specific news article."""
    assessment = await _get_copilot().get_assessment_for_news(news_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return _assessment_payload(news_id, assessment)


@app.post("/api/assessments/{news_id}/approve")
@app.post("/assessments/{news_id}/approve")
async def approve_assessment(news_id: str, payload: ApprovalRequest):
    """Run Gemma again to decide whether a stakeholder should be messaged."""

    if not payload.stakeholders:
        raise HTTPException(
            status_code=400,
            detail="Stakeholder directory is empty. Add at least one stakeholder before approving.",
        )

    try:
        decision = await _get_copilot().decide_approval_action(
            news_id=news_id,
            stakeholders=payload.stakeholders,
        )
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        raise HTTPException(status_code=status_code, detail=message) from exc

    result = decision.model_dump(mode="json")
    result["news_id"] = news_id
    return result


@app.get("/api/news")
@app.get("/news")
async def get_news(limit: int = 20):
    """Get recent news articles."""
    return []


@app.get("/api/config")
async def get_config():
    """Return the current business profile."""
    return _load_business_profile().model_dump(mode="json")


@app.put("/api/config")
async def update_config(profile: UserConfig):
    """Update the business profile used by the agent."""
    updated = _save_business_profile(profile)

    # Refresh the in-memory profile so the next cycle uses the new rules immediately.
    if copilot is not None:
        copilot.business_profile = updated
    return updated.model_dump(mode="json")


# Serve frontend
@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML."""
    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "..", "frontend", "index.html"
    )
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"error": "Frontend not found. Please run from project root."}


@app.get("/manager")
async def serve_manager():
    """Serve the runtime config manager page."""

    if MANAGER_PAGE_PATH.exists():
        return FileResponse(MANAGER_PAGE_PATH)
    return {"error": "manager.html not found. Please run from project root."}
