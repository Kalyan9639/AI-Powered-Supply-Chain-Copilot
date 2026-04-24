"""Hugging Face Gemma client for article risk analysis."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Sequence

from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError, HfHubHTTPError, InferenceTimeoutError, OverloadedError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from config import settings
from models.schema import ApprovalDecision, NewsArticle, RiskAssessment, StakeholderEntry, UserConfig
from utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_PATHS = (
    PROJECT_ROOT / "config.json",
    PROJECT_ROOT / "data" / "config" / "business_profile.json",
)
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class RiskAssessmentDraft(BaseModel):
    """Structured output expected from Gemma."""

    model_config = ConfigDict(extra="forbid")

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1",
    )
    proposed_action: str = Field(
        description="Concrete next action for the business",
    )
    reasoning: str = Field(
        description="Why the event matters for this business",
    )
    risk_level: str = Field(
        pattern="^(High|Medium|Low)$",
        description="Risk level: High, Medium, or Low",
    )


class ApprovalDecisionDraft(BaseModel):
    """Structured approval-time output expected from Gemma."""

    model_config = ConfigDict(extra="forbid")

    should_message: bool
    stakeholder_index: int | None = Field(default=None, ge=0)
    message_to_send: str | None = None
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


def load_business_profile(paths: Sequence[Path] = DEFAULT_PROFILE_PATHS) -> UserConfig:
    """Load the business profile used for prompts."""

    for path in paths:
        if not path.exists():
            continue

        try:
            return UserConfig.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load business profile from %s: %s", path, exc)

    logger.warning("No business profile file found; using a fallback profile")
    return UserConfig(
        business_name="Unknown Business",
        commodity="unknown commodity",
        region="unknown region",
        rules=[],
    )


class GemmaRiskAnalyzer:
    """Calls Hugging Face hosted Gemma and parses risk assessments."""

    def __init__(self, max_retries: int = 3, timeout: float = 120.0):
        self.max_retries = max_retries
        self.timeout = timeout
        self._client: InferenceClient | None = None

    @property
    def client(self) -> InferenceClient:
        """Create the Hugging Face client lazily so the app can still boot."""

        if self._client is None:
            if not settings.hf_api_key:
                raise RuntimeError(
                    "HF_API_KEY is missing. Add it to .env before running the agent."
                )

            self._client = InferenceClient(
                model=settings.hf_model,
                token=settings.hf_api_key,
                provider="auto",
                timeout=self.timeout,
            )

        return self._client

    async def analyze_article(
        self,
        article: NewsArticle,
        business_profile: UserConfig,
        similar_articles: list[NewsArticle] | None = None,
    ) -> RiskAssessment:
        """Run Gemma on the article and return a validated assessment."""

        messages = self._build_messages(article, business_profile, similar_articles or [])
        content = await self._call_model(
            messages,
            use_json_schema=True,
            schema_model=RiskAssessmentDraft,
        )
        draft = self._parse_draft(content)
        return self._to_risk_assessment(draft)

    async def decide_approval_action(
        self,
        article: NewsArticle,
        assessment: RiskAssessment,
        business_profile: UserConfig,
        stakeholders: list[StakeholderEntry],
    ) -> ApprovalDecision:
        """Decide whether the business should message a stakeholder after approval."""

        if not stakeholders:
            return ApprovalDecision(
                should_message=False,
                selected_stakeholder_index=None,
                selected_stakeholder=None,
                message_to_send=None,
                reasoning="No stakeholders were provided, so no message can be sent.",
                confidence=0.0,
            )

        messages = self._build_approval_messages(
            article=article,
            assessment=assessment,
            business_profile=business_profile,
            stakeholders=stakeholders,
        )
        content = await self._call_model(
            messages,
            use_json_schema=True,
            schema_model=ApprovalDecisionDraft,
        )
        draft = self._parse_approval_draft(content)
        return self._to_approval_decision(draft, stakeholders)

    def _build_messages(
        self,
        article: NewsArticle,
        business_profile: UserConfig,
        similar_articles: list[NewsArticle],
    ) -> list[dict[str, str]]:
        """Build a concise, structured prompt for Gemma."""

        profile_rules = "\n".join(f"- {rule}" for rule in business_profile.rules) or "- No explicit rules provided"
        similar_block = self._format_similar_articles(similar_articles)

        system_prompt = (
            "You are a supply chain risk analyst for an Indian agri-retail business. "
            "Use the business profile, the article, and past incidents to judge the impact on the business. "
            "Always answer with a concrete risk assessment."
        )
        user_prompt = f"""
Business profile:
- Business name: {business_profile.business_name}
- Commodity: {business_profile.commodity}
- Region: {business_profile.region}
- Rules:
{profile_rules}

News article:
- Title: {article.title}
- Source: {article.source}
- URL: {article.url}
- Published at: {article.published_at.isoformat() if article.published_at else "Unknown"}
- Content:
{article.content[:3000]}

Similar historical incidents:
{similar_block}

<output-format>
Return exactly one JSON object and nothing else.
Use these keys in this exact order:
1. confidence
2. proposed_action
3. reasoning
4. risk_level
Rules:
- All fields are required.
- confidence must be a number between 0 and 1.
- proposed_action must be one short sentence and no more than 18 words.
- reasoning must be one short sentence and no more than 25 words.
- risk_level must be exactly one of High, Medium, or Low.
- Do not add markdown, code fences, or extra keys.
</output-format>
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_similar_articles(self, similar_articles: list[NewsArticle]) -> str:
        if not similar_articles:
            return "- No similar incidents found in memory"

        lines = []
        for article in similar_articles[:3]:
            snippet = " ".join(article.content.split())[:250]
            lines.append(
                f"- {article.title} | {article.source} | {snippet or 'No summary available'}"
            )
        return "\n".join(lines)

    def _build_approval_messages(
        self,
        article: NewsArticle,
        assessment: RiskAssessment,
        business_profile: UserConfig,
        stakeholders: list[StakeholderEntry],
    ) -> list[dict[str, str]]:
        """Build a structured prompt for approval-time stakeholder selection."""

        profile_rules = "\n".join(f"- {rule}" for rule in business_profile.rules) or "- No explicit rules provided"
        stakeholder_lines = "\n".join(
            f"{idx}. {stakeholder.name} | WhatsApp: {stakeholder.whatsapp_number} | Location: {stakeholder.shop_location} | Supplies/Service: {stakeholder.materials_or_service}"
            for idx, stakeholder in enumerate(stakeholders)
        )

        system_prompt = (
            "You are an operations assistant for an Indian agri-retail business. "
            "Your job is to decide whether an approved risk assessment should trigger a WhatsApp message to one stakeholder."
        )
        user_prompt = f"""
Business profile:
- Business name: {business_profile.business_name}
- Commodity: {business_profile.commodity}
- Region: {business_profile.region}
- Rules:
{profile_rules}

Risk assessment being approved:
- Article title: {article.title}
- Article source: {article.source}
- Article URL: {article.url}
- Risk level: {assessment.risk_level}
- Assessment reasoning: {assessment.reasoning}
- Proposed action: {assessment.proposed_action or "No proposed action provided"}

Stakeholder directory:
{stakeholder_lines}

Decision rules:
- If a message is not needed, set should_message to false and stakeholder_index to null.
- If a stakeholder should be messaged, choose the exact 0-based stakeholder_index from the directory.
- Use the stakeholder's materials/service and location to pick the best match.
- Write a short WhatsApp message only when should_message is true.

Return a single JSON object with these fields:
"should_message": bool,
"stakeholder_index": int,
"message_to_send": "Short WhatsApp text",
"confidence": float
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def _call_model(
        self,
        messages: list[dict[str, str]],
        use_json_schema: bool,
        schema_model: type[BaseModel] | None = None,
    ) -> str:
        """Call Gemma with retry handling and optional structured response format."""

        response_format = None
        if use_json_schema:
            model_schema = schema_model or RiskAssessmentDraft
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": model_schema.__name__.lower(),
                    "description": f"Structured {model_schema.__name__} output",
                    "schema": model_schema.model_json_schema(),
                    "strict": True,
                },
            }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.chat_completion,
                    messages=messages,
                    model=settings.hf_model,
                    max_tokens=350,
                    frequency_penalty=0.2,
                    temperature=0.2,
                    top_p=0.9,
                    response_format=response_format,
                )
                content = response.choices[0].message.content or ""
                content = content.strip()
                if not content:
                    raise RuntimeError("Gemma returned an empty response")
                return content
            except BadRequestError as exc:
                if use_json_schema and self._should_fallback_to_plain_json(exc):
                    logger.warning(
                        "Gemma provider rejected structured JSON schema output; retrying with plain JSON prompting."
                    )
                    return await self._call_model(messages, use_json_schema=False)
                raise RuntimeError(f"Hugging Face rejected the request: {exc}") from exc
            except (OverloadedError, InferenceTimeoutError) as exc:
                last_error = exc
            except HfHubHTTPError as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code in RETRYABLE_STATUS_CODES:
                    last_error = exc
                else:
                    raise RuntimeError(f"Hugging Face inference request failed: {exc}") from exc
            except Exception as exc:
                last_error = exc

            if attempt < self.max_retries - 1:
                delay = 2 ** attempt
                logger.warning(
                    "Gemma request failed on attempt %s/%s. Retrying in %ss.",
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

        raise RuntimeError(f"Gemma inference failed after retries: {last_error}")

    def _should_fallback_to_plain_json(self, exc: BadRequestError) -> bool:
        message = str(exc).lower()
        return "response_format" in message or "json_schema" in message or "json object" in message

    def _parse_draft(self, content: str) -> RiskAssessmentDraft:
        """Parse the model response into a validated draft object."""

        cleaned = self._strip_code_fences(content)
        payload = self._load_json_payload(cleaned)

        if not payload.get("proposed_action"):
            payload["proposed_action"] = "Monitor the situation and update the business log."

        if isinstance(payload.get("risk_level"), str):
            normalized = payload["risk_level"].strip().lower()
            if normalized in {"high", "medium", "low"}:
                payload["risk_level"] = normalized.title()

        draft = RiskAssessmentDraft.model_validate(payload)
        return draft

    def _parse_approval_draft(self, content: str) -> ApprovalDecisionDraft:
        """Parse the model response into a validated approval decision draft."""

        cleaned = self._strip_code_fences(content)
        payload = self._load_json_payload(cleaned)

        draft = ApprovalDecisionDraft.model_validate(payload)
        return draft

    def _strip_code_fences(self, content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    def _load_json_payload(self, content: str) -> dict[str, Any]:
        candidate_payloads: list[str] = []
        stripped = content.strip()
        if stripped:
            candidate_payloads.append(stripped)

        if "{" in stripped:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate_payloads.append(stripped[start : end + 1])
            candidate_payloads.append(self._balance_json_braces(stripped[start:]))

        for candidate in candidate_payloads:
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                continue

        raise RuntimeError(f"Gemma response did not contain valid JSON: {content}")

    def _balance_json_braces(self, content: str) -> str:
        """Append missing closing braces when Gemma truncates a JSON object."""

        depth = 0
        in_string = False
        escape_next = False

        for char in content:
            if in_string:
                if escape_next:
                    escape_next = False
                elif char == "\\":
                    escape_next = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth = max(0, depth - 1)

        if depth > 0:
            content += "}" * depth

        return content

    def _to_risk_assessment(self, draft: RiskAssessmentDraft) -> RiskAssessment:
        proposed_action = (
            draft.proposed_action.strip()
            if draft.proposed_action and draft.proposed_action.strip()
            else "Monitor the situation and update the business log."
        )

        return RiskAssessment(
            risk_level=draft.risk_level,
            reasoning=draft.reasoning.strip(),
            proposed_action=proposed_action,
            confidence=max(0.0, min(1.0, draft.confidence)),
        )

    def _to_approval_decision(
        self,
        draft: ApprovalDecisionDraft,
        stakeholders: list[StakeholderEntry],
    ) -> ApprovalDecision:
        selected_stakeholder = None
        selected_index = draft.stakeholder_index

        if draft.should_message and selected_index is not None:
            if 0 <= selected_index < len(stakeholders):
                selected_stakeholder = stakeholders[selected_index]
            else:
                logger.warning(
                    "Gemma selected invalid stakeholder index %s for %s stakeholders",
                    selected_index,
                    len(stakeholders),
                )

        should_message = draft.should_message and selected_stakeholder is not None

        message_to_send = (
            draft.message_to_send.strip()
            if should_message and draft.message_to_send and draft.message_to_send.strip()
            else None
        )

        return ApprovalDecision(
            should_message=should_message,
            selected_stakeholder_index=selected_index if should_message else None,
            selected_stakeholder=selected_stakeholder if should_message else None,
            message_to_send=message_to_send,
            reasoning=draft.reasoning.strip(),
            confidence=max(0.0, min(1.0, draft.confidence)),
        )
