"""Project configuration using Pydantic settings plus runtime overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
APP_STORAGE_DIR = Path(os.getenv("APP_STORAGE_DIR", str(BASE_DIR / "data"))).expanduser()
RUNTIME_ENV_PATH = APP_STORAGE_DIR / "config" / "runtime.env"


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "debug"}:
            return True
        if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
            return False
        return False

    return bool(value)


def _parse_runtime_env_file(path: Path) -> dict[str, str]:
    """Parse a tiny .env-style file into uppercase string values."""

    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        cleaned = value.strip()
        if cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')) and len(cleaned) >= 2:
            cleaned = cleaned[1:-1]
        values[key.strip().upper()] = cleaned

    return values


def _serialize_runtime_env(values: dict[str, Any]) -> str:
    """Serialize runtime settings to a .env-style file."""

    lines: list[str] = [
        "# Runtime environment values managed from manager.html",
        "# Deployed instances should use a persistent volume or secret store for durability.",
    ]

    for key, value in values.items():
        if value is None:
            continue

        if isinstance(value, list):
            encoded = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            encoded = "true" if value else "false"
        else:
            encoded = str(value)

        lines.append(f"{key}={encoded}")

    lines.append("")
    return "\n".join(lines)


class Settings(BaseSettings):
    """Application settings loaded from environment and .env file."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Hugging Face settings
    hf_model: str = "google/gemma-4-26B-A4B-it"
    hf_api_key: str | None = None

    # Scraping settings
    scrape_interval_minutes: int = 1
    target_urls: list[str] = [
        "https://itln.in/supply-chain",
        "https://itln.in/road-transportation",
    ]

    # ChromaDB settings
    chromadb_path: Path = APP_STORAGE_DIR / "cache" / "chromadb"

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"

    # App settings
    debug: bool = False
    log_level: str = "INFO"

    @field_validator("debug", mode="before")
    @classmethod
    def _coerce_debug_flag(cls, value: object) -> bool:
        """Treat non-boolean debug values like 'release' as False."""

        return _coerce_bool(value)


def _build_settings() -> Settings:
    """Build settings from the standard env plus runtime overrides."""

    base_settings = Settings()
    runtime_values = _parse_runtime_env_file(RUNTIME_ENV_PATH)

    if runtime_values:
        field_map = {
            "HF_API_KEY": "hf_api_key",
            "HF_MODEL": "hf_model",
            "SCRAPE_INTERVAL_MINUTES": "scrape_interval_minutes",
            "TARGET_URLS": "target_urls",
            "LOG_LEVEL": "log_level",
            "DEBUG": "debug",
        }

        for env_key, field_name in field_map.items():
            if env_key not in runtime_values:
                continue

            raw_value = runtime_values[env_key]
            if field_name == "scrape_interval_minutes":
                try:
                    setattr(base_settings, field_name, int(raw_value))
                except ValueError:
                    continue
            elif field_name == "debug":
                setattr(base_settings, field_name, _coerce_bool(raw_value))
            elif field_name == "target_urls":
                try:
                    parsed_urls = json.loads(raw_value)
                    if isinstance(parsed_urls, list):
                        setattr(base_settings, field_name, [str(item) for item in parsed_urls if str(item).strip()])
                    else:
                        continue
                except json.JSONDecodeError:
                    lines = [line.strip() for line in raw_value.splitlines()]
                    if len(lines) == 1 and "," in lines[0]:
                        lines = [item.strip() for item in lines[0].split(",")]
                    setattr(base_settings, field_name, [line for line in lines if line])
            else:
                setattr(base_settings, field_name, raw_value)

    os.environ.update(
        {
            "HF_API_KEY": base_settings.hf_api_key or "",
            "HF_MODEL": base_settings.hf_model,
            "SCRAPE_INTERVAL_MINUTES": str(base_settings.scrape_interval_minutes),
            "LOG_LEVEL": base_settings.log_level,
            "DEBUG": "true" if base_settings.debug else "false",
            "TARGET_URLS": json.dumps(base_settings.target_urls, ensure_ascii=False),
        }
    )

    return base_settings


class SettingsProxy:
    """Mutable proxy so imported references see live settings reloads."""

    def __init__(self):
        object.__setattr__(self, "_settings", _build_settings())

    def reload(self) -> Settings:
        """Reload runtime overrides from disk and update the shared object."""

        object.__setattr__(self, "_settings", _build_settings())
        return object.__getattribute__(self, "_settings")

    def update_from_mapping(self, values: dict[str, Any]) -> Settings:
        """Persist runtime overrides and refresh the in-memory settings."""

        save_runtime_overrides(values)
        return self.reload()

    def __getattr__(self, item: str) -> Any:
        return getattr(object.__getattribute__(self, "_settings"), item)

    def __setattr__(self, key: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_settings"), key, value)

    def model_dump(self, *args, **kwargs):
        return object.__getattribute__(self, "_settings").model_dump(*args, **kwargs)

    def as_settings(self) -> Settings:
        return object.__getattribute__(self, "_settings")


def get_runtime_config_payload() -> dict[str, Any]:
    """Return the current settings in a form suitable for the manager UI."""

    current = settings.as_settings()
    runtime_values = _parse_runtime_env_file(RUNTIME_ENV_PATH)
    return {
        "hf_api_key": current.hf_api_key or "",
        "hf_model": current.hf_model,
        "scrape_interval_minutes": current.scrape_interval_minutes,
        "target_urls": current.target_urls,
        "log_level": current.log_level,
        "debug": current.debug,
        "runtime_env_path": str(RUNTIME_ENV_PATH),
        "source": "runtime.env" if runtime_values else ".env",
    }


def save_runtime_overrides(values: dict[str, Any]) -> None:
    """Persist the runtime settings to disk in .env format."""

    RUNTIME_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)

    current = settings.as_settings()
    target_urls = values.get("target_urls", current.target_urls)
    if target_urls is None:
        target_urls = current.target_urls

    payload = {
        "HF_API_KEY": values.get("hf_api_key", current.hf_api_key or ""),
        "HF_MODEL": values.get("hf_model", current.hf_model),
        "SCRAPE_INTERVAL_MINUTES": int(values.get("scrape_interval_minutes", current.scrape_interval_minutes)),
        "TARGET_URLS": target_urls,
        "LOG_LEVEL": values.get("log_level", current.log_level),
        "DEBUG": bool(values.get("debug", current.debug)),
    }

    RUNTIME_ENV_PATH.write_text(_serialize_runtime_env(payload), encoding="utf-8")


settings = SettingsProxy()
