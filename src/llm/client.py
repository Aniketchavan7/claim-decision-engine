"""Provider-agnostic LLM client using OpenAI-compatible SDK."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional, Type

from openai import OpenAI
from pydantic import BaseModel

from src.config import PROVIDER_URLS, get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper over the OpenAI-compatible SDK.

    Switching providers = changing LLM_PROVIDER + LLM_API_KEY env vars.
    No agent code changes needed.
    """

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        settings = get_settings()
        self.provider = provider or settings.llm_provider
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_reasoning_model
        self.temperature = settings.llm_temperature

        base_url = PROVIDER_URLS.get(self.provider)
        if not base_url:
            raise ValueError(
                f"Unknown LLM provider '{self.provider}'. "
                f"Known: {list(PROVIDER_URLS.keys())}"
            )

        self.client = OpenAI(base_url=base_url, api_key=self.api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a plain-text response."""
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            logger.info(
                "LLM call [%s] %.1fs, %d tokens",
                model or self.model,
                time.time() - t0,
                resp.usage.total_tokens if resp.usage else 0,
            )
            return content
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            # Try fallback model
            settings = get_settings()
            if (model or self.model) != settings.llm_fallback_model:
                logger.info("Retrying with fallback model: %s", settings.llm_fallback_model)
                return self.generate(
                    messages,
                    model=settings.llm_fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            raise

    def generate_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response.

        Instructs the model to respond in JSON and strips markdown fences
        if the model wraps its answer.
        """
        # Add JSON instruction to system message
        json_messages = list(messages)
        if json_messages and json_messages[0]["role"] == "system":
            json_messages[0] = {
                "role": "system",
                "content": json_messages[0]["content"]
                + "\n\nYou MUST respond with valid JSON only. No markdown fences, no explanation outside the JSON.",
            }

        # Prefer JSON mode if supported
        kwargs: dict[str, Any] = {}
        # Hint to OpenAI-compatible provider
        kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self.client.chat.completions.create(
                model=model or self.model,
                messages=json_messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            raw = resp.choices[0].message.content or ""
        except Exception:
            # Fall back to standard generate without response_format if provider complains
            raw = self.generate(
                json_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # 1. Direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Extract substring between first { and last }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        # 3. Robust partial JSON repair for truncated tokens
        try:
            repaired = self._repair_truncated_json(text)
            if repaired:
                return json.loads(repaired)
        except Exception:
            pass

        # 4. If all fails, retry once with fast fallback model
        logger.warning("JSON parsing failed on primary response; re-requesting with fallback model...")
        settings = get_settings()
        fallback_raw = self.generate(
            json_messages,
            model=settings.llm_fallback_model,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        clean_fallback = fallback_raw.strip()
        if "```" in clean_fallback:
            clean_fallback = clean_fallback.split("```")[1]
            if clean_fallback.startswith("json"):
                clean_fallback = clean_fallback[4:]
        try:
            return json.loads(clean_fallback.strip())
        except Exception:
            s = clean_fallback.find("{")
            e = clean_fallback.rfind("}") + 1
            if s >= 0 and e > s:
                return json.loads(clean_fallback[s:e])

        raise ValueError(f"LLM did not return valid JSON: {raw[:250]}")

    def _repair_truncated_json(self, text: str) -> str | None:
        """Attempt heuristic repair on truncated JSON streams (e.g., auto-closing open quotes/brackets)."""
        s = text.strip()
        first_brace = s.find("{")
        if first_brace == -1:
            return None
        s = s[first_brace:]

        # Balance open quotes
        in_string = False
        escape = False
        for char in s:
            if char == "\\" and not escape:
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
            escape = False

        if in_string:
            s += '"'

        # Count open brackets and braces
        open_curly = 0
        open_square = 0
        for char in s:
            if char == "{":
                open_curly += 1
            elif char == "}":
                open_curly -= 1
            elif char == "[":
                open_square += 1
            elif char == "]":
                open_square -= 1

        # Append closing brackets
        while open_square > 0:
            s += "]"
            open_square -= 1
        while open_curly > 0:
            s += "}"
            open_curly -= 1

        return s


def get_reasoning_client() -> LLMClient:
    """Return an LLM client configured for reasoning tasks."""
    settings = get_settings()
    return LLMClient(model=settings.llm_reasoning_model)


def get_fast_client() -> LLMClient:
    """Return an LLM client configured for fast/simple tasks."""
    settings = get_settings()
    return LLMClient(model=settings.llm_fast_model)
