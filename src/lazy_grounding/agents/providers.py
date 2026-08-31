"""Explicit provider adapters for the three evaluated base models."""

from __future__ import annotations

import os
import re
import secrets
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

from lazy_grounding.config import AgentConfig

Message = Mapping[str, str]
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_HTTP_ERROR = 400


@dataclass(frozen=True, slots=True)
class ModelReply:
    text: str
    usage: Mapping[str, Any] = field(default_factory=dict)


class ModelProvider(Protocol):
    model: str

    def generate(self, messages: Sequence[Message]) -> ModelReply: ...


def _redact(text: str) -> str:
    value = re.sub(r"sk-[A-Za-z0-9._*=-]+", "sk-<redacted>", text or "")
    value = re.sub(r"\bAIza[A-Za-z0-9_-]{20,}\b", "<key-redacted>", value)
    return re.sub(r"(bearer\s+)[^\s,}]+", r"\1<redacted>", value, flags=re.IGNORECASE)


def _responses_text(payload: Mapping[str, Any]) -> str:
    if payload.get("output_text"):
        return str(payload["output_text"])
    chunks: list[str] = []
    for item in payload.get("output") or ():
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        for content in item.get("content") or ():
            if not isinstance(content, Mapping):
                continue
            text = content.get("text") or content.get("refusal")
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


class _HTTPProvider:
    def __init__(
        self,
        config: AgentConfig,
        *,
        api_base: str,
        api_key: str,
        retries: int = 5,
        session: requests.Session | None = None,
    ):
        if not api_base.strip() or not api_key.strip():
            raise ValueError("api_base and api_key must be explicit")
        self.config = config
        self.model = config.model
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._retries = retries
        self._session = session or requests.Session()

    def _post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        last_error = ""
        for attempt in range(self._retries + 1):
            try:
                response = self._session.post(
                    f"{self._api_base}{path}",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=dict(payload),
                    timeout=(30, self.config.max_wall_time_seconds),
                )
                if response.status_code < _HTTP_ERROR:
                    value = response.json()
                    if not isinstance(value, Mapping):
                        raise RuntimeError("Provider returned a non-object JSON response")
                    return value
                last_error = f"HTTP {response.status_code}: {_redact(response.text)[:500]}"
                if response.status_code not in _RETRYABLE_STATUS:
                    break
                retry_after = response.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else min(60.0, 2.0**attempt)
            except requests.RequestException as error:
                last_error = _redact(str(error))
                delay = min(60.0, 2.0**attempt)
            if attempt < self._retries:
                time.sleep(delay + secrets.randbelow(251) / 1_000)
        raise RuntimeError(f"Provider request failed after retries: {last_error}")


class OpenAIResponsesProvider(_HTTPProvider):
    def generate(self, messages: Sequence[Message]) -> ModelReply:
        payload = self._post(
            "/responses",
            {
                "model": self.model,
                "input": list(messages),
                "max_output_tokens": self.config.max_output_tokens,
                "store": False,
            },
        )
        text = _responses_text(payload)
        if not text:
            raise RuntimeError("OpenAI Responses API returned an empty model response")
        raw_usage = payload.get("usage")
        usage = dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
        return ModelReply(text=text, usage=usage)


class OpenAIChatProvider(_HTTPProvider):
    def generate(self, messages: Sequence[Message]) -> ModelReply:
        payload = self._post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": list(messages),
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "presence_penalty": self.config.presence_penalty,
                "max_tokens": self.config.max_output_tokens,
                "stop": ["\n<tool_response>", "<tool_response>"],
            },
        )
        choices = payload.get("choices") or ()
        if not choices or not isinstance(choices[0], Mapping):
            raise RuntimeError("Chat Completions API returned no choices")
        message = choices[0].get("message")
        text = str(message.get("content") or "") if isinstance(message, Mapping) else ""
        if not text:
            raise RuntimeError("Chat Completions API returned an empty model response")
        raw_usage = payload.get("usage")
        usage = dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
        return ModelReply(text=text, usage=usage)


class GeminiProvider:
    def __init__(self, config: AgentConfig, *, api_key: str, retries: int = 5):
        if not api_key.strip():
            raise ValueError("GEMINI_API_KEY is required")
        try:
            from google import genai  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on optional installation
            raise RuntimeError("Install lazy-grounding[gemini] to use Gemini") from exc
        self.config = config
        self.model = config.model
        self._client = genai.Client(api_key=api_key)
        self._retries = retries

    @staticmethod
    def _status_code(error: Exception) -> int | None:
        code = getattr(error, "code", None)
        if isinstance(code, int):
            return code
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
        match = re.search(r"(?:^|\D)(4\d\d|5\d\d)(?:\D|$)", str(error))
        return int(match.group(1)) if match else None

    @classmethod
    def _retryable(cls, error: Exception) -> bool:
        code = cls._status_code(error)
        return code is None or code in _RETRYABLE_STATUS

    def generate(self, messages: Sequence[Message]) -> ModelReply:
        from google.genai import types  # noqa: PLC0415

        system = "\n\n".join(
            message["content"] for message in messages if message["role"] == "system"
        )
        contents = [
            types.Content(
                role="model" if message["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=message["content"])],
            )
            for message in messages
            if message["role"] != "system"
        ]
        response = None
        last_error = ""
        for attempt in range(self._retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system or None,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        max_output_tokens=self.config.max_output_tokens,
                        stop_sequences=["</tool_call>", "\n<tool_response>", "<tool_response>"],
                    ),
                )
                break
            except Exception as error:
                last_error = _redact(str(error))[:500]
                if attempt >= self._retries or not self._retryable(error):
                    raise RuntimeError(f"Gemini request failed: {last_error}") from error
                time.sleep(min(60.0, 2.0**attempt) + secrets.randbelow(251) / 1_000)
        if response is None:
            raise RuntimeError(f"Gemini request failed after retries: {last_error}")
        text = str(getattr(response, "text", "") or "").strip()
        if "<tool_call>" in text and "</tool_call>" not in text:
            text = f"{text}\n</tool_call>"
        if not text:
            raise RuntimeError("Gemini returned an empty model response")
        usage_value = getattr(response, "usage_metadata", None)
        usage = (
            usage_value.model_dump(exclude_none=True)
            if usage_value is not None and hasattr(usage_value, "model_dump")
            else {}
        )
        return ModelReply(text=text, usage=usage)


def provider_from_config(config: AgentConfig) -> ModelProvider:
    if config.provider == "openai":
        return OpenAIResponsesProvider(
            config,
            api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
    if config.provider == "vllm":
        return OpenAIChatProvider(
            config,
            api_base=os.environ.get("TONGYI_API_BASE", ""),
            api_key=os.environ.get("TONGYI_API_KEY", "EMPTY"),
        )
    if config.provider == "gemini":
        return GeminiProvider(config, api_key=os.environ.get("GEMINI_API_KEY", ""))
    raise ValueError(f"Unsupported provider: {config.provider}")
