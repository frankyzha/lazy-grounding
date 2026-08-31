from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from lazy_grounding.agents.providers import (
    GeminiProvider,
    OpenAIChatProvider,
    OpenAIResponsesProvider,
    provider_from_config,
)
from lazy_grounding.config import AgentConfig


class Response:
    def __init__(
        self,
        status: int,
        payload: Any,
        *,
        text: str = "",
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self) -> Any:
        return self._payload


class Session:
    def __init__(self, responses: list[Response]):
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> Response:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def config() -> AgentConfig:
    return AgentConfig(provider="openai", model="model")


def test_responses_and_chat_adapters_parse_text_and_usage() -> None:
    responses_session = Session([Response(200, {"output_text": "hello", "usage": {"total": 3}})])
    responses = OpenAIResponsesProvider(
        config(), api_base="https://api.test/v1", api_key="secret", session=responses_session
    )
    assert responses.generate([{"role": "user", "content": "hi"}]).text == "hello"
    assert responses_session.calls[0]["url"].endswith("/responses")

    chat_session = Session(
        [Response(200, {"choices": [{"message": {"content": "world"}}], "usage": {}})]
    )
    chat = OpenAIChatProvider(
        config(), api_base="https://api.test/v1", api_key="secret", session=chat_session
    )
    assert chat.generate([{"role": "user", "content": "hi"}]).text == "world"


def test_provider_retries_retryable_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lazy_grounding.agents.providers.time.sleep", lambda _: None)
    session = Session(
        [
            Response(429, {}, text="rate limit", headers={"Retry-After": "0"}),
            Response(200, {"output_text": "ok"}),
        ]
    )
    provider = OpenAIResponsesProvider(
        config(), api_base="https://api.test/v1", api_key="secret", retries=1, session=session
    )
    assert provider.generate([]).text == "ok"
    assert len(session.calls) == 2


def test_provider_rejects_bad_configuration_and_empty_output() -> None:
    with pytest.raises(ValueError, match="explicit"):
        OpenAIResponsesProvider(config(), api_base="", api_key="")
    provider = OpenAIResponsesProvider(
        config(),
        api_base="https://api.test/v1",
        api_key="secret",
        session=Session([Response(200, {})]),
    )
    with pytest.raises(RuntimeError, match="empty"):
        provider.generate([])


def test_nested_responses_text_and_nonretryable_error() -> None:
    nested = Session(
        [
            Response(
                200,
                {
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": "nested"}]}
                    ]
                },
            )
        ]
    )
    provider = OpenAIResponsesProvider(
        config(), api_base="https://api.test/v1", api_key="secret", session=nested
    )
    assert provider.generate([]).text == "nested"

    failure = OpenAIResponsesProvider(
        config(),
        api_base="https://api.test/v1",
        api_key="secret",
        session=Session([Response(401, {}, text="Bearer secret")]),
    )
    with pytest.raises(RuntimeError, match="failed after retries"):
        failure.generate([])


def install_fake_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    class Part:
        @staticmethod
        def from_text(*, text: str) -> str:
            return text

    class Content:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs

    class GenerateContentConfig:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs

    class Usage:
        def model_dump(self, **kwargs: Any) -> dict[str, int]:
            return {"total_token_count": 4}

    class Models:
        def generate_content(self, **kwargs: Any) -> Any:
            return types.SimpleNamespace(
                text='<tool_call>{"name":"search","arguments":{}}',
                usage_metadata=Usage(),
            )

    class Client:
        def __init__(self, **kwargs: Any):
            self.models = Models()

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai.Client = Client  # type: ignore[attr-defined]
    genai.types = types.SimpleNamespace(  # type: ignore[attr-defined]
        Part=Part, Content=Content, GenerateContentConfig=GenerateContentConfig
    )
    google.genai = genai  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)


def test_gemini_adapter_and_provider_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_gemini(monkeypatch)
    gemini_config = AgentConfig(provider="gemini", model="gemini")
    provider = GeminiProvider(gemini_config, api_key="key")
    reply = provider.generate(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ]
    )
    assert reply.text.endswith("</tool_call>")
    assert reply.usage["total_token_count"] == 4

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    assert provider_from_config(config()).model == "model"
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    assert provider_from_config(gemini_config).model == "gemini"


def test_gemini_error_classification() -> None:
    class CodedError(Exception):
        code = 429

    class ResponseError(Exception):
        response = types.SimpleNamespace(status_code=401)

    assert GeminiProvider._status_code(CodedError()) == 429
    assert GeminiProvider._status_code(ResponseError()) == 401
    assert GeminiProvider._status_code(RuntimeError("service returned 503")) == 503
    assert GeminiProvider._status_code(RuntimeError("network failure")) is None
    assert GeminiProvider._retryable(CodedError())
    assert not GeminiProvider._retryable(ResponseError())
