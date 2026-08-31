"""A small text ReAct loop matching the scaffold used in the paper."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import json5

from lazy_grounding.agents.providers import Message, ModelProvider
from lazy_grounding.config import AgentConfig


class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, Any]: ...

    def call(self, arguments: Mapping[str, Any]) -> str: ...


@dataclass(frozen=True, slots=True)
class ReactResult:
    answer: str
    termination: str
    messages: tuple[Mapping[str, str], ...]
    model_usage: tuple[Mapping[str, Any], ...]
    elapsed_seconds: float
    llm_calls: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["messages"] = list(self.messages)
        value["model_usage"] = list(self.model_usage)
        return value


_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_FINAL_LINE = re.compile(
    r"^\*{0,2}\s*(?:final\s+answer|answer)\s*\*{0,2}\s*[:-]\s*(.+?)\s*\*{0,2}$",
    re.IGNORECASE,
)


def _tool_schema(tool: Tool) -> str:
    return json.dumps(
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            },
        },
        ensure_ascii=False,
    )


def build_system_prompt(tools: Sequence[Tool]) -> str:
    schemas = "\n".join(_tool_schema(tool) for tool in tools)
    return f"""You are a deep research assistant. Your core function is to conduct thorough,
multi-source investigations into any topic. You must handle both broad, open-domain inquiries
and queries within specialized academic fields. For every request, synthesize information from
credible, diverse sources to deliver a comprehensive, accurate, and objective response. When
you have gathered sufficient information and are ready to provide the definitive response, you
must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{schemas}
</tools>

For each function call, return a JSON object with function name and arguments within
<tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <arguments-object>}}
</tool_call>

Current date: {datetime.now(timezone.utc).date().isoformat()}"""


def _explicit_answer(content: str) -> str:
    stripped = _THINK.sub("", content or "").strip()
    if _TOOL_CALL.search(stripped):
        return ""
    matches: list[str] = _ANSWER.findall(stripped)
    if matches:
        return matches[-1].strip()
    for line in reversed([line.strip() for line in stripped.splitlines() if line.strip()]):
        match = _FINAL_LINE.match(line)
        if match:
            return match.group(1).strip()
    return ""


def _parse_tool_call(content: str) -> tuple[str, Mapping[str, Any]] | None:
    match = _TOOL_CALL.search(content)
    if not match:
        return None
    value = json5.loads(match.group(1).strip())
    if not isinstance(value, Mapping):
        raise ValueError("Tool call must be an object")
    name = str(value.get("name") or "")
    arguments = value.get("arguments") or {}
    if not name or not isinstance(arguments, Mapping):
        raise ValueError("Tool call requires a name and an arguments object")
    return name, arguments


def _approximate_tokens(messages: Sequence[Message]) -> int:
    text = "\n".join(message.get("content", "") for message in messages)
    cjk = sum("\u4e00" <= character <= "\u9fff" for character in text)
    non_cjk = sum(
        not character.isspace() and not "\u4e00" <= character <= "\u9fff" for character in text
    )
    return cjk + max(1, non_cjk // 4)


class ReactAgent:
    def __init__(
        self,
        provider: ModelProvider,
        tools: Sequence[Tool],
        config: AgentConfig,
        *,
        max_context_tokens: int = 110 * 1_024,
        initial_instruction: str = "",
    ):
        if max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        self._provider = provider
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("Tool names must be unique")
        self._config = config
        self._max_context_tokens = max_context_tokens
        self._initial_instruction = initial_instruction.strip()

    def _finalize(
        self,
        messages: list[dict[str, str]],
        usage: list[Mapping[str, Any]],
    ) -> str:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Stop using tools now. Based only on the evidence already in the "
                    "conversation, give the most likely final answer to the user's question. "
                    "Do not explain and do not call tools. Use exactly this format: "
                    "<answer>your answer</answer>. If uncertain, still provide the "
                    "best-supported answer."
                ),
            }
        )
        reply = self._provider.generate(messages)
        usage.append(reply.usage)
        messages.append({"role": "assistant", "content": reply.text})
        return _explicit_answer(reply.text) or reply.text.strip()

    def _execute_tool(self, content: str) -> str:
        try:
            call = _parse_tool_call(content)
        except (ValueError, json.JSONDecodeError) as error:
            return f"Error: invalid tool call: {error}"
        if call is None:
            return "Error: response contained neither a final answer nor a tool call."
        name, arguments = call
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'."
        try:
            return tool.call(arguments)
        except Exception as error:  # noqa: BLE001
            return f"Error: tool '{name}' failed: {type(error).__name__}: {error}"

    def run(self, question: str) -> ReactResult:
        if not question.strip():
            raise ValueError("question cannot be empty")
        started = time.monotonic()
        user_content = question
        if self._initial_instruction:
            user_content = f"{self._initial_instruction}\n\nOriginal question:\n{question}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": build_system_prompt(tuple(self._tools.values()))},
            {"role": "user", "content": user_content},
        ]
        usage: list[Mapping[str, Any]] = []
        termination = "call_budget"
        answer = ""
        calls = 0

        while calls < self._config.max_llm_calls:
            elapsed = time.monotonic() - started
            if elapsed >= self._config.max_wall_time_seconds:
                termination = "wall_time"
                answer = self._finalize(messages, usage)
                calls += 1
                break
            if _approximate_tokens(messages) >= self._max_context_tokens:
                termination = "context_budget"
                answer = self._finalize(messages, usage)
                calls += 1
                break

            reply = self._provider.generate(messages)
            calls += 1
            usage.append(reply.usage)
            content = reply.text.split("<tool_response>", 1)[0].strip()
            messages.append({"role": "assistant", "content": content})

            answer = _explicit_answer(content)
            if answer:
                termination = "answer"
                break
            result = self._execute_tool(content)
            messages.append(
                {"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"}
            )

        if not answer:
            answer = self._finalize(messages, usage)
            calls += 1
        return ReactResult(
            answer=answer,
            termination=termination,
            messages=tuple(messages),
            model_usage=tuple(usage),
            elapsed_seconds=time.monotonic() - started,
            llm_calls=calls,
        )
