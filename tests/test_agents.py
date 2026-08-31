from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from lazy_grounding.agents.providers import ModelReply
from lazy_grounding.agents.react import ReactAgent, build_system_prompt
from lazy_grounding.config import AgentConfig


class Provider:
    model = "fake"

    def __init__(self, replies: Sequence[str]):
        self.replies = list(replies)

    def generate(self, messages: Sequence[Mapping[str, str]]) -> ModelReply:
        return ModelReply(self.replies.pop(0), {"calls": 1})


class Tool:
    name = "search"
    description = "Search"
    parameters: Mapping[str, Any] = {"type": "object"}

    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    def call(self, arguments: Mapping[str, Any]) -> str:
        self.calls.append(arguments)
        return "The source says Alice."


def config(*, calls: int = 5) -> AgentConfig:
    return AgentConfig(provider="openai", model="fake", max_llm_calls=calls)


def test_agent_executes_tool_then_extracts_answer() -> None:
    provider = Provider(
        [
            '<tool_call>{"name":"search","arguments":{"query":["winner"]}}</tool_call>',
            "<think>Evidence supports Alice.</think><answer>Alice</answer>",
        ]
    )
    tool = Tool()
    result = ReactAgent(provider, (tool,), config(), initial_instruction="Track constraints.").run(
        "Who won?"
    )
    assert result.answer == "Alice"
    assert result.termination == "answer"
    assert result.llm_calls == 2
    assert tool.calls == [{"query": ["winner"]}]
    assert "Original question" in result.messages[1]["content"]
    assert "<tool_response>" in result.messages[3]["content"]


def test_agent_finalizes_after_call_budget() -> None:
    provider = Provider(["No tool call yet.", "<answer>Best guess</answer>"])
    result = ReactAgent(provider, (Tool(),), config(calls=1)).run("Question?")
    assert result.answer == "Best guess"
    assert result.termination == "call_budget"
    assert result.llm_calls == 2


def test_invalid_tools_and_empty_question_fail() -> None:
    tool = Tool()
    with pytest.raises(ValueError, match="unique"):
        ReactAgent(Provider([]), (tool, tool), config())
    agent = ReactAgent(Provider([]), (tool,), config())
    with pytest.raises(ValueError, match="empty"):
        agent.run(" ")
    assert "Current date" in build_system_prompt((tool,))
