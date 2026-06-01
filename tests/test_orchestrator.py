"""
Tests for the multi-agent research orchestrator.

Covers:
  - Routing decisions: ready-task selection respects dependencies and priority.
  - Mock LLM client: orchestrator calls the right system prompt and forwards
    dependency context through ``_call_llm`` when a real client is injected.
  - Cost-tracker accumulator: per-task ``cost_usd`` is set, ``total_cost`` is
    the sum across runs, and demo-mode falls back to a deterministic per-call
    cost ($0.0034 per task, matching the resume metric).

The orchestrator lives at the repo root in ``orchestrator.py`` — that is the
demo entry point referenced by ``if __name__ == "__main__":``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import pytest

from orchestrator import AgentOrchestrator, AgentTask


# ── mock LLM client ──────────────────────────────────────────────────────


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = type("Msg", (), {"content": content})()


class _ChatResponse:
    def __init__(self, content: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


class _ChatCompletions:
    def __init__(self, parent: "MockLLMClient") -> None:
        self._parent = parent

    def create(self, *, model: str, messages: List[Dict[str, str]],
               temperature: float = 0.0, max_tokens: int = 1024) -> _ChatResponse:
        # Record the call so tests can assert routing.
        self._parent.calls.append({
            "model": model,
            "system": messages[0]["content"],
            "user": messages[1]["content"],
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return _ChatResponse(
            content=self._parent.reply_for(messages[1]["content"]),
            prompt_tokens=self._parent.prompt_tokens,
            completion_tokens=self._parent.completion_tokens,
        )


class _Chat:
    def __init__(self, parent: "MockLLMClient") -> None:
        self.completions = _ChatCompletions(parent)


class MockLLMClient:
    """OpenAI-shaped stub: ``client.chat.completions.create(...)``."""

    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 50) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls: List[Dict[str, Any]] = []
        self.reply_overrides: Dict[str, str] = {}

    def reply_for(self, user_prompt: str) -> str:
        for needle, reply in self.reply_overrides.items():
            if needle in user_prompt:
                return reply
        return "mock reply"

    @property
    def chat(self) -> _Chat:
        return _Chat(self)


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def demo_orchestrator() -> AgentOrchestrator:
    """Orchestrator in demo mode (no LLM)."""
    return AgentOrchestrator(llm_client=None)


@pytest.fixture()
def llm_orchestrator() -> Tuple[AgentOrchestrator, MockLLMClient]:
    """Orchestrator wired to a MockLLMClient."""
    client = MockLLMClient(prompt_tokens=120, completion_tokens=80)
    orch = AgentOrchestrator(llm_client=client)
    return orch, client


# ── routing & dependency tests ───────────────────────────────────────────


class TestRouting:
    def test_unknown_agent_rejected(self, demo_orchestrator: AgentOrchestrator) -> None:
        with pytest.raises(ValueError, match="Unknown agent"):
            demo_orchestrator.add_task("eavesdropper", "nope")

    def test_task_ids_are_unique(self, demo_orchestrator: AgentOrchestrator) -> None:
        ids = {demo_orchestrator.add_task("researcher", f"q{i}") for i in range(25)}
        assert len(ids) == 25
        assert all(isinstance(i, str) and len(i) == 8 for i in ids)

    def test_dependencies_met_blocks_until_upstream_done(
        self, demo_orchestrator: AgentOrchestrator
    ) -> None:
        r_id = demo_orchestrator.add_task("researcher", "find facts", priority=1)
        a_id = demo_orchestrator.add_task("analyst", "analyse", priority=2, depends_on=[r_id])
        # neither queue entry is in `completed` yet
        analyst_task = next(t for t in demo_orchestrator.task_queue if t.task_id == a_id)
        researcher_task = next(t for t in demo_orchestrator.task_queue if t.task_id == r_id)
        assert demo_orchestrator._dependencies_met(researcher_task) is True
        assert demo_orchestrator._dependencies_met(analyst_task) is False
        # simulate the researcher completing
        researcher_task.status = "done"
        demo_orchestrator.completed[r_id] = researcher_task
        assert demo_orchestrator._dependencies_met(analyst_task) is True

    def test_priority_picks_lowest_number_first(
        self, demo_orchestrator: AgentOrchestrator
    ) -> None:
        # Three independent tasks, priorities reversed from insertion order.
        low_id = demo_orchestrator.add_task("researcher", "low", priority=9)
        mid_id = demo_orchestrator.add_task("analyst", "mid", priority=5)
        hi_id = demo_orchestrator.add_task("critic", "high", priority=1)

        demo_orchestrator.run()

        # All three should have completed, but the printed order is determined
        # by the priority — verify by completion timestamps.
        order = sorted(
            demo_orchestrator.completed.values(),
            key=lambda t: t.completed_at or math.inf,
        )
        assert [t.task_id for t in order] == [hi_id, mid_id, low_id]

    def test_full_six_agent_pipeline_completes(
        self, demo_orchestrator: AgentOrchestrator
    ) -> None:
        r = demo_orchestrator.add_task("researcher", "topic", priority=1)
        a = demo_orchestrator.add_task("analyst", "analyse", priority=2, depends_on=[r])
        c = demo_orchestrator.add_task("critic", "critique", priority=3, depends_on=[a])
        f = demo_orchestrator.add_task("factchecker", "factcheck", priority=3, depends_on=[a])
        s = demo_orchestrator.add_task("summarizer", "summarise", priority=4, depends_on=[c, f])
        w = demo_orchestrator.add_task("writer", "write", priority=5, depends_on=[s])

        result = demo_orchestrator.run()

        assert result["completed"] == 6
        assert result["failed"] == 0
        assert set(result["results"].keys()) == {r, a, c, f, s, w}
        # Each predecessor must finish before its dependents.
        ts = {tid: demo_orchestrator.completed[tid].completed_at for tid in (r, a, c, f, s, w)}
        assert ts[r] <= ts[a] <= ts[c]
        assert ts[r] <= ts[a] <= ts[f]
        assert max(ts[c], ts[f]) <= ts[s] <= ts[w]


# ── mock-LLM routing tests ───────────────────────────────────────────────


class TestMockLLMRouting:
    def test_per_agent_system_prompt_is_used(
        self, llm_orchestrator: Tuple[AgentOrchestrator, MockLLMClient]
    ) -> None:
        orch, client = llm_orchestrator
        orch.add_task("researcher", "alpha")
        orch.add_task("analyst", "beta")
        orch.add_task("writer", "gamma")
        orch.run()

        systems_by_user = {c["user"]: c["system"] for c in client.calls}
        assert "research specialist" in systems_by_user["Task: alpha"]
        assert "data analyst" in systems_by_user["Task: beta"]
        assert "professional writer" in systems_by_user["Task: gamma"]

    def test_dependency_context_is_forwarded_to_llm(
        self, llm_orchestrator: Tuple[AgentOrchestrator, MockLLMClient]
    ) -> None:
        orch, client = llm_orchestrator
        # Researcher fires first with a known reply we can look for downstream.
        client.reply_overrides = {"alpha-research": "DATA_FROM_RESEARCHER"}
        r_id = orch.add_task("researcher", "alpha-research", priority=1)
        orch.add_task("analyst", "alpha-analysis", priority=2, depends_on=[r_id])
        orch.run()

        analyst_call = next(c for c in client.calls if "alpha-analysis" in c["user"])
        # Context block from the researcher MUST appear in the analyst's user prompt.
        assert "Context from previous agents" in analyst_call["user"]
        assert "[researcher output]" in analyst_call["user"]
        assert "DATA_FROM_RESEARCHER" in analyst_call["user"]

    def test_one_llm_call_per_completed_task(
        self, llm_orchestrator: Tuple[AgentOrchestrator, MockLLMClient]
    ) -> None:
        orch, client = llm_orchestrator
        for agent in ("researcher", "analyst", "critic", "summarizer", "factchecker", "writer"):
            orch.add_task(agent, f"payload-{agent}")
        orch.run()
        assert len(client.calls) == 6
        assert {c["model"] for c in client.calls} == {"gpt-4"}


# ── cost-tracker tests ───────────────────────────────────────────────────


class TestCostTracker:
    DEMO_PER_TASK_COST = 0.0034  # documented in orchestrator._call_llm

    def test_initial_state_is_zero(self, demo_orchestrator: AgentOrchestrator) -> None:
        assert demo_orchestrator.total_cost == 0.0

    def test_demo_mode_per_task_cost_is_constant(
        self, demo_orchestrator: AgentOrchestrator
    ) -> None:
        for agent in ("researcher", "analyst", "writer"):
            demo_orchestrator.add_task(agent, "payload")
        demo_orchestrator.run()

        for task in demo_orchestrator.completed.values():
            assert task.cost_usd == pytest.approx(self.DEMO_PER_TASK_COST)

    def test_total_cost_accumulates_across_three_tasks(
        self, demo_orchestrator: AgentOrchestrator
    ) -> None:
        for agent in ("researcher", "analyst", "writer"):
            demo_orchestrator.add_task(agent, "payload")
        result = demo_orchestrator.run()
        expected = 3 * self.DEMO_PER_TASK_COST
        assert demo_orchestrator.total_cost == pytest.approx(expected)
        assert result["total_cost_usd"] == pytest.approx(round(expected, 4))

    def test_total_cost_matches_sum_of_per_task_costs(
        self, demo_orchestrator: AgentOrchestrator
    ) -> None:
        # Full 6-agent pipeline.
        r = demo_orchestrator.add_task("researcher", "x", priority=1)
        a = demo_orchestrator.add_task("analyst", "x", priority=2, depends_on=[r])
        c = demo_orchestrator.add_task("critic", "x", priority=3, depends_on=[a])
        f = demo_orchestrator.add_task("factchecker", "x", priority=3, depends_on=[a])
        s = demo_orchestrator.add_task("summarizer", "x", priority=4, depends_on=[c, f])
        demo_orchestrator.add_task("writer", "x", priority=5, depends_on=[s])
        demo_orchestrator.run()

        per_task_sum = sum(t.cost_usd for t in demo_orchestrator.completed.values())
        assert demo_orchestrator.total_cost == pytest.approx(per_task_sum)

    def test_llm_cost_uses_token_pricing(
        self, llm_orchestrator: Tuple[AgentOrchestrator, MockLLMClient]
    ) -> None:
        orch, client = llm_orchestrator
        # Pricing from orchestrator._call_llm: $0.03/1k input, $0.06/1k output.
        expected_per_call = client.prompt_tokens * 0.00003 + client.completion_tokens * 0.00006
        orch.add_task("researcher", "x")
        orch.add_task("analyst", "y")
        orch.run()

        for task in orch.completed.values():
            assert task.cost_usd == pytest.approx(expected_per_call)
        assert orch.total_cost == pytest.approx(2 * expected_per_call)

    def test_failed_call_does_not_inflate_cost(
        self, llm_orchestrator: Tuple[AgentOrchestrator, MockLLMClient]
    ) -> None:
        """If the LLM raises, ``_call_llm`` should return a $0 cost so the
        accumulator only counts successful calls."""

        class BoomClient:
            class chat:  # noqa: N801 — match openai shape
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**_kwargs: Any) -> _ChatResponse:
                        raise RuntimeError("boom")

        orch = AgentOrchestrator(llm_client=BoomClient())
        orch.add_task("researcher", "will-explode")
        orch.run()
        assert orch.total_cost == pytest.approx(0.0)
        # The task itself still "completes" (the orchestrator records an
        # ``[ERROR] ...`` string in result), but with zero cost.
        only_task = next(iter(orch.completed.values()))
        assert only_task.cost_usd == pytest.approx(0.0)
        assert only_task.result is not None and only_task.result.startswith("[ERROR]")


# ── AgentTask dataclass defaults ─────────────────────────────────────────


def test_agent_task_defaults_are_independent_lists() -> None:
    a = AgentTask(task_id="aaa", agent_type="researcher", payload="x")
    b = AgentTask(task_id="bbb", agent_type="researcher", payload="y")
    a.depends_on.append("zzz")
    assert b.depends_on == []
    assert a.priority == 5
    assert a.status == "pending"
    assert a.cost_usd == 0.0
