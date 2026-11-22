"""CrewAI pipeline test with a scripted fake LLM: no network, no API key."""

import pytest

pytest.importorskip("crewai")
from crewai import BaseLLM

from supportlens.agent import TicketService
from supportlens.crew_agent import build_crew


class ScriptedLLM(BaseLLM):
    calls: list = []

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        role = getattr(from_agent, "role", "?")
        self.calls.append(role)
        if role == "Support Lead" or self.calls.count(role) > 1:
            return f"Thought: I now know the final answer\nFinal Answer: [{role}] done"
        tool = {
            "Support Triage Specialist": "classify_ticket",
            "Case Researcher": "find_similar_tickets",
        }[role]
        return f'Thought: use the tool\nAction: {tool}\nAction Input: {{"text": "where is my refund"}}'

    def supports_function_calling(self):
        return False

    def get_context_window_size(self):
        return 8000


def test_crew_runs_three_agents_and_real_tools(artifacts):
    svc, used = TicketService(artifacts), []
    for name in ("classify", "similar"):
        orig = getattr(svc, name)
        setattr(
            svc,
            name,
            (lambda o, n: lambda *a, **k: (used.append(n), o(*a, **k))[1])(orig, name),
        )
    llm = ScriptedLLM(model="stub")
    out = build_crew(svc, llm).kickoff(inputs={"ticket": "where is my refund"})
    assert out.raw == "[Support Lead] done"
    assert {"Support Triage Specialist", "Case Researcher", "Support Lead"} <= set(
        llm.calls
    )
    assert used == ["classify", "similar"]
