"""CrewAI version of the triage agent: three cooperating agents instead of one tool loop.

    python -m supportlens.crew_agent "I never got my refund for order #ORD-48213"

Pipeline (Process.sequential):
    Triage Specialist  --classify_ticket-->  category + confidence
    Case Researcher    --find_similar_tickets-->  how similar past tickets were labelled
    Support Lead       (no tools; reads both results)  -->  final triage note

Needs GROQ_API_KEY (override the model with CREW_MODEL). Without a key it falls back to the
offline router from agent.py, so the command never fails just because no LLM is available.

Safety design: every tool is read-only, agents cannot delegate, iteration caps are set, and the
last agent has no tools at all, so only the earlier, constrained steps ever touch data.
The ticket text is untrusted user input: it is passed as data inside task descriptions and
tools only ever read it, never execute it.
"""

from __future__ import annotations

import json
import os
import sys

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

from .agent import TicketService, offline_answer


def build_crew_tools(svc: TicketService) -> dict:
    @tool("classify_ticket")
    def classify_ticket(text: str) -> str:
        """Classify a support ticket into one of the categories the model was trained on."""
        return json.dumps(svc.classify(text))

    @tool("find_similar_tickets")
    def find_similar_tickets(text: str, k: int = 3) -> str:
        """Find the k most similar past tickets (TF-IDF cosine similarity)."""
        return json.dumps(svc.similar(text, max(1, min(int(k), 5))))

    @tool("ticket_stats")
    def ticket_stats() -> str:
        """Overall statistics about the historical ticket corpus."""
        return json.dumps(svc.stats())

    return {
        "classify": classify_ticket,
        "similar": find_similar_tickets,
        "stats": ticket_stats,
    }


def build_crew(svc: TicketService, llm) -> Crew:
    tools = build_crew_tools(svc)
    common = {"llm": llm, "allow_delegation": False, "max_iter": 4, "verbose": False}

    triage = Agent(
        role="Support Triage Specialist",
        goal="Assign each support ticket to the correct category and report the model's confidence.",
        backstory="You route customer tickets. You always use the classifier tool and never guess.",
        tools=[tools["classify"]],
        **common,
    )
    researcher = Agent(
        role="Case Researcher",
        goal="Find similar past tickets and report which categories they were filed under.",
        backstory="You look up precedent. You only report what the search tool returns.",
        tools=[tools["similar"]],
        **common,
    )
    lead = Agent(
        role="Support Lead",
        goal="Write a short, safe triage note from the findings of your teammates.",
        backstory="You never invent facts. If the classifier and past tickets disagree, you say so.",
        tools=[],
        **common,
    )

    t_classify = Task(
        description="Classify this support ticket with the classify_ticket tool:\n\n{ticket}",
        expected_output="The category, the confidence, and one line on why.",
        agent=triage,
    )
    t_research = Task(
        description="Use find_similar_tickets (k=3) for this support ticket:\n\n{ticket}",
        expected_output="The 3 most similar past tickets with their categories and similarity scores.",
        agent=researcher,
    )
    t_note = Task(
        description=(
            "Write the final triage note using ONLY the two previous results: the category, whether "
            "the similar past tickets agree (flag any disagreement), and one suggested next step."
        ),
        expected_output="A 3 to 5 line triage note.",
        agent=lead,
        context=[t_classify, t_research],
    )

    return Crew(
        agents=[triage, researcher, lead],
        tasks=[t_classify, t_research, t_note],
        process=Process.sequential,
        verbose=False,
    )


def make_llm():
    model = os.getenv("CREW_MODEL") or "groq/" + os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return LLM(model=model, temperature=0)


def answer(question: str, artifacts: str = "artifacts", llm=None) -> str:
    svc = TicketService(artifacts)
    if llm is None:
        if not os.getenv("GROQ_API_KEY"):
            return (
                "(GROQ_API_KEY not set: offline fallback, no CrewAI run)\n"
                + offline_answer(svc, question)
            )
        llm = make_llm()
    result = build_crew(svc, llm).kickoff(inputs={"ticket": question})
    return str(result.raw if hasattr(result, "raw") else result)


if __name__ == "__main__":
    print(
        answer(
            " ".join(sys.argv[1:]) or "I was charged twice for my subscription",
            os.getenv("SUPPORTLENS_ARTIFACTS", "artifacts"),
        )
    )
