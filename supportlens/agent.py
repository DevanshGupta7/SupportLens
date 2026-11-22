"""LangChain tool-calling agent on top of the trained classifier.

    python -m supportlens.agent "my parcel is stuck at the hub, order #ORD-12345"

With GROQ_API_KEY set, a Groq-hosted LLM decides which tools to call.
Without it, a deterministic router runs the same tools, so the project works offline.

Safety design (agents going rogue): read-only tools, a tool allow-list, a hard cap on loop
iterations, and tool errors are returned to the model as text rather than raised.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from .linalg_stats import TfidfVectorizer, cosine_similarity
from .model import build_model
from .preprocess import Vocab, tokenize

MAX_STEPS = 5
SYSTEM = (
    "You are a support-triage assistant. Use the tools to classify the ticket and find similar "
    "past tickets, then reply with the category, a one-line reason, and a suggested next step. "
    "Only use information returned by the tools."
)


class TicketService:
    """Loads trained artifacts and exposes classify / similar / stats."""

    def __init__(self, artifacts: str = "artifacts", max_corpus: int = 5000):
        root = Path(artifacts)
        ckpt = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
        self.categories = ckpt["categories"]
        self.max_len = ckpt["max_len"]
        self.vocab = Vocab.from_itos(ckpt["itos"])
        self.model = build_model(
            ckpt.get("arch", "transformer"),
            len(self.vocab),
            len(self.categories),
            self.max_len,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.corpus = json.loads((root / "corpus.json").read_text())
        if len(self.corpus) > max_corpus:  # keep the dense TF-IDF index small
            self.corpus = random.Random(0).sample(self.corpus, max_corpus)
        self.labels = np.array([c["label"] for c in self.corpus])
        self.tfidf = TfidfVectorizer(min_df=2, max_features=5000).fit(
            [c["text"] for c in self.corpus]
        )
        self.matrix = self.tfidf.transform([c["text"] for c in self.corpus])

    @torch.no_grad()
    def classify(self, text: str) -> dict:
        if not tokenize(
            text
        ):  # nothing left after cleaning (empty, only punctuation/emoji/HTML)
            return {"category": "unknown", "confidence": 0.0, "probabilities": {}}
        ids = torch.from_numpy(self.vocab.encode_batch([text], self.max_len))
        probs = torch.softmax(self.model(ids)[0], dim=-1).numpy()
        top = int(probs.argmax())
        return {
            "category": self.categories[top],
            "confidence": round(float(probs[top]), 3),
            "truncated": len(tokenize(text)) > self.max_len,
            "probabilities": {
                c: round(float(p), 3) for c, p in zip(self.categories, probs)
            },
        }

    def similar(self, text: str, k: int = 3, label: str | None = None) -> list[dict]:
        query = self.tfidf.transform([text])[0]
        if not query.any():  # no known words: any neighbour would be arbitrary
            return []
        sims = cosine_similarity(query, self.matrix)
        if label is not None:  # only consider past tickets filed under this category
            sims = np.where(self.labels == label, sims, -1.0)
        return [
            {
                "text": self.corpus[i]["text"][:140],
                "label": self.corpus[i]["label"],
                "similarity": round(float(sims[i]), 3),
            }
            for i in np.argsort(-sims)[:k]
        ]

    def stats(self) -> dict:
        counts = Counter(c["label"] for c in self.corpus)
        lengths = [len(tokenize(c["text"])) for c in self.corpus]
        return {
            "tickets": len(self.corpus),
            "per_category": dict(counts),
            "mean_tokens": round(float(np.mean(lengths)), 1),
        }


def build_tools(svc: TicketService) -> list:
    @tool
    def classify_ticket(text: str) -> str:
        """Classify a support ticket into one of the categories the model was trained on."""
        return json.dumps(svc.classify(text))

    @tool
    def find_similar_tickets(text: str, k: int = 3) -> str:
        """Find the k most similar past tickets (TF-IDF cosine similarity)."""
        return json.dumps(svc.similar(text, max(1, min(int(k), 5))))

    @tool
    def ticket_stats() -> str:
        """Overall statistics about the historical ticket corpus."""
        return json.dumps(svc.stats())

    return [classify_ticket, find_similar_tickets, ticket_stats]


def run_tool_loop(
    llm_with_tools, tools: list, question: str, max_steps: int = MAX_STEPS
) -> str:
    """Minimal agent loop: model -> tool calls -> tool results -> model, with a hard step cap."""
    registry = {t.name: t for t in tools}  # allow-list
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]
    for _ in range(max_steps):
        ai = llm_with_tools.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return str(ai.content)
        for call in ai.tool_calls:
            if call["name"] not in registry:
                result = f"error: unknown tool {call['name']!r}"
            else:
                try:
                    result = str(registry[call["name"]].invoke(call["args"]))
                except (
                    Exception
                ) as exc:  # return the error to the model instead of crashing
                    result = f"error: {exc}"
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return "Stopped: tool-call limit reached."


def offline_answer(svc: TicketService, question: str) -> str:
    c = svc.classify(question)
    sim = svc.similar(question, 2, label=None if c["category"] == "unknown" else c["category"])
    lines = [f"Category: {c['category']} (confidence {c['confidence']})"]
    if c.get("truncated"):
        lines.append(f"Note: long ticket, only the first {svc.max_len} tokens were classified.")
    if sim:
        lines.append("Similar past tickets in that category:")
    lines += [f"  - [{s['label']}] {s['text']} (sim {s['similarity']})" for s in sim]
    return "\n".join(lines)


def answer(question: str, artifacts: str = "artifacts") -> str:
    svc = TicketService(artifacts)
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        tools = build_tools(svc)
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0
        )
        return run_tool_loop(llm.bind_tools(tools), tools, question)
    return offline_answer(svc, question)


if __name__ == "__main__":
    print(
        answer(
            " ".join(sys.argv[1:]) or "I was charged twice for my subscription",
            os.getenv("SUPPORTLENS_ARTIFACTS", "artifacts"),
        )
    )
