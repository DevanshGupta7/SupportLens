"""Synthetic, messy support-ticket generator.

Stands in for real unstructured text. Categories deliberately share vocabulary and
sometimes mention a second issue, so the task is not trivially separable.
Swap `generate_tickets` for your own CSV loader to run on real data.
"""

from __future__ import annotations

import random

CATEGORIES = ["billing", "login", "shipping", "bug", "feature_request"]

CORE = {
    "billing": [
        "I was charged twice for my subscription",
        "my refund has not arrived",
        "the invoice amount looks wrong",
        "please cancel my plan and refund the payment",
        "my card was declined but the money was deducted",
    ],
    "login": [
        "I cannot log in to my account",
        "password reset email never arrives",
        "the two factor code is not working",
        "my account is locked after too many attempts",
        "it says invalid credentials but my password is right",
    ],
    "shipping": [
        "my package has not been delivered",
        "the tracking page shows no update for days",
        "the wrong item arrived in my order",
        "the delivery address was changed by mistake",
        "my parcel is stuck at the courier hub",
    ],
    "bug": [
        "the app crashes when I open settings",
        "the page freezes after I click save",
        "I get an error 500 on checkout",
        "the export button does nothing",
        "notifications show up twice",
    ],
    "feature_request": [
        "it would be great to have dark mode",
        "please add support for exporting to csv",
        "can you add a way to schedule reports",
        "I wish there was a mobile widget",
        "would love an integration with calendar apps",
    ],
}
SHARED = [
    "this is urgent",
    "I contacted support before",
    "since yesterday",
    "on my phone",
    "after the latest update",
    "please help",
    "very frustrating",
    "for my team account",
]


def _typo(word: str, rng: random.Random) -> str:
    if len(word) > 4:
        i = rng.randrange(1, len(word) - 2)
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    return word


def _noisy(text: str, rng: random.Random) -> str:
    words = [_typo(w, rng) if rng.random() < 0.06 else w for w in text.split()]
    text = " ".join(words)
    if rng.random() < 0.3:
        text += f" my email is user{rng.randint(1, 999)}@example.com"
    if rng.random() < 0.3:
        text += f" order #ORD-{rng.randint(10000, 99999)}"
    if rng.random() < 0.15:
        text += " see https://status.example.com/incident/" + str(rng.randint(1, 99))
    if rng.random() < 0.2:
        text = f"<p>{text}</p>"
    if rng.random() < 0.1:
        text = text.upper()
    return text


def generate_tickets(n: int = 3000, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        label = rng.choice(CATEGORIES)
        parts = [rng.choice(CORE[label])]
        if rng.random() < 0.25:  # a second, unrelated issue mentioned in passing
            other = rng.choice([c for c in CATEGORIES if c != label])
            parts.append("also " + rng.choice(CORE[other]))
        parts += rng.sample(SHARED, k=rng.randint(1, 2))
        out.append({"text": _noisy(". ".join(parts), rng), "label": label})
    return out
