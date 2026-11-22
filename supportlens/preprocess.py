"""Turn raw, unstructured ticket text into clean tokens and padded id matrices."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

PAD, UNK = "<pad>", "<unk>"

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"\S+@\S+\.\w+")
_ORDER = re.compile(r"#?ord-\d+", re.IGNORECASE)
_SLOT = re.compile(
    r"\{\{\s*([^}]+?)\s*\}\}"
)  # e.g. {{Order Number}} in the Bitext dataset
_NONWORD = re.compile(r"[^a-z0-9\s_<>]")


def clean_text(text: str) -> str:
    """Strip HTML, mask emails/URLs/order ids with placeholder tokens, lowercase."""
    text = _HTML.sub(" ", text)
    text = _URL.sub(" <url> ", text)
    text = _EMAIL.sub(" <email> ", text)
    text = _ORDER.sub(" <order> ", text)
    text = _SLOT.sub(
        lambda m: " slot_" + re.sub(r"\W+", "_", m.group(1).lower()) + " ", text
    )
    text = _NONWORD.sub(" ", text.lower())
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return clean_text(text).split()


class Vocab:
    def __init__(self, tokens_list: list[list[str]], min_freq: int = 2):
        counts = Counter(t for toks in tokens_list for t in toks)
        self.itos = [PAD, UNK] + sorted(t for t, c in counts.items() if c >= min_freq)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str], max_len: int) -> list[int]:
        ids = [self.stoi.get(t, 1) for t in tokens[:max_len]]
        return ids + [0] * (max_len - len(ids))

    def encode_batch(self, texts: list[str], max_len: int) -> np.ndarray:
        return np.array(
            [self.encode(tokenize(t), max_len) for t in texts], dtype=np.int64
        )

    @classmethod
    def from_itos(cls, itos: list[str]) -> Vocab:
        v = cls.__new__(cls)
        v.itos, v.stoi = itos, {t: i for i, t in enumerate(itos)}
        return v
