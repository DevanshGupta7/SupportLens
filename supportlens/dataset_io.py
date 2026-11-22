"""Load real data: any CSV, plus a one-line download of the recommended Bitext customer-support set.

Bitext Customer Support (CDLA-Sharing-1.0): ~27k English utterances, 11 categories, 27 intents.
Columns: flags, instruction, category, intent, response.
"""

from __future__ import annotations

import csv
import random
import urllib.request
from collections import Counter
from pathlib import Path

BITEXT_URL = (
    "https://raw.githubusercontent.com/bitext/customer-support-llm-chatbot-training-dataset/"
    "main/data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


def download_bitext(dest: str = "data/bitext.csv") -> Path:
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading Bitext dataset to {dest} ...")
        urllib.request.urlretrieve(BITEXT_URL, dest)
    return dest


def load_csv(
    path: str,
    text_col: str,
    label_col: str,
    max_per_class: int | None = None,
    seed: int = 0,
) -> list[dict]:
    """Read a CSV into [{'text', 'label'}], dropping blanks and exact duplicate texts.

    Duplicates are removed because the same sentence landing in both train and test
    would inflate the score. `max_per_class` caps each class (reproducible with `seed`).
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [
            c for c in (text_col, label_col) if c not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"columns {missing} not found; the CSV has {reader.fieldnames}"
            )
        rows = [(r[text_col].strip(), r[label_col].strip()) for r in reader]
    seen, data = set(), []
    for text, label in rows:
        if text and label and text.lower() not in seen:
            seen.add(text.lower())
            data.append({"text": text, "label": label})
    print(f"loaded {len(rows)} rows -> {len(data)} after dropping blanks/duplicates")
    if max_per_class:
        rng = random.Random(seed)
        by_label: dict[str, list[dict]] = {}
        for d in data:
            by_label.setdefault(d["label"], []).append(d)
        data = [
            d
            for items in by_label.values()
            for d in rng.sample(items, min(max_per_class, len(items)))
        ]
        rng.shuffle(data)
    print("class counts:", dict(sorted(Counter(d["label"] for d in data).items())))
    return data
