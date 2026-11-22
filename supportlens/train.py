"""Train + evaluate: Naive Bayes baseline vs. a PyTorch model, with statistics.

Synthetic demo data:
    python -m supportlens.train --n 3000

Real data (recommended: the Bitext customer-support set, downloaded for you):
    python -m supportlens.train --bitext
Any CSV:
    python -m supportlens.train --csv my.csv --text-col text --label-col label

Ablation (does attention help?):  add  --arch meanpool
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import CATEGORIES, generate_tickets
from .dataset_io import download_bitext, load_csv
from .linalg_stats import (
    MultinomialNaiveBayes,
    TfidfVectorizer,
    accuracy,
    bag_of_words,
    bootstrap_ci,
    confusion_matrix,
    macro_f1,
    paired_permutation_test,
    pca_svd,
    per_class_report,
)
from .model import ARCHITECTURES, build_model
from .preprocess import Vocab, tokenize


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split(n: int, seed: int, fracs=(0.7, 0.15, 0.15)):
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def stratified_split(y: np.ndarray, seed: int, fracs=(0.7, 0.15, 0.15)):
    """Split each class separately so train/val/test keep the same class proportions."""
    rng = np.random.default_rng(seed)
    parts: tuple[list, list, list] = ([], [], [])
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        a, b = int(fracs[0] * len(idx)), int((fracs[0] + fracs[1]) * len(idx))
        parts[0].append(idx[:a])
        parts[1].append(idx[a:b])
        parts[2].append(idx[b:])
    return tuple(rng.permutation(np.concatenate(p)) for p in parts)


@torch.no_grad()
def predict(
    model: nn.Module, X: np.ndarray, device: str, batch: int = 256
) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        logits = model(torch.from_numpy(X[i : i + batch]).to(device))
        out.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(out)


def fit(
    model,
    Xtr,
    ytr,
    Xva,
    yva,
    epochs,
    lr,
    batch_size,
    device,
    patience=4,
    class_weights=None,
):
    """Training loop: AdamW, cross-entropy (optionally class-weighted), early stopping on val accuracy."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    weight = (
        None
        if class_weights is None
        else torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    best, best_state, bad, history = 0.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        total = 0.0
        for i in range(0, len(perm), batch_size):
            b = perm[i : i + batch_size]
            xb, yb = Xtr_t[b].to(device), ytr_t[b].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        val_acc = accuracy(yva, predict(model, Xva, device))
        history.append(
            {"epoch": epoch, "train_loss": total / len(perm), "val_acc": val_acc}
        )
        print(f"epoch {epoch:2d}  loss {total / len(perm):.4f}  val_acc {val_acc:.4f}")
        if val_acc > best:
            best, bad = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break
    model.load_state_dict(best_state)
    return history


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_argument_group("data")
    src.add_argument(
        "--n",
        type=int,
        default=3000,
        help="synthetic tickets (ignored with --csv/--bitext)",
    )
    src.add_argument("--csv", help="path to a real CSV dataset")
    src.add_argument(
        "--bitext",
        action="store_true",
        help="download + use the Bitext customer-support dataset",
    )
    src.add_argument("--text-col", default="instruction")
    src.add_argument(
        "--label-col",
        default="category",
        help="e.g. category (11 classes) or intent (27 classes)",
    )
    src.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--arch", choices=ARCHITECTURES, default="transformer")
    p.add_argument(
        "--class-weights",
        action="store_true",
        help="re-weight the loss for imbalanced classes",
    )
    p.add_argument("--max-len", type=int, default=40)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.bitext:
        args.csv = str(download_bitext())
    if args.csv:
        data = load_csv(
            args.csv, args.text_col, args.label_col, args.max_per_class, args.seed
        )
        categories = sorted({d["label"] for d in data})
        source = Path(args.csv).name
    else:
        data, categories, source = (
            generate_tickets(args.n, args.seed),
            CATEGORIES,
            "synthetic",
        )
    K = len(categories)
    texts = [d["text"] for d in data]
    y = np.array([categories.index(d["label"]) for d in data])
    tr, va, te = stratified_split(y, args.seed)

    vocab = Vocab(
        [tokenize(texts[i]) for i in tr], min_freq=2
    )  # train split only: no leakage
    X = vocab.encode_batch(texts, args.max_len)
    print(
        f"{len(data)} tickets ({source}) | {K} classes | vocab {len(vocab)} | device {device}"
    )
    print(f"split: train {len(tr)} / val {len(va)} / test {len(te)}")

    # --- baseline: Naive Bayes from scratch (probability + linear algebra)
    counts = bag_of_words(X, len(vocab))
    nb = MultinomialNaiveBayes().fit(counts[tr], y[tr], K)
    nb_pred = nb.predict(counts[te])

    # --- neural model
    model = build_model(args.arch, len(vocab), K, args.max_len).to(device)
    print(f"{args.arch} params: {sum(q.numel() for q in model.parameters()):,}")
    weights = None
    if args.class_weights:
        weights = len(tr) / (K * np.bincount(y[tr], minlength=K))
    history = fit(
        model,
        X[tr],
        y[tr],
        X[va],
        y[va],
        args.epochs,
        args.lr,
        args.batch_size,
        device,
        class_weights=weights,
    )
    nn_pred = predict(model, X[te], device)

    # --- statistics
    lo, hi = bootstrap_ci(y[te], nn_pred)
    p_val = paired_permutation_test(nn_pred == y[te], nb_pred == y[te])
    cm = confusion_matrix(y[te], nn_pred, K)
    prec, rec, f1 = per_class_report(cm)
    metrics = {
        "dataset": source,
        "classes": categories,
        "arch": args.arch,
        "naive_bayes": {
            "acc": accuracy(y[te], nb_pred),
            "macro_f1": macro_f1(y[te], nb_pred, K),
        },
        "neural": {
            "acc": accuracy(y[te], nn_pred),
            "macro_f1": macro_f1(y[te], nn_pred, K),
            "acc_95ci": [lo, hi],
        },
        "permutation_test_p": p_val,
        "per_class": {
            c: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(cm[i].sum()),
            }
            for i, c in enumerate(categories)
        },
        "confusion_matrix": cm.tolist(),
        "history": history,
    }

    # --- linear algebra on TF-IDF (sampled: the dense matrix grows quickly on real data)
    sample = [texts[i] for i in tr[:3000]]
    _, _, evr = pca_svd(TfidfVectorizer(min_df=2).fit_transform(sample), 2)
    metrics["tfidf_variance_top10"] = float(evr[:10].sum())

    print("\n=== test set ===")
    print(
        f"Naive Bayes  acc {metrics['naive_bayes']['acc']:.4f}  macro-F1 {metrics['naive_bayes']['macro_f1']:.4f}"
    )
    print(
        f"{args.arch:<12} acc {metrics['neural']['acc']:.4f}  macro-F1 {metrics['neural']['macro_f1']:.4f}"
        f"  95% CI [{lo:.3f}, {hi:.3f}]"
    )
    print(f"paired permutation test ({args.arch} vs NB): p = {p_val:.4f}")
    print(
        f"top-10 TF-IDF principal components explain {metrics['tfidf_variance_top10']:.1%} of variance"
    )
    print(f"\n{'class':<18}{'prec':>7}{'recall':>8}{'f1':>7}{'n':>7}")
    for i, c in enumerate(categories):
        print(f"{c:<18}{prec[i]:>7.3f}{rec[i]:>8.3f}{f1[i]:>7.3f}{int(cm[i].sum()):>7}")
    worst = sorted(
        (
            (cm[i, j], categories[i], categories[j])
            for i in range(K)
            for j in range(K)
            if i != j
        ),
        reverse=True,
    )[:3]
    print(
        "most common confusions (true -> predicted):",
        [f"{a}->{b} x{n}" for n, a, b in worst if n > 0],
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "itos": vocab.itos,
            "max_len": args.max_len,
            "categories": categories,
            "arch": args.arch,
        },
        out / "model.pt",
    )
    (out / "corpus.json").write_text(json.dumps([data[i] for i in tr]))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved model, corpus and metrics to {out}/")


if __name__ == "__main__":
    main()
