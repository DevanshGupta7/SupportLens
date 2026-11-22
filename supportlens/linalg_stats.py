"""Linear algebra, probability and statistics building blocks, written from scratch in NumPy.

Nothing here calls scikit-learn/scipy: the point is to show the maths.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .preprocess import tokenize


# ---------------------------------------------------------------- linear algebra
class TfidfVectorizer:
    """TF-IDF with smoothed idf = ln((1+N)/(1+df)) + 1 and L2-normalised rows.

    min_df drops very rare terms (mostly typos); max_features keeps only the most frequent terms,
    which bounds the size of the dense matrix on large real datasets.
    """

    def __init__(self, min_df: int = 1, max_features: int | None = None):
        self.min_df, self.max_features = min_df, max_features

    def fit(self, docs: list[str]) -> TfidfVectorizer:
        df = Counter(t for d in docs for t in set(tokenize(d)))
        terms = [t for t, c in df.items() if c >= self.min_df]
        if self.max_features and len(terms) > self.max_features:
            terms = sorted(terms, key=lambda t: (-df[t], t))[: self.max_features]
        terms = sorted(terms)
        self.index = {t: i for i, t in enumerate(terms)}
        dfv = np.array([df[t] for t in terms], dtype=np.float64)
        self.idf = (np.log((1 + len(docs)) / (1 + dfv)) + 1).astype(np.float32)
        return self

    def transform(self, docs: list[str]) -> np.ndarray:
        X = np.zeros((len(docs), len(self.index)), dtype=np.float32)
        for r, d in enumerate(docs):
            for t in tokenize(d):
                if t in self.index:
                    X[r, self.index[t]] += 1
        X *= self.idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms == 0, 1, norms)

    def fit_transform(self, docs: list[str]) -> np.ndarray:
        return self.fit(docs).transform(docs)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """cos(q, m_i) = q.m_i / (|q||m_i|) for every row m_i."""
    q = query / (np.linalg.norm(query) or 1.0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1, norms)) @ q


def pca_svd(X: np.ndarray, k: int):
    """PCA via SVD of the centred matrix: Xc = U S V^T.

    Returns (projection onto top-k components, components V_k^T, explained variance ratio).
    Explained variance of component i is proportional to S_i^2.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    evr = S**2 / np.sum(S**2)
    return Xc @ Vt[:k].T, Vt[:k], evr


# ------------------------------------------------------------ probability / models
class MultinomialNaiveBayes:
    """argmax_c  log P(c) + sum_w count(w) * log P(w|c), with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(
        self, counts: np.ndarray, y: np.ndarray, n_classes: int
    ) -> MultinomialNaiveBayes:
        self.log_prior = np.log(np.bincount(y, minlength=n_classes) / len(y))
        class_counts = (
            np.stack([counts[y == c].sum(axis=0) for c in range(n_classes)])
            + self.alpha
        )
        self.log_lik = np.log(class_counts / class_counts.sum(axis=1, keepdims=True))
        return self

    def predict_log_proba(self, counts: np.ndarray) -> np.ndarray:
        return counts @ self.log_lik.T + self.log_prior

    def predict(self, counts: np.ndarray) -> np.ndarray:
        return self.predict_log_proba(counts).argmax(axis=1)


def bag_of_words(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """Count matrix from padded id matrix (column 0 = padding is dropped)."""
    out = np.zeros((len(ids), vocab_size))
    for r, row in enumerate(ids):
        np.add.at(out[r], row, 1)
    out[:, 0] = 0
    return out


# ------------------------------------------------------------------- statistics
def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def bootstrap_ci(
    y_true,
    y_pred,
    metric=accuracy,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
):
    """Percentile bootstrap confidence interval for any metric(y_true, y_pred)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        stats.append(metric(y_true[i], y_pred[i]))
    return float(np.quantile(stats, alpha / 2)), float(
        np.quantile(stats, 1 - alpha / 2)
    )


def paired_permutation_test(
    correct_a, correct_b, n_perm: int = 10000, seed: int = 0
) -> float:
    """Two-sided p-value for 'models A and B have the same accuracy' on the same examples.

    Under H0 the sign of each per-example difference is exchangeable, so we randomly flip signs.
    """
    d = np.asarray(correct_a, float) - np.asarray(correct_b, float)
    observed = abs(d.mean())
    if observed == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm >= observed) + 1) / (n_perm + 1))


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """cm[i, j] = number of examples with true class i predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class_report(cm: np.ndarray):
    """Precision, recall and F1 for every class, read off a confusion matrix."""
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1
