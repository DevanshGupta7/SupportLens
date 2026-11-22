import numpy as np
import torch
from langchain_core.messages import AIMessage

from supportlens.agent import run_tool_loop
from supportlens.linalg_stats import (
    MultinomialNaiveBayes,
    TfidfVectorizer,
    bag_of_words,
    bootstrap_ci,
    cosine_similarity,
    paired_permutation_test,
    pca_svd,
)
from supportlens.model import TransformerTextClassifier
from supportlens.preprocess import Vocab, clean_text


def test_clean_text_masks_noise():
    out = clean_text("<p>Mail me@x.com about ORD-12345 at https://a.b/c</p>")
    assert out == "mail <email> about <order> at <url>"


def test_tfidf_rows_are_unit_norm_and_self_similar():
    docs = ["refund my payment", "app crashes on save", "refund my payment please"]
    X = TfidfVectorizer().fit_transform(docs)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)
    sims = cosine_similarity(X[0], X)
    assert sims[0] > sims[2] > sims[1]


def test_svd_reconstruction_and_variance():
    X = np.random.default_rng(0).normal(size=(30, 6))
    proj, comps, evr = pca_svd(X, 6)
    assert np.isclose(evr.sum(), 1.0) and np.all(np.diff(evr) <= 1e-12)
    assert np.allclose(proj @ comps, X - X.mean(0))


def test_naive_bayes_learns_separable_data():
    toks = [["refund", "charge"], ["refund"], ["crash", "bug"], ["bug"]]
    v = Vocab(toks, min_freq=1)
    ids = np.array([v.encode(t, 4) for t in toks])
    y = np.array([0, 0, 1, 1])
    nb = MultinomialNaiveBayes().fit(bag_of_words(ids, len(v)), y, 2)
    assert (nb.predict(bag_of_words(ids, len(v))) == y).all()


def test_stats_helpers():
    y = np.array([0, 1] * 50)
    lo, hi = bootstrap_ci(y, y)
    assert lo == hi == 1.0
    a, b = np.ones(200), np.zeros(200)
    assert paired_permutation_test(a, b, n_perm=2000) < 0.01
    assert paired_permutation_test(a, a) == 1.0


def test_transformer_shapes_and_padding_invariance():
    torch.manual_seed(0)
    m = TransformerTextClassifier(50, 5, max_len=12).eval()
    ids = torch.tensor([[2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    longer_pad = torch.tensor([[2, 3, 4] + [0] * 9])
    assert m(ids).shape == (1, 5)
    assert torch.allclose(m(ids), m(longer_pad), atol=1e-5)


class _StubTool:
    name = "classify_ticket"

    def invoke(self, args):
        return '{"category": "billing"}'


class _StubLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "classify_ticket", "args": {"text": "x"}, "id": "1"},
                    {"name": "delete_everything", "args": {}, "id": "2"},
                ],
            )
        return AIMessage(content="billing")


def test_agent_loop_runs_tools_and_blocks_unknown_ones():
    llm = _StubLLM()
    assert run_tool_loop(llm, [_StubTool()], "q") == "billing"
    assert llm.calls == 2


def test_agent_loop_has_step_cap():
    class Looping:
        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[{"name": "classify_ticket", "args": {}, "id": "1"}],
            )

    assert "limit" in run_tool_loop(Looping(), [_StubTool()], "q", max_steps=3)
