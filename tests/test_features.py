import numpy as np
import pytest
import torch

from supportlens.dataset_io import load_csv
from supportlens.linalg_stats import TfidfVectorizer, confusion_matrix, per_class_report
from supportlens.model import build_model
from supportlens.preprocess import clean_text
from supportlens.train import stratified_split


def test_slot_placeholders_keep_their_meaning():
    assert (
        clean_text("track order {{Order Number}} now")
        == "track order slot_order_number now"
    )


def test_load_csv_drops_blanks_and_duplicates_and_handles_bom(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text(
        "\ufefftext,label,extra\nrefund me,billing,1\nREFUND ME,billing,2\n,billing,3\nhello,,4\ncrash,bug,5\n",
        encoding="utf-8",
    )
    data = load_csv(str(f), "text", "label")
    assert [d["text"] for d in data] == ["refund me", "crash"]


def test_load_csv_reports_missing_columns(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="not found"):
        load_csv(str(f), "text", "label")


def test_load_csv_caps_each_class(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text(
        "text,label\n" + "\n".join(f"t{i},{'a' if i % 3 else 'b'}" for i in range(60))
    )
    data = load_csv(str(f), "text", "label", max_per_class=5)
    assert (
        sum(d["label"] == "a" for d in data) == 5
        and sum(d["label"] == "b" for d in data) == 5
    )


def test_stratified_split_keeps_class_ratios_and_is_disjoint():
    y = np.array([0] * 900 + [1] * 100)
    tr, va, te = stratified_split(y, seed=0)
    assert len(set(tr) | set(va) | set(te)) == 1000
    assert (
        not (set(tr) & set(te)) and not (set(tr) & set(va)) and not (set(va) & set(te))
    )
    for part in (tr, va, te):
        assert abs(y[part].mean() - 0.1) < 0.02


def test_confusion_matrix_and_per_class_report():
    cm = confusion_matrix([0, 0, 1, 1, 1], [0, 1, 1, 1, 0], 2)
    assert cm.tolist() == [[1, 1], [1, 2]]
    precision, recall, f1 = per_class_report(cm)
    assert np.allclose(precision, [0.5, 2 / 3]) and np.allclose(recall, [0.5, 2 / 3])
    assert np.allclose(f1, [0.5, 2 / 3])


def test_tfidf_max_features_and_min_df():
    docs = ["refund refund payment", "refund invoice", "payment refund typoo"]
    v = TfidfVectorizer(min_df=2, max_features=1).fit(docs)
    assert list(v.index) == ["refund"]
    assert TfidfVectorizer(min_df=2).fit(docs).transform(docs).shape == (3, 2)


@pytest.mark.parametrize("arch", ["transformer", "meanpool"])
def test_build_model_shapes_and_padding_invariance(arch):
    torch.manual_seed(0)
    m = build_model(arch, 50, 4, 12).eval()
    a = torch.tensor([[2, 3, 4] + [0] * 9])
    assert m(a).shape == (1, 4)
    assert torch.allclose(m(a), m(torch.tensor([[2, 3, 4] + [0] * 9])), atol=1e-6)
    with pytest.raises(ValueError):
        build_model("lstm", 50, 4, 12)


def test_all_padding_input_does_not_produce_nan():
    m = build_model("transformer", 50, 4, 12).eval()
    out = m(torch.zeros(2, 12, dtype=torch.long))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("text", ["", "   ", "<p></p>", "!!! ???"])
def test_service_reports_unknown_for_empty_text(artifacts, text):
    from supportlens.agent import TicketService

    svc = TicketService(artifacts)
    assert svc.classify(text) == {
        "category": "unknown",
        "confidence": 0.0,
        "probabilities": {},
    }
    assert svc.similar(text) == []


def test_service_normal_ticket_has_valid_probabilities(artifacts):
    from supportlens.agent import TicketService

    c = TicketService(artifacts).classify("I was charged twice for my subscription")
    assert (
        abs(sum(c["probabilities"].values()) - 1.0) < 0.01 and 0 < c["confidence"] <= 1
    )


def test_langchain_tools_return_json_and_clamp_k(artifacts):
    import json

    from supportlens.agent import TicketService, build_tools

    tools = {t.name: t for t in build_tools(TicketService(artifacts))}
    assert set(tools) == {"classify_ticket", "find_similar_tickets", "ticket_stats"}
    assert "category" in json.loads(
        tools["classify_ticket"].invoke({"text": "my refund is late"})
    )
    assert (
        len(
            json.loads(
                tools["find_similar_tickets"].invoke(
                    {"text": "refund payment", "k": 1000}
                )
            )
        )
        <= 5
    )
    assert json.loads(tools["ticket_stats"].invoke({}))["tickets"] > 0


def test_agents_fall_back_offline_without_api_key(artifacts, monkeypatch):
    from supportlens import agent

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert "Category:" in agent.answer("where is my parcel", artifacts)
    pytest.importorskip("crewai")
    from supportlens import crew_agent

    out = crew_agent.answer("where is my parcel", artifacts)
    assert "offline fallback" in out and "Category:" in out
