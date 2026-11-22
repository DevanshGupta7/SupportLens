"""A small Transformer-encoder text classifier in PyTorch."""

from __future__ import annotations

import torch
from torch import nn


class TransformerTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ff: int = 128,
        max_len: int = 40,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, num_classes)
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # ids: (batch, seq)
        real = ids.ne(0)
        attn_pad = ~real
        attn_pad[~real.any(dim=1), 0] = (
            False  # an all-padding row would make softmax NaN; let it see 1 slot
        )
        positions = torch.arange(ids.size(1), device=ids.device)
        h = self.tok(ids) + self.pos(positions)[None]
        h = self.encoder(h, src_key_padding_mask=attn_pad)
        keep = real.unsqueeze(-1).float()  # masked mean-pool over real tokens only
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.head(pooled)  # logits (batch, num_classes)


class MeanPoolClassifier(nn.Module):
    """Ablation baseline: embeddings, masked mean, linear head. No attention, no word order."""

    def __init__(self, vocab_size: int, num_classes: int, d_model: int = 64):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, num_classes)
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        keep = ids.ne(0).unsqueeze(-1).float()
        return self.head((self.tok(ids) * keep).sum(1) / keep.sum(1).clamp(min=1.0))


ARCHITECTURES = ("transformer", "meanpool")


def build_model(
    arch: str, vocab_size: int, num_classes: int, max_len: int
) -> nn.Module:
    if arch == "transformer":
        return TransformerTextClassifier(vocab_size, num_classes, max_len=max_len)
    if arch == "meanpool":
        return MeanPoolClassifier(vocab_size, num_classes)
    raise ValueError(f"unknown arch {arch!r}; choose from {ARCHITECTURES}")
