"""model/transformer.py — Encoder-only Transformer（売買シグナル3クラス分類）.

入力  : (batch, seq_len=128, n_features=58)  ← model/dataset.py の系列窓（正規化済み）
出力  : (batch, 3)  ロジット。クラスは {0,1,2} = label {-1,0,+1}（dataset.LABEL_TO_CLASS）

確定仕様（baseline, user-confirmed）
-----------------------------------
* small: d_model=64, n_layers=2, n_heads=4, dim_ff=256, dropout=0.1
* 位置エンコーディング = 学習可能（位置埋め込みパラメータ）
* 系列集約 = 最終トークン（イベントバー）の表現で分類
* 入力射影 Linear(58→d_model), pre-norm(norm_first=True), 活性化 GELU,
  出力 Linear(d_model→3)

学習は model/train.py（次段）。本モジュールは定義のみで、forward と ONNX
エクスポート可否を単体テストする（推論は ONNX Runtime / Pi）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from shared.feature_names import N_FEATURES

SEQ_LEN: int = 128
N_CLASSES: int = 3


@dataclass(frozen=True)
class TransformerConfig:
    n_features: int = N_FEATURES
    seq_len: int = SEQ_LEN
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_ff: int = 256
    dropout: float = 0.1
    n_classes: int = N_CLASSES


class AevumTransformer(nn.Module):
    """Encoder-only Transformer。学習可能位置埋め込み・最終トークン分類。"""

    def __init__(self, config: TransformerConfig | None = None) -> None:
        super().__init__()
        cfg = config or TransformerConfig()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model({cfg.d_model}) は n_heads({cfg.n_heads}) で割り切れること")
        self.config = cfg

        self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)
        # 学習可能な位置埋め込み (1, seq_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.dropout = nn.Dropout(cfg.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, norm=nn.LayerNorm(cfg.d_model))
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) → logits (batch, n_classes)。"""
        t = x.size(1)
        h = self.input_proj(x) + self.pos_emb[:, :t, :]
        h = self.dropout(h)
        h = self.encoder(h)
        h = h[:, -1, :]              # 最終トークン（イベントバー）
        return self.head(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
