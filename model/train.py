"""model/train.py — 学習 + ONNX エクスポート.

Walk-forward で purged 性能を推定し、その後 全データで最終モデルを再学習して
ONNX へエクスポートする（CLAUDE.md: WF は性能推定、デプロイは全データ再学習）。

確定仕様（user-confirmed）
-------------------------
* 損失   : 重み付き CrossEntropy = sample_weight(AFML一意性) × 逆頻度クラス重み
* 最適化 : AdamW(lr=3e-4, weight_decay=1e-2), batch=256, early stopping(macro-F1)
* 本番   : WF 評価後、全データ（末尾を val に Purge+Embargo）で最終モデルを学習
* ONNX   : 正規化(z-score + ±5σ クリップ)を NormalizedModel としてグラフに同梱。
           Pi は生特徴量をそのまま渡すだけ（正規化の train/live ドリフトが原理的に無い）

成果物: <out_dir>/model.onnx, model.pt, metadata.json（FEATURE_NAMES, seq_len,
ラベル写像, 正規化統計, config, WF 指標を含む）。

純粋な部分（損失 / クラス重み / 指標 / NormalizedModel）は単体テストできる。
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.dataset import (
    BAR_MINUTES,
    CLIP_SIGMA,
    EMBARGO_BARS,
    LABEL_TO_CLASS,
    N_SPLITS,
    SEQ_LEN,
    SequenceDataset,
    ZScoreClipNormalizer,
    build_samples,
    load_from_db,
    make_folds,
)
from model.transformer import AevumTransformer, N_CLASSES, TransformerConfig
from shared.feature_names import FEATURE_NAMES, N_FEATURES

log = logging.getLogger("aevum.train")

LR = 3e-4
WEIGHT_DECAY = 1e-2
BATCH = 256
MAX_EPOCHS = 50
PATIENCE = 5
VAL_FRAC = 0.15


# --------------------------------------------------------------------------- #
# 損失・クラス重み・指標
# --------------------------------------------------------------------------- #
def compute_class_weights(y, n_classes: int = N_CLASSES) -> np.ndarray:
    """逆頻度クラス重み（平均 ≈ 1）。空クラスは頻度1扱い。"""
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    return (counts.sum() / (n_classes * counts)).astype(np.float32)


def weighted_cross_entropy(logits, targets, sample_w, class_w=None):
    """sample_weight × class_weight の重み付き CE（sample_weight 和で正規化）。"""
    ce = F.cross_entropy(logits, targets, weight=class_w, reduction="none")
    denom = sample_w.sum().clamp_min(1e-8)
    return (ce * sample_w).sum() / denom


def confusion_matrix(y_true, y_pred, n_classes: int = N_CLASSES) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    return cm


def classification_metrics(y_true, y_pred, n_classes: int = N_CLASSES) -> dict:
    """accuracy / macro-F1 / クラス別 F1 / 混同行列。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"),
                "f1_per_class": [], "confusion": []}
    cm = confusion_matrix(y_true, y_pred, n_classes)
    acc = float((y_true == y_pred).mean())
    f1s = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s)),
            "f1_per_class": [float(x) for x in f1s], "confusion": cm.tolist()}


# --------------------------------------------------------------------------- #
# 正規化を内包したモデル（ONNX 用）
# --------------------------------------------------------------------------- #
class NormalizedModel(nn.Module):
    """生特徴量を受け、(x-mean)/std + ±clip クリップ後に内部モデルへ通す。

    mean/std は buffer として保持し ONNX に定数として焼き込まれる（自己完結）。
    """

    def __init__(self, model: nn.Module, mean, std, clip: float = CLIP_SIGMA) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.as_tensor(np.asarray(mean), dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.asarray(std), dtype=torch.float32))
        self.clip = float(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / self.std
        z = torch.clamp(z, -self.clip, self.clip)
        return self.model(z)


# --------------------------------------------------------------------------- #
# 学習・評価
# --------------------------------------------------------------------------- #
def evaluate(model: nn.Module, ds, *, batch_size: int = BATCH, device: str = "cpu"):
    """予測クラスを返す: (y_true, y_pred)。"""
    if len(ds) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    model.eval()
    ys, ps = [], []
    loader = DataLoader(ds, batch_size=batch_size)
    with torch.no_grad():
        for xb, yb, _ in loader:
            logits = model(xb.to(device))
            ps.append(logits.argmax(dim=1).cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_model(
    model: nn.Module,
    train_ds,
    val_ds,
    *,
    class_weights=None,
    epochs: int = MAX_EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    batch_size: int = BATCH,
    patience: int = PATIENCE,
    device: str = "cpu",
):
    """重み付き CE + AdamW + early stopping(macro-F1)。best 重みを復元して返す。"""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cw = torch.as_tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_f1, best_state, bad = -float("inf"), None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss = weighted_cross_entropy(model(xb), yb, wb, cw)
            loss.backward()
            opt.step()
        yt, yp = evaluate(model, val_ds, batch_size=batch_size, device=device)
        met = classification_metrics(yt, yp)
        history.append(met)
        f1 = met["macro_f1"]
        if np.isfinite(f1) and f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# --------------------------------------------------------------------------- #
# ONNX / 成果物
# --------------------------------------------------------------------------- #
def export_onnx(normalized_model: nn.Module, path, *, seq_len: int = SEQ_LEN, n_features: int = N_FEATURES) -> None:
    normalized_model.eval()
    dummy = torch.randn(1, seq_len, n_features)
    # dynamo=False（レガシ exporter）: dynamic_axes を正しく反映し動的バッチに対応。
    # 新 exporter は dummy のバッチ次元で reshape を特殊化し別バッチで壊れるため使わない。
    torch.onnx.export(
        normalized_model, dummy, str(path),
        input_names=["features"], output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def save_artifacts(out_dir, raw_model: nn.Module, normalizer: ZScoreClipNormalizer,
                   config: TransformerConfig, wf_metrics: list) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(raw_model.state_dict(), out / "model.pt")
    meta = {
        "feature_names": list(FEATURE_NAMES),
        "seq_len": config.seq_len,
        "n_features": config.n_features,
        "label_to_class": {str(k): v for k, v in LABEL_TO_CLASS.items()},
        "normalizer": {
            "mean": [float(x) for x in normalizer.mean_],
            "std": [float(x) for x in normalizer.std_],
            "clip": normalizer.clip,
        },
        "config": dataclasses.asdict(config),
        "wf_metrics": wf_metrics,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# 統合
# --------------------------------------------------------------------------- #
def train_and_export(
    features_by_sym, times_by_sym, events_by_sym, out_dir, *,
    config: Optional[TransformerConfig] = None,
    seq_len: int = SEQ_LEN, n_splits: int = N_SPLITS, embargo_bars: int = EMBARGO_BARS,
    epochs: int = MAX_EPOCHS, val_frac: float = VAL_FRAC, device: str = "cpu",
) -> list:
    """WF で purged 評価 → 全データで最終モデルを学習 → ONNX/成果物を出力。"""
    cfg = config or TransformerConfig(seq_len=seq_len)

    # --- Walk-forward 評価 ---
    folds = make_folds(features_by_sym, times_by_sym, events_by_sym,
                       seq_len=seq_len, n_splits=n_splits, embargo_bars=embargo_bars)
    wf_metrics = []
    for i, fold in enumerate(folds):
        cw = compute_class_weights(fold.train.y)
        model = AevumTransformer(cfg)
        model, _ = train_model(model, fold.train, fold.test, class_weights=cw, epochs=epochs, device=device)
        yt, yp = evaluate(model, fold.test, device=device)
        met = classification_metrics(yt, yp)
        wf_metrics.append(met)
        log.info("WF fold %d: acc=%.3f macro_f1=%.3f", i, met["accuracy"], met["macro_f1"])

    # --- 全データで最終モデル（末尾 val_frac を val に、Purge+Embargo） ---
    samples = build_samples(features_by_sym, times_by_sym, events_by_sym, seq_len=seq_len)
    order = np.argsort(samples["t0"], kind="stable")
    cut = max(1, int(len(order) * (1.0 - val_frac)))
    train_sel, val_sel = order[:cut], order[cut:]
    embargo = np.timedelta64(int(embargo_bars * BAR_MINUTES), "m")
    if val_sel.size:
        val_start = samples["t0"][val_sel].min()
        train_sel = train_sel[samples["t1"][train_sel] < (val_start - embargo)]
        cutoff = val_start - embargo
    else:
        cutoff = samples["t0"].max() + np.timedelta64(1, "ns")

    train_rows = []
    for sym, Fm in features_by_sym.items():
        tt = np.asarray(times_by_sym[sym], dtype="datetime64[ns]")
        rows = np.asarray(Fm, dtype=np.float64)[tt < cutoff]
        rows = rows[np.isfinite(rows).all(axis=1)] if rows.size else rows
        if rows.size:
            train_rows.append(rows)
    norm = ZScoreClipNormalizer(CLIP_SIGMA).fit(np.concatenate(train_rows, axis=0))
    F_norm = {sym: norm.transform(Fm).astype(np.float32) for sym, Fm in features_by_sym.items()}

    final_train = SequenceDataset(F_norm, samples, train_sel, seq_len)
    final_val = SequenceDataset(F_norm, samples, val_sel if val_sel.size else train_sel, seq_len)
    cw = compute_class_weights(samples["y"][train_sel])
    final = AevumTransformer(cfg)
    final, _ = train_model(final, final_train, final_val, class_weights=cw, epochs=epochs, device=device)

    # --- 正規化を同梱して ONNX 出力 + 成果物保存 ---
    wrapped = NormalizedModel(final, norm.mean_, norm.std_, norm.clip or CLIP_SIGMA)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    export_onnx(wrapped, out / "model.onnx", seq_len=seq_len, n_features=cfg.n_features)
    save_artifacts(out, final, norm, cfg, wf_metrics)
    log.info("artifacts written to %s", out)
    return wf_metrics


async def run(out_dir="artifacts", symbols=("BTC", "ETH"), **params) -> list:
    import asyncpg

    from data.ingestion import resolve_dsn

    pool = await asyncpg.create_pool(resolve_dsn(), min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            feats, times, events = await load_from_db(conn, symbols)
    finally:
        await pool.close()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return train_and_export(feats, times, events, out_dir, device=device, **params)


def main() -> None:
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(run())


if __name__ == "__main__":
    main()
