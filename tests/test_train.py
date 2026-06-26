"""model/train.py の単体テスト（torch 必須なので無ければ skip）。"""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model.dataset import N_FEATURES  # noqa: E402
from model.train import (  # noqa: E402
    NormalizedModel,
    classification_metrics,
    compute_class_weights,
    confusion_matrix,
    train_and_export,
    weighted_cross_entropy,
)
from model.transformer import AevumTransformer, TransformerConfig  # noqa: E402

FIVE_MIN = np.timedelta64(5, "m")
T0 = np.datetime64("2026-01-01T00:00")


# --------------------------------------------------------------------------- #
# クラス重み・損失・指標
# --------------------------------------------------------------------------- #
def test_compute_class_weights_inverse_freq():
    w = compute_class_weights([0, 0, 0, 1], n_classes=3)  # counts=[3,1,1(空→1)]
    np.testing.assert_allclose(w, [5 / 9, 5 / 3, 5 / 3], rtol=1e-6)


def test_weighted_ce_zero_sample_weight_ignored():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    targets = torch.tensor([0, 0])
    sw = torch.tensor([1.0, 0.0])  # 2件目は無視
    loss = weighted_cross_entropy(logits, targets, sw)
    ref = torch.nn.functional.cross_entropy(logits[:1], targets[:1])
    torch.testing.assert_close(loss, ref)


def test_confusion_and_metrics():
    y_true = [0, 1, 2, 0]
    y_pred = [0, 1, 2, 1]
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[1, 1] == 1 and cm[2, 2] == 1
    met = classification_metrics(y_true, y_pred, 3)
    assert met["accuracy"] == pytest.approx(0.75)
    assert met["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1.0) / 3, rel=1e-6)


def test_metrics_empty():
    met = classification_metrics([], [], 3)
    assert np.isnan(met["accuracy"]) and met["confusion"] == []


# --------------------------------------------------------------------------- #
# NormalizedModel（正規化同梱）
# --------------------------------------------------------------------------- #
def _small_cfg():
    return TransformerConfig(seq_len=8, d_model=16, n_heads=2, n_layers=1, dim_ff=32)


def test_normalized_model_matches_manual():
    torch.manual_seed(0)
    cfg = _small_cfg()
    inner = AevumTransformer(cfg).eval()
    mean = np.arange(N_FEATURES, dtype=np.float64)
    std = np.full(N_FEATURES, 2.0)
    wrapped = NormalizedModel(inner, mean, std, clip=5.0).eval()
    x = torch.randn(4, cfg.seq_len, N_FEATURES) * 3.0
    with torch.no_grad():
        z = torch.clamp((x - torch.tensor(mean, dtype=torch.float32)) / torch.tensor(std, dtype=torch.float32), -5, 5)
        ref = inner(z)
        got = wrapped(x)
    torch.testing.assert_close(got, ref)


def test_normalized_model_onnx_parity(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from model.train import export_onnx

    torch.manual_seed(1)
    cfg = _small_cfg()
    wrapped = NormalizedModel(AevumTransformer(cfg), np.zeros(N_FEATURES), np.full(N_FEATURES, 3.0), 5.0).eval()
    x = torch.randn(2, cfg.seq_len, N_FEATURES)
    with torch.no_grad():
        ref = wrapped(x).numpy()
    path = tmp_path / "m.onnx"
    export_onnx(wrapped, path, seq_len=cfg.seq_len, n_features=N_FEATURES)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"features": x.numpy()})[0]
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-4)


# --------------------------------------------------------------------------- #
# 統合スモーク: 微小データで WF + 全データ再学習 + ONNX 出力
# --------------------------------------------------------------------------- #
def _synthetic(n=300, seq_len=8):
    rng = np.random.default_rng(0)
    feats, times, events = {}, {}, {}
    for s in ("BTC", "ETH"):
        feats[s] = rng.normal(0, 1, (n, N_FEATURES))
        times[s] = T0 + np.arange(n) * FIVE_MIN
        e = np.arange(seq_len, n - 40, 3)
        events[s] = {
            "e_idx": e, "t1": times[s][e] + 48 * FIVE_MIN,
            "label": rng.integers(-1, 2, e.size), "weight": np.full(e.size, 1.0),
        }
    return feats, times, events


def test_train_and_export_smoke(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    feats, times, events = _synthetic()
    cfg = _small_cfg()
    wf = train_and_export(
        feats, times, events, tmp_path,
        config=cfg, seq_len=cfg.seq_len, n_splits=2, embargo_bars=8, epochs=1, device="cpu",
    )
    assert isinstance(wf, list) and len(wf) >= 1
    assert (tmp_path / "model.onnx").is_file()
    assert (tmp_path / "model.pt").is_file()

    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["feature_names"][:1] == ["ret_1"]
    assert len(meta["normalizer"]["mean"]) == N_FEATURES
    assert len(meta["normalizer"]["std"]) == N_FEATURES
    assert meta["seq_len"] == cfg.seq_len

    # ONNX は生特徴量を受けて (B,3) を返す
    sess = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    out = sess.run(["logits"], {"features": np.random.randn(5, cfg.seq_len, N_FEATURES).astype(np.float32)})[0]
    assert out.shape == (5, 3)
