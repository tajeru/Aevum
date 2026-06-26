"""model/transformer.py の単体テスト（torch 必須なので無ければ skip）。"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.transformer import AevumTransformer, TransformerConfig  # noqa: E402
from shared.feature_names import N_FEATURES  # noqa: E402


def _model():
    torch.manual_seed(0)
    return AevumTransformer()


def test_default_config_is_small():
    cfg = TransformerConfig()
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.dim_ff) == (64, 2, 4, 256)
    assert cfg.n_features == N_FEATURES and cfg.seq_len == 128 and cfg.n_classes == 3


def test_forward_shape():
    m = _model().eval()
    x = torch.randn(8, 128, N_FEATURES)
    out = m(x)
    assert out.shape == (8, 3)


def test_variable_batch():
    m = _model().eval()
    for b in (1, 4, 16):
        assert m(torch.randn(b, 128, N_FEATURES)).shape == (b, 3)


def test_deterministic_in_eval():
    m = _model().eval()
    x = torch.randn(3, 128, N_FEATURES)
    with torch.no_grad():
        a, b = m(x), m(x)
    torch.testing.assert_close(a, b)


def test_pos_emb_shape_and_params():
    m = _model()
    assert tuple(m.pos_emb.shape) == (1, 128, 64)
    assert m.num_parameters() > 0


def test_gradients_flow():
    m = _model().train()
    x = torch.randn(8, 128, N_FEATURES)
    y = torch.randint(0, 3, (8,))
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_d_model_must_divide_heads():
    with pytest.raises(ValueError):
        AevumTransformer(TransformerConfig(d_model=65, n_heads=4))


def test_onnx_export_runs(tmp_path):
    m = _model().eval()
    dummy = torch.randn(1, 128, N_FEATURES)
    path = tmp_path / "model.onnx"
    torch.onnx.export(
        m, dummy, str(path),
        input_names=["x"], output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    assert path.is_file() and path.stat().st_size > 0


def test_onnx_runtime_parity(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    m = _model().eval()
    # batch=1 でエクスポートし batch=3 で実行（動的バッチを検証）
    path = tmp_path / "model.onnx"
    torch.onnx.export(
        m, torch.randn(1, 128, N_FEATURES), str(path),
        input_names=["x"], output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    x = torch.randn(3, 128, N_FEATURES)
    with torch.no_grad():
        ref = m(x).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"x": x.numpy()})[0]
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-4)
