# -*- coding: utf-8 -*-
"""Проверки движка SLR-Net: быстрые реализации сверяются с наивными эталонами."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slr_net_engine as E  # noqa: E402

torch = pytest.importorskip("torch")
F = torch.nn.functional


def naive_ssm(X, A, B, C):
    """h_t = exp(A_t) h_{t-1} + X_t ⊗ B_t ; y_t = h_t · C_t  — цикл по времени."""
    b, l, h, p = X.shape
    n = B.shape[-1]
    state = torch.zeros(b, h, p, n, dtype=torch.float64)
    ys = []
    for t in range(l):
        state = state * torch.exp(A[:, t])[..., None, None] + X[:, t][..., None] * B[:, t][:, None, None, :]
        ys.append((state * C[:, t][:, None, None, :]).sum(-1))
    return torch.stack(ys, 1), state


@pytest.mark.parametrize("l,chunk", [(1, 4), (7, 4), (16, 4), (19, 8), (5, 64)])
def test_ssd_scan_matches_recurrence(l, chunk):
    g = torch.Generator().manual_seed(l * 31 + chunk)
    b, h, p, n = 2, 3, 4, 5
    X = torch.randn(b, l, h, p, generator=g, dtype=torch.float64)
    A = -torch.rand(b, l, h, generator=g, dtype=torch.float64)
    B = torch.randn(b, l, n, generator=g, dtype=torch.float64)
    C = torch.randn(b, l, n, generator=g, dtype=torch.float64)
    y_ref, s_ref = naive_ssm(X, A, B, C)
    y, s = E.ssd_scan(X, A, B, C, chunk)
    assert torch.allclose(y.double(), y_ref, atol=1e-4)
    assert torch.allclose(s.double(), s_ref, atol=1e-4)


def test_local_attention_matches_dense_band_mask():
    torch.manual_seed(0)
    cfg = E.make_config("tiny", window_size=8, ternary=False)
    blk = E.LocalAttentionBlock(cfg)
    b, h, l, d = 2, cfg.n_heads, 29, cfg.head_dim
    q, k, v = (torch.randn(b, h, l, d) for _ in range(3))
    i = torch.arange(l)[:, None]
    j = torch.arange(l)[None, :]
    mask = (j <= i) & (i - j < cfg.window_size)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(blk._local_attention(q, k, v), ref, atol=1e-5)


def test_product_key_lookup_is_exact_topk():
    torch.manual_seed(0)
    cfg = E.make_config("tiny", ternary=False)
    mem = E.ProductKeyMemory(cfg)
    x = torch.randn(10, cfg.d_model)
    idx, _ = mem.lookup(x)
    q = mem.query(x)
    s1 = q[:, :mem.half] @ mem.keys[0].t()
    s2 = q[:, mem.half:] @ mem.keys[1].t()
    full = (s1[:, :, None] + s2[:, None, :]).flatten(1)
    ref = full.topk(cfg.memory_top_k, dim=-1).indices
    assert torch.equal(idx.sort(-1).values, ref.sort(-1).values)


@pytest.mark.parametrize("router_bias", [0.0, 5.0, -5.0])
def test_step_generation_matches_parallel_forward(router_bias):
    """Пошаговая генерация с кэшем (O(1) на токен) должна давать те же логиты, что и параллельный проход."""
    torch.manual_seed(1)
    cfg = E.make_config("tiny", vocab_size=97, window_size=6, ssm_chunk=4, ternary=False)
    model = E.SLRNetLanguageModel(cfg).double().eval()
    with torch.no_grad():
        model.core.router.bias.fill_(router_bias)
    ids = torch.randint(0, cfg.vocab_size, (2, 21))
    with torch.no_grad():
        par, _, _ = model(ids, routing="threshold")
        cache = model.init_cache(2)
        steps = torch.stack([model.step(ids[:, t], cache) for t in range(ids.shape[1])], 1)
    assert torch.allclose(par, steps, atol=1e-6), (par - steps).abs().max()


def test_causality_in_inference_mode():
    torch.manual_seed(2)
    model = E.SLRNetLanguageModel(E.make_config("tiny", vocab_size=64)).eval()
    a = torch.randint(0, 64, (1, 70))
    b = a.clone()
    b[0, 50] = (b[0, 50] + 1) % 64
    with torch.no_grad():
        diff = (model(a)[0][0, :50] - model(b)[0][0, :50]).abs().max()
    assert float(diff) == 0.0


def test_training_step_reduces_loss_and_router_gets_gradient():
    torch.manual_seed(3)
    cfg = E.make_config("tiny", vocab_size=50)
    model = E.SLRNetLanguageModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ids = torch.randint(0, 50, (4, 33))
    first = None
    for _ in range(30):
        _, loss, _ = model(ids[:, :-1], ids[:, 1:])
        opt.zero_grad()
        loss.backward()
        assert model.core.router.weight.grad is not None
        assert float(model.core.router.weight.grad.abs().sum()) > 0
        opt.step()
        first = first if first is not None else float(loss.detach())
    assert float(loss.detach()) < first * 0.8
    assert abs(first - math.log(50)) < 1.0, "стартовый лосс должен быть около ln(V)"


def test_bitlinear_ste_gradient_is_identity_wrt_effective_weight():
    torch.manual_seed(4)
    layer = E.BitLinear158(16, 8, use_norm=False)
    x = torch.randn(5, 16)
    layer(x).sum().backward()
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True)
    x_q = (x * scale).round().clamp(-128, 127) / scale
    expected = torch.ones(5, 8).t() @ x_q
    assert torch.allclose(layer.weight.grad, expected, atol=1e-5)


@pytest.mark.parametrize("preset", list(E.PRESETS))
def test_footprint_formula_matches_real_parameter_count(preset):
    cfg = E.make_config(preset)
    with torch.device("meta"):
        model = E.SLRNetLanguageModel(cfg)
    real = model.num_params()["total"]
    est = E.calculate_hardware_footprint(cfg)["total_params_millions"] * 1e6
    assert abs(real - est) < 1


def test_export_packed_roundtrip():
    torch.manual_seed(5)
    model = E.SLRNetLanguageModel(E.make_config("tiny", vocab_size=40))
    packed = model.export_packed()
    name = "encoder.0.in_proj"
    trits, gamma = model.encoder[0].in_proj.export_ternary()
    unpacked = E.unpack_ternary_matrix(packed[f"{name}.weight.packed"], tuple(packed[f"{name}.weight.shape"]))
    assert np.array_equal(unpacked, trits)
    assert abs(float(packed[f"{name}.weight.gamma"][0]) - gamma) < 1e-6


def test_numpy_prototype_gradients_are_exact():
    proto = E.NumpySLRNetPrototype(vocab_size=20, d_model=12, memory_slots=8, quantize=False,
                                   dtype=np.float64, seed=11)
    tokens = np.array([1, 5, 3, 7, 2, 9, 4])
    targets = np.array([5, 3, 7, 2, 9, 4, 1])
    hints = np.array([0, 1, 0, 1, 1, 0, 0], dtype=np.float64)
    assert proto.gradient_check(tokens, targets, hints, n_checks=8) < 1e-4


def test_generate_runs_and_respects_length():
    torch.manual_seed(6)
    model = E.SLRNetLanguageModel(E.make_config("tiny", vocab_size=30))
    out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=7, temperature=0.9, top_k=10)
    assert out.shape == (1, 10)
