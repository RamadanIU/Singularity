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


def test_model_from_packed_gives_same_logits():
    """Упакованные троичные веса загружаются обратно и дают те же логиты (остальное — float16)."""
    torch.manual_seed(7)
    cfg = E.make_config("tiny", vocab_size=50)
    model = E.SLRNetLanguageModel(cfg).eval()
    with torch.no_grad():   # «обученные» веса с заметным разбросом, а не только инициализация
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    restored = E.SLRNetLanguageModel.from_packed(cfg, model.export_packed()).eval()
    # Троичные матрицы совпадают в точности
    for (n1, m1), (_, m2) in zip(model.named_modules(), restored.named_modules()):
        if isinstance(m1, E.BitLinear158):
            t1, g1 = m1.export_ternary()
            t2, g2 = m2.export_ternary()
            assert np.array_equal(t1, t2), n1
            assert abs(g1 - g2) < 1e-4 * max(g1, 1e-6), n1
    x = torch.randint(0, 50, (2, 40))
    with torch.no_grad():
        l1, _, _ = model(x, routing="all")
        l2, _, _ = restored(x, routing="all")
    assert (l1 - l2).abs().max() < 0.05 * l1.abs().max()
    assert (l1.argmax(-1) == l2.argmax(-1)).float().mean() > 0.9


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


# ------------------------------------------------------------------ без памяти фактов и сжатый эмбеддинг

@pytest.mark.parametrize("bits", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("n", [1, 7, 64])
def test_pack_int_matrix_roundtrip(bits, n):
    rng = np.random.default_rng(bits * 100 + n)
    qmax = 2 ** (bits - 1) - 1
    q = rng.integers(-qmax, qmax + 1, size=(3, n)).astype(np.int8)
    data = E.pack_int_matrix(q, bits)
    assert data.dtype == np.uint8
    assert data.size == (math.ceil(q.size / 2) if bits <= 4 else q.size)   # до 4 бит — два числа в байте
    assert np.array_equal(E.unpack_int_matrix(data, q.shape, bits), q)


def test_group_quantize_is_idempotent_and_ste_passes_gradient():
    torch.manual_seed(8)
    w = (torch.randn(11, 32) * 0.05).requires_grad_()
    q, scale = E.group_quantize(w.detach(), 4, 8)
    assert q.abs().max() <= 7 and scale.shape == (11, 4)
    assert torch.equal(scale, scale.half().float())                       # масштаб уже такой, как в файле
    w2 = (q.reshape(11, 4, 8) * scale[..., None]).reshape(11, 32)
    q2, scale2 = E.group_quantize(w2, 4, 8)
    assert torch.equal(q, q2) and torch.equal(scale, scale2)               # после загрузки из файла — те же числа
    assert (w2 - w.detach()).abs().max() <= scale.max() / 2 + 1e-7
    fq = E.fake_quant_groups(w, 4, 8)
    assert torch.allclose(fq.detach(), w2)
    (fq * torch.arange(32.0)).sum().backward()                             # STE: градиент идёт к мастер-весам как есть
    assert torch.allclose(w.grad, torch.arange(32.0).expand(11, 32))


@pytest.mark.parametrize("router_bias", [5.0, -5.0])
def test_model_without_memory_with_compressed_embedding_steps_like_parallel(router_bias):
    torch.manual_seed(9)
    cfg = E.make_config("tiny", vocab_size=97, window_size=6, ssm_chunk=4, ternary=False,
                        memory_keys=0, embed_bits=4, embed_group=32)
    model = E.SLRNetLanguageModel(cfg).double().eval()
    assert model.core.memory is None and model.num_params()["memory_values"] == 0
    assert not any("memory" in k for k in model.state_dict())
    with torch.no_grad():
        model.core.router.bias.fill_(router_bias)
    ids = torch.randint(0, cfg.vocab_size, (2, 21))
    with torch.no_grad():
        par, _, _ = model(ids, routing="threshold")
        cache = model.init_cache(2)
        steps = torch.stack([model.step(ids[:, t], cache) for t in range(ids.shape[1])], 1)
    assert torch.allclose(par, steps, atol=1e-6), (par - steps).abs().max()
    # и вход, и выход (связанный с эмбеддингом) считаются по округлённой таблице
    w = model.embedding_weight()
    assert len(torch.unique(model.tok_embeddings.weight)) > 1000 and not torch.equal(w, model.tok_embeddings.weight)
    q, _ = E.group_quantize(w, 4, 32)
    assert torch.equal(E.group_quantize(model.tok_embeddings.weight, 4, 32)[0], q)


def test_compressed_embedding_trains_and_exports_exactly():
    """Модель с 4-битным эмбеддингом учится (градиент доходит до эмбеддинга), а в файл попадают ровно те
    числа, с которыми она работала: после загрузки эмбеддинг совпадает бит в бит."""
    torch.manual_seed(10)
    cfg = E.make_config("tiny", vocab_size=60, memory_keys=0, embed_bits=4, embed_group=32)
    model = E.SLRNetLanguageModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ids = torch.randint(0, 60, (4, 33))
    losses = []
    for _ in range(25):
        _, loss, _ = model(ids[:, :-1], ids[:, 1:])
        opt.zero_grad()
        loss.backward()
        assert float(model.tok_embeddings.weight.grad.abs().sum()) > 0
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.8
    model.eval()
    packed = model.export_packed()
    assert "tok_embeddings.weight" not in packed and "tok_embeddings.weight.qdata" in packed
    assert packed["tok_embeddings.weight.qdata"].nbytes == 60 * cfg.d_model // 2          # полбайта на число
    restored = E.SLRNetLanguageModel.from_packed(cfg, packed).eval()
    with torch.no_grad():
        assert torch.equal(restored.embedding_weight(), model.embedding_weight())
        assert torch.equal(restored.embedding_weight(), restored.tok_embeddings.weight)
        x = torch.randint(0, 60, (2, 30))
        l1, l2 = model(x, routing="all")[0], restored(x, routing="all")[0]
    assert (l1.argmax(-1) == l2.argmax(-1)).float().mean() > 0.9


def test_mini2_redistributes_the_file_from_tables_to_computation():
    """mini2 весит столько же, сколько mini, но почти весь файл — вычисляющие слои, а внимания больше."""
    old, new = (E.calculate_hardware_footprint(E.make_config(p, vocab_size=8194)) for p in ("mini", "mini2"))
    cfg = E.make_config("mini2")
    assert cfg.memory_keys == 0 and cfg.embed_bits == 4
    assert new["packed_memory_mb"] == 0
    bits_per_number = new["packed_embedding_mb"] * 1024 ** 2 * 8 / (new["embedding_params_millions"] * 1e6)
    assert bits_per_number < 4.3                                           # было 16 бит на число
    assert new["packed_embedding_mb"] < 0.45 * old["packed_embedding_mb"]  # хотя векторы шире (512 вместо 320)
    assert new["packed_model_mb"] <= old["packed_model_mb"]
    assert new["packed_ternary_mb"] > 3 * old["packed_ternary_mb"]
    assert new["packed_ternary_mb"] > 0.6 * new["packed_model_mb"]
    old_cfg = E.make_config("mini")
    n_attn = lambda c: (E.layer_kinds(c.n_encoder_layers, c.encoder_attn_every)
                        + E.layer_kinds(c.n_decoder_layers, c.decoder_attn_every)).count("attn")
    assert n_attn(cfg) > n_attn(old_cfg)
    assert cfg.n_kv_heads == cfg.n_heads > old_cfg.n_kv_heads                 # у каждой головы свои ключи
    assert cfg.window_size >= 2 * old_cfg.window_size
