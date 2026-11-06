# -*- coding: utf-8 -*-
"""
Проверки новых частей Singularity (старший брат mini2, PLAN_SINGULARITY_300M_RU.md):
Gated DeltaNet, глобальное внимание, затвор внимания и QK-norm, ядро мышления 2.0, пресеты s300.
Как и в test_engine.py, быстрые реализации сверяются с наивными эталонами.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slr_net_engine as E  # noqa: E402

torch = pytest.importorskip("torch")
F = torch.nn.functional

ALL_NEW = dict(memory_keys=0, ssm_type="gdn", gdn_head_dim=32, attn_gate=True, qk_norm=True,
               global_attn_every=2, global_kv_heads=1, core_inject=True, core_min_steps=1)


def naive_gated_delta(q, k, v, g, beta, S=None):
    """Цикл по времени: тот же шаг, что в пошаговой генерации."""
    b, h, l, dk = k.shape
    S = torch.zeros(b, h, dk, v.shape[-1], dtype=torch.float64) if S is None else S.clone()
    outs = []
    for t in range(l):
        o, S = E.gated_delta_step(S, q[:, :, t], k[:, :, t], v[:, :, t], g[:, :, t], beta[:, :, t])
        outs.append(o)
    return torch.stack(outs, 2), S


def random_gdn_inputs(seed, b=2, h=3, l=19, dk=8, dv=6):
    gen = torch.Generator().manual_seed(seed)
    q = E.l2norm(torch.randn(b, h, l, dk, generator=gen, dtype=torch.float64)) * dk ** -0.5
    k = E.l2norm(torch.randn(b, h, l, dk, generator=gen, dtype=torch.float64))
    v = torch.randn(b, h, l, dv, generator=gen, dtype=torch.float64)
    g = -torch.rand(b, h, l, generator=gen, dtype=torch.float64)
    beta = torch.rand(b, h, l, generator=gen, dtype=torch.float64)
    return q, k, v, g, beta


@pytest.mark.parametrize("l,chunk", [(1, 4), (7, 4), (16, 4), (19, 8), (5, 64), (33, 16)])
def test_gated_delta_chunked_matches_recurrence(l, chunk):
    q, k, v, g, beta = random_gdn_inputs(l * 7 + chunk, l=l)
    S0 = torch.randn(2, 3, 8, 6, dtype=torch.float64) * 0.3
    for init in (None, S0):
        ref_o, ref_S = naive_gated_delta(q, k, v, g, beta, init)
        o, S = E.gated_delta_rule_chunked(q, k, v, g, beta, chunk, initial_state=init)
        assert torch.allclose(o.double(), ref_o, atol=1e-4), (o.double() - ref_o).abs().max()
        assert torch.allclose(S.double(), ref_S, atol=1e-4)


def test_delta_rule_overwrites_old_value_under_same_key():
    """Главное отличие от Mamba-2: при beta = 1 новое значение под тем же ключом заменяет старое."""
    k = E.l2norm(torch.randn(1, 1, 1, 4, dtype=torch.float64))
    S = torch.zeros(1, 1, 4, 3, dtype=torch.float64)
    one, zero = torch.ones(1, 1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64)
    v_old, v_new = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64), \
        torch.tensor([[[-5.0, 0.5, 7.0]]], dtype=torch.float64)
    _, S = E.gated_delta_step(S, k[:, :, 0], k[:, :, 0], v_old, zero, one)
    o, S = E.gated_delta_step(S, k[:, :, 0], k[:, :, 0], v_new, zero, one)
    assert torch.allclose(o, v_new)           # вспоминается только новое, старое не «смазано» с ним


@pytest.mark.parametrize("router_bias", [5.0, -5.0, 0.0])
def test_singularity_model_steps_like_parallel(router_bias):
    """Пошаговая генерация (Gated DeltaNet, глобальный кэш, затвор, ядро 2.0 с лишними кругами при ответе)
    даёт те же логиты, что и параллельный проход."""
    torch.manual_seed(11)
    cfg = E.make_config("tiny", vocab_size=97, window_size=6, ssm_chunk=4, ternary=False,
                        core_infer_steps=5, **ALL_NEW)
    enc, dec = E.model_layer_kinds(cfg)
    assert "gattn" in enc + dec and "attn" in enc + dec
    model = E.SLRNetLanguageModel(cfg).double().eval()
    assert isinstance(model.core.layers[0], E.GatedDeltaNetBlock)
    with torch.no_grad():
        model.core.router.bias.fill_(router_bias)
        model.core.inject_gate.fill_(0.7)     # подмешивание входа включено по-настоящему
    ids = torch.randint(0, cfg.vocab_size, (2, 23))
    with torch.no_grad():
        par, _, stats = model(ids, routing="threshold")
        cache = model.init_cache(2)
        steps = torch.stack([model.step(ids[:, t], cache) for t in range(ids.shape[1])], 1)
    assert torch.allclose(par, steps, atol=1e-6), (par - steps).abs().max()
    if router_bias > 0:
        assert float(stats["core_depth_mean"]) == 5.0           # больше кругов, чем max_latent_steps = 3


def test_global_attention_is_full_causal_without_rope():
    torch.manual_seed(12)
    cfg = E.make_config("tiny", window_size=4, ternary=False, global_kv_heads=1, qk_norm=True)
    blk = E.LocalAttentionBlock(cfg, is_global=True).double()
    x = torch.randn(2, 19, cfg.d_model, dtype=torch.float64)
    h = blk.norm(x)
    q, k, v = blk.qkv(h).split([cfg.n_heads * cfg.head_dim, cfg.head_dim, cfg.head_dim], dim=-1)
    q = blk.q_norm(q.view(2, 19, cfg.n_heads, -1).transpose(1, 2))
    k = blk.k_norm(k.view(2, 19, 1, -1).transpose(1, 2)).expand(-1, cfg.n_heads, -1, -1)
    v = v.view(2, 19, 1, -1).transpose(1, 2).expand(-1, cfg.n_heads, -1, -1)
    att = F.scaled_dot_product_attention(q, k, v, is_causal=True)           # без окна и без RoPE
    ref = x + blk.o_proj(att.transpose(1, 2).reshape(2, 19, -1))
    ref = blk.mlp(ref)
    assert torch.allclose(blk(x), ref, atol=1e-10)
    # первый токен виден последнему, хотя окно локальных слоёв — 4 токена
    x2 = x.clone()
    x2[:, 0] += 1.0
    assert (blk(x2)[:, -1] - blk(x)[:, -1]).abs().max() > 1e-6


def test_attention_gate_trains_and_starts_half_open():
    torch.manual_seed(13)
    cfg = E.make_config("tiny", vocab_size=50, attn_gate=True, ternary=False)
    model = E.SLRNetLanguageModel(cfg)
    blk = next(m for m in model.modules() if isinstance(m, E.LocalAttentionBlock))
    h = torch.randn(3, 5, cfg.d_model)
    with torch.no_grad():
        gate = torch.sigmoid(blk.gate_proj(h))
    assert 0.3 < float(gate.mean()) < 0.7
    ids = torch.randint(0, 50, (2, 17))
    _, loss, _ = model(ids[:, :-1], ids[:, 1:])
    loss.backward()
    assert float(blk.gate_proj.weight.grad.abs().sum()) > 0


def test_core_random_depth_and_truncated_backprop():
    torch.manual_seed(14)
    cfg = E.make_config("tiny", vocab_size=50, ternary=False, max_latent_steps=4, core_min_steps=1,
                        core_inject=True, memory_keys=0)
    model = E.SLRNetLanguageModel(cfg).train()
    seen = {model.core._num_steps("topk") for _ in range(200)}
    assert seen == {1, 2, 3, 4}
    model.eval()
    assert model.core._num_steps("topk") == 4 and model.core._num_steps("threshold") == 4

    # Усечённый градиент меняет только градиенты, но не сам расчёт
    ids = torch.randint(0, 50, (2, 21))
    grads = {}
    for bptt in (0, 1):
        torch.manual_seed(15)
        cfg_b = E.make_config("tiny", vocab_size=50, ternary=False, max_latent_steps=3, core_min_steps=3,
                              core_inject=True, memory_keys=0, core_bptt=bptt)
        m = E.SLRNetLanguageModel(cfg_b).train()
        _, loss, _ = m(ids[:, :-1], ids[:, 1:])
        loss.backward()
        grads[bptt] = (float(loss.detach()), m.encoder[0].in_proj.weight.grad.clone(), m.core.inject_gate.grad.clone())
    assert abs(grads[0][0] - grads[1][0]) < 1e-5
    assert float(grads[1][1].abs().sum()) > 0                    # прелюдия учится и с усечённым градиентом
    assert not torch.allclose(grads[0][1], grads[1][1])
    assert grads[1][2][0] == 0 and grads[1][2][-1] != 0          # ранние круги — без градиента


def test_singularity_export_roundtrip_gives_same_logits():
    torch.manual_seed(16)
    cfg = E.make_config("tiny", vocab_size=50, embed_bits=4, embed_group=32, **ALL_NEW)
    model = E.SLRNetLanguageModel(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    packed = model.export_packed()
    assert "core.layers.0.in_proj.weight.packed" in packed            # Gated DeltaNet — троичный
    assert any(k.endswith("gate_proj.weight.packed") for k in packed)  # затвор внимания — тоже
    restored = E.SLRNetLanguageModel.from_packed(cfg, packed).eval()
    for (n1, m1), (_, m2) in zip(model.named_modules(), restored.named_modules()):
        if isinstance(m1, E.BitLinear158):
            assert np.array_equal(m1.export_ternary()[0], m2.export_ternary()[0]), n1
    x = torch.randint(0, 50, (2, 40))
    with torch.no_grad():
        l1, l2 = model(x, routing="all")[0], restored(x, routing="all")[0]
    assert (l1.argmax(-1) == l2.argmax(-1)).float().mean() > 0.9


def test_s300_is_mini2_scaled_with_proportions_kept():
    """s300 — mini2 с шириной ×2, глубиной ×2, словарём ×4; около 300 млн параметров и ~65 МБ файла."""
    small, big = E.make_config("mini2"), E.make_config("s300")
    assert big.d_model == 2 * small.d_model
    for name in ("n_encoder_layers", "n_reasoning_layers", "n_decoder_layers", "n_heads"):
        assert getattr(big, name) == 2 * getattr(small, name), name
    assert big.encoder_attn_every == small.encoder_attn_every == 2          # внимание через слой
    assert big.vocab_size == 4 * small.vocab_size and big.embed_bits == 4 and big.memory_keys == 0
    enc, dec = E.model_layer_kinds(big)
    assert (enc + dec).count("gattn") == 3 and (enc + dec).count("attn") == 9
    fp = E.calculate_hardware_footprint(big)
    assert 260 < fp["total_params_millions"] < 320
    assert fp["packed_model_mb"] < 80
    assert fp["packed_ternary_mb"] > 0.7 * fp["packed_model_mb"]           # байты — на вычисления
    # разговор на 32 тыс. токенов: глобальный кэш < 100 МБ
    assert fp["global_cache_mb_per_1k_tokens"] * 32.768 < 100
    proxy = E.calculate_hardware_footprint(E.make_config("s300proxy"))
    assert 50 < proxy["total_params_millions"] < 100


def test_old_presets_are_untouched_by_new_options():
    """Новые настройки по умолчанию выключены: mini и mini2 строятся из тех же весов, что и раньше."""
    for preset in ("mini", "mini2"):
        cfg = E.make_config(preset)
        assert cfg.ssm_type == "mamba2" and not cfg.attn_gate and not cfg.qk_norm
        assert cfg.global_attn_every == 0 and not cfg.core_inject and cfg.infer_steps == cfg.max_latent_steps
        with torch.device("meta"):
            model = E.SLRNetLanguageModel(cfg)
        names = set(model.state_dict())
        assert not any(("gate_proj" in n or "q_norm" in n or "inject" in n or "ab_proj" in n) for n in names)


def test_ternary_quantization_can_be_switched_on_during_training():
    """Разгон без квантования (train_lm.py --fp-warmup-steps): тот же расчёт, что у модели с ternary=False."""
    torch.manual_seed(17)
    cfg = E.make_config("tiny", vocab_size=50, **ALL_NEW)
    model = E.SLRNetLanguageModel(cfg).eval()
    plain = E.SLRNetLanguageModel(E.make_config("tiny", vocab_size=50, ternary=False, **ALL_NEW)).eval()
    plain.load_state_dict(model.state_dict())
    x = torch.randint(0, 50, (2, 30))
    with torch.no_grad():
        quantized = model(x, routing="all")[0]
        E.set_ternary_quantization(model, False)
        assert torch.allclose(model(x, routing="all")[0], plain(x, routing="all")[0], atol=1e-6)
        E.set_ternary_quantization(model, True)
        assert torch.allclose(model(x, routing="all")[0], quantized, atol=1e-6)
