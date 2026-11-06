#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLR-Net (Sparse Latent-Reasoning Network) — эталонная архитектура языковой модели.
==================================================================================

Что внутри (и откуда идеи):

1. Троичные веса BitNet b1.58 (Ma et al., arXiv:2402.17764; BitNet b1.58 2B4T, arXiv:2504.12285).
   Веса линейных слоёв квантуются в {-1, 0, +1} * gamma, активации — в int8. Обучение идёт через
   Straight-Through Estimator (STE): в прямом проходе используются квантованные веса, градиент
   передаётся мастер-весам как есть. Экономия: ~1.58 бита на вес вместо 16.

2. Mamba-2 / SSD (Dao & Gu, arXiv:2405.21060) вместо Mamba-1.
   Состояние контекста фиксированного размера (нет растущего KV-кэша), а обучение считается
   блочными матричными умножениями («chunked scan»), а не питоновским циклом по токенам.

3. Локальное скользящее внимание (Sliding Window Attention) с RoPE и GQA — как в гибридах
   Samba (arXiv:2406.07522) / Jamba (arXiv:2403.19887). Окно W: память на внимание O(W), а не O(L).

4. Рекуррентное «ядро мышления» с общими весами (recurrent depth: Universal Transformers,
   arXiv:1807.03819; Geiping et al., arXiv:2502.05171) и маршрутизацией токенов по глубине в стиле
   Mixture-of-Depths (arXiv:2404.02258) / Mixture-of-Recursions (arXiv:2507.10524):
   на каждом шаге мышления роутер выбирает, какие токены «думают» дальше. Невыбранные токены
   реально НЕ вычисляются (gather/scatter), поэтому экономия вычислений настоящая.
   Идея «думать в скрытом пространстве, а не текстом» близка к Coconut (arXiv:2412.06769),
   но технически здесь рекуррентная глубина, а не подача скрытого состояния обратно на вход.

5. Память фактов с произведением ключей (Product-Key Memory: Lample et al., arXiv:1907.05242;
   Memory Layers at Scale, arXiv:2412.09764). Поиск top-k среди N = n_keys^2 ячеек стоит O(sqrt(N)),
   а не O(N): ключ делится на две половины, каждая ищется среди n_keys подключей.
   memory_keys=0 убирает память совсем: в модели «mini» она оказалась бесполезной (см. weights_xray.py).

6. Эмбеддинг, сжатый во время обучения (quantization-aware training, как в BitNet для троичных слоёв,
   и как 4-битные таблицы в llama.cpp/Q4_0): при embed_bits > 0 векторы слов в каждом проходе
   округляются до embed_bits бит (свой масштаб на каждую группу из embed_group чисел строки), градиент
   идёт к «мастер-весам» через STE. Модель с самого начала учится жить с округлёнными векторами, поэтому
   сжатие эмбеддинга (он же выходной слой) в 4 раза почти ничего не стоит — в отличие от округления
   готовой модели (−6 пунктов на задачах).

Файл содержит:
- Раздел 1: реализацию на PyTorch (обучение на GPU/CPU, инкрементальная генерация с кэшем O(1)
  на токен для SSM и O(W) для внимания, экспорт троичных весов).
- Раздел 2: автономное ядро на NumPy: троичный сумматор без умножений, микро-прототип с ТОЧНЫМ
  обратным распространением и численной проверкой градиентов, честный расчёт памяти модели.

Запуск самопроверки:  python slr_net_engine.py
Обучение на больших текстах: см. train_lm.py
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:  # NumPy-часть работает и без PyTorch
    TORCH_AVAILABLE = False


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

@dataclass
class SLRNetConfig:
    """Конфигурация SLR-Net. Значения по умолчанию — пресет «large» (~206 млн параметров вместе с памятью фактов)."""
    vocab_size: int = 16384           # Субсловный словарь (SentencePiece Unigram), 2^14 токенов
    d_model: int = 1024               # Ширина остаточного потока
    # --- Mamba-2 (SSD) ---
    expand: int = 2                   # d_inner = expand * d_model
    d_state: int = 64                 # Размер состояния SSM на голову (Mamba-2 обычно 64..128)
    d_conv: int = 4                   # Ширина каузальной свёртки
    ssm_headdim: int = 64             # Размер головы SSM
    ssm_chunk: int = 64               # Длина блока для параллельного (chunked) скана
    # --- Слои ---
    n_encoder_layers: int = 8         # «Прелюдия»: SSM + редкое локальное внимание
    encoder_attn_every: int = 4       # Каждый k-й слой прелюдии — внимание (остальные — Mamba-2)
    n_reasoning_layers: int = 4       # Слои ядра мышления (общие веса, крутятся до max_latent_steps раз)
    n_decoder_layers: int = 4         # «Кода»: синтез выходного распределения
    decoder_attn_every: int = 2
    # --- Внимание ---
    n_heads: int = 16
    n_kv_heads: int = 4               # GQA: меньше голов для K/V -> меньше кэш
    window_size: int = 256            # Окно локального внимания
    rope_theta: float = 10000.0
    mlp_ratio: float = 4.0            # MLP (ReLU^2) после каждого слоя внимания; 0 — без MLP
    # --- Ядро мышления + роутер ---
    max_latent_steps: int = 4         # Максимум шагов скрытого мышления
    router_capacity: float = 0.5      # Доля токенов, проходящих на каждый следующий шаг (0.5, 0.25, ...)
    router_aux_coef: float = 0.01     # Вес вспомогательного BCE-лосса роутера
    # --- Память фактов (product keys) ---
    memory_keys: int = 256            # Ячеек памяти = memory_keys^2 (256^2 = 65536); 0 — без памяти
    memory_key_dim: int = 256
    memory_top_k: int = 8
    # --- Эмбеддинг (он же выходной слой) ---
    embed_bits: int = 0               # 0 — float16 как раньше; 2..8 — сжатие во время обучения до стольких бит
    embed_group: int = 64             # Сколько чисел строки эмбеддинга делят один масштаб (float16)
    # --- Прочее ---
    ternary: bool = True              # False — обычные FP-веса (полезно для сравнения качества)
    rms_norm_eps: float = 1e-6
    init_std: float = 0.02

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def ssm_nheads(self) -> int:
        return self.d_inner // self.ssm_headdim

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def memory_slots(self) -> int:
        return self.memory_keys ** 2

    @property
    def use_memory(self) -> bool:
        return self.memory_keys > 0

    @property
    def mlp_hidden(self) -> int:
        return int(self.mlp_ratio * self.d_model)

    def validate(self) -> None:
        assert self.d_inner % self.ssm_headdim == 0, "d_inner должно делиться на ssm_headdim"
        assert self.d_model % self.n_heads == 0, "d_model должно делиться на n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads должно делиться на n_kv_heads"
        assert self.head_dim % 2 == 0, "head_dim должен быть чётным (RoPE)"
        assert 0.0 < self.router_capacity <= 1.0
        assert self.memory_keys >= 0
        if self.use_memory:
            assert self.memory_key_dim % 2 == 0, "memory_key_dim должен быть чётным (product keys)"
            assert self.memory_top_k <= self.memory_keys
        if self.embed_bits:
            assert 2 <= self.embed_bits <= 8, "embed_bits: 0 (float16) или от 2 до 8"
            assert self.embed_group > 0 and self.d_model % self.embed_group == 0, \
                "d_model должно делиться на embed_group"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# Готовые размеры. «tiny» — для отладки на CPU, «small»/«base» — реальные кандидаты
# на «маленькую, но умную» модель. Точное число параметров печатает calculate_hardware_footprint().
PRESETS: Dict[str, Dict[str, object]] = {
    "tiny": dict(
        vocab_size=4096, d_model=128, d_state=16, ssm_headdim=32, ssm_chunk=32,
        n_encoder_layers=4, encoder_attn_every=4, n_reasoning_layers=2,
        n_decoder_layers=2, decoder_attn_every=2, n_heads=4, n_kv_heads=2,
        window_size=64, max_latent_steps=3, memory_keys=32, memory_key_dim=64, memory_top_k=4,
    ),
    # «mini» — то, что реально обучить на обычном CPU за несколько часов (так обучены веса
    # slr_net_subtitles_ru_mini.npz на русских субтитрах; словарь 8192 — меньше считать на выходе)
    "mini": dict(
        vocab_size=8192, d_model=320, d_state=32, ssm_headdim=64, ssm_chunk=32,
        n_encoder_layers=6, encoder_attn_every=3, n_reasoning_layers=2,
        n_decoder_layers=3, decoder_attn_every=3, n_heads=5, n_kv_heads=1,
        window_size=128, max_latent_steps=3, memory_keys=64, memory_key_dim=128, memory_top_k=4,
    ),
    # «mini2» — тот же файл ~9 МБ, что у «mini», но перераспределённый: память фактов (2.4 МБ, не работала)
    # убрана, эмбеддинг сжимается до 4 бит прямо при обучении (4.9 -> 2.1 МБ), а освободившиеся ~6 МБ отданы
    # вычислениям — в первую очередь вниманию: 6 слоёв внимания вместо 3, у каждой из 8 голов свои ключи
    # (было 5 голов на 1 общий ключ), окно 256 токенов (весь контекст обучения) вместо 128.
    # Вычисляющие троичные слои: 1.6 -> 6 МБ. На CPU с bf16 (AMX) учится ~2300 токенов/с.
    "mini2": dict(
        vocab_size=8192, d_model=512, d_state=32, ssm_headdim=64, ssm_chunk=32,
        n_encoder_layers=8, encoder_attn_every=2, n_reasoning_layers=2,
        n_decoder_layers=4, decoder_attn_every=2, n_heads=8, n_kv_heads=8,
        window_size=256, max_latent_steps=3, memory_keys=0, embed_bits=4, embed_group=64,
    ),
    "small": dict(
        vocab_size=16384, d_model=384, d_state=64, ssm_headdim=64,
        n_encoder_layers=6, encoder_attn_every=3, n_reasoning_layers=2,
        n_decoder_layers=4, decoder_attn_every=2, n_heads=6, n_kv_heads=2,
        window_size=256, max_latent_steps=4, memory_keys=128, memory_key_dim=128, memory_top_k=8,
    ),
    "base": dict(
        vocab_size=16384, d_model=768, d_state=64, ssm_headdim=64,
        n_encoder_layers=8, encoder_attn_every=4, n_reasoning_layers=3,
        n_decoder_layers=4, decoder_attn_every=2, n_heads=12, n_kv_heads=4,
        window_size=512, max_latent_steps=4, memory_keys=256, memory_key_dim=256, memory_top_k=8,
    ),
    "large": dict(),
}


def make_config(preset: str = "small", **overrides) -> SLRNetConfig:
    if preset not in PRESETS:
        raise ValueError(f"Неизвестный пресет '{preset}'. Доступны: {list(PRESETS)}")
    params = dict(PRESETS[preset])
    params.update({k: v for k, v in overrides.items() if v is not None})
    cfg = SLRNetConfig(**params)
    cfg.validate()
    return cfg


def layer_kinds(n_layers: int, attn_every: int) -> List[str]:
    """Схема слоёв: 'attn' на каждом attn_every-м месте, остальные — 'ssm'."""
    return ["attn" if attn_every > 0 and (i + 1) % attn_every == 0 else "ssm" for i in range(n_layers)]


# =============================================================================
# РАЗДЕЛ 1. РЕАЛИЗАЦИЯ НА PYTORCH
# =============================================================================

if TORCH_AVAILABLE:

    class RMSNorm(nn.Module):
        """Среднеквадратичная нормализация (без вычитания среднего)."""

        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            dtype = x.dtype
            x = x.float()
            x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x.to(dtype) * self.weight

    class BitLinear158(nn.Module):
        """
        Линейный слой BitNet b1.58: веса {-1, 0, +1} * gamma, активации int8 (absmax на токен).

        Исправление относительно первой версии: масштаб gamma входит ВНУТРЬ STE
        (w + (gamma * round(w / gamma) - w).detach()), как в официальной реализации.
        Раньше в обратном проходе градиент дополнительно умножался на gamma (~0.01-0.06)
        и тёк через сам gamma — это искажало обучение.

        use_norm=True добавляет RMSNorm перед квантованием активаций (SubLN из BitNet).
        Если вход уже нормализован снаружи, норму внутри не дублируем (use_norm=False).
        """

        def __init__(self, in_features: int, out_features: int, *, ternary: bool = True,
                     use_norm: bool = True, bias: bool = False, eps: float = 1e-5):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.ternary = ternary
            self.eps = eps
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
            self.norm = RMSNorm(in_features) if use_norm else None
            nn.init.normal_(self.weight, std=0.02)
            self._wq_cache: Optional[Tuple[Tuple[int, int, torch.dtype], torch.Tensor]] = None

        def _quantize_weight(self) -> torch.Tensor:
            w = self.weight
            gamma = w.abs().mean().clamp(min=self.eps)
            w_q = (w / gamma).round().clamp(-1.0, 1.0) * gamma
            return w + (w_q - w).detach()

        def effective_weight(self) -> torch.Tensor:
            if not self.ternary:
                return self.weight
            if torch.is_grad_enabled():
                return self._quantize_weight()
            # Инференс: кэшируем квантованные веса, чтобы не квантовать их на каждом токене
            key = (self.weight.data_ptr(), self.weight._version, self.weight.dtype)
            if self._wq_cache is None or self._wq_cache[0] != key:
                self._wq_cache = (key, self._quantize_weight().detach())
            return self._wq_cache[1]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.norm is not None:
                x = self.norm(x)
            if self.ternary:
                scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps)
                x_q = (x * scale).round().clamp(-128.0, 127.0) / scale
                x = x + (x_q - x).detach()
            w = self.effective_weight()
            return F.linear(x, w.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

        @torch.no_grad()
        def export_ternary(self) -> Tuple[np.ndarray, float]:
            """Возвращает (троичная матрица int8, gamma) для упаковки 5 тритов в байт."""
            w = self.weight.float()
            gamma = float(w.abs().mean().clamp(min=self.eps))
            trits = (w / gamma).round().clamp(-1, 1).to(torch.int8).cpu().numpy()
            return trits, gamma

    # ---------------------------------------------------------------- Сжатый эмбеддинг

    def group_quantize(w: torch.Tensor, bits: int, group: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Симметричное квантование строк по группам: каждая строка делится на группы по `group` чисел,
        у группы свой масштаб s = max|w| / qmax (qmax = 2^(bits-1) - 1, для 4 бит — 7), число -> round(w / s).
        Масштаб сразу округляется до float16 — ровно так он хранится в файле, поэтому модель при обучении
        видит в точности те же числа, что будут после упаковки.
        Возвращает (целые q той же формы, что w, во float; масштабы (строк, групп)).
        Квантование идемпотентно: w' = q * s даёт снова те же q и s (это нужно для загрузки из файла).
        """
        rows, cols = w.shape
        qmax = 2 ** (bits - 1) - 1
        g = w.float().reshape(rows, cols // group, group)
        scale = (g.abs().amax(dim=-1) / qmax).clamp(min=1e-5).half().float()
        q = (g / scale[..., None]).round().clamp(-qmax, qmax)
        return q.reshape(rows, cols), scale

    def fake_quant_groups(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
        """Прямой проход — округлённые веса, обратный — градиент как есть (STE), как в BitLinear158."""
        q, scale = group_quantize(w.detach(), bits, group)
        w_q = (q.reshape(w.shape[0], -1, group) * scale[..., None]).reshape(w.shape).to(w.dtype)
        return w + (w_q - w).detach()

    # ---------------------------------------------------------------- RoPE

    def rope_cos_sin(positions: torch.Tensor, dim: int, theta: float, dtype: torch.dtype):
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device).float() / dim))
        freqs = positions.float()[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin

    # ---------------------------------------------------------------- Mamba-2 SSD

    def segsum(x: torch.Tensor) -> torch.Tensor:
        """Устойчивая сегментная сумма: out[..., i, j] = sum_{k=j+1..i} x[..., k] (i >= j), иначе -inf."""
        T = x.size(-1)
        x = x.unsqueeze(-1).expand(*x.shape, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
        x = x.masked_fill(~mask, 0.0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
        return x_segsum.masked_fill(~mask, float("-inf"))

    def ssd_scan(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                 chunk: int, initial_state: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Параллельный блочный скан Mamba-2 (SSD). Эквивалентен рекуррентности
            h_t = exp(A_t) * h_{t-1} + X_t ⊗ B_t,   y_t = h_t · C_t
        но считается матричными умножениями, а не циклом по времени.
          X: (b, l, h, p)  — вход, уже умноженный на dt
          A: (b, l, h)     — A * dt (отрицательное)
          B, C: (b, l, n)  — общие для всех голов (ngroups = 1)
        Возвращает y: (b, l, h, p) и финальное состояние (b, h, p, n).
        """
        b, l, h, p = X.shape
        n = B.shape[-1]
        T = max(1, min(chunk, l))
        pad = (T - l % T) % T
        X, A, B, C = (t.float() for t in (X, A, B, C))
        if pad:
            X = F.pad(X, (0, 0, 0, 0, 0, pad))
            A = F.pad(A, (0, 0, 0, pad))
            B = F.pad(B, (0, 0, 0, pad))
            C = F.pad(C, (0, 0, 0, pad))
        c = X.shape[1] // T
        X = X.reshape(b, c, T, h, p)
        B = B.reshape(b, c, T, n)
        C = C.reshape(b, c, T, n)
        A = A.reshape(b, c, T, h).permute(0, 3, 1, 2)          # (b, h, c, T)
        A_cum = torch.cumsum(A, dim=-1)

        # 1) Внутри блоков (диагональные блоки)
        L = torch.exp(segsum(A))                               # (b, h, c, T, T)
        CB = torch.einsum("bcln,bcsn->bcls", C, B)             # (b, c, T, T)
        Y_diag = torch.einsum("bcls,bhcls,bcshp->bclhp", CB, L, X)

        # 2) Состояние на конце каждого блока
        decay_states = torch.exp(A_cum[..., -1:] - A_cum)      # (b, h, c, T)
        states = torch.einsum("bcln,bhcl,bclhp->bchpn", B, decay_states, X)

        # 3) Рекуррентность между блоками
        if initial_state is None:
            initial_state = torch.zeros(b, h, p, n, device=X.device, dtype=X.dtype)
        states = torch.cat([initial_state.float().unsqueeze(1), states], dim=1)
        decay_chunk = torch.exp(segsum(F.pad(A_cum[..., -1], (1, 0))))   # (b, h, c+1, c+1)
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]

        # 4) Вклад состояния из прошлых блоков в выход
        state_decay_out = torch.exp(A_cum)                     # (b, h, c, T)
        Y_off = torch.einsum("bcln,bchpn,bhcl->bclhp", C, states, state_decay_out)

        Y = (Y_diag + Y_off).reshape(b, c * T, h, p)[:, :l]
        return Y, final_state

    class Mamba2Block(nn.Module):
        """
        Блок Mamba-2 с троичными проекциями. Обучение — параллельный скан (ssd_scan),
        генерация — шаг рекуррентности с состоянием фиксированного размера (step).
        """

        def __init__(self, cfg: SLRNetConfig):
            super().__init__()
            self.d_model, self.d_inner = cfg.d_model, cfg.d_inner
            self.d_state, self.headdim = cfg.d_state, cfg.ssm_headdim
            self.nheads, self.chunk, self.d_conv = cfg.ssm_nheads, cfg.ssm_chunk, cfg.d_conv
            self.conv_dim = self.d_inner + 2 * self.d_state

            self.norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
            # Вход уже нормализован self.norm, поэтому внутри BitLinear норму не дублируем
            self.in_proj = BitLinear158(cfg.d_model, self.d_inner + self.conv_dim,
                                        ternary=cfg.ternary, use_norm=False)
            # Шаг dt — маленькая, но чувствительная проекция: держим её в полной точности
            self.dt_proj = nn.Linear(cfg.d_model, self.nheads, bias=True)
            self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=cfg.d_conv,
                                    groups=self.conv_dim, padding=cfg.d_conv - 1, bias=True)
            self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1.0, 16.0)))
            self.D = nn.Parameter(torch.ones(self.nheads))
            self.out_norm = RMSNorm(self.d_inner, eps=cfg.rms_norm_eps)
            self.out_proj = BitLinear158(self.d_inner, cfg.d_model, ternary=cfg.ternary, use_norm=False)

            # Инициализация dt как в Mamba: dt ~ logUniform[1e-3, 1e-1] (иначе память «забывается» мгновенно)
            dt = torch.exp(torch.empty(self.nheads).uniform_(math.log(1e-3), math.log(1e-1)))
            with torch.no_grad():
                self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # обратный softplus
                nn.init.normal_(self.dt_proj.weight, std=0.02)

        def _project(self, x: torch.Tensor):
            h = self.norm(x)
            zxbc = self.in_proj(h)
            z, xBC = zxbc.split([self.d_inner, self.conv_dim], dim=-1)
            dt = F.softplus(self.dt_proj(h).float())
            return z, xBC, dt

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, l, _ = x.shape
            z, xBC, dt = self._project(x)
            xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :l].transpose(1, 2))
            xs, Bm, Cm = xBC.split([self.d_inner, self.d_state, self.d_state], dim=-1)
            A = -torch.exp(self.A_log.float())                        # (h,)
            X = xs.reshape(b, l, self.nheads, self.headdim)
            y, _ = ssd_scan(X.float() * dt[..., None], A * dt, Bm, Cm, self.chunk)
            y = y + X.float() * self.D.float()[:, None]
            y = y.reshape(b, l, self.d_inner).to(x.dtype)
            y = self.out_norm(y * F.silu(z))
            return x + self.out_proj(y)

        def init_cache(self, batch: int, device, dtype) -> Dict[str, torch.Tensor]:
            return {
                "conv": torch.zeros(batch, self.conv_dim, self.d_conv - 1, device=device, dtype=dtype),
                "ssm": torch.zeros(batch, self.nheads, self.headdim, self.d_state, device=device,
                                   dtype=torch.float32),
            }

        def step(self, x: torch.Tensor, cache: Dict[str, torch.Tensor],
                 active: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Один токен: x (b, d_model). active (b,) — для каких строк обновлять состояние."""
            b = x.shape[0]
            z, xBC, dt = self._project(x)
            window = torch.cat([cache["conv"], xBC.unsqueeze(-1)], dim=-1)       # (b, C, K)
            xBC = F.silu((window * self.conv1d.weight[:, 0, :]).sum(-1) + self.conv1d.bias)
            xs, Bm, Cm = xBC.split([self.d_inner, self.d_state, self.d_state], dim=-1)
            A = -torch.exp(self.A_log.float())
            dA = torch.exp(dt * A)                                              # (b, h)
            X = xs.reshape(b, self.nheads, self.headdim).float()
            state = cache["ssm"] * dA[..., None, None] + \
                (X * dt[..., None]).unsqueeze(-1) * Bm.float()[:, None, None, :]
            y = (state * Cm.float()[:, None, None, :]).sum(-1) + X * self.D.float()[:, None]
            y = y.reshape(b, self.d_inner).to(x.dtype)
            out = x + self.out_proj(self.out_norm(y * F.silu(z)))
            new_conv = window[..., 1:]
            if active is None:
                cache["conv"], cache["ssm"] = new_conv, state
            else:
                cache["conv"] = torch.where(active[:, None, None], new_conv, cache["conv"])
                cache["ssm"] = torch.where(active[:, None, None, None], state, cache["ssm"])
            return out

    # ---------------------------------------------------------------- Внимание + MLP

    class ReluSqMLP(nn.Module):
        """MLP с активацией ReLU^2 (как в BitNet b1.58 2B4T: даёт разреженные активации)."""

        def __init__(self, cfg: SLRNetConfig):
            super().__init__()
            self.norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
            self.up = BitLinear158(cfg.d_model, cfg.mlp_hidden, ternary=cfg.ternary, use_norm=False)
            self.down = BitLinear158(cfg.mlp_hidden, cfg.d_model, ternary=cfg.ternary, use_norm=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.down(F.relu(self.up(self.norm(x))).square())

    class LocalAttentionBlock(nn.Module):
        """
        Каузальное внимание в скользящем окне W (+ RoPE, GQA) и MLP.
        Считается блоками: каждый блок из W запросов смотрит на 2W ключей (свой и предыдущий блок),
        поэтому память и время O(L * W), а не O(L^2), как было с полной матрицей L x L.
        """

        def __init__(self, cfg: SLRNetConfig):
            super().__init__()
            self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
            self.window, self.rope_theta = cfg.window_size, cfg.rope_theta
            self.norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
            qkv_out = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
            self.qkv = BitLinear158(cfg.d_model, qkv_out, ternary=cfg.ternary, use_norm=False)
            self.o_proj = BitLinear158(self.n_heads * self.head_dim, cfg.d_model,
                                       ternary=cfg.ternary, use_norm=True)
            self.mlp = ReluSqMLP(cfg) if cfg.mlp_ratio > 0 else None

        def _qkv(self, x: torch.Tensor, positions: torch.Tensor):
            b, l, _ = x.shape
            q, k, v = self.qkv(self.norm(x)).split(
                [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim,
                 self.n_kv_heads * self.head_dim], dim=-1)
            q = q.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(b, l, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(b, l, self.n_kv_heads, self.head_dim).transpose(1, 2)
            cos, sin = rope_cos_sin(positions, self.head_dim, self.rope_theta, q.dtype)
            return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

        def _expand_kv(self, t: torch.Tensor) -> torch.Tensor:
            rep = self.n_heads // self.n_kv_heads
            return t if rep == 1 else t.repeat_interleave(rep, dim=1)

        def _local_attention(self, q, k, v) -> torch.Tensor:
            b, h, l, d = q.shape
            W = self.window
            if l <= W:  # всё помещается в окно — обычное каузальное внимание
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            nb = math.ceil(l / W)
            pad = nb * W - l
            if pad:
                q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
            q = q.reshape(b, h, nb, W, d)
            k = k.reshape(b, h, nb, W, d)
            v = v.reshape(b, h, nb, W, d)
            k_prev = torch.cat([torch.zeros_like(k[:, :, :1]), k[:, :, :-1]], dim=2)
            v_prev = torch.cat([torch.zeros_like(v[:, :, :1]), v[:, :, :-1]], dim=2)
            kk = torch.cat([k_prev, k], dim=3)                                # (b, h, nb, 2W, d)
            vv = torch.cat([v_prev, v], dim=3)
            i = torch.arange(W, device=q.device)[:, None]
            j = torch.arange(2 * W, device=q.device)[None, :]
            dist = (i + W) - j
            band = (dist >= 0) & (dist < W)                                   # (W, 2W)
            mask = band.unsqueeze(0).repeat(nb, 1, 1)
            mask[0, :, :W] = False                                            # у первого блока нет «прошлого»
            out = F.scaled_dot_product_attention(
                q.reshape(b, h * nb, W, d), kk.reshape(b, h * nb, 2 * W, d),
                vv.reshape(b, h * nb, 2 * W, d), attn_mask=mask.repeat(h, 1, 1).unsqueeze(0))
            return out.reshape(b, h, nb * W, d)[:, :, :l]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, l, _ = x.shape
            q, k, v = self._qkv(x, torch.arange(l, device=x.device))
            out = self._local_attention(q, self._expand_kv(k), self._expand_kv(v))
            x = x + self.o_proj(out.transpose(1, 2).reshape(b, l, -1))
            return self.mlp(x) if self.mlp is not None else x

        def init_cache(self, batch: int, device, dtype) -> Dict[str, object]:
            return {"k": None, "v": None, "pos": 0}

        def step(self, x: torch.Tensor, cache: Dict[str, object]) -> torch.Tensor:
            b = x.shape[0]
            pos = torch.tensor([cache["pos"]], device=x.device)
            q, k, v = self._qkv(x.unsqueeze(1), pos)
            if cache["k"] is not None:
                k = torch.cat([cache["k"], k], dim=2)[:, :, -self.window:]
                v = torch.cat([cache["v"], v], dim=2)[:, :, -self.window:]
            cache["k"], cache["v"], cache["pos"] = k, v, cache["pos"] + 1
            out = F.scaled_dot_product_attention(q, self._expand_kv(k), self._expand_kv(v))
            x = x + self.o_proj(out.transpose(1, 2).reshape(b, -1))
            return self.mlp(x) if self.mlp is not None else x

    # ---------------------------------------------------------------- Память фактов

    class ProductKeyMemory(nn.Module):
        """
        Память фактов с произведением ключей. N = n_keys^2 ячеек, но поиск top-k стоит
        O(n_keys * key_dim) благодаря разбиению запроса на две половины. Результат поиска ТОЧНЫЙ:
        top-k суммы s1[i] + s2[j] всегда лежит внутри topk(s1) x topk(s2).
        (В первой версии запрос сравнивался со всеми 65536 ключами — O(N) и гигабайты памяти.)
        """

        def __init__(self, cfg: SLRNetConfig):
            super().__init__()
            self.n_keys, self.top_k = cfg.memory_keys, cfg.memory_top_k
            self.half = cfg.memory_key_dim // 2
            self.query = BitLinear158(cfg.d_model, cfg.memory_key_dim, ternary=cfg.ternary, use_norm=True)
            self.keys = nn.Parameter(torch.randn(2, self.n_keys, self.half) / math.sqrt(self.half))
            self.values = nn.Embedding(self.n_keys ** 2, cfg.d_model)
            nn.init.normal_(self.values.weight, std=cfg.d_model ** -0.5)
            self.gate = nn.Parameter(torch.tensor(0.1))

        def lookup(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """x: (N, d_model) -> (индексы (N, k), веса (N, k))."""
            q = self.query(x).float()
            q1, q2 = q[:, :self.half], q[:, self.half:]
            s1, i1 = (q1 @ self.keys[0].float().t()).topk(self.top_k, dim=-1)
            s2, i2 = (q2 @ self.keys[1].float().t()).topk(self.top_k, dim=-1)
            cand = (s1[:, :, None] + s2[:, None, :]).flatten(1)             # (N, k*k)
            cand_idx = (i1[:, :, None] * self.n_keys + i2[:, None, :]).flatten(1)
            best, pos = cand.topk(self.top_k, dim=-1)
            return cand_idx.gather(1, pos), torch.softmax(best, dim=-1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            idx, w = self.lookup(flat)
            mem = F.embedding_bag(idx, self.values.weight, per_sample_weights=w.to(self.values.weight.dtype),
                                  mode="sum")
            return x + (torch.tanh(self.gate) * mem).to(x.dtype).reshape(shape)

    # ---------------------------------------------------------------- Ядро мышления

    class RecurrentDepthCore(nn.Module):
        """
        Ядро «скрытого мышления»: стек Mamba-2 слоёв с общими весами, который крутится до K раз.
        На каждом шаге k роутер решает, какие токены думают дальше:
          • обучение ("topk"): ровно ceil(L * capacity^(k+1)) лучших токенов среди прошедших шаг k-1
            (фиксированный бюджет вычислений; общий бюджет K шагов <= одного полного прохода при capacity=0.5)
            + BCE-лосс, обучающий роутер предсказывать этот выбор каузально;
          • инференс ("threshold"): токен думает, пока sigmoid(router) > 0.5 — решение каузальное,
            поэтому работает и в пошаговой генерации.
        Выбранные токены собираются (gather) в подпоследовательность, обрабатываются и
        возвращаются на место (scatter) — невыбранные токены не стоят ни одной операции.
        """

        def __init__(self, cfg: SLRNetConfig):
            super().__init__()
            self.cfg = cfg
            self.layers = nn.ModuleList([Mamba2Block(cfg) for _ in range(cfg.n_reasoning_layers)])
            self.memory = ProductKeyMemory(cfg) if cfg.use_memory else None
            self.router = nn.Linear(cfg.d_model, 1)
            self.step_emb = nn.Parameter(torch.randn(cfg.max_latent_steps, cfg.d_model) * cfg.init_std)
            nn.init.normal_(self.router.weight, std=cfg.init_std)
            nn.init.zeros_(self.router.bias)

        def _think(self, z_in: torch.Tensor) -> torch.Tensor:
            h = self.memory(z_in) if self.memory is not None else z_in
            for layer in self.layers:
                h = layer(h)
            return h

        def forward(self, h: torch.Tensor, routing: str = "topk"):
            b, l, d = h.shape
            z = h
            cand = torch.ones(b, l, dtype=torch.bool, device=h.device)
            aux = torch.zeros((), device=h.device)
            depth = torch.zeros(b, l, device=h.device)
            for k in range(self.cfg.max_latent_steps):
                z_in = z + self.step_emb[k].to(z.dtype)
                logit = self.router(z_in).squeeze(-1).float()                   # (b, l)
                if routing == "topk":
                    cap = min(l, max(1, math.ceil(l * self.cfg.router_capacity ** (k + 1))))
                    top = logit.masked_fill(~cand, float("-inf")).topk(cap, dim=1).indices
                    sel = torch.zeros_like(cand).scatter_(1, top, True) & cand
                    aux = aux + F.binary_cross_entropy_with_logits(logit[cand], sel[cand].float())
                elif routing == "threshold":
                    sel = cand & (logit > 0)
                elif routing == "all":
                    sel = cand
                else:
                    raise ValueError(f"Неизвестный режим маршрутизации: {routing}")
                if not bool(sel.any()):
                    break
                counts = sel.sum(dim=1)
                lmax = int(counts.max())
                order = torch.argsort((~sel).to(torch.int8), dim=1, stable=True)[:, :lmax]
                valid = torch.arange(lmax, device=h.device)[None, :] < counts[:, None]
                gidx = order.unsqueeze(-1).expand(-1, -1, d)
                gate = torch.sigmoid(logit).gather(1, order) * valid
                out = self._think(torch.gather(z_in, 1, gidx))
                delta = (out - torch.gather(z, 1, gidx)) * gate.unsqueeze(-1)
                z = z.scatter_add(1, gidx, delta.to(z.dtype))
                depth = depth + sel.float()
                cand = sel
            stats = {"core_depth_mean": depth.mean().detach(),
                     "core_step1_frac": (depth > 0).float().mean().detach()}
            return z, aux, stats

        def init_cache(self, batch: int, device, dtype):
            return [[layer.init_cache(batch, device, dtype) for layer in self.layers]
                    for _ in range(self.cfg.max_latent_steps)]

        def step(self, z: torch.Tensor, cache) -> torch.Tensor:
            active = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)
            for k in range(self.cfg.max_latent_steps):
                z_in = z + self.step_emb[k].to(z.dtype)
                logit = self.router(z_in).squeeze(-1).float()
                active = active & (logit > 0)
                if not bool(active.any()):
                    break
                h = self.memory(z_in) if self.memory is not None else z_in
                for layer, c in zip(self.layers, cache[k]):
                    h = layer.step(h, c, active=active)
                gate = torch.sigmoid(logit).to(z.dtype)[:, None]
                z = torch.where(active[:, None], z + gate * (h - z), z)
            return z

    # ---------------------------------------------------------------- Модель целиком

    class SLRNetLanguageModel(nn.Module):
        """
        Прелюдия (Mamba-2 + локальное внимание) -> ядро мышления (рекуррентная глубина + роутер +
        память фактов, если она есть) -> кода (Mamba-2 + локальное внимание) -> RMSNorm -> выход (связан с
        эмбеддингом). Все слои каузальные: модель — авторегрессионная языковая модель.
        При embed_bits > 0 эмбеддинг (и выходной слой) во всех проходах округляется до embed_bits бит.
        """

        def __init__(self, config: SLRNetConfig):
            super().__init__()
            config.validate()
            self.config = config

            def build(kind: str) -> nn.Module:
                return LocalAttentionBlock(config) if kind == "attn" else Mamba2Block(config)

            self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
            self.encoder = nn.ModuleList(
                [build(k) for k in layer_kinds(config.n_encoder_layers, config.encoder_attn_every)])
            self.core = RecurrentDepthCore(config)
            self.decoder = nn.ModuleList(
                [build(k) for k in layer_kinds(config.n_decoder_layers, config.decoder_attn_every)])
            self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.tok_embeddings.weight      # weight tying

            # Инициализация: эмбеддинги N(0, 0.02) (раньше было N(0, 1) -> стартовый лосс ~45 вместо ln V),
            # выходные проекции остаточных веток уменьшены на 1/sqrt(2 * число слоёв) (как в GPT-2).
            nn.init.normal_(self.tok_embeddings.weight, std=config.init_std)
            n_res = config.n_encoder_layers + config.n_decoder_layers + config.n_reasoning_layers
            for name, p in self.named_parameters():
                if name.endswith(("out_proj.weight", "o_proj.weight", "down.weight")):
                    nn.init.normal_(p, std=config.init_std / math.sqrt(2 * n_res))
            self._emb_cache: Optional[Tuple[Tuple[int, int, torch.dtype], torch.Tensor]] = None

        def num_params(self) -> Dict[str, int]:
            total = sum(p.numel() for p in self.parameters())
            ternary = sum(m.weight.numel() for m in self.modules()
                          if isinstance(m, BitLinear158) and m.ternary)
            memory = self.core.memory.values.weight.numel() if self.core.memory is not None else 0
            return {"total": total, "ternary": ternary, "memory_values": memory,
                    "embedding": self.tok_embeddings.weight.numel()}

        def embedding_weight(self) -> torch.Tensor:
            """Таблица векторов слов, которой модель реально пользуется (на входе и на выходе)."""
            w = self.tok_embeddings.weight
            bits = self.config.embed_bits
            if not bits:
                return w
            if torch.is_grad_enabled():
                return fake_quant_groups(w, bits, self.config.embed_group)
            # Инференс: округляем один раз, а не на каждом токене генерации
            key = (w.data_ptr(), w._version, w.dtype)
            if self._emb_cache is None or self._emb_cache[0] != key:
                self._emb_cache = (key, fake_quant_groups(w, bits, self.config.embed_group).detach())
            return self._emb_cache[1]

        def _embed(self, ids: torch.Tensor) -> torch.Tensor:
            if not self.config.embed_bits:
                return self.tok_embeddings(ids)
            return F.embedding(ids, self.embedding_weight())

        def _logits(self, h: torch.Tensor) -> torch.Tensor:
            h = self.final_norm(h)
            if not self.config.embed_bits:
                return self.lm_head(h)
            return F.linear(h, self.embedding_weight().to(h.dtype))

        def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None,
                    routing: Optional[str] = None):
            """
            routing: "topk" (по умолчанию при обучении), "threshold" (по умолчанию при eval) или "all".
            Возвращает (logits, loss, stats). stats — тензоры без градиента (без .item(), чтобы не
            синхронизировать GPU на каждом шаге).
            """
            routing = routing or ("topk" if self.training else "threshold")
            h = self._embed(input_ids)
            for layer in self.encoder:
                h = layer(h)
            h, aux, stats = self.core(h, routing)
            for layer in self.decoder:
                h = layer(h)
            logits = self._logits(h)
            loss = None
            if targets is not None:
                ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                                     ignore_index=-100)
                # Раньше «энергетический штраф» был числом float без градиента и ни на что не влиял.
                # Теперь бюджет вычислений задаётся router_capacity, а роутер учится через aux-лосс.
                loss = ce + self.config.router_aux_coef * aux
                stats["ce_loss"] = ce.detach()
                stats["router_aux"] = aux.detach()
            return logits, loss, stats

        # ------------------------------------------------ генерация с кэшем

        def init_cache(self, batch: int, device=None, dtype=None):
            device = device or self.tok_embeddings.weight.device
            dtype = dtype or self.tok_embeddings.weight.dtype
            return {
                "encoder": [l.init_cache(batch, device, dtype) for l in self.encoder],
                "core": self.core.init_cache(batch, device, dtype),
                "decoder": [l.init_cache(batch, device, dtype) for l in self.decoder],
            }

        @torch.no_grad()
        def step(self, token_ids: torch.Tensor, cache) -> torch.Tensor:
            """Один токен на строку (b,) -> логиты (b, V). Память на токен не растёт (SSM) / O(W) (внимание)."""
            h = self._embed(token_ids)
            for layer, c in zip(self.encoder, cache["encoder"]):
                h = layer.step(h, c)
            h = self.core.step(h, cache["core"])
            for layer, c in zip(self.decoder, cache["decoder"]):
                h = layer.step(h, c)
            return self._logits(h)

        @torch.no_grad()
        def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 0.8,
                     top_k: int = 50, top_p: float = 0.95, repetition_penalty: float = 1.1,
                     eos_id: Optional[int] = None) -> torch.Tensor:
            was_training = self.training
            self.eval()
            b, l0 = input_ids.shape
            cache = self.init_cache(b)
            logits = None
            for t in range(l0):                       # префилл (пошагово — просто и точно)
                logits = self.step(input_ids[:, t], cache)
            out = input_ids
            finished = torch.zeros(b, dtype=torch.bool, device=input_ids.device)
            for _ in range(max_new_tokens):
                logits = logits.float()
                if repetition_penalty != 1.0:
                    seen = torch.gather(logits, 1, out)
                    seen = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
                    logits = logits.scatter(1, out, seen)
                if temperature <= 0:
                    nxt = logits.argmax(dim=-1)
                else:
                    logits = logits / temperature
                    if top_k and top_k < logits.size(-1):
                        kth = logits.topk(top_k, dim=-1).values[:, -1:]
                        logits = logits.masked_fill(logits < kth, float("-inf"))
                    if top_p < 1.0:
                        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                        remove = cum - torch.softmax(sorted_logits, dim=-1) > top_p
                        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
                    nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)
                if eos_id is not None:
                    nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
                    finished |= nxt == eos_id
                out = torch.cat([out, nxt[:, None]], dim=1)
                if eos_id is not None and bool(finished.all()):
                    break
                logits = self.step(nxt, cache)
            self.train(was_training)
            return out

        # ------------------------------------------------ экспорт

        @torch.no_grad()
        def export_packed(self) -> Dict[str, np.ndarray]:
            """
            Троичные матрицы -> 5 тритов в байт (+gamma); эмбеддинг при embed_bits > 0 -> целые по
            embed_bits бит (до 4 бит — по два в байт) + масштабы групп во float16; остальное -> float16.
            """
            packed: Dict[str, np.ndarray] = {}
            ternary_weights = set()
            if self.config.embed_bits:
                q, scale = group_quantize(self.tok_embeddings.weight, self.config.embed_bits,
                                          self.config.embed_group)
                packed["tok_embeddings.weight.qdata"] = pack_int_matrix(
                    q.to(torch.int8).cpu().numpy(), self.config.embed_bits)
                packed["tok_embeddings.weight.qscale"] = scale.cpu().numpy().astype(np.float16)
                packed["tok_embeddings.weight.shape"] = np.array(q.shape, dtype=np.int64)
                ternary_weights.add("tok_embeddings.weight")
            for name, mod in self.named_modules():
                if isinstance(mod, BitLinear158) and mod.ternary:
                    trits, gamma = mod.export_ternary()
                    data, shape = pack_ternary_matrix(trits)
                    packed[f"{name}.weight.packed"] = data
                    packed[f"{name}.weight.shape"] = np.array(shape, dtype=np.int64)
                    packed[f"{name}.weight.gamma"] = np.array([gamma], dtype=np.float32)
                    ternary_weights.add(f"{name}.weight")
            for name, t in self.state_dict().items():
                if name in ternary_weights or name == "lm_head.weight":   # lm_head связан с эмбеддингом
                    continue
                packed[name] = t.detach().float().cpu().numpy().astype(np.float16)
            return packed

        @classmethod
        def from_packed(cls, config: SLRNetConfig, packed: Dict[str, np.ndarray]) -> "SLRNetLanguageModel":
            """
            Обратная операция к export_packed(): модель из упакованных троичных весов.

            Тонкость: BitLinear квантует веса заново на каждом проходе с gamma' = mean|W|. Если просто
            записать W = trits * gamma, то gamma' = gamma * (доля ненулевых), и масштаб слоя «поплывёт».
            Поэтому записываем W = trits * gamma / (доля ненулевых): тогда mean|W| = gamma ровно, а
            round(W / gamma) = trits — модель считает в точности те же троичные веса, что и до экспорта.
            """
            model = cls(config)
            state: Dict[str, torch.Tensor] = {}
            for key in packed:
                if not key.endswith(".weight.packed"):
                    continue
                name = key[: -len(".packed")]
                shape = tuple(int(s) for s in packed[name + ".shape"])
                trits = unpack_ternary_matrix(packed[key], shape).astype(np.float32)
                gamma = float(np.asarray(packed[name + ".gamma"]).reshape(-1)[0])
                nonzero = float(np.count_nonzero(trits)) / max(1, trits.size)
                scale = gamma / nonzero if nonzero > 0 else 0.0
                state[name] = torch.from_numpy(trits * scale)
            if "tok_embeddings.weight.qdata" in packed:
                if not config.embed_bits:
                    raise ValueError("Эмбеддинг в файле сжат, а в настройках модели embed_bits = 0")
                shape = tuple(int(s) for s in packed["tok_embeddings.weight.shape"])
                q = unpack_int_matrix(packed["tok_embeddings.weight.qdata"], shape, config.embed_bits)
                scale = np.asarray(packed["tok_embeddings.weight.qscale"], dtype=np.float32)
                w = q.astype(np.float32).reshape(shape[0], -1, config.embed_group) * scale[..., None]
                state["tok_embeddings.weight"] = torch.from_numpy(w.reshape(shape))
            own = model.state_dict()
            for key, value in packed.items():
                if key in own and key not in state:
                    state[key] = torch.from_numpy(np.asarray(value, dtype=np.float32)).reshape(own[key].shape)
            state["lm_head.weight"] = state["tok_embeddings.weight"]
            missing = sorted(set(own) - set(state))
            if missing:
                raise ValueError(f"В упакованных весах нет параметров: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
            model.load_state_dict(state)
            return model


    def reconfigure(model: "SLRNetLanguageModel", **overrides) -> "SLRNetLanguageModel":
        """
        Та же модель с изменёнными настройками, которые не меняют форму остальных весов: например,
        memory_keys=0 (выбросить память фактов) или embed_bits=4 (дальше учиться со сжатым эмбеддингом).
        Все веса, которые есть в обеих моделях, копируются как есть.
        """
        cfg = SLRNetConfig(**{**model.config.to_dict(), **overrides})
        new = SLRNetLanguageModel(cfg).to(model.tok_embeddings.weight.device)
        old_state, own = model.state_dict(), new.state_dict()
        bad = [k for k in own if k not in old_state or old_state[k].shape != own[k].shape]
        if bad:
            raise ValueError(f"Эти настройки меняют форму весов, перенести нельзя: {bad[:5]}")
        new.load_state_dict({k: old_state[k] for k in own})
        return new


# =============================================================================
# РАЗДЕЛ 2. АВТОНОМНОЕ ЯДРО НА ЧИСТОМ NUMPY
# =============================================================================

def quantize_weights_b158_numpy(w: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, float, float]:
    """
    Квантование BitNet b1.58:  gamma = mean(|W|),  W_t = Clip(Round(W / gamma), -1, +1).
    Возвращает троичную матрицу int8, gamma и долю нулей.
    """
    gamma = float(max(np.mean(np.abs(w)), eps))
    w_ternary = np.clip(np.round(w / gamma), -1.0, 1.0).astype(np.int8)
    return w_ternary, gamma, float(np.mean(w_ternary == 0))


def ternary_add_only_linear(x: np.ndarray, w_ternary: np.ndarray, gamma: float) -> np.ndarray:
    """
    Демонстрация: линейный слой без умножений на веса
        y_j = gamma * (sum_{W_ji = +1} x_i - sum_{W_ji = -1} x_i).
    Это показывает принцип для специализированного железа/ядер (например, bitnet.cpp).
    В NumPy/PyTorch на обычном CPU/GPU это всё равно считается через умножения — выигрыш
    в скорости дают только специальные ядра, а не сама формула.
    """
    pos_mask = (w_ternary == 1)
    neg_mask = (w_ternary == -1)
    out = np.zeros((x.shape[0], w_ternary.shape[0]), dtype=np.float32)
    for j in range(w_ternary.shape[0]):
        out[:, j] = (x[:, pos_mask[j]].sum(axis=1) - x[:, neg_mask[j]].sum(axis=1)) * gamma
    return out


def pack_ternary_matrix(w_ternary: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Упаковка 5 тритов в 1 байт (3^5 = 243 < 256) — 1.6 бита на вес."""
    shape = tuple(int(s) for s in w_ternary.shape)
    flat = w_ternary.ravel().astype(np.int16) + 1
    pad_len = (5 - flat.size % 5) % 5
    if pad_len:
        flat = np.concatenate([flat, np.ones(pad_len, dtype=np.int16)])
    powers = np.array([1, 3, 9, 27, 81], dtype=np.int16)
    packed = (flat.reshape(-1, 5) * powers).sum(axis=1).astype(np.uint8)
    return packed, shape


def unpack_ternary_matrix(packed: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    n_total = int(np.prod(shape))
    val = packed.astype(np.int16)[:, None]
    powers = np.array([1, 3, 9, 27, 81], dtype=np.int16)
    trits = ((val // powers) % 3) - 1
    return trits.ravel()[:n_total].reshape(shape).astype(np.int8)


def pack_int_matrix(q: np.ndarray, bits: int) -> np.ndarray:
    """
    Целые из [-(2^(bits-1)-1), 2^(bits-1)-1] -> байты. До 4 бит — по два числа в байт (младшие 4 бита —
    чётный элемент), число хранится со сдвигом +8: 4-битный эмбеддинг занимает ровно полбайта на число.
    5–8 бит — по байту (int8 как есть).
    """
    flat = np.asarray(q, dtype=np.int8).reshape(-1)
    if bits > 4:
        return flat.view(np.uint8).copy()
    u = (flat.astype(np.int16) + 8).astype(np.uint8)
    if u.size % 2:
        u = np.concatenate([u, np.full(1, 8, dtype=np.uint8)])
    return (u[0::2] | (u[1::2] << 4)).astype(np.uint8)


def unpack_int_matrix(data: np.ndarray, shape: Tuple[int, ...], bits: int) -> np.ndarray:
    """Обратная операция к pack_int_matrix: байты -> int8 матрица формы shape."""
    n = int(np.prod(shape))
    data = np.asarray(data, dtype=np.uint8).reshape(-1)
    if bits > 4:
        return data[:n].view(np.int8).reshape(shape).copy()
    u = np.empty(data.size * 2, dtype=np.int16)
    u[0::2] = data & 0x0F
    u[1::2] = data >> 4
    return (u[:n] - 8).astype(np.int8).reshape(shape)


def _rms(x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / r, r


def _rms_backward(dy: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Градиент y = x / rms(x):  dx = (dy - y * mean(dy * y)) / r."""
    return (dy - y * np.mean(dy * y, axis=-1, keepdims=True)) / r


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


class NumpySLRNetPrototype:
    """
    Микро-прототип SLR-Net на NumPy с ТОЧНЫМ ручным обратным распространением.
    Прямой проход:
      x = E[tok] -> SSM: s_t = a*s_{t-1} + (1-a)*tanh(rms(x_t) W_ssm^T),  h = x + s
      роутер p = sigmoid(h w_r + b_r); токены с p > 0.5 идут в цикл мышления:
        m = память (top-2 по ключам);  z_{k+1} = z_k + 0.5 * tanh(rms(z_k + 0.3 m) W_think^T)
      z_final = h + p * (z_K - h);  logits = rms(z_final) E^T  (связанные веса)
    Лосс: кросс-энтропия + router_coef * MSE(p, подсказки сложности).
    Квантование весов — STE (градиент по эффективному весу передаётся мастер-весу).
    Метод gradient_check() сверяет аналитические градиенты с численными.
    """

    PARAMS = ("embed", "w_ssm", "w_think", "w_router", "b_router", "mem_keys", "mem_vals")

    def __init__(self, vocab_size: int = 64, d_model: int = 48, memory_slots: int = 128,
                 latent_steps: int = 3, seed: int = 42, quantize: bool = True,
                 dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.vocab_size, self.d_model, self.memory_slots = vocab_size, d_model, memory_slots
        self.latent_steps, self.quantize, self.dtype = latent_steps, quantize, dtype
        self.decay, self.router_coef = 0.75, 0.25
        s = 1.0 / math.sqrt(d_model)
        self.embed = (rng.standard_normal((vocab_size, d_model)) * 0.15).astype(dtype)
        self.w_ssm = (rng.standard_normal((d_model, d_model)) * s).astype(dtype)
        self.w_think = (rng.standard_normal((d_model, d_model)) * s).astype(dtype)
        self.w_router = (rng.standard_normal(d_model) * 0.2).astype(dtype)
        self.b_router = np.array([0.1], dtype=dtype)
        self.mem_keys = (rng.standard_normal((memory_slots, d_model)) * s).astype(dtype)
        self.mem_vals = (rng.standard_normal((memory_slots, d_model)) * 0.15).astype(dtype)
        self._adam_m = {k: np.zeros_like(getattr(self, k)) for k in self.PARAMS}
        self._adam_v = {k: np.zeros_like(getattr(self, k)) for k in self.PARAMS}
        self._t = 0

    def _eff(self, w: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self.quantize:
            return w, 0.0
        w_t, gamma, sparsity = quantize_weights_b158_numpy(w)
        return w_t.astype(self.dtype) * self.dtype(gamma), sparsity

    def forward(self, tokens: np.ndarray, early_exit: bool = True) -> Dict[str, object]:
        L = len(tokens)
        w_ssm, sp_ssm = self._eff(self.w_ssm)
        w_think, sp_think = self._eff(self.w_think)
        x = self.embed[tokens]
        xn, r1 = _rms(x)
        a = np.tanh(xn @ w_ssm.T)
        s = np.zeros_like(x)
        prev = np.zeros(self.d_model, dtype=x.dtype)
        for t in range(L):
            prev = self.decay * prev + (1.0 - self.decay) * a[t]
            s[t] = prev
        h = x + s
        p = _sigmoid(h @ self.w_router + self.b_router[0])
        active = p > 0.5
        z_final = h.copy()
        loops: Dict[int, Dict[str, object]] = {}
        for t in np.nonzero(active)[0]:
            z0 = h[t]
            scores = self.mem_keys @ z0
            top = np.argpartition(scores, -2)[-2:]
            w_mem = _softmax(scores[top])
            m = w_mem @ self.mem_vals[top]
            zs, steps = [z0], []
            for k in range(self.latent_steps):
                v = zs[-1] + 0.3 * m
                vn, r = _rms(v)
                th = np.tanh(vn @ w_think.T)
                zs.append(zs[-1] + 0.5 * th)
                steps.append((vn, r, th))
                if early_exit and k >= 1 and float(np.mean(np.abs(th))) < 0.08:
                    break
            z_final[t] = h[t] + p[t] * (zs[-1] - h[t])
            loops[int(t)] = {"top": top, "w_mem": w_mem, "m": m, "zs": zs, "steps": steps}
        hf, rf = _rms(z_final)
        logits = hf @ self.embed.T
        return {"tokens": tokens, "x": x, "xn": xn, "r1": r1, "a": a, "h": h, "p": p,
                "active": active, "loops": loops, "z_final": z_final, "hf": hf, "rf": rf,
                "logits": logits, "w_ssm_eff": w_ssm, "w_think_eff": w_think,
                "sparsity": (sp_ssm + sp_think) / 2.0}

    def loss_and_grads(self, tokens: np.ndarray, targets: np.ndarray, hints: np.ndarray,
                       early_exit: bool = True) -> Tuple[float, Dict[str, np.ndarray], Dict[str, object]]:
        c = self.forward(tokens, early_exit=early_exit)
        L = len(tokens)
        probs = _softmax(c["logits"])
        nll = float(-np.mean(np.log(probs[np.arange(L), targets] + 1e-12)))
        p, h = c["p"], c["h"]
        router_mse = float(np.mean((p - hints) ** 2))
        loss = nll + self.router_coef * router_mse

        g = {k: np.zeros_like(getattr(self, k)) for k in self.PARAMS}
        dlogits = probs.copy()
        dlogits[np.arange(L), targets] -= 1.0
        dlogits /= L
        g["embed"] += dlogits.T @ c["hf"]
        dzf = _rms_backward(dlogits @ self.embed, c["hf"], c["rf"])

        dh = dzf.copy()
        dp = np.zeros(L, dtype=h.dtype)
        w_think = c["w_think_eff"]
        for t, lp in c["loops"].items():
            zs, steps = lp["zs"], lp["steps"]
            zK = zs[-1]
            dp[t] = float(np.dot(dzf[t], zK - h[t]))
            dh[t] = dzf[t] * (1.0 - p[t])
            dz = p[t] * dzf[t]
            dm = np.zeros(self.d_model, dtype=h.dtype)
            for k in range(len(steps) - 1, -1, -1):
                vn, r, th = steps[k]
                du = 0.5 * dz * (1.0 - th * th)
                g["w_think"] += np.outer(du, vn)
                dv = _rms_backward(du @ w_think, vn, r)
                dz = dz + dv
                dm += 0.3 * dv
            top, w_mem = lp["top"], lp["w_mem"]
            g["mem_vals"][top] += np.outer(w_mem, dm)
            dw = self.mem_vals[top] @ dm
            dscore = w_mem * (dw - np.dot(w_mem, dw))
            g["mem_keys"][top] += np.outer(dscore, zs[0])
            dh[t] += dz + dscore @ self.mem_keys[top]

        dlogit_r = dp * p * (1.0 - p) + self.router_coef * 2.0 * (p - hints) / L * p * (1.0 - p)
        g["w_router"] += h.T @ dlogit_r
        g["b_router"][0] += float(np.sum(dlogit_r))
        dh += np.outer(dlogit_r, self.w_router)

        dx = dh.copy()
        gs = np.zeros(self.d_model, dtype=h.dtype)
        da = np.zeros_like(dh)
        for t in range(L - 1, -1, -1):
            gs = dh[t] + self.decay * gs
            da[t] = (1.0 - self.decay) * gs
        du = da * (1.0 - c["a"] ** 2)
        g["w_ssm"] += du.T @ c["xn"]
        dx += _rms_backward(du @ c["w_ssm_eff"], c["xn"], c["r1"])
        np.add.at(g["embed"], tokens, dx)
        return loss, g, {"nll": nll, "cache": c}

    def train_step(self, tokens, targets, hints, lr: float = 0.01) -> Dict[str, float]:
        loss, grads, info = self.loss_and_grads(tokens, targets, hints)
        self._t += 1
        b1, b2 = 0.9, 0.99
        for k in self.PARAMS:
            m = self._adam_m[k] = b1 * self._adam_m[k] + (1 - b1) * grads[k]
            v = self._adam_v[k] = b2 * self._adam_v[k] + (1 - b2) * grads[k] ** 2
            upd = (m / (1 - b1 ** self._t)) / (np.sqrt(v / (1 - b2 ** self._t)) + 1e-8)
            setattr(self, k, (getattr(self, k) - lr * upd).astype(self.dtype))
        c = info["cache"]
        steps_per_tok = [len(lp["steps"]) for lp in c["loops"].values()]
        return {
            "loss": loss, "nll": info["nll"],
            "accuracy": float(np.mean(np.argmax(c["logits"], axis=-1) == targets)),
            "routed_fraction": float(np.mean(c["active"])),
            "avg_latent_steps": float(np.sum(steps_per_tok) / len(tokens)),
            "sparsity": float(c["sparsity"]),
        }

    # совместимость со старым именем метода
    def forward_and_train_step(self, tokens, targets, token_complexity_hints, lr: float = 0.01):
        return self.train_step(tokens, targets, token_complexity_hints, lr)

    def gradient_check(self, tokens, targets, hints, n_checks: int = 6, eps: float = 1e-6,
                       seed: int = 0) -> float:
        """Сравнивает аналитический градиент с численным (центральная разность). Возвращает max отн. ошибку."""
        assert not self.quantize, "Градиент проверяется на непрерывной (неквантованной) версии сети"
        rng = np.random.default_rng(seed)
        _, grads, _ = self.loss_and_grads(tokens, targets, hints, early_exit=False)
        worst = 0.0
        for name in self.PARAMS:
            param = getattr(self, name)
            for _ in range(n_checks):
                idx = tuple(int(rng.integers(0, s)) for s in param.shape)
                if name == "embed" and rng.random() < 0.7:     # чаще проверяем реально использованные строки
                    idx = (int(rng.choice(tokens)),) + idx[1:]
                old = param[idx]
                param[idx] = old + eps
                lp, _, _ = self.loss_and_grads(tokens, targets, hints, early_exit=False)
                param[idx] = old - eps
                lm, _, _ = self.loss_and_grads(tokens, targets, hints, early_exit=False)
                param[idx] = old
                num = (lp - lm) / (2 * eps)
                ana = grads[name][idx]
                rel = abs(num - ana) / max(1e-7, abs(num) + abs(ana))
                if abs(num) + abs(ana) > 1e-9:
                    worst = max(worst, rel)
        return worst


def calculate_hardware_footprint(config: SLRNetConfig) -> Dict[str, float]:
    """
    Честный подсчёт параметров и памяти для заданной конфигурации (формулы совпадают с PyTorch-
    моделью один в один — это проверяется в tests/). Ничего не «забыто»: в первой версии не
    учитывалась память фактов (84 млн параметров) и неверно считался размер SSM-блока.
    """
    D, V, di, N = config.d_model, config.vocab_size, config.d_inner, config.d_state
    H, P, K = config.ssm_nheads, config.ssm_headdim, config.d_conv
    conv_dim = di + 2 * N
    hd, Hq, Hkv, Fh = config.head_dim, config.n_heads, config.n_kv_heads, config.mlp_hidden

    ssm_tern = D * (di + conv_dim) + di * D
    ssm_fp = D + (D * H + H) + (conv_dim * K + conv_dim) + 2 * H + di
    attn_tern = D * (Hq + 2 * Hkv) * hd + Hq * hd * D
    attn_fp = D + Hq * hd
    if config.mlp_ratio > 0:
        attn_tern += 2 * D * Fh
        attn_fp += D + Fh

    kinds = layer_kinds(config.n_encoder_layers, config.encoder_attn_every) + \
        layer_kinds(config.n_decoder_layers, config.decoder_attn_every)
    n_ssm = kinds.count("ssm") + config.n_reasoning_layers
    n_attn = kinds.count("attn")

    if config.use_memory:
        mem_tern = D * config.memory_key_dim
        mem_fp = D + 2 * config.memory_keys * (config.memory_key_dim // 2) + 1
        mem_values = config.memory_slots * D
    else:
        mem_tern = mem_fp = mem_values = 0
    core_fp = (D + 1) + config.max_latent_steps * D

    ternary = n_ssm * ssm_tern + n_attn * attn_tern + mem_tern
    fp_small = n_ssm * ssm_fp + n_attn * attn_fp + mem_fp + core_fp + D
    embedding = V * D
    total = ternary + fp_small + embedding + mem_values

    mb = 1024.0 ** 2
    if config.embed_bits:
        # целые по embed_bits бит (до 4 бит — два в байт) + масштаб float16 на каждую группу строки
        emb_bytes = embedding * (0.5 if config.embed_bits <= 4 else 1.0) + 2 * V * (D // config.embed_group)
    else:
        emb_bytes = 2.0 * embedding
    parts = {
        "packed_ternary_mb": ternary / 5.0 / mb,
        "packed_embedding_mb": emb_bytes / mb,
        "packed_memory_mb": 2.0 * mem_values / mb,
        "packed_other_mb": 2.0 * fp_small / mb,
    }
    packed_mb = sum(parts.values())
    fp16_mb = 2.0 * total / mb

    # Память на генерацию (кэш) — не зависит от длины контекста
    ssm_state_layers = kinds.count("ssm") + config.n_reasoning_layers * config.max_latent_steps
    ssm_cache = ssm_state_layers * (H * P * N * 4 + conv_dim * (K - 1) * 2)
    attn_cache = n_attn * 2 * config.window_size * Hkv * hd * 2
    return {
        "total_params_millions": total / 1e6,
        "ternary_params_millions": ternary / 1e6,
        "embedding_params_millions": embedding / 1e6,
        "memory_params_millions": mem_values / 1e6,
        "other_fp_params_millions": fp_small / 1e6,
        "packed_model_mb": packed_mb,
        **parts,
        "fp16_model_mb": fp16_mb,
        "generation_cache_mb": (ssm_cache + attn_cache) / mb,
    }


def run_verification_suite() -> None:
    """Самопроверка: математика квантования, точность градиентов, обучение, PyTorch-модель."""
    print("=" * 78)
    print("САМОПРОВЕРКА АРХИТЕКТУРЫ SLR-Net")
    print("=" * 78)

    print("\n[1] ЧЕСТНЫЙ РАСЧЁТ РАЗМЕРА МОДЕЛИ ПО ПРЕСЕТАМ:")
    for name in ("tiny", "mini", "mini2", "small", "base", "large"):
        fp = calculate_hardware_footprint(make_config(name))
        print(f"    • {name:5s}: {fp['total_params_millions']:7.1f} млн параметров "
              f"(троичных {fp['ternary_params_millions']:.1f}, эмбеддинг {fp['embedding_params_millions']:.1f}, "
              f"память фактов {fp['memory_params_millions']:.1f}) | "
              f"упаковано ≈ {fp['packed_model_mb']:.0f} МБ (FP16: {fp['fp16_model_mb']:.0f} МБ) | "
              f"кэш генерации {fp['generation_cache_mb']:.1f} МБ")

    rng = np.random.default_rng(7)
    x_test = rng.standard_normal((8, 64)).astype(np.float32)
    w_tern, gamma, sparsity = quantize_weights_b158_numpy(rng.standard_normal((32, 64)).astype(np.float32))
    y_add = ternary_add_only_linear(x_test, w_tern, gamma)
    y_ref = (x_test @ w_tern.T.astype(np.float32)) * gamma
    packed, shape = pack_ternary_matrix(w_tern)
    print("\n[2] ТРОИЧНЫЙ СУММАТОР И УПАКОВКА:")
    print(f"    • Нулевых весов: {sparsity * 100:.1f}% | расхождение сумматора с matmul: "
          f"{float(np.max(np.abs(y_add - y_ref))):.1e}")
    print(f"    • Упаковка 5 тритов/байт обратима: {np.array_equal(unpack_ternary_matrix(packed, shape), w_tern)} "
          f"({w_tern.size} весов -> {packed.size} байт)")

    vocab = ["<pad>", "почему", "небо", "синее", "?", "потому", "что", "атмосфера", "рассеивает",
             "короткие", "синие", "волны", "солнечного", "света", "."]
    seq = np.arange(1, len(vocab), dtype=np.int64)
    inp, tgt = seq[:-1], seq[1:]
    hints = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0], dtype=np.float64)

    checker = NumpySLRNetPrototype(vocab_size=32, d_model=16, memory_slots=16, quantize=False,
                                   dtype=np.float64, seed=3)
    err = checker.gradient_check(inp, tgt, hints)
    print("\n[3] ПРОВЕРКА ТОЧНОСТИ ОБРАТНОГО РАСПРОСТРАНЕНИЯ (аналитика vs численная производная):")
    print(f"    • Максимальная относительная ошибка градиента: {err:.2e} "
          f"({'OK' if err < 1e-4 else 'ОШИБКА'})")

    proto = NumpySLRNetPrototype(vocab_size=64, d_model=48, memory_slots=128)
    print("\n[4] ОБУЧЕНИЕ МИКРО-ПРОТОТИПА (STE + Adam) — запоминание одной фразы:")
    t0 = time.time()
    for epoch in range(1, 151):
        st = proto.train_step(inp, tgt, hints.astype(np.float32), lr=0.01)
        if epoch in (1, 25, 75, 150):
            print(f"    • Эпоха {epoch:3d} | Loss {st['loss']:.4f} | Точность {st['accuracy'] * 100:5.1f}% | "
                  f"в цикле мысли {st['routed_fraction'] * 100:4.1f}% | тактов/токен {st['avg_latent_steps']:.2f} | "
                  f"нулей {st['sparsity'] * 100:.1f}%")
    print(f"    (заняло {(time.time() - t0) * 1000:.0f} мс; это проверка, что градиенты текут, "
          f"а не доказательство «разумности» — одну фразу запомнит любая сеть)")

    if not TORCH_AVAILABLE:
        print("\n[5] PyTorch не установлен — проверка полной модели пропущена (pip install torch).")
        print("=" * 78)
        return

    torch.manual_seed(0)
    cfg = make_config("tiny", vocab_size=256)
    model = SLRNetLanguageModel(cfg)
    ids = torch.randint(0, 256, (2, 80))
    logits, loss, stats = model(ids[:, :-1], ids[:, 1:])
    loss.backward()
    model.eval()
    with torch.no_grad():
        a = torch.randint(0, 256, (1, 40))
        b = a.clone()
        b[0, 30] = (b[0, 30] + 1) % 256
        leak = (model(a)[0][0, :30] - model(b)[0][0, :30]).abs().max().item()
    n = model.num_params()
    print("\n[5] PyTorch-МОДЕЛЬ (пресет tiny):")
    print(f"    • Параметров: {n['total'] / 1e6:.2f} млн (совпадает с формулой: "
          f"{abs(n['total'] - calculate_hardware_footprint(cfg)['total_params_millions'] * 1e6) < 1})")
    print(f"    • Стартовый лосс {float(stats['ce_loss']):.2f} (ожидается ≈ ln V = {math.log(256):.2f})")
    print(f"    • Утечка из будущего (каузальность): {leak:.1e} (должно быть 0)")
    print(f"    • Средняя глубина мышления на токен (top-k): {float(stats['core_depth_mean']):.2f}")
    print("=" * 78)


if __name__ == "__main__":
    run_verification_suite()
