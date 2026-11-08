#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Самообучение для браузера (mini3): эталон на Python того, что делает страница slr_net_mobile_chat.html.
=====================================================================================================

В self_grow.py новые нейроны стоят во всех 6 слоях внимания, и ошибка проходит через всю сеть. На телефоне это
часы. Здесь — облегчённая схема, которая делает самообучение в браузере делом минут:

  • новые нейроны — только в MLP ПОСЛЕДНЕГО слоя сети (decoder.3). Всё, что до него, от них не зависит: для
    каждого токена ответа достаточно один раз посчитать два вектора — вход нейронов h (нормированный вход MLP) и
    выход слоя y (после старого MLP). Дальше обучение идёт только по этим сохранённым векторам:
        логиты = E · RMSNorm(y + down·relu(up·h + b)²);
  • ошибка считается не по всем 8 194 словам словаря, а по выборке: 32 самых вероятных слова исходной модели в
    этой позиции, нужное слово и 128 случайных слов — их вклад в сумму вероятностей домножается на долю
    невыбранных слов (оценка по выборке, как sampled softmax). Так шаг обучения в ~30 раз дешевле;
  • «не менять старое» — те же KL к исходной модели, но признаки старых диалогов и вопросов о невыученных словах
    посчитаны заранее и лежат прямо в файле модели (slr_net_chat_ru_mini3.npz, ключи selflearn.*): в браузере
    их не нужно считать;
  • рост нейронов и самопроверка — как в self_grow.py (v2): свои отложенные вопросы, только новые для модели токены
    ответа учителя; вред — совпадение первого выбора с исходной на старом.

Выученное ложится в те же GrownMLP-куски слоя decoder.3, что и в self_grow.py, поэтому проверки, модули знаний и
оркестратор работают без изменений.

  python selflearn_tail.py export --out slr_net_chat_ru_mini3.npz              # mini2 + данные самообучения
  python selflearn_tail.py experiment --words 20 --json training/selflearn/tail20.json   # качество, как в браузере
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import chat_corpus as C
import slr_net_engine as E
import self_grow as S
import train_chat as T

PACK_VERSION = 1
SETTINGS = dict(                      # настройки самообучения — те же в браузере (кладутся в файл модели)
    k_init=4, grow_factor=2.0, max_rounds=6, k_max=256, steps=200, lr=3e-3, batch=256, replay_batch=256,
    kl_coef=1.0, neg_frac=0.5, n_random=128, topk=32, target=0.9, train_target=0.95, keep_target=0.97, check_frac=0.2,
    bias_init=-0.5, keep_items=400, near_words=0,     # near_words > 0 — похожие несуществующие слова (строже, но хуже переносит)
)

if E.TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


# =============================================================================
# 1. ПРИЗНАКИ ПОСЛЕДНЕГО СЛОЯ
# =============================================================================

def tail_block(model):
    """Последний блок сети: внимание с MLP. После него — только final_norm и выходной слой."""
    block = model.decoder[-1]
    if not (isinstance(block, E.LocalAttentionBlock) and block.mlp is not None):
        raise ValueError("Последний слой модели — не внимание с MLP: облегчённое самообучение не подходит")
    return block


def capture_tail(model, tok, seqs: Sequence[Tuple[List[int], List[int]]], batch: int = 32):
    """
    Для каждой размеченной позиции (там, где модель пишет ответ): h — вход новых нейронов, y — выход последнего
    слоя без модулей, target — правильный следующий токен. Возвращает H (N, D), Y (N, D), targets (N,), seq_of (N,).
    """
    block = tail_block(model)
    mlp = block.mlp
    base = mlp.base if isinstance(mlp, S.GrownMLP) else mlp
    store = {}

    def hook(mod, inp, out):
        store["x"], store["y"] = inp[0].detach(), out.detach()

    was = [g.enabled for g in S.grown_layers(model).values()]
    S.set_enabled(model, False)
    handle = mlp.register_forward_hook(hook)
    H, Y, tg, so = [], [], [], []
    model.eval()
    try:
        with torch.no_grad():
            for i in range(0, len(seqs), batch):
                part = seqs[i:i + batch]
                x, y = T.pad_batch(part, tok.nl_id)
                model(x, routing="threshold")
                hn = base.norm(store["x"]).float()
                yy = store["y"].float()
                for r in range(len(part)):
                    v = y[r] != -100
                    H.append(hn[r][v])
                    Y.append(yy[r][v])
                    tg.append(y[r][v])
                    so += [i + r] * int(v.sum())
    finally:
        handle.remove()
        for g, e in zip(S.grown_layers(model).values(), was):
            g.enabled = e
    D = model.config.d_model
    if not H:
        return torch.zeros(0, D), torch.zeros(0, D), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    return torch.cat(H), torch.cat(Y), torch.cat(tg), torch.tensor(so)


def head_weights(model):
    """Выходной слой (= эмбеддинг, 4 бита) и вес final_norm — то, через что идут логиты."""
    with torch.no_grad():
        return model.embedding_weight().float().clone(), model.final_norm.weight.float().clone(), model.config.rms_norm_eps


def rmsnorm(z, w, eps):
    return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * w


def full_logits(Ew, fw, eps, Y, chunks=None, H=None, batch: int = 512):
    """Точные логиты по всему словарю для позиций (Y — выход слоя; chunks — новые нейроны, если есть)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(Y), batch):
            z = Y[i:i + batch]
            if chunks:
                z = z + extra(chunks, H[i:i + batch])
            out.append(rmsnorm(z, fw, eps) @ Ew.T)
    return torch.cat(out) if out else torch.zeros(0, Ew.shape[0])


def quantize_rows(X: "torch.Tensor") -> Tuple[np.ndarray, np.ndarray]:
    """Строки -> int8 с масштабом float16 на строку (так признаки лежат в файле модели)."""
    s = X.abs().amax(1).clamp(min=1e-8) / 127.0
    s16 = s.half().float()
    q = torch.round(X / s16[:, None]).clamp(-127, 127).to(torch.int8)
    return q.numpy(), s16.half().numpy()


def dequantize_rows(q: np.ndarray, s: np.ndarray) -> "torch.Tensor":
    return torch.from_numpy(q.astype(np.float32) * s.astype(np.float32)[:, None])


# =============================================================================
# 2. НОВЫЕ НЕЙРОНЫ И ОШИБКА ПО ВЫБОРКЕ СЛОВАРЯ
# =============================================================================

def new_chunk(D: int, k: int, bias_init: float, rng: np.random.Generator) -> Dict[str, "torch.Tensor"]:
    return {"up": torch.tensor(rng.standard_normal((k, D)) / math.sqrt(D), dtype=torch.float32),
            "bias": torch.full((k,), float(bias_init)), "down": torch.zeros(D, k)}


def extra(chunks, H):
    out = 0
    for ch in chunks:
        a = F.relu(H @ ch["up"].T + ch["bias"]).square()
        out = out + a @ ch["down"].T
    return out


def sampled_logz(lc, cand_idx, rand_idx, V: int):
    """
    log Σ exp(логиты) по всему словарю — оценка по выборке. lc — логиты выбранных слов (N, Kc), cand_idx (N, Kc) —
    их номера; rand_idx (m,) — случайные слова (общие для пачки) с логитами в последних m столбцах lc.
    Случайные слова, совпавшие с выбранными в этой позиции, не учитываются; их вклад домножается на
    (V − число выбранных) / число оставшихся случайных.
    """
    kc = cand_idx.shape[1]
    top = lc[:, :kc]
    rnd = lc[:, kc:]
    dup = (cand_idx[:, :, None] == rand_idx[None, None, :]).any(1)                 # (N, m)
    m_eff = (~dup).sum(1).clamp(min=1).float()
    n_cand = torch.tensor([len(set(r.tolist())) for r in cand_idx], dtype=torch.float32)
    w = torch.log((V - n_cand) / m_eff)                                              # (N,)
    rnd = rnd.masked_fill(dup, float("-inf")) + w[:, None]
    # повторы среди выбранных (нужное слово уже в топе) — не считаем дважды
    first = torch.ones_like(cand_idx, dtype=torch.bool)
    for j in range(1, kc):
        first[:, j] = ~(cand_idx[:, :j] == cand_idx[:, j:j + 1]).any(1)
    top = top.masked_fill(~first, float("-inf"))
    return torch.logsumexp(torch.cat([top, rnd], 1), dim=1)


def batch_loss(chunks, Ew, fw, eps, H, Y, cand, rand_idx, V, kind: str, ref_lp=None):
    """
    kind = "ce": cand[:, 0] — нужное слово, дальше топ исходной модели; ошибка = logZ − логит нужного.
    kind = "kl": cand — топ исходной модели, ref_lp — их log p у исходной; ошибка = Σ p_ref (log p_ref − log p_new).
    """
    z = Y + extra(chunks, H)
    n = rmsnorm(z, fw, eps)
    idx = torch.cat([cand, rand_idx[None, :].expand(len(cand), -1)], 1)
    lc = (n[:, None, :] * Ew[idx]).sum(-1)
    logz = sampled_logz(lc, cand, rand_idx, V)
    if kind == "ce":
        return (logz - lc[:, 0]).mean()
    lp_new = lc[:, :cand.shape[1]] - logz[:, None]
    p = ref_lp.exp()
    return (p * (ref_lp - lp_new)).sum(1).mean()


class Adam:
    """Adam как в браузере (β1 0.9, β2 0.98, без затухания весов), с ограничением нормы градиента 1.0."""

    def __init__(self, params, b1=0.9, b2=0.98, eps=1e-8):
        self.params, self.b1, self.b2, self.eps, self.t = params, b1, b2, eps, 0
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]

    def step(self, grads, lr):
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
        clip = min(1.0, 1.0 / (norm + 1e-6))
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g * clip
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p.sub_(lr * mh / (vh.sqrt() + self.eps))
        return norm


def lr_at(step: int, steps: int, lr: float) -> float:
    return lr * min(1.0, (step + 1) / 10) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(1, steps))))


def train_step(chunks_all, trainable, Ew, fw, eps, V, stud, rep, opt: Adam, lr: float, kl_coef: float, rand_idx):
    """Один шаг: stud = (H, Y, cand), rep = (H, Y, cand, ref_lp) или None. Возвращает (ce, kl)."""
    params = [p for ch in trainable for p in (ch["up"], ch["bias"], ch["down"])]
    for p in params:
        p.requires_grad_(True)
    ce = batch_loss(chunks_all, Ew, fw, eps, stud[0], stud[1], stud[2], rand_idx, V, "ce")
    kl = torch.zeros(())
    if rep is not None and kl_coef > 0:
        kl = batch_loss(chunks_all, Ew, fw, eps, rep[0], rep[1], rep[2], rand_idx, V, "kl", rep[3])
    loss = ce + kl_coef * kl
    grads = torch.autograd.grad(loss, params)
    with torch.no_grad():
        opt.step(list(grads), lr)
    for p in params:
        p.requires_grad_(False)
    return float(ce.detach()), float(kl.detach())


# =============================================================================
# 3. ДАННЫЕ «НЕ МЕНЯТЬ СТАРОЕ» (КЛАДУТСЯ В ФАЙЛ МОДЕЛИ)
# =============================================================================

def build_pack(model, tok, data_dir: str, n_old: int = 1500, n_neg: int = 1000, n_neg_words: int = 160, seed: int = 0,
               neg_mode: str = "classic", log=print) -> Dict[str, np.ndarray]:
    """
    Признаки последнего слоя для старых диалогов и для вопросов о невыученных выдуманных словах (с ответами
    исходной модели — «не знаю»), в int8, и цели исходной модели: 32 самых вероятных слова и их log p.
    Цели считаются по уже округлённым признакам: в начале обучения KL ровно 0 (как будет в браузере).
    """
    t0 = time.time()
    if neg_mode == "classic":
        # как в self_grow.py: 80 незнакомых слов × 4 вопроса, позиции ответа — случайной выборкой. Так в mini3:
        # лучше всего запоминает новое (PLAN_MULTIMODULE_RU.md, раздел 1.8)
        cfg = S.GrowConfig(seed=seed, replay_rows=max(600, n_old // 2), replay_len=96, negative_words=80)
        old_s, neg_s = S.replay_seqs(model, tok, data_dir, cfg, exclude=[])
        return _finish_pack(model, tok, old_s, neg_s, n_old, n_neg, seed, False, t0, log)
    # "first": вопросы теми же формулировками, что у инструмента, и все первые токены ответа — модуль строже
    # молчит на незнакомых словах, но хуже переносит выученное на новые формулировки (раздел 1.8)
    cfg = S.GrowConfig(seed=seed, replay_rows=max(600, n_old // 2), replay_len=96, negative_words=0)
    old_s, _ = S.replay_seqs(model, tok, data_dir, cfg, exclude=[])
    prng = random.Random(seed + 5)
    words = list(C.NONCE_TRAIN)
    prng.shuffle(words)
    neg_turns = []
    for w in words[:n_neg_words]:
        qs = S.text_questions(w, f"{C.cap(w)} — это маленькая птица синего цвета. Она живёт в лесу.")
        neg_turns += [[("user", q)] for q, _ in prng.sample(qs, 4)]
    S.set_enabled(model, False)
    answers = S.ask(model, tok, neg_turns)
    S.set_enabled(model, True)
    neg_s = [T.encode_dialogue(tok, t + [("bot", a)]) for t, a in zip(neg_turns, answers)]
    return _finish_pack(model, tok, old_s, neg_s, n_old, n_neg, seed, True, t0, log)


def _finish_pack(model, tok, old_s, neg_s, n_old: int, n_neg: int, seed: int, first_tokens: bool, t0: float, log):
    rng = np.random.default_rng(seed)
    parts = []
    for seqs, n, flag in ((old_s, n_old, 0), (neg_s, n_neg, 1)):
        Hh, Yy, _, so = capture_tail(model, tok, seqs)
        if len(Hh) == 0:
            continue
        if flag and first_tokens:
            # первый токен ответа решает «знаю / не знаю» — его берём у всех вопросов, остальные — выборкой
            so = so.numpy()
            first = np.flatnonzero(np.r_[True, so[1:] != so[:-1]])
            rest = np.setdiff1d(np.arange(len(Hh)), first)
            extra_n = max(0, min(n - len(first), len(rest)))
            idx = np.sort(np.concatenate([first, rng.choice(rest, size=extra_n, replace=False)]))
        else:
            idx = np.sort(rng.choice(len(Hh), size=min(n, len(Hh)), replace=False))
        parts.append((Hh[idx], Yy[idx], np.full(len(idx), flag, np.uint8)))
    H = torch.cat([p[0] for p in parts])
    Y = torch.cat([p[1] for p in parts])
    hq, hs = quantize_rows(H)
    yq, ys = quantize_rows(Y)
    Ew, fw, eps = head_weights(model)
    lp = torch.log_softmax(full_logits(Ew, fw, eps, dequantize_rows(yq, ys)), -1)
    top = lp.topk(SETTINGS["topk"], dim=-1)
    pack = {"selflearn.replay_h": hq, "selflearn.replay_h_scale": hs,
            "selflearn.replay_y": yq, "selflearn.replay_y_scale": ys,
            "selflearn.replay_top_idx": top.indices.numpy().astype(np.int16),
            "selflearn.replay_top_lp": top.values.numpy().astype(np.float16),
            "selflearn.replay_neg": np.concatenate([p[2] for p in parts])}
    log(f"  данные «не менять старое»: {len(H)} позиций ({int(pack['selflearn.replay_neg'].sum())} — вопросы о "
        f"невыученных словах), {sum(v.nbytes for v in pack.values()) / 1024 ** 2:.2f} МБ, {time.time() - t0:.0f} c")
    return pack


VOWELS, CONSONANTS = "аеиоуыэюя", "бвгджзклмнпрстфхцчшщ"


def near_misses(word: str, n: int, rng: random.Random) -> List[str]:
    """Похожие несуществующие слова: в выученном слове заменены 1–2 буквы (гласная на гласную, согласная на
    согласную). На них модель с модулем должна отвечать как без него — так нейроны учат само слово, а не
    «вопрос о незнакомом слове»."""
    w = word.lower()
    out: List[str] = []
    for _ in range(50 * n):
        if len(out) >= n:
            break
        chars = list(w)
        # окончание (две последние буквы) не трогаем: «курдяпу» — это та же курдяпа, а не другое слово
        pos = [i for i, ch in enumerate(chars[:-2]) if ch in VOWELS or ch in CONSONANTS]
        if not pos:
            break
        for i in rng.sample(pos, min(len(pos), 1 + (rng.random() < 0.5))):
            pool = VOWELS if chars[i] in VOWELS else CONSONANTS
            chars[i] = rng.choice([c for c in pool if c != chars[i]])
        v = "".join(chars)
        if v != w and v not in out:
            out.append(v)
    return out


def near_miss_data(model, tok, facts, settings: dict, seed: int):
    """Признаки и цели исходной модели для вопросов о похожих несуществующих словах (как пакет, но для этих слов)."""
    rng = random.Random(seed + 23)
    turns = [[("user", q)] for f in facts for v in near_misses(f.word, settings["near_words"], rng)
             for q in (f"Что такое {v}?", f"Какого цвета {v}?")]
    if not turns:
        return None
    S.set_enabled(model, False)
    answers = S.ask(model, tok, turns)
    S.set_enabled(model, True)
    H, Y, _, _ = capture_tail(model, tok, [T.encode_dialogue(tok, t + [("bot", a)]) for t, a in zip(turns, answers)])
    Ew, fw, eps = head_weights(model)
    top = torch.log_softmax(full_logits(Ew, fw, eps, Y), -1).topk(settings["topk"], dim=-1)
    return H, Y, top.indices, top.values


def load_pack(arrays: Dict[str, np.ndarray]):
    H = dequantize_rows(arrays["selflearn.replay_h"], arrays["selflearn.replay_h_scale"])
    Y = dequantize_rows(arrays["selflearn.replay_y"], arrays["selflearn.replay_y_scale"])
    idx = torch.from_numpy(arrays["selflearn.replay_top_idx"].astype(np.int64))
    lp = torch.from_numpy(arrays["selflearn.replay_top_lp"].astype(np.float32))
    neg = torch.from_numpy(arrays["selflearn.replay_neg"].astype(bool))
    return H, Y, idx, lp, neg


# =============================================================================
# 4. САМООБУЧЕНИЕ (ТО ЖЕ, ЧТО В БРАУЗЕРЕ)
# =============================================================================

def self_learn_tail(model, tok, facts, pack, settings: Optional[dict] = None, seed: int = 0, log=print,
                    tag: str = "") -> dict:
    """
    Модель сама выучивает факты: учитель (она же с объяснением в разговоре) отвечает на вопросы инструмента,
    признаки ответов считаются один раз, новые нейроны последнего слоя растут, пока самопроверка не пройдёт.
    Выученное кладётся в GrownMLP последнего слоя (с меткой tag) — дальше модель отвечает как обычно.
    """
    st = {**SETTINGS, **(settings or {})}
    t0 = time.time()
    gcfg = S.GrowConfig(seed=seed, teacher="inline")
    teacher_seqs, _ctx, items, answers, t_acc = S.build_teacher_data(model, tok, facts, gcfg, log)
    rng = random.Random(seed + 3)
    by_word: Dict[str, List[int]] = {}
    for i, (f, _, _, _) in enumerate(items):
        by_word.setdefault(f.word, []).append(i)
    check_idx = set()
    for idx in by_word.values():
        idx = idx[:]
        rng.shuffle(idx)
        check_idx.update(idx[:int(len(idx) * st["check_frac"])])
    train_idx = [i for i in range(len(items)) if i not in check_idx]
    Ht, Yt, Tt, _ = capture_tail(model, tok, [teacher_seqs[i] for i in train_idx])
    Hc, Yc, Tc, _ = capture_tail(model, tok, [teacher_seqs[i] for i in sorted(check_idx)])
    Ew, fw, eps = head_weights(model)
    V = Ew.shape[0]
    base_t = full_logits(Ew, fw, eps, Yt)
    cand_t = torch.cat([Tt[:, None], base_t.topk(st["topk"], -1).indices], 1)            # нужное слово + топ-32
    info_c = full_logits(Ew, fw, eps, Yc).argmax(-1) != Tc                               # новые для модели токены
    info_t = base_t.argmax(-1) != Tt
    Hr, Yr, Ir, Lr, Nr = pack
    keep_sel = torch.cat([torch.nonzero(Nr).flatten()[:st["keep_items"] // 2],
                          torch.nonzero(~Nr).flatten()[:st["keep_items"] // 2]])
    neg_pos, old_pos = torch.nonzero(Nr).flatten(), torch.nonzero(~Nr).flatten()
    near = near_miss_data(model, tok, facts, st, seed) if st.get("near_words", 0) > 0 else None
    near_pos = torch.zeros(0, dtype=torch.long)
    if near is not None:                       # похожие несуществующие слова — в конец пакета
        near_pos = torch.arange(len(Hr), len(Hr) + len(near[0]))
        Hr, Yr = torch.cat([Hr, near[0]]), torch.cat([Yr, near[1]])
        Ir, Lr = torch.cat([Ir, near[2]]), torch.cat([Lr, near[3]])
        keep_sel = torch.cat([keep_sel, near_pos])
    log(f"  позиций ответа: {len(Ht)} для обучения, {len(Hc)} для самопроверки (новых для модели — {int(info_c.sum())}); "
        f"подготовка {time.time() - t0:.0f} c")
    block = tail_block(model)
    if not isinstance(block.mlp, S.GrownMLP):
        block.mlp = S.GrownMLP(block.mlp)
    g = block.mlp
    D = Ht.shape[1]
    nrng = np.random.default_rng(seed + 11)
    trng = torch.Generator().manual_seed(seed + 17)
    trainable: List[dict] = []
    frozen: List[dict] = []          # каждый модуль учится сам по себе: включает его оркестратор (по словам модуля)
    report = {"teacher_train_acc": t_acc, "rounds": [], "positions": {"train": len(Ht), "check": len(Hc),
                                                                     "check_informative": int(info_c.sum())}}
    k_total, k_next = 0, st["k_init"]
    for r in range(st["max_rounds"]):
        trainable.append(new_chunk(D, k_next, st["bias_init"], nrng))
        k_total += k_next
        opt = Adam([p for ch in trainable for p in (ch["up"], ch["bias"], ch["down"])])
        for step in range(st["steps"]):
            sb = torch.randint(0, len(Ht), (min(st["batch"], len(Ht)),), generator=trng)
            nn_ = int(round(st["replay_batch"] * st["neg_frac"]))
            nn_ = st["replay_batch"] if len(old_pos) == 0 else (0 if len(neg_pos) == 0 else nn_)
            n_near = nn_ // 2 if len(near_pos) else 0          # половина «незнакомых» — похожие на выученное слово
            rb = torch.cat([neg_pos[torch.randint(0, max(1, len(neg_pos)), (nn_ - n_near,), generator=trng)],
                            near_pos[torch.randint(0, max(1, len(near_pos)), (n_near,), generator=trng)],
                            old_pos[torch.randint(0, max(1, len(old_pos)), (st["replay_batch"] - nn_,), generator=trng)]])
            rand_idx = torch.randint(0, V, (st["n_random"],), generator=trng)
            ce, kl = train_step(frozen + trainable, trainable, Ew, fw, eps, V, (Ht[sb], Yt[sb], cand_t[sb]),
                                (Hr[rb], Yr[rb], Ir[rb], Lr[rb]), opt, lr_at(step, st["steps"], st["lr"]),
                                st["kl_coef"], rand_idx)
        allc = frozen + trainable
        gen = float((full_logits(Ew, fw, eps, Yc, allc, Hc).argmax(-1) == Tc)[info_c].float().mean()) if info_c.any() else 1.0
        mem = float((full_logits(Ew, fw, eps, Yt, allc, Ht).argmax(-1) == Tt)[info_t].float().mean()) if info_t.any() else 1.0
        keep = float((full_logits(Ew, fw, eps, Yr[keep_sel], allc, Hr[keep_sel]).argmax(-1) == Ir[keep_sel, 0]).float().mean())
        report["rounds"].append({"neurons": k_total, "self_check_heldout": gen, "self_check_train": mem, "keep": keep,
                                 "ce": ce, "kl": kl, "seconds": round(time.time() - t0, 1)})
        log(f"  самопроверка ({k_total} нейронов): новые токены своих отложенных вопросов {gen:.1%} (обучающих "
            f"{mem:.1%}), совпадение с исходной на старом {keep:.1%}; ошибка {ce:.3f}, KL {kl:.4f}", flush=True)
        # своих отложенных вопросов мало, когда слово одно (3–4 вопроса): тогда решают и выученные свои ответы
        if gen >= st["target"] and mem >= st["train_target"] and keep >= st["keep_target"]:
            break
        k_next = int(round(k_total * (st["grow_factor"] - 1)))
        if k_total + k_next > st["k_max"]:
            break
    for ch in trainable:                                            # выученное -> куски GrownMLP последнего слоя
        c = g.grow(ch["up"].shape[0], 0.0, seed=0)
        c.tag = tag
        with torch.no_grad():
            c.up.copy_(ch["up"])
            c.bias.copy_(ch["bias"])
            c.down.copy_(ch["down"])
        for p in c.parameters():
            p.requires_grad_(False)
    report["neurons"] = k_total
    report["new_params"] = k_total * (2 * D + 1)
    report["seconds"] = round(time.time() - t0, 1)
    return report


# =============================================================================
# 5. КОМАНДЫ
# =============================================================================

def cmd_export(args: argparse.Namespace) -> None:
    """mini3 = веса mini2 без изменений + данные самообучения (selflearn.*) + описание в meta.selflearn."""
    torch.set_flush_denormal(True)
    model, tok, _ = T.load_chat_model(args.ckpt, torch.device("cpu"), allow_base=False)
    with np.load(args.ckpt, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    tail_block(model)
    pack = build_pack(model, tok, args.data, n_old=args.n_old, n_neg=args.n_neg, seed=args.seed, neg_mode=args.neg_mode)
    meta = {"version": PACK_VERSION, "site": "decoder." + str(len(model.decoder) - 1), "settings": SETTINGS,
            "base": Path(args.ckpt).name, "positions": int(len(pack["selflearn.replay_neg"]))}
    arrays.update(pack)
    arrays["meta.selflearn"] = np.array(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(args.out, **arrays)
    print(f"Сохранено: {args.out} ({os.path.getsize(args.out) / 1024 ** 2:.2f} МБ; было "
          f"{os.path.getsize(args.ckpt) / 1024 ** 2:.2f} МБ)")


def cmd_experiment(args: argparse.Namespace) -> None:
    """Те же 20 слов и проверки, что в self_grow.py experiment, но облегчённое самообучение (как в браузере)."""
    torch.set_flush_denormal(True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    pool = list(C.NONCE_TEST)
    rng.shuffle(pool)
    learn_words, control_words = pool[:args.words], pool[args.words:args.words + args.control]
    facts = S.make_facts(learn_words, args.seed)
    model, tok, _ = T.load_chat_model(args.ckpt, torch.device("cpu"), allow_base=False)
    if args.ckpt.endswith(".npz"):
        with np.load(args.ckpt, allow_pickle=False) as z:
            has = "selflearn.replay_h" in z.files
            arrays = {k: z[k] for k in z.files if k.startswith("selflearn.")} if has else None
    else:
        arrays = None
    if arrays is None:
        arrays = build_pack(model, tok, args.data, seed=args.seed)
    pack = load_pack(arrays)
    settings = {k: v for k, v in (("steps", args.steps), ("k_init", args.k_init), ("n_random", args.n_random),
                                  ("near_words", args.near_words)) if v is not None}
    rep = self_learn_tail(model, tok, facts, pack, settings, seed=args.seed, tag="m")
    rec, kinds, rows = S.eval_recall(model, tok, facts, train=False, seed=args.seed)
    rec_tr, kinds_tr, _ = S.eval_recall(model, tok, facts, train=True, seed=args.seed)
    hon, _ = S.eval_honesty(model, tok, control_words)
    out = {"words": args.words, "learn": rep, "recall_heldout": rec, "recall_heldout_by_kind": kinds,
           "recall_train_forms": rec_tr, "recall_train_by_kind": kinds_tr, "honesty_unlearned": hon,
           "examples": rows[:24]}
    if not args.no_old:
        out["old"] = S.eval_old_skills(model, tok, args.data, probes_every=args.probes_every)
        S.set_active(model, [])
        out["old_without_module"] = S.eval_old_skills(model, tok, args.data, probes_every=args.probes_every)
        S.set_active(model, None)
    print(f"\n### облегчённое самообучение: отложенные вопросы {rec:.1%}, открытые {kinds['ОТКРЫТЫЕ']:.1%} "
          f"(обучающие формулировки {rec_tr:.1%}, открытые {kinds_tr['ОТКРЫТЫЕ']:.1%}); честность {hon:.1%}"
          + (f"; старые задачи {out['old']['probes']:.1%} (без модуля {out['old_without_module']['probes']:.1%})"
             if "old" in out else ""))
    for r in rows[:8]:
        print(f"   {r['q']!r} -> {r['a']!r} {'✓' if r['ok'] else '✗'}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Облегчённое самообучение (как в браузере)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="mini3: веса mini2 + данные самообучения")
    e.add_argument("--ckpt", default="slr_net_chat_ru_mini2.npz")
    e.add_argument("--data", default="data/chat")
    e.add_argument("--out", default="slr_net_chat_ru_mini3.npz")
    e.add_argument("--n-old", type=int, default=1500)
    e.add_argument("--n-neg", type=int, default=1000)
    e.add_argument("--neg-mode", default="classic", choices=["classic", "first"])
    e.add_argument("--seed", type=int, default=0)
    x = sub.add_parser("experiment", help="качество облегчённого самообучения на выдуманных словах")
    x.add_argument("--ckpt", default="slr_net_chat_ru_mini3.npz")
    x.add_argument("--data", default="data/chat")
    x.add_argument("--words", type=int, default=20)
    x.add_argument("--control", type=int, default=20)
    x.add_argument("--steps", type=int, default=None)
    x.add_argument("--k-init", type=int, default=None)
    x.add_argument("--n-random", type=int, default=None)
    x.add_argument("--near-words", type=int, default=None, help="похожих несуществующих слов на каждое выученное")
    x.add_argument("--no-old", action="store_true")
    x.add_argument("--probes-every", type=int, default=3)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    {"export": cmd_export, "experiment": cmd_experiment}[args.cmd](args)


if __name__ == "__main__":
    main()
