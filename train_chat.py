#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пост-тренинг SLR-Net: из «продолжателя субтитров» — в собеседника.
===================================================================

После предобучения (train_lm.py) модель умеет только продолжать текст: на вопрос она пишет
«следующую реплику фильма», почти не глядя на сам вопрос. Этот файл учит её вести диалог.

Этапы (каждый — отдельная команда):

  1) mine   — добыча живых пар «вопрос → ответ» из субтитров. Берутся две соседние реплики с
              тире (смена говорящего), мусор отсекается правилами, а затем САМА модель оценивает,
              зависит ли ответ от вопроса (PMI = log p(ответ | свой вопрос) − log p(ответ | чужой
              вопрос)). Остаются пары, где ответ действительно отвечает.
  2) build  — сборка обучающих диалогов: ручные диалоги (личность, разговор, факты, честное
              «не знаю»), шаблонные навыки (запомнить имя и вспомнить его через несколько реплик,
              счёт, дни недели...), добытые пары и многоходовые диалоги из них.
  3) sft    — обучение с учителем (SFT): модель учится писать ТОЛЬКО реплики бота (ошибка
              считается только по ним). В словарь добавляются два ролевых токена <user> и <bot>.
  4) align  — «поощрение и наказание»: двусторонняя контрастная оптимизация предпочтений.
              Для каждого вопроса правильный ответ должен быть вероятнее, чем
                • собственный неудачный ответ модели (проверяется автоматической наградой:
                  правильное имя/число/факт, русский язык, реплика закончена, без повторов),
                • чужой ответ на этот вопрос (ответ не по теме),
                • этот же ответ, но на чужой вопрос (ответ должен зависеть от вопроса — эта сторона
                  прямо увеличивает PMI вопроса и ответа и лечит главную болезнь малых моделей).
              Лосс как в DPO (относительно замороженной SFT-модели) + обычная ошибка на правильном
              ответе, чтобы его вероятность не падала. Свои ответы модель пересэмплирует раундами.
  5) eval   — честная проверка до/после: точность на задачах с НОВЫМИ именами/городами/формулировками,
              выбор правильного ответа из 5, завершённость реплик, только русский.
  6) chat   — консольный диалог с моделью.
  7) export — упаковка троичных весов (5 тритов в байт) + токенизатор + признак чат-модели.

  python train_chat.py mine   --subs data_raw/ru_subtitles.txt --ckpt slr_net_subtitles_ru_mini.npz
  python train_chat.py build  --out data/chat
  python train_chat.py sft    --base slr_net_subtitles_ru_mini.npz --data data/chat --out runs/chat_sft
  python train_chat.py align  --ckpt runs/chat_sft/best.pt --data data/chat --out runs/chat_align
  python train_chat.py eval   --ckpt slr_net_subtitles_ru_mini.npz runs/chat_align/best.pt --data data/chat
  python train_chat.py export --ckpt runs/chat_align/best.pt --out slr_net_chat_ru_mini.npz
  python train_chat.py chat   --ckpt slr_net_chat_ru_mini.npz

Формат диалога (токены):  <user> текст <n> <bot> ответ <n> <user> ... <bot> ответ <n>
Ролевые токены лежат ВНЕ словаря SentencePiece (id = 8192 и 8193): их нельзя «набрать» текстом,
поэтому пользователь не может подделать реплику бота.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import slr_net_engine as E
import train_lm as LM

ROLE_TOKENS = ("<user>", "<bot>")
CHAT_FORMAT = "slr-chat-v1"


# =============================================================================
# ТОКЕНИЗАТОР И ФОРМАТ ДИАЛОГА
# =============================================================================

def clean_turn(text: str) -> str:
    """Реплика — одна строка: переводы строк и лишние пробелы убираются."""
    return re.sub(r"\s+", " ", str(text)).strip()


class ChatTokenizer(LM.Tokenizer):
    """SentencePiece + два ролевых токена с id = sp_size и sp_size + 1 (вне словаря SentencePiece)."""

    def __init__(self, model_path: Optional[str] = None, model_proto: Optional[bytes] = None):
        super().__init__(model_path, model_proto)
        self.sp_size = self.vocab_size
        self.user_id = self.sp_size
        self.bot_id = self.sp_size + 1
        self.vocab_size = self.sp_size + len(ROLE_TOKENS)
        self.unk_id = self.sp.unk_id()
        # Эти токены модель никогда не должна писать сама
        self.banned_ids = sorted({self.user_id, self.bot_id, self.eos_id, self.bos_id, self.unk_id} - {-1})

    @classmethod
    def from_base(cls, tok: LM.Tokenizer) -> "ChatTokenizer":
        return cls(model_proto=tok.sp.serialized_model_proto())

    def encode_line(self, text: str) -> List[int]:
        text = clean_turn(text)
        return list(self.sp.encode(text)) if text else []

    def decode(self, ids) -> str:
        return super().decode([int(t) for t in ids if int(t) < self.sp_size])


def encode_dialogue(tok: ChatTokenizer, turns: Sequence[Sequence[str]]) -> Tuple[List[int], List[int]]:
    """
    turns: [("user", текст), ("bot", текст), ...] -> (ids, mask).
    mask[j] = 1, если токен j — то, что модель должна научиться писать: текст реплик бота и <n>,
    которым бот заканчивает реплику (так модель учится вовремя замолкать). Реплики пользователя и
    ролевые токены в ошибку не входят: модель не учится говорить «за пользователя».
    """
    ids: List[int] = []
    mask: List[int] = []
    for role, text in turns:
        body = tok.encode_line(text)
        if role == "user":
            ids += [tok.user_id] + body + [tok.nl_id]
            mask += [0] * (len(body) + 2)
        elif role == "bot":
            ids += [tok.bot_id] + body + [tok.nl_id]
            mask += [0] + [1] * (len(body) + 1)
        else:
            raise ValueError(f"Неизвестная роль: {role}")
    return ids, mask


def prompt_ids(tok: ChatTokenizer, turns: Sequence[Sequence[str]]) -> List[int]:
    """Контекст для ответа: вся история + <bot> (дальше модель пишет реплику бота)."""
    ids, _ = encode_dialogue(tok, turns)
    return ids + [tok.bot_id]


def is_chat_config(cfg: E.SLRNetConfig, tok: LM.Tokenizer) -> bool:
    return cfg.vocab_size >= tok.vocab_size + len(ROLE_TOKENS)


# =============================================================================
# МОДЕЛЬ: РАСШИРЕНИЕ СЛОВАРЯ, ЗАГРУЗКА, ВЕРОЯТНОСТИ ОТВЕТОВ
# =============================================================================

def extend_vocab(model, new_vocab: int, init_from: int, seed: int = 0):
    """
    Новая модель со словарём new_vocab: старые веса копируются, строки новых токенов эмбеддинга
    (он же выходной слой) — копия строки init_from (<n>: «начинается новая реплика») + шум,
    чтобы токены различались. Остальные веса не меняются.
    """
    import torch
    old_cfg = model.config
    if new_vocab <= old_cfg.vocab_size:
        return model
    cfg = E.SLRNetConfig(**{**old_cfg.to_dict(), "vocab_size": new_vocab})
    new = E.SLRNetLanguageModel(cfg)
    state = model.state_dict()
    old_emb = state["tok_embeddings.weight"]
    g = torch.Generator().manual_seed(seed)
    std = float(old_emb.float().std())
    extra = old_emb[init_from].float()[None, :] + 0.3 * std * torch.randn(
        new_vocab - old_cfg.vocab_size, old_emb.shape[1], generator=g)
    emb = torch.cat([old_emb.float(), extra], dim=0).to(old_emb.dtype)
    state["tok_embeddings.weight"] = emb
    state["lm_head.weight"] = emb
    new.load_state_dict(state)
    return new


def load_chat_model(path: str, device, *, allow_base: bool = True):
    """
    Модель + ChatTokenizer из .pt (чекпоинт train_lm/train_chat) или .npz (export).
    Базовая (не чат) модель возвращается как есть, с флагом chat=False.
    """
    model, base_tok = LM.load_model(path, device)
    tok = ChatTokenizer.from_base(base_tok)
    chat = is_chat_config(model.config, base_tok)
    if not chat and not allow_base:
        raise ValueError(f"{path}: это не чат-модель (словарь {model.config.vocab_size}). Сначала sft.")
    return model, tok, chat


def pad_batch(seqs: Sequence[Tuple[List[int], List[int]]], pad_id: int, length: Optional[int] = None):
    """[(ids, mask)] -> x (B, L), y (B, L) с -100 там, где ошибку не считаем; L = длина - 1."""
    import torch
    L = (length or max(len(s[0]) for s in seqs)) - 1
    x = np.full((len(seqs), L), pad_id, dtype=np.int64)
    y = np.full((len(seqs), L), -100, dtype=np.int64)
    for i, (ids, mask) in enumerate(seqs):
        ids, mask = ids[:L + 1], mask[:L + 1]
        n = len(ids) - 1
        x[i, :n] = ids[:-1]
        tgt = np.asarray(ids[1:], dtype=np.int64)
        tgt[np.asarray(mask[1:], dtype=bool) == 0] = -100
        y[i, :n] = tgt
    return torch.from_numpy(x), torch.from_numpy(y)


def sequence_logprobs(model, x, y, routing: str = "threshold"):
    """Сумма log p по токенам, где y != -100, и их число: (B,), (B,). С градиентом, если он включён."""
    import torch
    import torch.nn.functional as F
    logits, _, _ = model(x, routing=routing)
    logp = F.log_softmax(logits.float(), dim=-1)
    valid = y != -100
    tok_lp = logp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1) * valid
    return tok_lp.sum(dim=1), valid.sum(dim=1)


def score_sequences(model, seqs: Sequence[Tuple[List[int], List[int]]], pad_id: int,
                    batch_size: int = 64, progress: bool = False) -> np.ndarray:
    """log p размеченных токенов для каждой последовательности: массив (N, 2) = [сумма log p, число токенов]."""
    import torch
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))   # похожие длины — меньше заполнения
    out = np.zeros((len(seqs), 2), dtype=np.float64)
    was_training = model.training
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for bi in range(0, len(order), batch_size):
            idx = order[bi:bi + batch_size]
            x, y = pad_batch([seqs[i] for i in idx], pad_id)
            s, n = sequence_logprobs(model, x, y)
            out[idx, 0] = s.numpy()
            out[idx, 1] = n.numpy()
            if progress and (bi // batch_size) % 200 == 0:
                print(f"  оценено {bi + len(idx):,}/{len(seqs):,} ({time.time() - t0:.0f} c)", flush=True)
    model.train(was_training)
    return out


# =============================================================================
# ГЕНЕРАЦИЯ ОТВЕТА
# =============================================================================

def _filter_logits(logits, temperature: float, top_k: int, top_p: float):
    import torch
    logits = logits / temperature
    if top_k and top_k < logits.size(-1):
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        remove = probs.cumsum(dim=-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
    return logits


_WORD_INFO: Dict[int, Tuple[List[bool], List[bool]]] = {}
COPY_START_MIN_ID = 1000          # первые кусочки с меньшим id — частые («▁Я», «▁Ко»): их не подталкиваем


def _word_info(tok: ChatTokenizer) -> Tuple[List[bool], List[bool]]:
    """Для каждого токена: начинает ли он слово («▁...») и есть ли в нём буквы."""
    key = id(tok)
    if key not in _WORD_INFO:
        starts, letters = [], []
        for i in range(tok.vocab_size):
            piece = tok.sp.id_to_piece(i) if i < tok.sp_size else ""
            starts.append(piece.startswith("\u2581"))
            letters.append(any(ch.isalpha() for ch in piece))
        _WORD_INFO[key] = (starts, letters)
    return _WORD_INFO[key]


def context_words(tok: ChatTokenizer, ids: Sequence[int]) -> List[List[int]]:
    """Слова разговора как последовательности токенов (только слова из двух и больше кусочков: имена,
    города, новые слова — то, что маленькая модель хуже всего переписывает дословно)."""
    starts, letters = _word_info(tok)
    words, cur = [], []
    for t in ids:
        t = int(t)
        if t >= tok.sp_size or not letters[t]:
            if len(cur) >= 2:
                words.append(cur)
            cur = []
            continue
        if starts[t] and cur:
            if len(cur) >= 2:
                words.append(cur)
            cur = []
        cur.append(t)
    if len(cur) >= 2:
        words.append(cur)
    uniq = {tuple(w): w for w in words}
    return list(uniq.values())


def copy_bonus(tok: ChatTokenizer, words: List[List[int]], out: List[int]) -> Dict[int, float]:
    """
    «Механизм копирования» (как кэш-модель: Grave и др., 2017). Если начало текущего слова ответа совпадает с
    началом слова из разговора, следующий кусочек этого слова получает бонус. Так «▁Ко» + «стро» → «ме»
    (Костроме), а не «Количество». Слои Mamba плохо копируют дословно (Jelassi и др., 2024), а внимания в
    модели мало — этот бонус восполняет недостающую «головку копирования» без изменения весов.
    """
    starts, letters = _word_info(tok)
    j = len(out) - 1
    while j >= 0 and letters[out[j]] and not starts[out[j]]:
        j -= 1
    if j < 0 or not (letters[out[j]] and starts[out[j]]):
        return {}
    partial = out[j:]                                   # текущее (недописанное) слово ответа
    bonus: Dict[int, float] = {}
    for w in words:
        if len(w) > len(partial) and w[:len(partial)] == partial:
            bonus[w[len(partial)]] = 1.0
    return bonus
    for w in words:
        if len(w) > len(partial) and w[:len(partial)] == partial:
            bonus[w[len(partial)]] = bonus.get(w[len(partial)], 0.0) + 1.0
    return bonus


def generate_batch(model, tok: ChatTokenizer, ctxs: Sequence[List[int]], n: int = 1, max_new_tokens: int = 40,
                   temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9,
                   repetition_penalty: float = 1.1, guidance: float = 0.0,
                   unconds: Optional[Sequence[List[int]]] = None, stop_id: Optional[int] = None,
                   banned: Optional[List[int]] = None, generator=None, copy: float = 0.0,
                   copy_start: float = 0.0) -> List[List[List[int]]]:
    """
    По n ответов на каждый из контекстов ctxs ОДИНАКОВОЙ длины — одним батчем (len(ctxs) * n строк).
    Остановка на <n> (конец реплики). Ролевые и служебные токены запрещены, пустая реплика тоже.
    temperature <= 0 — жадный выбор. Штраф за повтор действует только на токены самого ответа: иначе он
    мешал бы повторить то, что сказал собеседник («Меня зовут Всеволод» → «Тебя зовут Всена»).

    guidance > 0 — «удержание темы» (classifier-free guidance для текста): второй поток считает
    ответ без последнего вопроса (unconds, тоже одинаковой длины), и итоговые логиты
        l = l(ответ | вопрос) + guidance * (l(ответ | вопрос) − l(ответ | без вопроса))
    усиливают то, что следует именно из вопроса. Стоит второго прохода сети.

    copy > 0 — «механизм копирования» (см. copy_bonus): следующий кусочек слова из разговора получает бонус
    copy к логиту. copy_start > 0 — бонус первым кусочкам редких многокусочковых слов разговора (имена, новые слова).
    """
    import torch
    stop_id = tok.nl_id if stop_id is None else stop_id
    banned = tok.banned_ids if banned is None else banned
    if len({len(c) for c in ctxs}) != 1:
        raise ValueError("generate_batch: контексты должны быть одной длины")
    was_training = model.training
    model.eval()
    dev = model.tok_embeddings.weight.device
    B = len(ctxs) * n
    with torch.no_grad():
        def prefill(rows: Sequence[List[int]]):
            ids = torch.tensor([r for r in rows for _ in range(n)], dtype=torch.long, device=dev)
            cache = model.init_cache(B)
            logits = None
            for t in range(ids.shape[1]):
                logits = model.step(ids[:, t], cache)
            return cache, logits, ids

        cache, logits, _ = prefill(ctxs)
        history = torch.zeros(B, 0, dtype=torch.long, device=dev)     # уже написанные токены ответа
        use_g = guidance > 0 and unconds is not None and len(unconds[0]) > 0
        if use_g:
            if len({len(u) for u in unconds}) != 1:
                raise ValueError("generate_batch: контексты без вопроса должны быть одной длины")
            ucache, ulogits, _ = prefill(unconds)
        out: List[List[int]] = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        words = [context_words(tok, ctxs[i // n]) for i in range(B)] if (copy or copy_start) else None
        starts_bonus = [sorted({w[0] for w in ws if w[0] >= COPY_START_MIN_ID}) for ws in words] if copy_start else None
        ban = None
        for step in range(max_new_tokens):
            lg = logits.float()
            if use_g:
                lg = lg + guidance * (lg - ulogits.float())
            if words is not None:
                for i in torch.nonzero(~done).flatten().tolist():
                    if copy:
                        for t, v in copy_bonus(tok, words[i], out[i]).items():
                            lg[i, t] += copy * v
                    if copy_start and starts_bonus[i]:
                        lg[i, starts_bonus[i]] += copy_start
            if repetition_penalty != 1.0 and history.shape[1]:
                seen = torch.gather(lg, 1, history)
                seen = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
                lg = lg.scatter(1, history, seen)
            if ban is None:
                ban = [t for t in banned if t < lg.size(-1)]      # у базовой модели ролевых токенов нет
            lg[:, ban] = float("-inf")
            if step == 0:
                lg[:, stop_id] = float("-inf")                  # пустых реплик не бывает
            if temperature <= 0:
                nxt = lg.argmax(dim=-1)
            else:
                probs = torch.softmax(_filter_logits(lg, temperature, top_k, top_p), dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, stop_id), nxt)
            for i in torch.nonzero(~done).flatten().tolist():
                out[i].append(int(nxt[i]))
            done |= nxt == stop_id
            if bool(done.all()):
                break
            history = torch.cat([history, nxt[:, None]], dim=1)
            logits = model.step(nxt, cache)
            if use_g:
                ulogits = model.step(nxt, ucache)
    model.train(was_training)
    return [out[i * n:(i + 1) * n] for i in range(len(ctxs))]


def generate_replies(model, tok: ChatTokenizer, ctx_ids: List[int], n: int = 1, max_new_tokens: int = 40,
                     temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9,
                     repetition_penalty: float = 1.1, guidance: float = 0.0,
                     uncond_ids: Optional[List[int]] = None, stop_id: Optional[int] = None,
                     banned: Optional[List[int]] = None, generator=None, copy: float = 0.0,
                     copy_start: float = 0.0) -> List[List[int]]:
    """n ответов на один контекст (см. generate_batch)."""
    return generate_batch(model, tok, [ctx_ids], n=n, max_new_tokens=max_new_tokens, temperature=temperature,
                          top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, guidance=guidance,
                          unconds=[uncond_ids] if uncond_ids else None, stop_id=stop_id, banned=banned,
                          generator=generator, copy=copy, copy_start=copy_start)[0]


def reply_text(tok: ChatTokenizer, ids: List[int]) -> str:
    return tok.decode([t for t in ids if t != tok.nl_id]).strip()


# =============================================================================
# 1) ДОБЫЧА ПАР «ВОПРОС → ОТВЕТ» ИЗ СУБТИТРОВ
# =============================================================================

_CYR_ONLY = re.compile(r"^[А-Яа-яЁё0-9 ,.!?:;\-–—'\"«»()]+$")
_BAD_WORDS = re.compile(
    r"ху[йяеёию]|пизд|еба|ебл|ёба|бля|сук[аиу]|мудак|мудил|долбо|дерьм|говн|жоп|срат|сран|шлюх|трах|"
    r"пидор|педик|ублюд|засран|сучк|мраз|урод|задниц|член|чёрт|черт|дьявол|\bад[ау]?\b|"
    r"уби[йлтвю]|убьё|убье|убью|труп|кров|пистолет|пушк|стреля|застрел|нож[аеу ]|смерт|умер|умрё|умре|"
    r"мёртв|мертв|оружи|бомб|взорв|наркот|тюрьм|полици|копы|пытк|насил|гроб|похорон|секс|голая|голый",
    re.IGNORECASE)
_PERSONA_Q = re.compile(
    r"зовут|твоё имя|твое имя|ваше имя|как тебя|как вас|кто ты|кто вы|сколько тебе|сколько вам|откуда ты|"
    r"откуда вы|где ты жив|где вы жив|ты человек|ты робот|замужем|женат|у тебя есть|у вас есть|кем ты|"
    r"чем ты занима|чем вы занима|твоя мама|твой отец|твой папа|твоя мать|твои родители|ты мужчина|"
    r"ты женщина|ты девушка|ты парень|ты мальчик|ты девочка|ты меня любишь|ты любишь меня|"
    r"ты помнишь|ты знаешь его|ты знаешь её|ты знаешь ее",
    re.IGNORECASE)
# Женские формы о себе («я рада», «я устала»): у бота мужской род, как в остальном проекте
_FEM_SELF = re.compile(
    r"\bя\b[^.!?]*\b(\w+лась|\w{2,}ла|рада|готова|уверена|должна|согласна|одна|сама|счастлива|"
    r"влюблена|права|больна|голодна|свободна|занята|беременна|замужем)\b", re.IGNORECASE)
_NAME_CHAR = re.compile(r"[А-ЯЁ]")


def _has_inner_capital(text: str) -> bool:
    """Слово с заглавной буквы не в начале предложения — скорее всего имя персонажа."""
    words = text.split()
    for i, w in enumerate(words):
        if i == 0 or words[i - 1][-1:] in ".!?—-–":
            continue
        core = w.strip("«»\"'(),.!?:;-–—")
        if core and _NAME_CHAR.match(core[0]) and core not in ("Я",):
            return True
    return False


def good_subtitle_pair(q: str, a: str) -> bool:
    qw, aw = q.split(), a.split()
    if not (2 <= len(qw) <= 14 and 1 <= len(aw) <= 16):
        return False
    if not (q.endswith("?") and a[-1:] in ".!") or "?" in a:
        return False
    if "..." in q or "..." in a or "…" in q or "…" in a or "--" in q or "--" in a:
        return False
    if not (_CYR_ONLY.match(q) and _CYR_ONLY.match(a)):
        return False
    if not (_NAME_CHAR.match(q[0]) and _NAME_CHAR.match(a[0])):
        return False
    if _BAD_WORDS.search(q) or _BAD_WORDS.search(a) or _PERSONA_Q.search(q) or _FEM_SELF.search(a):
        return False
    if _has_inner_capital(q) or _has_inner_capital(a):
        return False
    return True


def iter_dash_pairs(path: str) -> Iterator[Tuple[str, str]]:
    """
    Соседние реплики с тире в начале — это два разных говорящих одного субтитра:
        - Ты голоден?
        - Нет, спасибо.
    Вопрос — реплика с «?» на конце, ответ — следующая реплика с тире.
    """
    prev = None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = re.sub(r"\s+", " ", line).strip()
            if prev is not None and prev.startswith("- ") and s.startswith("- "):
                q, a = prev[2:].strip(), s[2:].strip()
                if q.endswith("?"):
                    yield q, a
            prev = s


def cmd_mine(args: argparse.Namespace) -> None:
    import torch
    torch.set_flush_denormal(True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    t0 = time.time()
    per_q: Dict[str, int] = {}
    per_a: Dict[str, int] = {}
    seen = set()
    cands: List[Tuple[str, str]] = []
    n_raw = 0
    for q, a in iter_dash_pairs(args.subs):
        n_raw += 1
        if not good_subtitle_pair(q, a):
            continue
        kq, ka = q.lower(), a.lower()
        if (kq, ka) in seen or per_q.get(kq, 0) >= args.max_per_question or per_a.get(ka, 0) >= args.max_per_answer:
            continue
        seen.add((kq, ka))
        per_q[kq] = per_q.get(kq, 0) + 1
        per_a[ka] = per_a.get(ka, 0) + 1
        cands.append((q, a))
    print(f"Пар с тире: {n_raw:,}, после фильтров и удаления повторов: {len(cands):,} ({time.time() - t0:.0f} c)")
    rng.shuffle(cands)
    cands = cands[:args.max_candidates]

    model, tok, chat = load_chat_model(args.ckpt, torch.device("cpu"))
    if chat:
        raise ValueError("Для добычи пар нужна базовая (не чат) модель: она судит «по-субтитровому».")
    dash = list(tok.sp.encode("-"))

    def seq(q: str, a: str):
        ctx = list(tok.sp.encode("- " + q)) + [tok.nl_id] + dash
        ans = list(tok.sp.encode(a)) + [tok.nl_id]
        return ctx + ans, [0] * len(ctx) + [1] * len(ans)

    # Чужой вопрос — случайный вопрос другой пары: оценка «насколько ответ вероятен вообще»
    neg_q = [cands[(i + 1 + rng.randrange(len(cands) - 1)) % len(cands)][0] for i in range(len(cands))]
    seqs = [seq(q, a) for q, a in cands] + [seq(nq, a) for nq, (_, a) in zip(neg_q, cands)]
    print(f"Оценка {len(cands):,} пар моделью {args.ckpt} (свой и чужой вопрос) ...", flush=True)
    scores = score_sequences(model, seqs, tok.nl_id, batch_size=args.batch_size, progress=True)
    n = len(cands)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, (q, a) in enumerate(cands):
            lp, cnt = scores[i]
            lp_neg = scores[n + i, 0]
            f.write(json.dumps({"q": q, "a": a, "lp": round(lp, 3), "lp_neg": round(lp_neg, 3),
                                "pmi": round(lp - lp_neg, 3), "n": int(cnt)}, ensure_ascii=False) + "\n")
    pmi = scores[:n, 0] - scores[n:, 0]
    print(f"Готово: {out} ({n:,} пар, {time.time() - t0:.0f} c). PMI: медиана {np.median(pmi):.2f}, "
          f"доля > 1: {np.mean(pmi > 1):.2f}, доля > 3: {np.mean(pmi > 3):.2f}")


# =============================================================================
# 2) СБОРКА ОБУЧАЮЩИХ ДИАЛОГОВ
# =============================================================================

def read_jsonl(path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def select_mined(rows: List[dict], min_pmi: float, min_lp_tok: float, limit: int, seed: int) -> List[dict]:
    """Пары, где ответ зависит от вопроса (PMI) и сам по себе выглядит как нормальная речь."""
    good = [r for r in rows if r["pmi"] >= min_pmi and r["lp"] / max(1, r["n"]) >= min_lp_tok]
    good.sort(key=lambda r: -r["pmi"])
    good = good[:limit]
    random.Random(seed).shuffle(good)
    return good


def cmd_build(args: argparse.Namespace) -> None:
    import chat_corpus as C
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mined: List[dict] = []
    if args.mined and os.path.exists(args.mined):
        mined = select_mined(read_jsonl(args.mined), args.min_pmi, args.min_lp_tok, args.mined_limit, args.seed)
        print(f"Добытых пар из субтитров: {len(mined):,} (PMI ≥ {args.min_pmi})")
    else:
        print("Добытых пар нет (команда mine не запускалась) — только ручные и шаблонные диалоги")
    data = C.build_dataset(mined, seed=args.seed, n_compose=args.compose, mined_val=args.mined_val,
                           augment_p=args.augment)
    for name, rows in data.items():
        write_jsonl(out / f"{name}.jsonl", rows)
        print(f"  {name}: {len(rows):,}")
    stats = C.describe(data["sft_train"])
    with open(out / "build_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


# =============================================================================
# 3) SFT: ОБУЧЕНИЕ С УЧИТЕЛЕМ ТОЛЬКО ПО РЕПЛИКАМ БОТА
# =============================================================================

def truncate_dialogue(ids: List[int], mask: List[int], limit: int, user_id: int):
    """Длинный диалог: отбрасываем самые ранние реплики целиком (начинаем с <user>)."""
    if len(ids) <= limit:
        return ids, mask
    start = len(ids) - limit
    while start < len(ids) and ids[start] != user_id:
        start += 1
    if start >= len(ids) - 2:
        start = len(ids) - limit
    return ids[start:], mask[start:]


def pack_rows(examples: List[Tuple[List[int], List[int]]], row_len: int, pad_id: int,
              rng: random.Random, user_id: int) -> List[Tuple[List[int], List[int]]]:
    """
    Несколько диалогов в одну строку длины row_len (без разрезания диалога пополам), остаток —
    заполнение без ошибки. Каждая строка начинается с начала диалога.
    """
    order = list(range(len(examples)))
    rng.shuffle(order)
    rows, cur_ids, cur_mask = [], [], []
    for i in order:
        ids, mask = truncate_dialogue(*examples[i], row_len, user_id)
        if len(cur_ids) + len(ids) > row_len:
            rows.append((cur_ids + [pad_id] * (row_len - len(cur_ids)), cur_mask + [0] * (row_len - len(cur_mask))))
            cur_ids, cur_mask = [], []
        cur_ids += ids
        cur_mask += mask
    if cur_ids:
        rows.append((cur_ids + [pad_id] * (row_len - len(cur_ids)), cur_mask + [0] * (row_len - len(cur_mask))))
    return rows


def encode_rows(tok: ChatTokenizer, rows: List[dict]) -> List[Tuple[List[int], List[int], float]]:
    out = []
    for r in rows:
        ids, mask = encode_dialogue(tok, r["turns"])
        out.append((ids, mask, float(r.get("w", 1.0))))
    return out


def expand_by_weight(encoded, rng: random.Random):
    """Вес примера w: целая часть — сколько раз повторить, дробная — вероятность ещё одного раза."""
    out = []
    for ids, mask, w in encoded:
        k = int(w) + (1 if rng.random() < w - int(w) else 0)
        out.extend([(ids, mask)] * k)
    return out


def save_chat_ckpt(path: Path, model, optimizer, tok: ChatTokenizer, step: int, best_val: float,
                   extra: Optional[dict] = None) -> None:
    import torch
    tmp = path.with_name(path.name + ".tmp")
    payload = {"model": model.state_dict(), "config": model.config.to_dict(), "step": step,
               "best_val": best_val, "tokenizer_proto": tok.sp.serialized_model_proto(),
               "chat_format": CHAT_FORMAT}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, tmp)
    os.replace(tmp, path)


def lr_cosine(step: int, total: int, warmup: int, base: float, min_ratio: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))


def masked_val_loss(model, rows, batch_size: int, pad_id: int) -> float:
    import torch
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            x, y = pad_batch(rows[i:i + batch_size], pad_id)
            s, n = sequence_logprobs(model, x, y)
            tot += float(-s.sum())
            cnt += int(n.sum())
    model.train()
    return tot / max(1, cnt)


def cmd_sft(args: argparse.Namespace) -> None:
    import torch
    torch.manual_seed(args.seed)
    torch.set_flush_denormal(True)
    rng = random.Random(args.seed)
    device = LM.pick_device(args.device)
    model, tok, chat = load_chat_model(args.base, torch.device("cpu"))
    if not chat:
        model = extend_vocab(model, tok.vocab_size, init_from=tok.nl_id, seed=args.seed)
        print(f"Словарь расширен до {tok.vocab_size}: ролевые токены <user>={tok.user_id}, <bot>={tok.bot_id}")
    model = model.to(device).train()
    data = Path(args.data)
    train = encode_rows(tok, read_jsonl(data / "sft_train.jsonl"))
    val = [(ids, mask) for ids, mask, _ in encode_rows(tok, read_jsonl(data / "sft_val.jsonl"))]
    val = [truncate_dialogue(i, m, args.seq_len + 1, tok.user_id) for i, m in val]
    n_tok = sum(len(i) for i, _, _ in train)
    n_tgt = sum(sum(m) for _, m, _ in train)
    print(f"Диалогов: {len(train):,} ({n_tok:,} токенов, из них {n_tgt:,} — реплики бота), "
          f"валидация: {len(val):,}")

    args.lr = args.lr or 5e-4
    optimizer = LM.build_optimizer(model, argparse.Namespace(weight_decay=args.weight_decay, lr=args.lr), device)
    train_model = torch.compile(model) if args.compile else model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({"config": model.config.to_dict(), "args": vars(args)}, f, ensure_ascii=False, indent=2)
    metrics = open(out / "metrics.jsonl", "a", encoding="utf-8")

    def epoch_rows():
        return pack_rows(expand_by_weight(train, rng), args.seq_len + 1, tok.nl_id, rng, tok.user_id)

    rows = epoch_rows()
    steps_per_epoch = max(1, len(rows) // args.batch_size)
    total = args.max_steps or int(args.epochs * steps_per_epoch)
    print(f"Строк по {args.seq_len} токенов: {len(rows):,} за эпоху, шагов: {total} "
          f"(≈ {total / steps_per_epoch:.1f} эпох), LR {args.lr:g}")
    best = masked_val_loss(model, val, args.batch_size, tok.nl_id)
    print(f"  ВАЛИДАЦИЯ до обучения: {best:.4f}", flush=True)
    metrics.write(json.dumps({"kind": "val", "step": 0, "loss": best}) + "\n")
    if chat:                      # продолжение с чат-модели: «лучший» — только если стало лучше, чем было
        save_chat_ckpt(out / "best.pt", model, None, tok, 0, best)
    else:
        best = float("inf")
    ptr, t_last, t0 = 0, time.time(), time.time()
    for step in range(total):
        if ptr + args.batch_size > len(rows):
            rows, ptr = epoch_rows(), 0
        batch = rows[ptr:ptr + args.batch_size]
        ptr += args.batch_size
        lr = lr_cosine(step, total, args.warmup, args.lr, args.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = pad_batch(batch, tok.nl_id)
        x, y = x.to(device), y.to(device)
        _, loss, stats = train_model(x, y)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % args.log_every == 0:
            dt = time.time() - t_last
            t_last = time.time()
            tps = args.batch_size * args.seq_len * (args.log_every if step else 1) / max(dt, 1e-9)
            print(f"шаг {step:5d}/{total} | loss {float(stats['ce_loss']):.4f} | lr {lr:.2e} | |g| {float(gn):.2f} | "
                  f"{tps:,.0f} ток/с", flush=True)
            metrics.write(json.dumps({"kind": "train", "step": step, "loss": float(stats["ce_loss"]), "lr": lr,
                                      "grad_norm": float(gn), "time": time.time() - t0}) + "\n")
            metrics.flush()
        if (step + 1) % args.eval_every == 0 or step == total - 1:
            v = masked_val_loss(model, val, args.batch_size, tok.nl_id)
            improved = v < best
            best = min(best, v)
            print(f"  ВАЛИДАЦИЯ шаг {step + 1}: {v:.4f}{' | новый лучший' if improved else ''}", flush=True)
            metrics.write(json.dumps({"kind": "val", "step": step + 1, "loss": v}) + "\n")
            metrics.flush()
            if improved:
                save_chat_ckpt(out / "best.pt", model, None, tok, step + 1, best)
            if args.sample_every and ((step + 1) // args.eval_every) % args.sample_every == 0:
                for q in ("Привет! Как тебя зовут?", "Сколько будет два плюс два?", "Меня зовут Оля. Как меня зовут?"):
                    r = generate_replies(model, tok, prompt_ids(tok, [("user", q)]), n=1, temperature=0)[0]
                    print(f"    {q} -> {reply_text(tok, r)}", flush=True)
    save_chat_ckpt(out / "last.pt", model, optimizer, tok, total, best)
    metrics.close()
    print(f"Готово за {(time.time() - t0) / 60:.1f} мин: {out}/best.pt (лучший по валидации), {out}/last.pt")


# =============================================================================
# 4) ALIGN: ПООЩРЕНИЕ И НАКАЗАНИЕ (двусторонняя контрастная оптимизация предпочтений)
# =============================================================================

_GENERIC = {"да", "нет", "не знаю", "хорошо", "ладно", "конечно", "спасибо", "может быть", "наверное",
            "не уверен", "понятно", "ясно", "ок", "угу", "ага", "ну да", "да, конечно", "нет, спасибо"}


def norm_text(s: str) -> str:
    s = re.sub(r"[^\w ]+", "", s.lower().replace("ё", "е"))
    return re.sub(r"\s+", " ", s).strip()


def is_generic(reply: str) -> bool:
    """Короткие универсальные ответы подходят ко многим вопросам — их не наказываем за «чужой» вопрос."""
    t = norm_text(reply)
    return t in _GENERIC or len(t.split()) <= 1


def reply_defects(text: str, ids: List[int], tok: ChatTokenizer, context_text: str, max_new: int) -> List[str]:
    """Автоматическая награда (без учителя): что не так с ответом. Пустой список — замечаний нет."""
    bad = []
    if not ids or ids[-1] != tok.nl_id:
        bad.append("не закончил реплику")
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    if letters and sum(1 for c in letters if c.isascii()) / len(letters) > 0.2:
        bad.append("не по-русски")
    words = norm_text(text).split()
    if len(words) >= 4 and len(set(words)) / len(words) < 0.6:
        bad.append("повторы")
    if len(words) > 30:
        bad.append("слишком длинно")
    if len(words) >= 3 and norm_text(text) == norm_text(context_text):   # «Привет!» на «Привет» — не эхо
        bad.append("повторил вопрос")
    return bad


def _key_pattern(key: str):
    """Ключ проверки: короткие слова и числа — целиком («да» не совпадает с «давай», «7» — с «70»),
    длинные — по началу слова («желт» совпадает с «жёлтое»)."""
    k = norm_text(key)
    exact = len(k) <= 3 or k[-1:].isdigit()
    return re.compile(r"(?<!\w)" + re.escape(k) + (r"(?!\w)" if exact else ""))


def check_reply(text: str, check: Optional[dict]) -> Optional[bool]:
    """Проверяемая награда: ответ содержит нужное (имя, число, факт) и не содержит запрещённого."""
    if not check:
        return None
    t = norm_text(text)
    if check.get("tail"):                      # проверяем только вывод — текст после последнего «значит»
        i = t.rfind(norm_text(check["tail"]))
        if i < 0:
            return False
        t = t[i + len(norm_text(check["tail"])):]
    ok = any(_key_pattern(a).search(t) for a in check.get("any", []) if norm_text(a))
    if ok and check.get("none"):
        ok = not any(_key_pattern(a).search(t) for a in check["none"] if norm_text(a))
    return ok


class AlignItem:
    __slots__ = ("context", "reply", "check", "kind", "negatives", "positives", "sample_context")

    def __init__(self, context, reply, check, kind):
        self.context, self.reply, self.check, self.kind = context, reply, check, kind
        self.negatives: List[str] = []
        self.positives: List[str] = []
        self.sample_context = context        # на каком варианте вопроса модель отвечала сама


def align_items(rows: List[dict]) -> List[AlignItem]:
    items = []
    for r in rows:
        turns = r["turns"]
        if not turns or turns[-1][0] != "bot":
            continue
        items.append(AlignItem([tuple(t) for t in turns[:-1]], turns[-1][1], r.get("check"), r.get("src", "")))
    return items


def collect_own_samples(model, tok, items: List[AlignItem], k: int, max_new: int, temperature: float,
                        seed: int, log_every: int = 200, max_batch: int = 64, perturb: float = 0.0) -> Dict[str, float]:
    """
    Раунд «модель отвечает сама»: k ответов на каждый вопрос. Награда:
      • задачи с проверкой (имя, число, факт): верно/неверно;
      • остальное: автоматические замечания (не закончил, не по-русски, повторы, длинно, эхо).
    Плохие ответы — в наказание (negatives), хорошие на проверяемых задачах — в поощрение (positives).
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    n_ok = n_check = n_bad = n_all = 0
    t0 = time.time()
    import chat_corpus as C
    prng = random.Random(seed)
    groups: Dict[int, List[Tuple[AlignItem, List[int]]]] = {}
    for it in items:
        # «Небрежный» вариант последнего вопроса (без заглавных, без знаков, «слушай, ...»): на новых
        # формулировках модель ошибается чаще — наказание бьёт туда, где она слабее всего
        it.sample_context = it.context
        if perturb > 0 and it.context and prng.random() < perturb and not it.kind.endswith(C.NO_AUG_SRC):
            last = it.context[-1]
            it.sample_context = list(it.context[:-1]) + [(last[0], C.casual(last[1], prng))]
        ctx = prompt_ids(tok, it.sample_context)
        groups.setdefault(len(ctx), []).append((it, ctx))
    per_batch = max(1, max_batch // k)
    done_items = 0
    for _, grp in sorted(groups.items()):
        for bi in range(0, len(grp), per_batch):
            chunk = grp[bi:bi + per_batch]
            all_samples = generate_batch(model, tok, [c for _, c in chunk], n=k, max_new_tokens=max_new,
                                         temperature=temperature, top_k=0, top_p=1.0, repetition_penalty=1.0,
                                         generator=g)
            for (it, _), samples in zip(chunk, all_samples):
                it.negatives, it.positives = [], []
                last_user = it.context[-1][1] if it.context else ""
                for ids in samples:
                    text = reply_text(tok, ids)
                    n_all += 1
                    defects = reply_defects(text, ids, tok, last_user, max_new)
                    verdict = check_reply(text, it.check)
                    if verdict is not None:
                        n_check += 1
                        n_ok += int(verdict and not defects)
                    if defects or verdict is False:
                        n_bad += 1
                        if norm_text(text) != norm_text(it.reply) and text not in it.negatives:
                            it.negatives.append(text)
                    elif verdict is True and text not in it.positives:
                        it.positives.append(text)
            prev = done_items
            done_items += len(chunk)
            if log_every and done_items // log_every > prev // log_every:
                print(f"    свои ответы: {done_items}/{len(items)} | верно на проверяемых {n_ok}/{max(1, n_check)} | "
                      f"плохих {n_bad}/{n_all} ({time.time() - t0:.0f} c)", flush=True)
    return {"checked_acc": n_ok / max(1, n_check), "bad_frac": n_bad / max(1, n_all), "n": n_all}


def cmd_align(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    torch.manual_seed(args.seed)
    torch.set_flush_denormal(True)
    rng = random.Random(args.seed)
    model, tok, chat = load_chat_model(args.ckpt, torch.device("cpu"), allow_base=False)
    model.train()
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    data = Path(args.data)
    rows = read_jsonl(data / "sft_train.jsonl")
    items = align_items(rows)
    checked = [it for it in items if it.check]
    others = [it for it in items if not it.check]
    rng.shuffle(checked)
    rng.shuffle(others)
    pool = checked[:args.checked_items] + others[:args.open_items]
    rng.shuffle(pool)
    replies = [it.reply for it in items]
    contexts = [it.context for it in items if it.context]
    val_rows = read_jsonl(data / "sft_val.jsonl")
    val = [encode_dialogue(tok, r["turns"]) for r in val_rows]
    print(f"Задач: {len(pool):,} (с проверкой ответа: {min(len(checked), args.checked_items):,}), "
          f"β = {args.beta}, λ(NLL) = {args.nll_coef}, раундов: {args.rounds}")

    optimizer = LM.build_optimizer(model, argparse.Namespace(weight_decay=0.0, lr=args.lr), torch.device("cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = open(out / "metrics.jsonl", "a", encoding="utf-8")
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args)}, f, ensure_ascii=False, indent=2)
    total = args.rounds * args.steps_per_round
    step = 0
    best_val = masked_val_loss(model, val, 32, tok.nl_id)
    print(f"  ошибка SFT-модели на валидации: {best_val:.4f}")

    def seq_for(context, reply):
        ids, mask = encode_dialogue(tok, list(context) + [("bot", reply)])
        return truncate_dialogue(ids, mask, args.max_len, tok.user_id)

    t0 = time.time()
    for rnd in range(args.rounds):
        sample_set = pool[:]
        rng.shuffle(sample_set)
        sample_set = sample_set[:args.sample_items]
        print(f"Раунд {rnd + 1}/{args.rounds}: модель отвечает сама на {len(sample_set)} вопросов ...", flush=True)
        st = collect_own_samples(model, tok, sample_set, args.samples, args.max_new, args.temperature,
                                 args.seed + rnd, perturb=args.perturb)
        print(f"  верно на проверяемых: {st['checked_acc']:.3f} | доля плохих ответов: {st['bad_frac']:.3f}", flush=True)
        metrics.write(json.dumps({"kind": "samples", "round": rnd, **st}) + "\n")
        # Половина батча — задачи, где модель ошиблась сама (наказание за свои ошибки), половина — любые
        with_neg = [it for it in pool if it.negatives]
        print(f"  задач, где модель ошиблась сама: {len(with_neg)}", flush=True)
        for _ in range(args.steps_per_round):
            n_neg = min(len(with_neg), args.batch_size // 2)
            batch = [rng.choice(with_neg) for _ in range(n_neg)] + \
                [rng.choice(pool) for _ in range(args.batch_size - n_neg)]
            seqs, pairs, nll_idx = [], [], []
            for it in batch:
                c = len(seqs)
                seqs.append(seq_for(it.context, it.reply))
                nll_idx.append(c)
                chosen = [c]
                # свой плохой ответ сравнивается с правильным на том же варианте вопроса, на который отвечала модель
                sc = it.sample_context
                own_chosen = [c] if sc is it.context else [len(seqs)]
                if sc is not it.context:
                    seqs.append(seq_for(sc, it.reply))
                # поощрение: свой верный ответ модели тоже «хороший» (rejection sampling)
                if it.positives and args.own_positives:
                    seqs.append(seq_for(sc, rng.choice(it.positives)))
                    own_chosen.append(len(seqs) - 1)
                # 1) свой плохой ответ (по награде)
                if it.negatives:
                    seqs.append(seq_for(sc, rng.choice(it.negatives)))
                    pairs.append((rng.choice(own_chosen), len(seqs) - 1))
                # 2) чужой ответ на этот вопрос
                if args.mismatch:
                    other = rng.choice(replies)
                    if norm_text(other) != norm_text(it.reply) and not (is_generic(other) and is_generic(it.reply)):
                        seqs.append(seq_for(it.context, other))
                        pairs.append((c, len(seqs) - 1))
                # 3) этот ответ на чужой вопрос: ответ должен зависеть от вопроса
                if args.context_swap and it.context and not is_generic(it.reply):
                    other_ctx = rng.choice(contexts)
                    if norm_text(other_ctx[-1][1]) != norm_text(it.context[-1][1]):
                        seqs.append(seq_for(other_ctx, it.reply))
                        pairs.append((c, len(seqs) - 1))
            x, y = pad_batch(seqs, tok.nl_id)
            lr = lr_cosine(step, total, args.warmup, args.lr, 0.1)
            for g in optimizer.param_groups:
                g["lr"] = lr
            with torch.no_grad():
                ref_lp, _ = sequence_logprobs(ref, x, y)
            pol_lp, cnt = sequence_logprobs(model, x, y)
            ci = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            ri = torch.tensor([p[1] for p in pairs], dtype=torch.long)
            if args.length_norm:
                # Средний log p на токен (как в SimPO): иначе короткий ответ выигрывает просто потому, что
                # он короткий, и модель учится отвечать голым числом (проверено: счёт упал с 35% до 12%)
                n_tok = cnt.clamp(min=1).float()
                pol_n, ref_n = pol_lp / n_tok, ref_lp / n_tok
            else:
                pol_n, ref_n = pol_lp, ref_lp
            margin = (pol_n[ci] - ref_n[ci]) - (pol_n[ri] - ref_n[ri])
            # Потолок отрыва: уже надёжно разведённые пары (β·отрыв > max_margin) больше не тянут модель —
            # иначе она уходит всё дальше от SFT-модели и портит то, что уже умела (проверено: −4% точности)
            active = (args.beta * margin.detach() < args.max_margin).float()
            pref = (-F.logsigmoid(args.beta * margin) * active).sum() / max(1, len(pairs)) if len(pairs) \
                else pol_lp.sum() * 0
            ni = torch.tensor(nll_idx, dtype=torch.long)
            nll = -(pol_lp[ni].sum() / cnt[ni].sum().clamp(min=1))
            loss = pref + args.nll_coef * nll
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % args.log_every == 0:
                acc = float((margin > 0).float().mean()) if len(pairs) else float("nan")
                print(f"шаг {step:5d}/{total} | pref {float(pref.detach()):.4f} | nll {float(nll.detach()):.4f} | "
                      f"верный порядок {acc:.2f} | отрыв {float(margin.detach().mean()) if len(pairs) else 0:.2f} | "
                      f"пар {len(pairs)} | lr {lr:.1e} | |g| {float(gn):.2f} | {time.time() - t0:.0f} c", flush=True)
                metrics.write(json.dumps({"kind": "train", "step": step, "pref": float(pref.detach()),
                                          "nll": float(nll.detach()),
                                          "pair_acc": acc, "pairs": len(pairs), "lr": lr}) + "\n")
                metrics.flush()
            step += 1
        v = masked_val_loss(model, val, 32, tok.nl_id)
        print(f"  ошибка на валидации после раунда {rnd + 1}: {v:.4f}", flush=True)
        metrics.write(json.dumps({"kind": "val", "step": step, "loss": v}) + "\n")
        metrics.flush()
        save_chat_ckpt(out / f"round{rnd + 1}.pt", model, None, tok, step, v)
    save_chat_ckpt(out / "last.pt", model, None, tok, step, best_val)
    metrics.close()
    print(f"Готово за {(time.time() - t0) / 60:.1f} мин: {out}/last.pt")


# =============================================================================
# 5) EVAL: ЧЕСТНАЯ ПРОВЕРКА ДО/ПОСЛЕ
# =============================================================================

def base_prompt_ids(tok: ChatTokenizer, turns) -> List[int]:
    """Для базовой модели (без ролевых токенов) — тот же диалог в формате субтитров: «- реплика»."""
    ids: List[int] = []
    for _, text in turns:
        ids += list(tok.sp.encode("- " + clean_turn(text))) + [tok.nl_id]
    return ids + list(tok.sp.encode("-"))


def answer(model, tok, chat: bool, turns, greedy: bool = True, generator=None, guidance: float = 0.0,
           max_new: int = 40, copy: float = 0.0, copy_start: float = 0.0) -> str:
    if chat:
        ctx = prompt_ids(tok, turns)
        uncond = prompt_ids(tok, list(turns[:-1])) if guidance > 0 else None
        ids = generate_replies(model, tok, ctx, n=1, max_new_tokens=max_new, temperature=0 if greedy else 0.7,
                               guidance=guidance, uncond_ids=uncond, generator=generator,
                               repetition_penalty=1.0 if greedy else 1.1, copy=copy, copy_start=copy_start)[0]
    else:
        ids = generate_replies(model, tok, base_prompt_ids(tok, turns), n=1, max_new_tokens=max_new,
                               temperature=0 if greedy else 0.7, generator=generator,
                               repetition_penalty=1.0 if greedy else 1.1)[0]
    text = reply_text(tok, ids)
    return re.sub(r"^\s*[-–—]\s*", "", text)


NEUTRAL_QUESTIONS = ("Привет!", "Расскажи что-нибудь.", "Что нового?")


def answer_rerank(model, tok, turns, n: int = 8, alpha: float = 1.0, temperature: float = 0.8, max_new: int = 64,
                  generator=None, details: bool = False):
    """
    «Гипотезы и самопроверка» без дообучения. Модель пишет несколько вариантов ответа (жадный + n − 1
    случайных), а потом сама проверяет каждый вариант двумя вопросами:
      • насколько ответ вероятен после ЭТОГО вопроса (средний log p на токен — без выгоды коротким ответам);
      • насколько он вероятнее после этого вопроса, чем после нейтральных («Привет!», ...): отвечает ли он
        именно на этот вопрос, а не «вообще» (PMI). Узнавать свой вопрос по ответу модель умеет гораздо
        лучше, чем сразу писать верный ответ (84–89% против 45%), и этот разрыв здесь и используется.
    Итог = правдоподобие + alpha · PMI. Незаконченные реплики отбрасываются. Стоит ~n проходов генерации.
    """
    ctx = prompt_ids(tok, turns)
    cands = [generate_replies(model, tok, ctx, n=1, max_new_tokens=max_new, temperature=0,
                              repetition_penalty=1.0)[0]]
    if n > 1:
        cands += generate_replies(model, tok, ctx, n=n - 1, max_new_tokens=max_new, temperature=temperature,
                                  generator=generator)
    uniq, seen = [], set()
    for c in cands:
        if c and c[-1] == tok.nl_id and tuple(c) not in seen:
            seen.add(tuple(c))
            uniq.append(c)
    uniq = uniq or [cands[0]]
    seqs = [(ctx + c, [0] * len(ctx) + [1] * len(c)) for c in uniq]
    for q in NEUTRAL_QUESTIONS:
        nctx = prompt_ids(tok, list(turns[:-1]) + [("user", q)])
        seqs += [(nctx + c, [0] * len(nctx) + [1] * len(c)) for c in uniq]
    sc = score_sequences(model, seqs, tok.nl_id, batch_size=64)
    k = len(uniq)
    mean_lp = sc[:, 0] / np.maximum(sc[:, 1], 1)
    lp = mean_lp[:k]
    neutral = mean_lp[k:].reshape(len(NEUTRAL_QUESTIONS), k).mean(axis=0)
    score = lp + alpha * (lp - neutral)
    best = int(np.argmax(score))
    text = reply_text(tok, uniq[best])
    if details:
        return text, [(reply_text(tok, c), float(l), float(l - u), float(v))      # (ответ, log p, PMI, итог)
                      for c, l, u, v in zip(uniq, lp, neutral, score)]
    return text


def choice_accuracy(model, tok, chat: bool, rows: List[dict]) -> Dict[str, float]:
    """
    Выбор ответа из 5: правильный + 4 ответа на другие вопросы. Две оценки:
      • «ответ к вопросу»: какой из 5 ответов модель считает самым вероятным после вопроса;
      • «вопрос к ответу»: после какого из 5 вопросов этот ответ самый вероятный (PMI, частота ответа
        сокращается) — прямо измеряет, слышит ли модель вопрос.
    Случайное угадывание — 20%.
    """
    seqs = []

    def make(q_turns, a):
        if chat:
            return encode_dialogue(tok, list(q_turns) + [("bot", a)])
        ctx = base_prompt_ids(tok, q_turns)
        ans = list(tok.sp.encode(a)) + [tok.nl_id]
        return ctx + ans, [0] * len(ctx) + [1] * len(ans)

    for r in rows:
        for a in [r["answer"]] + r["distractors"]:
            seqs.append(make(r["context"], a))
        for c in r["other_contexts"]:
            seqs.append(make(c, r["answer"]))
    sc = score_sequences(model, seqs, tok.nl_id, batch_size=64)
    k = 1 + len(rows[0]["distractors"]) + len(rows[0]["other_contexts"])
    res: Dict[str, List[Tuple[int, int]]] = {}
    for i in range(len(rows)):
        block = sc[i * k:(i + 1) * k, 0]
        nd = 1 + len(rows[i]["distractors"])
        a_ok = int(np.argmax(block[:nd]) == 0)
        q_ok = int(np.argmax(np.concatenate([[block[0]], block[nd:]])) == 0)
        for key in (rows[i].get("src", "все"), "ВСЕГО"):
            res.setdefault(key, []).append((a_ok, q_ok))
    return {key: {"ответ к вопросу": float(np.mean([v[0] for v in vals])),
                  "вопрос к ответу": float(np.mean([v[1] for v in vals])), "n": len(vals)}
            for key, vals in res.items()}


def cmd_eval(args: argparse.Namespace) -> None:
    import torch
    import chat_corpus as C
    torch.set_flush_denormal(True)
    data = Path(args.data)
    probes = read_jsonl(data / "probes.jsonl")[:args.probes or None]
    choice = read_jsonl(data / "choice.jsonl")[:args.choice_items]
    open_set = read_jsonl(data / "open_eval.jsonl")[:args.open_items]
    report = {}
    showcase = {}
    for path in args.ckpt:
        model, tok, chat = load_chat_model(path, torch.device("cpu"))
        name = Path(path).name
        t0 = time.time()
        per_cat: Dict[str, List[int]] = {}
        probe_out = []
        gen = torch.Generator().manual_seed(args.seed)
        for p in probes:
            if chat and args.rerank > 1:
                text = answer_rerank(model, tok, p["turns"], n=args.rerank, alpha=args.rerank_alpha,
                                     max_new=args.max_new, generator=gen)
            else:
                text = answer(model, tok, chat, p["turns"], guidance=args.guidance if chat else 0.0,
                              max_new=args.max_new, copy=args.copy if chat else 0.0,
                              copy_start=args.copy_start if chat else 0.0)
            ok = bool(check_reply(text, p["check"]))
            per_cat.setdefault(p["cat"], []).append(int(ok))
            probe_out.append({"cat": p["cat"], "q": p["turns"][-1][1], "reply": text, "ok": ok})
        probe_acc = {c: float(np.mean(v)) for c, v in sorted(per_cat.items())}
        probe_acc["ВСЕГО"] = float(np.mean([o["ok"] for o in probe_out]))
        ch = choice_accuracy(model, tok, chat, choice)
        # Форма ответов на открытых вопросах: закончена ли реплика, русский ли язык, длина
        ended = rus = 0
        lens, echo = [], 0
        for r in open_set:
            turns = r["turns"][:-1]                  # последняя реплика — эталонный ответ бота
            ctx = prompt_ids(tok, turns) if chat else base_prompt_ids(tok, turns)
            ids = generate_replies(model, tok, ctx, n=1, max_new_tokens=40, temperature=0,
                                   repetition_penalty=1.0)[0]
            text = reply_text(tok, ids)
            ended += int(bool(ids) and ids[-1] == tok.nl_id)
            d = reply_defects(text, ids, tok, turns[-1][1], 40)
            rus += int("не по-русски" not in d)
            echo += int("повторил вопрос" in d)
            lens.append(len(text.split()))
        form = {"ended": ended / len(open_set), "russian": rus / len(open_set), "echo": echo / len(open_set),
                "mean_words": float(np.mean(lens))}
        sc = []
        g = torch.Generator().manual_seed(0)
        for q in C.SHOWCASE[:args.showcase or None]:
            turns = [("user", t) if i % 2 == 0 else ("bot", t) for i, t in enumerate(q)]
            sc.append({"q": q, "greedy": answer(model, tok, chat, turns, guidance=args.guidance if chat else 0.0),
                       "sampled": answer(model, tok, chat, turns, greedy=False, generator=g,
                                         guidance=args.guidance if chat else 0.0)})
        report[name] = {"chat": chat, "probes": probe_acc, "choice": ch, "form": form,
                        "seconds": round(time.time() - t0, 1)}
        showcase[name] = {"showcase": sc, "probes": probe_out}
        print(f"\n=== {name} ({'чат' if chat else 'базовая'}) за {time.time() - t0:.0f} c")
        print("  задачи с проверкой:", json.dumps(probe_acc, ensure_ascii=False))
        print("  выбор из 5:", json.dumps(ch, ensure_ascii=False))
        print("  форма:", json.dumps(form, ensure_ascii=False))
        for s in sc[:args.print_showcase]:
            print(f"   {s['q'][-1]!r} -> {s['greedy']!r}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open(Path(args.json).with_name(Path(args.json).stem + "_replies.json"), "w", encoding="utf-8") as f:
            json.dump(showcase, f, ensure_ascii=False, indent=1)


# =============================================================================
# 6) CHAT  7) EXPORT
# =============================================================================

def load_dictionary(path: str) -> Dict[str, Tuple[str, str]]:
    """
    Внешний словарь: строки «слово — определение» (или «слово | определение»), # — комментарий.
    Модель не хранит словарь в весах: определение подкладывается в разговор, когда слово встречается
    в вопросе, а модель обучена отвечать по такой справке (chat_corpus.py, шаблон «справка»).
    """
    out: Dict[str, Tuple[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+[—–-]\s+|\s*\|\s*", line, maxsplit=1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                continue
            word, definition = parts[0].strip(), parts[1].strip()
            if not definition.lower().startswith("это "):
                definition = "это " + definition[:1].lower() + definition[1:]
            if not definition.endswith((".", "!", "?")):
                definition += "."
            out[word.lower().replace("ё", "е")] = (word, definition)
    return out


def dictionary_hit(text: str, dictionary: Dict[str, Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    """Первое слово сообщения, которое есть в словаре (с грубым учётом падежей: «синхрофазотрона»)."""
    for w in re.findall(r"[а-яёa-z-]+", text.lower().replace("ё", "е")):
        if w in dictionary:
            return dictionary[w]
        for cut in (1, 2):                     # окончание: астролябию → астролябия, барометра → барометр
            stem = w[:-cut] if len(w) - cut >= 4 else None
            if not stem:
                continue
            for head in (stem, stem + "а", stem + "я", stem + "ь", stem + "й"):
                if head in dictionary and len(head) >= 5:
                    return dictionary[head]
    return None


def cmd_chat(args: argparse.Namespace) -> None:
    import torch
    torch.manual_seed(args.seed)
    model, tok, chat = load_chat_model(args.ckpt, torch.device("cpu"), allow_base=False)
    g = torch.Generator().manual_seed(args.seed)
    history: List[Tuple[str, str]] = []
    dictionary = load_dictionary(args.dictionary) if args.dictionary else {}
    if dictionary:
        print(f"Словарь: {len(dictionary):,} слов из {args.dictionary}")
    print("Диалог с SLR-Net (пустая строка — выход, /новый — начать заново)")
    while True:
        try:
            text = input("\nВы: ").strip() if args.prompt is None else args.prompt
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        if text == "/новый":
            history = []
            continue
        hit = dictionary_hit(text, dictionary) if dictionary else None
        if hit:
            import chat_corpus as C
            print(f"   (справка из словаря: {hit[0]} — {hit[1]})")
            text = C.reference_turn(hit[0], hit[1], text)
        history.append(("user", text))
        ctx = prompt_ids(tok, history)
        while len(ctx) > args.context and len(history) > 1:      # длинная история: забываем начало
            history = history[2:]
            ctx = prompt_ids(tok, history)
        uncond = prompt_ids(tok, history[:-1]) if args.guidance > 0 else None
        t0 = time.time()
        ids = generate_replies(model, tok, ctx, n=1, max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                               repetition_penalty=args.repetition_penalty, guidance=args.guidance,
                               uncond_ids=uncond, generator=g, copy=args.copy, copy_start=args.copy_start)[0]
        reply = reply_text(tok, ids)
        history.append(("bot", reply))
        print(f"SLR-Net: {reply}   ({time.time() - t0:.2f} c)")
        if args.prompt is not None:
            break


def cmd_export(args: argparse.Namespace) -> None:
    import torch
    model, tok, chat = load_chat_model(args.ckpt, torch.device("cpu"), allow_base=False)
    packed = model.export_packed()
    extra = {
        "tokenizer_proto": np.frombuffer(tok.sp.serialized_model_proto(), dtype=np.uint8),
        "meta.chat_format": np.array(CHAT_FORMAT),
        "meta.user_id": np.array([tok.user_id], dtype=np.int64),
        "meta.bot_id": np.array([tok.bot_id], dtype=np.int64),
    }
    if args.ckpt.endswith(".pt"):
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        extra["meta.step"] = np.array([ck.get("step", 0)], dtype=np.int64)
        extra["meta.best_val"] = np.array([ck.get("best_val", float("nan"))], dtype=np.float32)
    np.savez_compressed(args.out, config=json.dumps(model.config.to_dict()), **extra, **packed)
    print(f"Экспортировано в {args.out}: {os.path.getsize(args.out) / 1024 ** 2:.2f} МБ")


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Пост-тренинг SLR-Net: из продолжателя текста — в собеседника")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine", help="пары «вопрос → ответ» из субтитров + отбор самой моделью (PMI)")
    m.add_argument("--subs", required=True, help="текст субтитров, по реплике на строку")
    m.add_argument("--ckpt", default="slr_net_subtitles_ru_mini.npz", help="базовая модель-судья")
    m.add_argument("--out", default="data/chat/mined.jsonl")
    m.add_argument("--max-candidates", type=int, default=200_000)
    m.add_argument("--max-per-question", type=int, default=8)
    m.add_argument("--max-per-answer", type=int, default=20)
    m.add_argument("--batch-size", type=int, default=128)
    m.add_argument("--seed", type=int, default=7)

    b = sub.add_parser("build", help="сборка обучающих диалогов")
    b.add_argument("--out", default="data/chat")
    b.add_argument("--mined", default="data/chat/mined.jsonl")
    b.add_argument("--min-pmi", type=float, default=1.5)
    b.add_argument("--min-lp-tok", type=float, default=-4.5, help="средний log p токена ответа не ниже")
    b.add_argument("--mined-limit", type=int, default=60_000)
    b.add_argument("--mined-val", type=int, default=1000, help="сколько добытых пар отложить для проверки")
    b.add_argument("--compose", type=int, default=30_000, help="сколько многоходовых диалогов собрать")
    b.add_argument("--augment", type=float, default=0.3, help="доля «небрежных» реплик пользователя")
    b.add_argument("--seed", type=int, default=11)

    s = sub.add_parser("sft", help="обучение с учителем по репликам бота")
    s.add_argument("--base", default="slr_net_subtitles_ru_mini.npz")
    s.add_argument("--data", default="data/chat")
    s.add_argument("--out", default="runs/chat_sft")
    s.add_argument("--seq-len", type=int, default=256)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--epochs", type=float, default=2.0)
    s.add_argument("--max-steps", type=int, default=0, help="0 — по числу эпох")
    s.add_argument("--lr", type=float, default=None)
    s.add_argument("--min-lr-ratio", type=float, default=0.1)
    s.add_argument("--warmup", type=int, default=100)
    s.add_argument("--weight-decay", type=float, default=0.05)
    s.add_argument("--grad-clip", type=float, default=1.0)
    s.add_argument("--eval-every", type=int, default=200)
    s.add_argument("--sample-every", type=int, default=1, help="печатать примеры каждые N проверок (0 — нет)")
    s.add_argument("--log-every", type=int, default=20)
    s.add_argument("--compile", action="store_true")
    s.add_argument("--device", default="auto")
    s.add_argument("--seed", type=int, default=1234)

    a = sub.add_parser("align", help="поощрение и наказание: двусторонняя контрастная оптимизация")
    a.add_argument("--ckpt", required=True, help="SFT-модель")
    a.add_argument("--data", default="data/chat")
    a.add_argument("--out", default="runs/chat_align")
    a.add_argument("--rounds", type=int, default=1)
    a.add_argument("--steps-per-round", type=int, default=300)
    a.add_argument("--batch-size", type=int, default=12)
    a.add_argument("--checked-items", type=int, default=6000, help="задач с проверяемым ответом")
    a.add_argument("--open-items", type=int, default=12000, help="открытых вопросов")
    a.add_argument("--sample-items", type=int, default=3000, help="на скольких задачах модель отвечает сама за раунд")
    a.add_argument("--samples", type=int, default=4, help="своих ответов на задачу")
    a.add_argument("--temperature", type=float, default=1.0)
    a.add_argument("--max-new", type=int, default=48)
    a.add_argument("--max-len", type=int, default=160)
    a.add_argument("--beta", type=float, default=0.1)
    a.add_argument("--nll-coef", type=float, default=1.0)
    a.add_argument("--max-margin", type=float, default=3.0, help="потолок β·отрыв: дальше пару не раздвигаем")
    a.add_argument("--perturb", type=float, default=0.5, help="доля «небрежных» вариантов вопроса при своих ответах")
    a.add_argument("--lr", type=float, default=3e-5)
    a.add_argument("--warmup", type=int, default=30)
    a.add_argument("--length-norm", action="store_true", help="сравнивать средний log p на токен, а не сумму")
    a.add_argument("--no-own-positives", dest="own_positives", action="store_false",
                   help="поощрять только эталонный ответ, а не свои верные ответы модели")
    a.add_argument("--no-mismatch", dest="mismatch", action="store_false")
    a.add_argument("--no-context-swap", dest="context_swap", action="store_false")
    a.add_argument("--log-every", type=int, default=20)
    a.add_argument("--seed", type=int, default=99)

    e = sub.add_parser("eval", help="проверка до/после")
    e.add_argument("--ckpt", nargs="+", required=True)
    e.add_argument("--data", default="data/chat")
    e.add_argument("--choice-items", type=int, default=600)
    e.add_argument("--open-items", type=int, default=200)
    e.add_argument("--guidance", type=float, default=0.0)
    e.add_argument("--rerank", type=int, default=0,
                   help="гипотезы и самопроверка: столько вариантов ответа, лучший — по самопроверке (0 — выкл)")
    e.add_argument("--rerank-alpha", type=float, default=1.0, help="вес «отвечает ли именно на этот вопрос» (PMI)")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--copy", type=float, default=0.0, help="механизм копирования слов из разговора (0 — выкл)")
    e.add_argument("--copy-start", type=float, default=0.0, help="бонус началу редких слов из разговора")
    e.add_argument("--max-new", type=int, default=64,
                   help="предел длины ответа в токенах (64 — как в браузере; рассуждению нужно больше 40)")
    e.add_argument("--probes", type=int, default=0, help="сколько задач с проверкой (0 — все)")
    e.add_argument("--showcase", type=int, default=0, help="сколько примеров диалога (0 — все)")
    e.add_argument("--print-showcase", type=int, default=12)
    e.add_argument("--json", default=None)

    c = sub.add_parser("chat", help="диалог в консоли")
    c.add_argument("--ckpt", default="slr_net_chat_ru_mini.npz")
    c.add_argument("--prompt", default=None, help="один вопрос без интерактивного режима")
    c.add_argument("--max-new-tokens", type=int, default=64)
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--top-k", type=int, default=40)
    c.add_argument("--top-p", type=float, default=0.9)
    c.add_argument("--repetition-penalty", type=float, default=1.1)
    c.add_argument("--guidance", type=float, default=0.0, help="удержание темы (0 — выкл, 1–2 — сильнее)")
    c.add_argument("--context", type=int, default=240)
    c.add_argument("--copy", type=float, default=3.0,
                   help="механизм копирования слов из разговора (0 — выкл; 3 подобрано на проверке)")
    c.add_argument("--copy-start", type=float, default=2.0, help="бонус началу редких слов из разговора")
    c.add_argument("--dictionary", default=None,
                   help="внешний словарь «слово — определение»: определение подкладывается в разговор")
    c.add_argument("--seed", type=int, default=0)

    x = sub.add_parser("export", help="упаковать веса чат-модели")
    x.add_argument("--ckpt", required=True)
    x.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    {"mine": cmd_mine, "build": cmd_build, "sft": cmd_sft, "align": cmd_align, "eval": cmd_eval,
     "chat": cmd_chat, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
