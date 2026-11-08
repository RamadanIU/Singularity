#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Самодообучение с ростом нейронов (идея 1 из PLAN_MULTIMODULE_RU.md).
=====================================================================

Модель получает новую информацию («Курдяпа — это маленькая птица синего цвета, она живёт в лесу») и сама
переносит её из разговора в свои веса: после этого она отвечает о курдяпе в НОВОМ разговоре, где объяснения
уже нет. Старые веса не трогаются: для новой информации модель отращивает себе новые нейроны.

Как это устроено:

  1. Новые нейроны. К MLP слоёв внимания добавляется параллельная ветка из k нейронов:
         y = MLP(x) + down(relu(up(norm(x)) + b)^2),
     down в начале — нули, поэтому в момент вставки модель считает РОВНО то же, что и без них (как LoRA
     и Net2Net). Старые троичные веса заморожены — испортить их нечем.
  2. Учитель — она сама. Инструмент вопросов задаёт вопросы о новой информации («Какого цвета курдяпа?»,
     «Где она живёт?», «Это животное?»), а модель отвечает на них, ВИДЯ объяснение в разговоре (этому она
     обучена: «не знаю — объясни → понял → применяю»). Новые нейроны учатся давать те же ответы БЕЗ
     объяснения: знание из контекста переносится в веса (дистилляция контекста, Snell и др., 2022; похожая
     самоправка весов — SEAL, Zweiger и др., 2025).
  3. Не мешать старому. Одновременно на старых диалогах (и на вопросах о словах, которых модель не учила)
     ответ с новыми нейронами должен совпадать с ответом без них (KL-дивергенция к исходной модели).
     Иначе новые нейроны «выдумывали» бы знания о любых незнакомых словах.
  4. Рост по требованию. Начинаем с малого числа нейронов. После обучения модель сама проверяет себя:
     совпадают ли её ответы без объяснения с ответами учителя. Если нет — добавляет ещё столько же нейронов
     (старые продолжают учиться) и учится дальше. Так нейронов получается столько, сколько нужно для этой
     информации, а не заранее заданное число.
  5. Модуль знаний — отдельный файл (несколько сотен КБ), его можно подключить, отключить или передать
     другому; исходный файл модели не меняется.

  python self_grow.py learn  --text "Курдяпа — это маленькая птица синего цвета. Она живёт в лесу." --out runs/kurd.npz
  python self_grow.py chat   --module runs/kurd.npz          # /запомни <текст> — выучить прямо в разговоре
  python self_grow.py experiment --words 20 --json training/self_grow/exp_20.json   # опыт с цифрами
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import chat_corpus as C
import slr_net_engine as E
import train_chat as T

MODULE_FORMAT = "slr-growth-v1"

if E.TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # =========================================================================
    # 1. НОВЫЕ НЕЙРОНЫ
    # =========================================================================

    class NeuronChunk(nn.Module):
        """k новых нейронов: вход d -> k (с порогом b), ReLU², выход k -> d. Выход в начале — нули."""

        def __init__(self, d: int, k: int, bias_init: float = -0.5, seed: int = 0):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.up = nn.Parameter(torch.randn(k, d, generator=g) / math.sqrt(d))
            self.bias = nn.Parameter(torch.full((k,), float(bias_init)))
            self.down = nn.Parameter(torch.zeros(d, k))
            self.tag = ""                       # чей это кусок: имя модуля знаний (для оркестратора)

        @property
        def k(self) -> int:
            return self.up.shape[0]

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            a = F.relu(F.linear(h, self.up.to(h.dtype), self.bias.to(h.dtype))).square()
            return F.linear(a, self.down.to(h.dtype))

    class GrownMLP(nn.Module):
        """
        Обёртка вокруг MLP слоя внимания: старый MLP (заморожен) + куски новых нейронов. Вход новых нейронов —
        тот же нормированный вход, что у старого MLP (его RMSNorm). enabled=False — модуль отключён
        (модель считает в точности как исходная): так же считается и «учитель» без модуля.
        """

        def __init__(self, base_mlp: nn.Module):
            super().__init__()
            self.base = base_mlp
            self.chunks = nn.ModuleList()
            self.enabled = True
            self.active: Optional[set] = None   # None — все куски; иначе только куски этих модулей

        @property
        def n_neurons(self) -> int:
            return sum(ch.k for ch in self.chunks)

        def grow(self, k: int, bias_init: float, seed: int) -> NeuronChunk:
            d = self.base.norm.weight.shape[0]
            ch = NeuronChunk(d, k, bias_init=bias_init, seed=seed).to(self.base.norm.weight.device)
            self.chunks.append(ch)
            return ch

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.base(x)
            if not self.enabled or len(self.chunks) == 0:
                return y
            chunks = [ch for ch in self.chunks if self.active is None or ch.tag in self.active]
            if not chunks:
                return y
            h = self.base.norm(x)
            extra = chunks[0](h)
            for ch in chunks[1:]:
                extra = extra + ch(h)
            return y + extra

    def mlp_sites(model) -> List[Tuple[str, nn.Module]]:
        """Все блоки внимания с MLP (у mini2 — 6: 4 в прелюдии, 2 в коде): куда можно добавить нейроны."""
        sites = []
        for part in ("encoder", "decoder"):
            for i, block in enumerate(getattr(model, part)):
                if isinstance(block, E.LocalAttentionBlock) and block.mlp is not None:
                    sites.append((f"{part}.{i}", block))
        return sites

    def attach(model, sites: Optional[Sequence[str]] = None) -> Dict[str, GrownMLP]:
        """Оборачивает MLP выбранных блоков в GrownMLP (без нейронов: модель считает как раньше)."""
        out: Dict[str, GrownMLP] = {}
        for name, block in mlp_sites(model):
            if sites and name not in sites:
                continue
            if not isinstance(block.mlp, GrownMLP):
                block.mlp = GrownMLP(block.mlp)
            out[name] = block.mlp
        if not out:
            raise ValueError("В модели нет блоков внимания с MLP — некуда добавлять нейроны")
        return out

    def grown_layers(model) -> Dict[str, GrownMLP]:
        return {name: block.mlp for name, block in mlp_sites(model) if isinstance(block.mlp, GrownMLP)}

    def set_enabled(model, enabled: bool) -> None:
        for g in grown_layers(model).values():
            g.enabled = enabled

    def set_active(model, tags: Optional[Sequence[str]]) -> None:
        """Оркестратор: включить только модули с этими именами (None — все, [] — ни одного)."""
        for g in grown_layers(model).values():
            g.active = None if tags is None else set(tags)

    def grow_all(layers: Dict[str, GrownMLP], k: int, bias_init: float, seed: int) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for j, (name, g) in enumerate(sorted(layers.items())):
            ch = g.grow(k, bias_init, seed=seed * 1009 + j)
            params += list(ch.parameters())
        return params

    def n_new_params(model) -> int:
        return sum(p.numel() for g in grown_layers(model).values() for p in g.chunks.parameters())

    def freeze_base(model) -> None:
        for name, p in model.named_parameters():
            p.requires_grad_(".chunks." in name)

    # =========================================================================
    # 2. ФАЙЛ МОДУЛЯ ЗНАНИЙ
    # =========================================================================

    def save_module(path: str, model, meta: dict) -> None:
        """Только новые нейроны (float16) + описание: что выучено, ключ для оркестратора."""
        arrays: Dict[str, np.ndarray] = {}
        for name, g in grown_layers(model).items():
            for j, ch in enumerate(g.chunks):
                for pname in ("up", "bias", "down"):
                    arrays[f"{name}.{j}.{pname}"] = getattr(ch, pname).detach().float().cpu().numpy().astype(np.float16)
        meta = {**meta, "format": MODULE_FORMAT, "n_params": n_new_params(model),
                "neurons": {name: g.n_neurons for name, g in grown_layers(model).items()}}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=json.dumps(meta, ensure_ascii=False), **arrays)

    def read_module(path: str) -> Tuple[dict, Dict[str, np.ndarray]]:
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads(str(arrays.pop("meta")))
        if meta.get("format") != MODULE_FORMAT:
            raise ValueError(f"{path}: не модуль знаний ({meta.get('format')})")
        return meta, arrays

    def load_module(model, path: str, tag: Optional[str] = None) -> dict:
        """Подключает модуль знаний к модели (можно несколько: их нейроны складываются). tag — имя модуля
        для оркестратора (по умолчанию — имя файла)."""
        meta, arrays = read_module(path)
        tag = tag or Path(path).stem
        meta["tag"] = tag
        layers = attach(model, sorted(meta["neurons"]))
        chunks: Dict[Tuple[str, int], Dict[str, np.ndarray]] = defaultdict(dict)
        for key, value in arrays.items():
            name, j, pname = key.rsplit(".", 2)
            chunks[(name, int(j))][pname] = value
        for (name, j), parts in sorted(chunks.items()):
            g = layers[name]
            ch = g.grow(parts["up"].shape[0], 0.0, seed=0)
            ch.tag = tag
            with torch.no_grad():
                for pname in ("up", "bias", "down"):
                    getattr(ch, pname).copy_(torch.from_numpy(parts[pname].astype(np.float32)))
        return meta


# =============================================================================
# 3. ЗНАНИЯ И ИНСТРУМЕНТ ВОПРОСОВ
# =============================================================================

@dataclass
class Fact:
    """Одно новое слово: само слово, текст, который «прислал пользователь», и (для опыта) разметка смысла."""
    word: str
    text: str
    info: Optional[dict] = None


def fact_from_info(word: str, info: dict, rng: random.Random) -> Fact:
    return Fact(word, C.explain(word, info, rng), info)


def question_bank(fact: Fact, train: bool, seed: int = 0, tries: int = 160) -> List[Tuple[str, Optional[dict], str]]:
    """
    Вопросы о новом слове: [(вопрос, проверка или None, вид)]. Если смысл размечен (опыт) — все виды вопросов
    из chat_corpus.concept_question: train=True — обучающие формулировки, train=False — ОТЛОЖЕННЫЕ (их модель
    при самодообучении не видит). Если нет — вопросы по шаблонам из текста (text_questions).
    """
    if fact.info is None:
        return [(q, None, kind) for q, kind in text_questions(fact.word, fact.text)]
    rng = random.Random(seed)
    seen: Dict[str, Tuple[str, Optional[dict], str]] = {}
    for _ in range(tries):
        q, _a, c, src = C.concept_question(fact.word, fact.info, rng, train)
        kind = "вывод" if src.endswith("вывод") else _kind_of(q, fact)
        seen.setdefault(q, (q, c, kind))
    return list(seen.values())


def _kind_of(q: str, fact: Fact) -> str:
    ql = q.lower()
    for key, kind in (("цвет", "цвет"), ("величин", "размер"), ("размер", "размер"), ("больш", "размер"),
                      ("маленьк", "размер"), ("где", "место"), ("каком месте", "место"), ("для чего", "зачем"),
                      ("зачем", "зачем"), ("с помощью", "зачем"), (" или ", "выбор")):
        if key in ql:
            return kind
    return "что"


PLACE_RE = re.compile(r"\b(живёт|живет|растёт|растет|лежит|ездит|бывает|водится|стоит|работает)\s+((?:в|во|на|у|по|под)\s+[а-яё]+)")
COLOR_RE = re.compile(r"\b([а-яё]+(?:ого|его))\s+цвета\b")


def text_questions(word: str, text: str) -> List[Tuple[str, str]]:
    """
    Инструмент вопросов для свободного текста: общие вопросы о слове + вопросы о признаках, которые нашлись
    в тексте по шаблонам (цвет, место, размер). Ответы на них даёт сама модель, видя текст. Для сложных
    текстов этот инструмент нужно заменить моделью-«составителем вопросов» (PLAN_MULTIMODULE_RU.md, раздел 1.6).
    Тот же список вопросов строит страница (textQuestions в slr_net_mobile_chat.html) — тест сверяет их.
    """
    W, w = C.cap(word), word.lower()
    qs = [(f"Что такое {w}?", "что"), (f"Расскажи про слово «{w}».", "что"), (f"{W} — это что?", "что"),
          (f"Что ты знаешь про слово «{w}»?", "что"), (f"Объясни ещё раз, что такое {w}.", "что"),
          (f"Так что такое {w}?", "что"), (f"Ты знаешь, что такое {w}?", "что"), (f"Что значит слово «{w}»?", "что")]
    t = text.lower()
    if COLOR_RE.search(t):
        qs += [(f"Какого цвета {w}?", "цвет"), (f"{W} какого цвета?", "цвет"), (f"А какого цвета {w}?", "цвет")]
    m = PLACE_RE.search(t)
    if m:
        verb = m.group(1)
        qs += [(f"Где {verb} {w}?", "место"), (f"А где {verb} {w}?", "место"), (f"В каком месте {verb} {w}?", "место"),
               (f"Где обычно {verb} {w}?", "место")]
    if re.search(r"\b(маленьк|крошечн|больш|огромн)", t):
        qs += [(f"Какой величины {w}?", "размер"), (f"{W} какой величины?", "размер")]
    return qs


DEF_RE = re.compile(r"^\s*([А-ЯЁA-Z][а-яёa-z-]+)\s+[—–-]\s+(?:это\s+)?(.+)$")


def facts_from_text(text: str) -> List[Fact]:
    """
    Свободный текст -> новые слова. Каждое предложение вида «Слово — это ...» начинает новое слово, следующие
    предложения без такого начала относятся к нему же («Она живёт в лесу.»).
    """
    facts: List[Fact] = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not sent:
            continue
        m = DEF_RE.match(sent)
        if m:
            facts.append(Fact(m.group(1).lower(), sent))
        elif facts:
            facts[-1] = Fact(facts[-1].word, facts[-1].text + " " + sent)
    return facts


# =============================================================================
# 4. ГЕНЕРАЦИЯ ПАЧКАМИ, УЧИТЕЛЬ, ПРОВЕРКИ
# =============================================================================

def greedy_many(model, tok, ctxs: Sequence[List[int]], max_new: int = 48, batch: int = 96) -> List[List[int]]:
    """Жадные ответы на много контекстов: контексты одной длины идут одной пачкой."""
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, c in enumerate(ctxs):
        groups[len(c)].append(i)
    out: List[Optional[List[int]]] = [None] * len(ctxs)
    for _, idx in sorted(groups.items()):
        for b in range(0, len(idx), batch):
            part = idx[b:b + batch]
            res = T.generate_batch(model, tok, [ctxs[i] for i in part], n=1, max_new_tokens=max_new,
                                   temperature=0, repetition_penalty=1.0)
            for i, r in zip(part, res):
                out[i] = r[0]
    return out  # type: ignore[return-value]


def teacher_turns(fact: Fact, question: str, style: str) -> List[Tuple[str, str]]:
    """Разговор, в котором модель-учитель видит объяснение. inline — объяснение и вопрос одной репликой."""
    if style == "inline":
        return [("user", fact.text + " " + question)]
    if style == "reference":
        definition = fact.text.split(" — ", 1)[1] if " — " in fact.text else fact.text
        return [("user", C.reference_turn(fact.word, definition, question))]
    raise ValueError(f"Неизвестный вид учителя: {style}")


def ask(model, tok, turns_list: Sequence[List[Tuple[str, str]]], max_new: int = 48) -> List[str]:
    ctxs = [T.prompt_ids(tok, turns) for turns in turns_list]
    return [T.reply_text(tok, ids) for ids in greedy_many(model, tok, ctxs, max_new=max_new)]


def accuracy(texts: Sequence[str], checks: Sequence[Optional[dict]]) -> float:
    vals = [bool(T.check_reply(t, c)) for t, c in zip(texts, checks) if c]
    return float(np.mean(vals)) if vals else float("nan")


DONT_KNOW = C.chk("не знаю", "незнаком", "не рассказывал", "не объяснял")


def control_questions(words: Sequence[str]) -> List[Tuple[str, dict]]:
    """Слова, которых модель НЕ учила: честный ответ — «не знаю» (иначе новые нейроны выдумывают знания)."""
    out = []
    for w in words:
        out += [(f"Что такое {w}?", DONT_KNOW), (f"Расскажи, что такое {w}.", DONT_KNOW)]
    return out


# =============================================================================
# 5. САМООБУЧЕНИЕ
# =============================================================================

@dataclass
class GrowConfig:
    mode: str = "grow"            # grow — новые нейроны; full — дообучение всех весов; raw — нейроны, но учатся на сыром тексте
    k_init: int = 8               # нейронов на слой в начале (у mini2 6 слоёв с MLP)
    k_max: int = 256              # не больше стольких нейронов на слой
    grow_factor: float = 2.0      # при росте — во сколько раз больше нейронов (добавляется k·(f−1))
    steps: int = 150              # шагов обучения на каждом размере
    lr: float = 3e-3
    full_lr: float = 3e-4         # для mode=full
    batch: int = 32
    replay_batch: int = 8
    kl_coef: float = 1.0          # вес «не менять старое» (KL к исходной модели на старых диалогах)
    soft_coef: float = 0.0        # > 0: ещё и KL к распределению учителя на каждом токене ответа (не только его слова)
    bias_init: float = -0.5
    target: float = 0.85          # самопроверка 1: доля НОВЫХ для модели токенов ответа учителя (исходная модель без
    #                               объяснения их не угадывает), которые модель угадывает без объяснения на СВОИХ
    #                               ОТЛОЖЕННЫХ вопросах (их нет в обучении: проверяется понимание, а не заучивание)
    keep_target: float = 0.97     # самопроверка 2: на старых диалогах и невыученных словах первый выбор модели совпадает
    #                               с исходной моделью не реже этого (проверяется, что новое не вредит старому)
    check_frac: float = 0.2       # какая доля вопросов учителя откладывается для самопроверки 1
    neg_frac: float = 0.5         # доля вопросов о невыученных словах в пачке «не менять старое»
    max_rounds: int = 6
    teacher: str = "inline"
    sites: Optional[List[str]] = None
    seed: int = 0
    replay_rows: int = 1000
    replay_len: int = 96
    negative_words: int = 60      # слов из NONCE_TRAIN, которых не учим: про них ответ без изменений («не знаю»)
    topk: int = 32                # сколько самых вероятных токенов исходной модели хранить для KL
    bf16: bool = True             # умножения в bf16 при обучении (на CPU с AMX ~1.5× быстрее); проверка — float32


def autocast(cfg: GrowConfig):
    from contextlib import nullcontext
    return torch.autocast("cpu", dtype=torch.bfloat16) if cfg.bf16 else nullcontext()


def build_teacher_data(model, tok, facts: Sequence[Fact], cfg: GrowConfig, log=print):
    """Модель-учитель (без модуля, с объяснением в разговоре) отвечает на обучающие вопросы."""
    items = []
    for f in facts:
        for q, c, kind in question_bank(f, train=True, seed=cfg.seed):
            items.append((f, q, c, kind))
    t0 = time.time()
    set_enabled(model, False)            # учитель — исходная модель (уже выученные модули тоже выключены)
    answers = ask(model, tok, [teacher_turns(f, q, cfg.teacher) for f, q, _, _ in items])
    set_enabled(model, True)
    acc = accuracy(answers, [c for _, _, c, _ in items])
    log(f"  учитель ответил на {len(items)} вопросов за {time.time() - t0:.0f} c; верно (по разметке): {acc:.1%}")
    seqs = [T.encode_dialogue(tok, [("user", q), ("bot", a)]) for (_, q, _, _), a in zip(items, answers)]
    teacher_ctx = [T.encode_dialogue(tok, teacher_turns(f, q, cfg.teacher) + [("bot", a)])
                   for (f, q, _, _), a in zip(items, answers)]
    return seqs, teacher_ctx, items, answers, acc


def raw_text_seqs(tok, facts: Sequence[Fact]):
    """Для сравнения: учиться прямо на тексте объяснения, как при обычном дообучении на тексте."""
    seqs = []
    for f in facts:
        ids = tok.encode_line(f.text) + [tok.nl_id]
        seqs.append((ids, [0] + [1] * (len(ids) - 1)))
    return seqs


def replay_seqs(model, tok, data_dir: Optional[str], cfg: GrowConfig, exclude: Sequence[str]):
    """
    Старые диалоги + вопросы о НЕвыученных словах (с ответами исходной модели — «не знаю»): на них модель с
    модулем должна отвечать так же, как без него (KL по всем токенам ответа). Возвращает (старые, невыученные).
    """
    rng = random.Random(cfg.seed + 7)
    rows: List[dict] = []
    if data_dir and (Path(data_dir) / "sft_train.jsonl").exists():
        allrows = T.read_jsonl(Path(data_dir) / "sft_train.jsonl")
        rows = rng.sample(allrows, min(cfg.replay_rows, len(allrows)))
    seqs = [T.truncate_dialogue(*T.encode_dialogue(tok, r["turns"]), cfg.replay_len, tok.user_id) for r in rows]
    excl = {w.lower() for w in exclude}
    neg_words = [w for w in C.NONCE_TRAIN if w not in excl]
    rng.shuffle(neg_words)
    neg = [[("user", q)] for w in neg_words[:cfg.negative_words]
           for q in (f"Что такое {w}?", f"Ты знаешь, что такое {w}?", f"Какого цвета {w}?", f"Расскажи про слово «{w}».")]
    if neg:
        set_enabled(model, False)
        answers = ask(model, tok, neg)
        set_enabled(model, True)
        neg_seqs = [T.encode_dialogue(tok, turns + [("bot", a)]) for turns, a in zip(neg, answers)]
    else:
        neg_seqs = []
    keep = lambda xs: [x for x in xs if sum(x[1][1:]) > 0]
    return keep(seqs), keep(neg_seqs)


def topk_targets(model, tok, seqs, cfg: GrowConfig, batch: int = 32):
    """
    Распределение ИСХОДНОЙ модели (модули выключены) на размеченных позициях каждой последовательности:
    k самых вероятных токенов и их log p. Считается один раз — на шагах обучения второй проход не нужен.
    """
    out = []
    set_enabled(model, False)
    model.eval()
    with torch.no_grad(), autocast(cfg):
        for i in range(0, len(seqs), batch):
            part = seqs[i:i + batch]
            x, y = T.pad_batch(part, tok.nl_id)
            lp = F.log_softmax(model(x, routing="threshold")[0].float(), dim=-1)
            for r in range(len(part)):
                top = lp[r][y[r] != -100].topk(cfg.topk, dim=-1)
                out.append((top.indices, top.values))
    set_enabled(model, True)
    return out


def masked_ce(logits, y):
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)


def topk_kl(logits_new, y, targets) -> "torch.Tensor":
    """KL(исходная || новая) по k самым вероятным токенам исходной модели (остаток вероятности — малая доля)."""
    lp_new = F.log_softmax(logits_new.float()[y != -100], dim=-1)
    idx = torch.cat([t[0] for t in targets])
    lp_ref = torch.cat([t[1] for t in targets])
    return (lp_ref.exp() * (lp_ref - lp_new.gather(1, idx))).sum(-1).mean()


def _argmax_hits(model, tok, seqs, batch: int = 64) -> List["torch.Tensor"]:
    """Для каждой последовательности: угадан ли каждый размеченный токен первым выбором модели."""
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            part = seqs[i:i + batch]
            x, y = T.pad_batch(part, tok.nl_id)
            pred = model(x, routing="threshold")[0].argmax(-1)
            for r in range(len(part)):
                v = y[r] != -100
                out.append(pred[r][v] == y[r][v])
    return out


def informative_masks(model, tok, seqs) -> List["torch.Tensor"]:
    """Токены ответа учителя, которые исходная модель БЕЗ объяснения не угадывает: в них и есть новое знание
    (повтор слова из вопроса, «это», «цвета» модель угадывает и так). Модули знаний на время выключаются."""
    was = [g.enabled for g in grown_layers(model).values()]
    set_enabled(model, False)
    masks = [~h for h in _argmax_hits(model, tok, seqs)]
    for g, e in zip(grown_layers(model).values(), was):
        g.enabled = e
    return masks


def self_check(model, tok, seqs, masks: Optional[List["torch.Tensor"]] = None, batch: int = 64) -> float:
    """
    Самопроверка без разметки: доля токенов ответа учителя, которые модель без объяснения ставит первыми.
    masks — считать только эти токены (informative_masks: только новое знание; у исходной модели там 0%).
    """
    hits = _argmax_hits(model, tok, seqs, batch)
    if masks is not None:
        hit = sum(int(h[m].sum()) for h, m in zip(hits, masks))
        tot = sum(int(m.sum()) for m in masks)
    else:
        hit = sum(int(h.sum()) for h in hits)
        tot = sum(int(h.numel()) for h in hits)
    return hit / max(1, tot)


def keep_check(model, tok, items, batch: int = 64) -> float:
    """Самопроверка вреда: доля позиций, где первый выбор модели совпадает с исходной (из сохранённых целей)."""
    hit = tot = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(items), batch):
            part = items[i:i + batch]
            x, y = T.pad_batch([(it[0], it[1]) for it in part], tok.nl_id)
            logits, _, _ = model(x, routing="threshold")
            pred = logits[y != -100].argmax(-1)
            ref = torch.cat([it[2][0][:, 0] for it in part])
            hit += int((pred == ref).sum())
            tot += int(ref.numel())
    return hit / max(1, tot)


def train_steps(model, tok, params, student, replay, cfg: GrowConfig, steps: int, lr: float,
                log=print, tag: str = "") -> Dict[str, float]:
    """
    Обучение новых нейронов (или всех весов при mode=full). student — [(ids, mask, мягкие цели учителя или None)],
    replay — (старые, невыученные): списки [(ids, mask, цели исходной модели)]. Ошибка = CE на ответах учителя (+ soft_coef · KL к учителю)
    + kl_coef · KL к исходной модели на старых диалогах.
    """
    rng = random.Random(cfg.seed * 31 + steps + len(params))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.98))
    model.eval()                         # маршрутизация ядра как при ответе ("threshold"); дропаута нет
    t0 = time.time()
    last = {}
    for step in range(steps):
        cur_lr = lr * min(1.0, (step + 1) / 10) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(1, steps))))
        for gr in opt.param_groups:
            gr["lr"] = cur_lr
        sb = [student[rng.randrange(len(student))] for _ in range(cfg.batch)]
        x, y = T.pad_batch([(s[0], s[1]) for s in sb], tok.nl_id)
        with autocast(cfg):
            logits, _, _ = model(x, routing="threshold")
        ce = masked_ce(logits, y)
        loss = ce
        soft = kl = torch.zeros(())
        if cfg.soft_coef > 0 and sb[0][2] is not None:
            soft = topk_kl(logits, y, [s[2] for s in sb])
            loss = loss + cfg.soft_coef * soft
        old_r, neg_r = replay if replay else ([], [])
        if (old_r or neg_r) and cfg.kl_coef > 0:
            n_neg = int(round(cfg.replay_batch * cfg.neg_frac)) if neg_r else 0
            n_neg = cfg.replay_batch if not old_r else n_neg
            rb = [neg_r[rng.randrange(len(neg_r))] for _ in range(n_neg)] + \
                 [old_r[rng.randrange(len(old_r))] for _ in range(cfg.replay_batch - n_neg)]
            xr, yr = T.pad_batch([(r[0], r[1]) for r in rb], tok.nl_id)
            with autocast(cfg):
                new_logits, _, _ = model(xr, routing="threshold")
            kl = topk_kl(new_logits, yr, [r[2] for r in rb])
            loss = loss + cfg.kl_coef * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        last = {"ce": float(ce.detach()), "soft": float(soft.detach()), "kl": float(kl.detach())}
        if step % 50 == 0 or step == steps - 1:
            log(f"    {tag}шаг {step:4d}/{steps}: ошибка на ответах учителя {last['ce']:.3f}"
                + (f", KL к учителю {last['soft']:.3f}" if cfg.soft_coef > 0 else "")
                + f", KL на старом {last['kl']:.4f} ({time.time() - t0:.0f} c)", flush=True)
    return last


def self_learn(model, tok, facts: Sequence[Fact], cfg: GrowConfig, data_dir: Optional[str] = "data/chat",
               log=print) -> dict:
    """
    Главная функция: модель сама выучивает факты. Возвращает отчёт: сколько нейронов понадобилось, самопроверка
    по раундам, точность учителя. Модель меняется на месте (у неё появляются новые нейроны / меняются веса).
    """
    torch.manual_seed(cfg.seed)
    words = [f.word for f in facts]
    report: dict = {"mode": cfg.mode, "words": words, "rounds": []}
    teacher_seqs, teacher_ctx, items, answers, t_acc = build_teacher_data(model, tok, facts, cfg, log)
    report["teacher_train_acc"] = t_acc
    t0 = time.time()
    # свои отложенные вопросы для самопроверки: по check_frac вопросов каждого слова в обучение не идут
    rng = random.Random(cfg.seed + 3)
    by_word: Dict[str, List[int]] = defaultdict(list)
    for i, (f, _, _, _) in enumerate(items):
        by_word[f.word].append(i)
    check_idx = set()
    for idx in by_word.values():
        idx = idx[:]
        rng.shuffle(idx)
        check_idx.update(idx[:int(len(idx) * cfg.check_frac)])
    train_idx = [i for i in range(len(items)) if i not in check_idx]
    check_seqs = [teacher_seqs[i] for i in sorted(check_idx)] or teacher_seqs
    if cfg.mode == "raw":
        student = [(i, m, None) for i, m in raw_text_seqs(tok, facts)]
    else:
        soft = topk_targets(model, tok, [teacher_ctx[i] for i in train_idx], cfg) if cfg.soft_coef > 0 \
            else [None] * len(train_idx)
        student = [(teacher_seqs[i][0], teacher_seqs[i][1], s) for i, s in zip(train_idx, soft)]
    old_s, neg_s = replay_seqs(model, tok, data_dir, cfg, exclude=words)
    old_r = [(i, m, t) for (i, m), t in zip(old_s, topk_targets(model, tok, old_s, cfg))] if old_s else []
    neg_r = [(i, m, t) for (i, m), t in zip(neg_s, topk_targets(model, tok, neg_s, cfg))] if neg_s else []
    replay = (old_r, neg_r) if cfg.kl_coef > 0 else ([], [])
    keep_items = neg_r + old_r[:200]
    log(f"  обучающих примеров: {len(student)}, отложено для самопроверки: {len(check_seqs)}, старых диалогов: "
        f"{len(old_r)}, вопросов о невыученных словах: {len(neg_r)} (подготовка {time.time() - t0:.0f} c)")

    train_seqs = [teacher_seqs[i] for i in train_idx]
    check_masks, train_masks = informative_masks(model, tok, check_seqs), informative_masks(model, tok, train_seqs)
    report["informative_tokens"] = {"held_out": int(sum(int(m.sum()) for m in check_masks)),
                                    "all_held_out": int(sum(int(m.numel()) for m in check_masks))}
    log(f"  новых для модели токенов в отложенных ответах учителя: {report['informative_tokens']['held_out']} из "
        f"{report['informative_tokens']['all_held_out']}")

    def check(r: int, k: int) -> bool:
        gen = self_check(model, tok, check_seqs, check_masks)
        mem = self_check(model, tok, train_seqs, train_masks)
        keep = keep_check(model, tok, keep_items) if keep_items else 1.0
        report["rounds"].append({"neurons_per_layer": k, "self_check_heldout": gen, "self_check_train": mem,
                                 "keep": keep, "new_params": n_new_params(model) if k else 0,
                                 "seconds": round(time.time() - t0, 1)})
        log(f"  самопроверка ({'все веса' if not k else f'{k} нейр./слой, {n_new_params(model):,} новых параметров'}):"
            f" новые токены в своих отложенных вопросах {gen:.1%} (в обучающих {mem:.1%}), совпадение с исходной на "
            f"старом {keep:.1%}")
        return gen >= cfg.target and keep >= cfg.keep_target
    if cfg.mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        params = [p for p in model.parameters()]
        for r in range(cfg.max_rounds):
            train_steps(model, tok, params, student, replay, cfg, cfg.steps, cfg.full_lr, log=log,
                        tag=f"[все веса, раунд {r + 1}] ")
            if check(r, 0):
                break
        for p in model.parameters():
            p.requires_grad_(False)
        report["new_params"] = 0
        report["seconds"] = round(time.time() - t0, 1)
        return report
    layers = attach(model, cfg.sites)
    freeze_base(model)
    params: List = []
    k_total = 0
    k_next = cfg.k_init
    for r in range(cfg.max_rounds):
        params += grow_all(layers, k_next, cfg.bias_init, seed=cfg.seed * 7 + r)
        k_total += k_next
        train_steps(model, tok, params, student, replay, cfg, cfg.steps, cfg.lr, log=log,
                    tag=f"[{k_total} нейр./слой] ")
        if check(r, k_total) or cfg.grow_factor <= 1.0:
            break
        k_next = int(round(k_total * (cfg.grow_factor - 1.0)))
        if k_total + k_next > cfg.k_max:
            log(f"  предел {cfg.k_max} нейронов на слой — дальше не растём")
            break
    for p in params:
        p.requires_grad_(False)
    report["neurons_per_layer"] = k_total
    report["new_params"] = n_new_params(model)
    report["seconds"] = round(time.time() - t0, 1)
    return report


# =============================================================================
# 6. ПРОВЕРКИ
# =============================================================================

OPEN_KINDS = ("что", "цвет", "место", "зачем")     # угадать почти нельзя (в «выбор», «вывод», «размер» — 50%)


def open_recall(kinds: Dict[str, float], counts: Dict[str, int]) -> float:
    n = sum(counts.get(k, 0) for k in OPEN_KINDS)
    return sum(kinds[k] * counts[k] for k in OPEN_KINDS if k in kinds) / max(1, n)


def eval_recall(model, tok, facts: Sequence[Fact], train: bool, seed: int = 0) -> Tuple[float, Dict[str, float], list]:
    """Новый разговор, объяснения нет: верно ли модель отвечает о выученных словах (по разметке)."""
    items = [(f, q, c, kind) for f in facts for q, c, kind in question_bank(f, train=train, seed=seed + 1)]
    texts = ask(model, tok, [[("user", q)] for _, q, _, _ in items])
    by_kind: Dict[str, List[bool]] = defaultdict(list)
    for (_, _, c, kind), t in zip(items, texts):
        by_kind[kind].append(bool(T.check_reply(t, c)))
    kinds = {k: float(np.mean(v)) for k, v in sorted(by_kind.items())}
    kinds["ОТКРЫТЫЕ"] = open_recall(kinds, {k: len(v) for k, v in by_kind.items()})
    rows = [{"word": f.word, "q": q, "a": t, "ok": bool(T.check_reply(t, c))} for (f, q, c, _), t in zip(items, texts)]
    return accuracy(texts, [c for _, _, c, _ in items]), kinds, rows


def eval_in_context(model, tok, facts: Sequence[Fact], style: str, seed: int = 0) -> Dict[str, float]:
    """Верхний ориентир: та же модель, но объяснение лежит в разговоре (отложенные формулировки вопросов)."""
    items = [(f, q, c, kind) for f in facts for q, c, kind in question_bank(f, train=False, seed=seed + 1)]
    texts = ask(model, tok, [teacher_turns(f, q, style) for f, q, _, _ in items])
    by_kind: Dict[str, List[bool]] = defaultdict(list)
    for (_, _, c, kind), t in zip(items, texts):
        by_kind[kind].append(bool(T.check_reply(t, c)))
    kinds = {k: float(np.mean(v)) for k, v in sorted(by_kind.items())}
    return {"ВСЕ": accuracy(texts, [c for _, _, c, _ in items]),
            "ОТКРЫТЫЕ": open_recall(kinds, {k: len(v) for k, v in by_kind.items()}), **kinds}


def eval_honesty(model, tok, control_words: Sequence[str]) -> Tuple[float, list]:
    qs = control_questions(control_words)
    texts = ask(model, tok, [[("user", q)] for q, _ in qs])
    return accuracy(texts, [c for _, c in qs]), [{"q": q, "a": t} for (q, _), t in zip(qs, texts)]


def eval_old_skills(model, tok, data_dir: str, probes_every: int = 3, val_rows: int = 511) -> dict:
    """Не забыла ли модель старое: задачи с проверкой (каждая probes_every-я из probes.jsonl) и ошибка на
    отложенных диалогах."""
    data = Path(data_dir)
    probes = T.read_jsonl(data / "probes.jsonl")[::probes_every]
    texts = ask(model, tok, [[tuple(t) for t in p["turns"]] for p in probes], max_new=64)
    ok = [bool(T.check_reply(t, p["check"])) for t, p in zip(texts, probes)]
    val = [T.truncate_dialogue(i, m, 257, tok.user_id)
           for i, m, _ in T.encode_rows(tok, T.read_jsonl(data / "sft_val.jsonl")[:val_rows])]
    loss = T.masked_val_loss(model, val, 16, tok.nl_id)
    model.eval()
    return {"probes": float(np.mean(ok)), "n_probes": len(ok), "val_loss": loss,
            "probe_ok": ok}


# =============================================================================
# 7. ОПЫТ
# =============================================================================

def make_facts(words: Sequence[str], seed: int) -> List[Fact]:
    rng = random.Random(seed)
    return [fact_from_info(w, C.concept_info(w, rng), rng) for w in words]


def cmd_experiment(args: argparse.Namespace) -> None:
    torch.set_flush_denormal(True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    pool = list(C.NONCE_TEST)
    rng.shuffle(pool)
    learn_words, control_words = pool[:args.words], pool[args.words:args.words + args.control]
    if len(learn_words) < args.words:
        extra = [w for w in C.NONCE_TRAIN if w not in pool]
        rng.shuffle(extra)
        learn_words += extra[:args.words - len(learn_words)]
    facts = make_facts(learn_words, args.seed)
    base, tok, _ = T.load_chat_model(args.ckpt, torch.device("cpu"), allow_base=False)
    base.eval()
    out: dict = {"ckpt": args.ckpt, "words": args.words, "facts": [{"word": f.word, "text": f.text} for f in facts],
                 "control_words": control_words, "runs": {}}
    print(f"Слов выучить: {len(facts)}; для проверки честности — {len(control_words)} невыученных")
    for f in facts[:3]:
        print(f"  {f.text}")

    def evaluate(model, name: str, report: Optional[dict] = None):
        t0 = time.time()
        rec_test, kinds_test, rows_test = eval_recall(model, tok, facts, train=False, seed=args.seed)
        rec_train, kinds_train, _ = eval_recall(model, tok, facts, train=True, seed=args.seed)
        hon, hon_rows = eval_honesty(model, tok, control_words)
        res = {"recall_heldout": rec_test, "recall_heldout_by_kind": kinds_test, "recall_train_forms": rec_train,
               "recall_train_by_kind": kinds_train,
               "honesty_unlearned": hon, "examples": rows_test[:24], "honesty_examples": hon_rows[:6]}
        if not args.no_old:
            res["old"] = eval_old_skills(model, tok, args.data, probes_every=args.probes_every)
        if report:
            res["learn"] = report
        res["eval_seconds"] = round(time.time() - t0, 1)
        out["runs"][name] = res
        old = res.get("old", {})
        print(f"\n### {name}: без объяснения, отложенные вопросы {rec_test:.1%}, из них открытые "
              f"{kinds_test['ОТКРЫТЫЕ']:.1%} (обучающие формулировки {rec_train:.1%}, открытые "
              f"{kinds_train['ОТКРЫТЫЕ']:.1%}); честность на невыученных {hon:.1%}"
              + (f"; старые задачи {old['probes']:.1%}, ошибка на старых диалогах {old['val_loss']:.4f}" if old else ""))
        print("   по видам:", json.dumps(kinds_test, ensure_ascii=False))
        for r in rows_test[:6]:
            print(f"   {r['q']!r} -> {r['a']!r} {'✓' if r['ok'] else '✗'}")
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    if "base" in args.runs:
        out["in_context"] = {s: eval_in_context(base, tok, facts, s, seed=args.seed) for s in args.in_context}
        print(f"Объяснение в разговоре (отложенные вопросы): {json.dumps(out['in_context'], ensure_ascii=False)}")
        evaluate(base, "base")
    for run in args.runs:
        if run == "base":
            continue
        model = copy.deepcopy(base)
        mode, _, k = run.partition(":")
        cfg = GrowConfig(mode=mode if mode in ("grow", "full", "raw") else "grow", seed=args.seed, steps=args.steps,
                         teacher=args.teacher, kl_coef=args.kl, lr=args.lr, k_init=args.k_init, target=args.target,
                         max_rounds=args.max_rounds, bias_init=args.bias_init, full_lr=args.full_lr,
                         soft_coef=args.soft, keep_target=args.keep_target, check_frac=args.check_frac,
                         neg_frac=args.neg_frac, sites=args.sites or None, k_max=args.k_max)
        if mode == "fixed":                       # фиксированное число нейронов: один раунд без роста
            cfg.k_init, cfg.max_rounds, cfg.grow_factor = int(k), 1, 1.0
        if mode == "full":
            cfg.max_rounds = min(cfg.max_rounds, 2)
            if k:
                cfg.kl_coef = float(k)
        if mode == "raw":
            cfg.k_init, cfg.max_rounds, cfg.grow_factor = int(k or 64), 1, 1.0
        print(f"\n=== {run}")
        report = self_learn(model, tok, facts, cfg, data_dir=args.data)
        if args.save_modules and mode != "full":
            save_module(str(Path(args.save_modules) / f"module_{run.replace(':', '_')}.npz"), model,
                        {"words": [f.word for f in facts], "texts": [f.text for f in facts],
                         "infos": [f.info for f in facts], "control_words": control_words, "report": report})
        evaluate(model, run, report)
    print("\nГотово." + (f" Итоги — в {args.json}" if args.json else ""))


# =============================================================================
# 8. ВЫУЧИТЬ ТЕКСТ И ПОГОВОРИТЬ
# =============================================================================

def learn_text(model, tok, text: str, cfg: GrowConfig, data_dir: Optional[str], log=print) -> Tuple[List[Fact], dict]:
    facts = facts_from_text(text)
    if not facts:
        raise ValueError("Не нашёл в тексте определений вида «Слово — это ...»")
    log(f"Новых слов: {len(facts)} ({', '.join(f.word for f in facts)})")
    report = self_learn(model, tok, facts, cfg, data_dir=data_dir, log=log)
    return facts, report


def cmd_learn(args: argparse.Namespace) -> None:
    torch.set_flush_denormal(True)
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    model, tok, _ = T.load_chat_model(args.ckpt, torch.device("cpu"), allow_base=False)
    for m in args.module or []:
        load_module(model, m)
    cfg = GrowConfig(seed=args.seed, steps=args.steps, k_init=args.k_init, target=args.target, lr=args.lr)
    facts, report = learn_text(model, tok, text, cfg, args.data)
    save_module(args.out, model, {"words": [f.word for f in facts], "texts": [f.text for f in facts],
                                  "report": report, "key": module_key(model, tok, facts)})
    print(f"Модуль знаний: {args.out} ({os.path.getsize(args.out) / 1024:.0f} КБ, "
          f"{report['new_params']:,} новых параметров, {report['seconds']:.0f} c)")
    for f in facts:
        for q, _, _ in question_bank(f, train=True)[:3]:
            print(f"  {q} -> {ask(model, tok, [[('user', q)]])[0]}")


def module_key(model, tok, facts: Sequence[Fact]) -> List[float]:
    """Ключ модуля для оркестратора: средний вектор вопросов о выученных словах (без модуля)."""
    import orchestrator as O
    qs = [q for f in facts for q, _, _ in question_bank(f, train=True)]
    return O.embed_texts(model, tok, qs).mean(axis=0).tolist()


def cmd_chat(args: argparse.Namespace) -> None:
    torch.set_flush_denormal(True)
    model, tok, _ = T.load_chat_model(args.ckpt, torch.device("cpu"), allow_base=False)
    for m in args.module or []:
        meta = load_module(model, m)
        print(f"Подключён модуль {m}: {', '.join(meta.get('words', []))}")
    history: List[Tuple[str, str]] = []
    print("Диалог (пустая строка — выход, /новый — заново, /запомни <текст> — выучить, /модуль вкл|выкл)")
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
        if text.startswith("/модуль"):
            set_enabled(model, "выкл" not in text)
            print("   модули", "выключены" if "выкл" in text else "включены")
            continue
        if text.startswith("/запомни"):
            body = text[len("/запомни"):].strip(" :")
            try:
                facts, rep = learn_text(model, tok, body, GrowConfig(seed=args.seed, steps=args.steps), args.data)
            except ValueError as e:
                print(f"   {e}")
                continue
            if args.save:
                save_module(args.save, model, {"words": [f.word for f in facts], "texts": [f.text for f in facts]})
            print(f"   выучено за {rep['seconds']:.0f} c: {rep['new_params']:,} новых параметров")
            history = []
            continue
        history.append(("user", text))
        reply = ask(model, tok, [history], max_new=64)[0]
        history.append(("bot", reply))
        print(f"SLR-Net: {reply}")
        if args.prompt is not None:
            break


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Самодообучение с ростом нейронов")
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("experiment", help="опыт: выучить выдуманные слова и проверить без объяснения")
    x.add_argument("--ckpt", default="slr_net_chat_ru_mini3.npz")
    x.add_argument("--data", default="data/chat")
    x.add_argument("--words", type=int, default=20)
    x.add_argument("--control", type=int, default=20)
    x.add_argument("--runs", nargs="+", default=["base", "grow"],
                   help="base, grow (рост по требованию), fixed:K, raw:K, full[:KL]")
    x.add_argument("--steps", type=int, default=150)
    x.add_argument("--k-init", type=int, default=8)
    x.add_argument("--target", type=float, default=0.85)
    x.add_argument("--keep-target", type=float, default=0.97)
    x.add_argument("--check-frac", type=float, default=0.2)
    x.add_argument("--neg-frac", type=float, default=0.5)
    x.add_argument("--max-rounds", type=int, default=6)
    x.add_argument("--sites", nargs="*", default=None, help="куда добавлять нейроны: decoder.3 — только последний слой")
    x.add_argument("--k-max", type=int, default=256)
    x.add_argument("--lr", type=float, default=3e-3)
    x.add_argument("--full-lr", type=float, default=3e-4)
    x.add_argument("--kl", type=float, default=1.0)
    x.add_argument("--soft", type=float, default=0.0, help="вес KL к распределению учителя")
    x.add_argument("--bias-init", type=float, default=-0.5)
    x.add_argument("--teacher", default="inline", choices=["inline", "reference"])
    x.add_argument("--in-context", nargs="*", default=["inline"], help="проверить и с объяснением в разговоре")
    x.add_argument("--probes-every", type=int, default=3)
    x.add_argument("--no-old", action="store_true", help="не проверять старые навыки (быстрее)")
    x.add_argument("--save-modules", default=None)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--json", default=None)
    lr_ = sub.add_parser("learn", help="выучить текст: «Слово — это ...» -> модуль знаний")
    lr_.add_argument("--ckpt", default="slr_net_chat_ru_mini3.npz")
    lr_.add_argument("--text", default=None)
    lr_.add_argument("--file", default=None)
    lr_.add_argument("--module", nargs="*", help="уже выученные модули (подключить перед обучением)")
    lr_.add_argument("--out", required=True)
    lr_.add_argument("--data", default="data/chat")
    lr_.add_argument("--steps", type=int, default=150)
    lr_.add_argument("--k-init", type=int, default=8)
    lr_.add_argument("--target", type=float, default=0.85)
    lr_.add_argument("--lr", type=float, default=3e-3)
    lr_.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("chat", help="диалог с подключёнными модулями знаний")
    c.add_argument("--ckpt", default="slr_net_chat_ru_mini3.npz")
    c.add_argument("--module", nargs="*")
    c.add_argument("--data", default="data/chat")
    c.add_argument("--steps", type=int, default=150)
    c.add_argument("--save", default=None, help="куда сохранять модуль после /запомни")
    c.add_argument("--prompt", default=None)
    c.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "learn" and not (args.text or args.file):
        ap.error("learn: нужен --text или --file")
    {"experiment": cmd_experiment, "learn": cmd_learn, "chat": cmd_chat}[args.cmd](args)


if __name__ == "__main__":
    main()
