#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оркестратор (идея 2 из PLAN_MULTIMODULE_RU.md): понять, о чём вопрос, и отдать его нужной части.
====================================================================================================

Оркестратор не отдельная большая сеть, а «голова» на скрытых состояниях той же mini2: вопрос прогоняется
через модель (без модулей), его скрытые состояния усредняются в один вектор, а дальше:

  • эксперт по теме (разговор / знания / счёт / логика / код) — линейная голова на векторе вопроса (или, для
    сравнения, ближайший центр класса). Учится за секунды на примерах из обучающих диалогов; проверка — на
    ОТЛОЖЕННЫХ задачах (probes.jsonl, их формулировок в обучении не было). Эксперт по JavaScript пока только
    запланирован: оркестратор его узнаёт, но отвечать некому;
  • модуль знаний — если вопрос близок к одному из выученных слов (ключ слова — средний вектор вопросов о
    нём), включаются нейроны только этого модуля, иначе модель отвечает «чистой», без модулей. Так модули не
    мешают старым навыкам и друг другу;
  • глубина мышления — сколько кругов ядра мышления делать для этой темы (опыт `depth` меряет, помогают ли
    лишние круги на каждой теме).

  python orchestrator.py router --json training/orchestrator/router.json       # точность выбора эксперта
  python orchestrator.py depth  --steps 1 2 3 4 6 --json training/orchestrator/depth.json
  python orchestrator.py chat   --module runs/kurd.npz
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import chat_corpus as C
import slr_net_engine as E
import train_chat as T

# Тема диалога по его источнику в chat_corpus (src у обучающих диалогов, cat у проверочных задач)
DOMAIN_OF = {
    "разговор": ("ручные:личность", "ручные:приветствие", "ручные:разговор", "ручные:эмоции", "ручные:о себе",
                 "ручные:мнения", "ручные:просьбы", "шаблон:я и ты", "шаблон:мнения", "шаблон:советы"),
    "знания": ("ручные:знания", "шаблон:определения", "шаблон:столицы", "шаблон:люди", "шаблон:звуки", "шаблон:ноги",
               "шаблон:цвета предметов", "шаблон:категории", "шаблон:антонимы", "шаблон:дни недели", "шаблон:месяцы"),
    "счёт": ("шаблон:счёт", "шаблон:задачи", "шаблон:следующее число"),
    "логика": ("шаблон:логика: силлогизм", "шаблон:логика: цепочка", "шаблон:сравнение"),
}

# Эксперта по коду ещё нет, но узнавать такие просьбы оркестратор должен уже сейчас. Примеры написаны вручную:
# первые CODE_TRAIN — для центра класса, остальные — отложенная проверка.
CODE_PROMPTS = """
Напиши игру змейка на JavaScript.
Сделай тетрис на HTML и JavaScript.
Как в JavaScript сделать цикл от одного до десяти?
Напиши функцию на JS, которая складывает два числа.
Сделай страницу HTML с кнопкой, которая меняет цвет.
Напиши игру танчики для браузера.
Как нарисовать квадрат на canvas?
Почему мой код на JavaScript выдаёт ошибку undefined?
Напиши код, который сортирует массив.
Сделай калькулятор на HTML.
Как добавить управление стрелками в игру на JS?
Напиши программу, которая выводит таблицу умножения.
Сделай игру пинг-понг в браузере.
Как в JavaScript получить элемент по id?
Напиши функцию, которая переворачивает строку.
Сделай анимацию мяча на canvas.
Как сделать, чтобы игра подстраивалась под размер окна?
Напиши код для счётчика очков в игре.
Как в JS создать массив и добавить в него элемент?
Сделай простой сайт-визитку на HTML.
Напиши игру крестики-нолики на JavaScript.
Как обработать нажатие клавиши в браузере?
Напиши функцию на JavaScript, которая считает факториал.
Сделай таймер обратного отсчёта на JS.
Как в JavaScript сделать случайное число?
Напиши код игры арканоид.
Как сделать столкновение двух прямоугольников в игре?
Напиши скрипт, который меняет текст на странице.
Сделай меню для игры на HTML.
Как запустить игровой цикл через requestAnimationFrame?
Напиши игру сапёр для браузера.
Как в JS проверить, что число чётное?
Сделай форму входа на HTML.
Напиши код, который рисует звёздное небо на canvas.
Как сохранить рекорд игры в localStorage?
Напиши на JavaScript функцию поиска максимума в массиве.
Сделай игру про космический корабль, который стреляет.
Как в JavaScript объявить переменную?
Напиши код, чтобы танк поворачивался и ехал.
Сделай платформер на JavaScript.
Напиши игру «угадай число» на JS.
Как в браузере нарисовать круг?
Сделай змейку, которая ускоряется.
Напиши функцию, которая проверяет палиндром.
Как сделать паузу в игре на JavaScript?
Сделай страницу со списком дел на HTML и JS.
Напиши код для движения персонажа прыжками.
Как в JS пройтись по всем элементам массива?
Сделай игру гонки сверху на canvas.
Напиши функцию, которая считает сумму чисел от 1 до n.
Как добавить звук выстрела в браузерную игру?
Сделай часы, которые показывают время на странице.
Напиши игру тетрис с подсчётом очков.
Как сделать, чтобы враги в игре двигались сами?
Напиши программу на JavaScript, которая рисует график.
Сделай игру мемори с карточками.
""".strip().splitlines()
CODE_TRAIN = 36

if E.TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


def domain_of(src: str) -> Optional[str]:
    src = src.replace("склейка:", "")
    for dom, prefixes in DOMAIN_OF.items():
        if src in prefixes:
            return dom
    return None


# =============================================================================
# ВЕКТОР ВОПРОСА
# =============================================================================

def embed_texts(model, tok, texts: Sequence[str], batch: int = 64, layer: str = "core") -> np.ndarray:
    """
    Вектор реплики пользователя: скрытые состояния модели (без модулей знаний), усреднённые по токенам текста.
    layer: "encoder" (выход прелюдии), "core" (выход ядра мышления) или "final" (перед выходным слоем).
    """
    import self_grow as S
    was = [g.enabled for g in S.grown_layers(model).values()] if E.TORCH_AVAILABLE else []
    S.set_enabled(model, False)
    model.eval()
    seqs = []
    for t in texts:
        body = tok.encode_line(t)[:200]
        seqs.append([tok.user_id] + body + [tok.nl_id])
    out = np.zeros((len(seqs), model.config.d_model), dtype=np.float32)
    with torch.no_grad():
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            L = max(len(seqs[i]) for i in idx)
            x = torch.full((len(idx), L), tok.nl_id, dtype=torch.long)
            m = torch.zeros(len(idx), L)
            for r, i in enumerate(idx):
                x[r, :len(seqs[i])] = torch.tensor(seqs[i])
                m[r, 1:len(seqs[i]) - 1] = 1.0                    # только текст, без ролевого токена и <n>
            h = model._embed(x)
            for blk in model.encoder:
                h = blk(h)
            if layer != "encoder":
                h, _, _ = model.core(h, "threshold")
                if layer == "final":
                    for blk in model.decoder:
                        h = blk(h)
                    h = model.final_norm(h)
            v = (h.float() * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            out[idx] = v.numpy()
    for g, e in zip(S.grown_layers(model).values(), was):
        g.enabled = e
    return out


def normalize(v: np.ndarray, center: Optional[np.ndarray] = None) -> np.ndarray:
    if center is not None:
        v = v - center
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


# =============================================================================
# ВЫБОР ЭКСПЕРТА
# =============================================================================

class DomainRouter:
    """Ближайший центр класса по косинусу (векторы центрированы средним по всем примерам)."""

    def __init__(self):
        self.center: Optional[np.ndarray] = None
        self.labels: List[str] = []
        self.centroids: Optional[np.ndarray] = None

    def fit(self, vecs: np.ndarray, labels: Sequence[str]) -> "DomainRouter":
        self.center = vecs.mean(0)
        v = normalize(vecs, self.center)
        self.labels = sorted(set(labels))
        lab = np.array(labels)
        self.centroids = normalize(np.stack([v[lab == c].mean(0) for c in self.labels]))
        return self

    def scores(self, vecs: np.ndarray) -> np.ndarray:
        return normalize(vecs, self.center) @ self.centroids.T

    def predict(self, vecs: np.ndarray) -> List[str]:
        return [self.labels[i] for i in self.scores(vecs).argmax(1)]

    def to_dict(self) -> dict:
        return {"labels": self.labels, "center": self.center.tolist(), "centroids": self.centroids.tolist()}


class LinearRouter:
    """
    Линейная «голова» поверх того же вектора вопроса: многоклассовая логистическая регрессия (секунды обучения на
    CPU). Классы взвешены обратно частоте: примеров кода мало (36), но пропускать их нельзя.
    """

    def __init__(self, weight_decay: float = 1e-3, steps: int = 400, lr: float = 0.05):
        self.wd, self.steps, self.lr = weight_decay, steps, lr
        self.labels: List[str] = []
        self.mean = self.std = None
        self.W = None

    def fit(self, vecs: np.ndarray, labels: Sequence[str]) -> "LinearRouter":
        self.labels = sorted(set(labels))
        self.mean, self.std = vecs.mean(0), vecs.std(0) + 1e-6
        x = torch.tensor((vecs - self.mean) / self.std, dtype=torch.float32)
        y = torch.tensor([self.labels.index(l) for l in labels])
        counts = torch.bincount(y, minlength=len(self.labels)).float()
        w = (counts.sum() / (len(self.labels) * counts))
        lin = torch.nn.Linear(x.shape[1], len(self.labels))
        opt = torch.optim.AdamW(lin.parameters(), lr=self.lr, weight_decay=self.wd)
        for _ in range(self.steps):
            loss = F.cross_entropy(lin(x), y, weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.W = lin
        return self

    def predict(self, vecs: np.ndarray) -> List[str]:
        with torch.no_grad():
            z = self.W(torch.tensor((vecs - self.mean) / self.std, dtype=torch.float32))
        return [self.labels[i] for i in z.argmax(1).tolist()]


def last_user_turn(turns) -> str:
    for role, text in reversed(turns):
        if role == "user":
            return text
    return ""


def router_data(data_dir: str, per_class: int, seed: int) -> Tuple[List[str], List[str], List[str], List[str]]:
    rng = random.Random(seed)
    rows = T.read_jsonl(Path(data_dir) / "sft_train.jsonl")
    by: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        d = domain_of(r["src"])
        if d:
            by[d].append(last_user_turn(r["turns"]))
    tr_x, tr_y = [], []
    for d, texts in sorted(by.items()):
        texts = list(dict.fromkeys(texts))
        rng.shuffle(texts)
        tr_x += texts[:per_class]
        tr_y += [d] * min(per_class, len(texts))
    tr_x += CODE_PROMPTS[:CODE_TRAIN]
    tr_y += ["код"] * CODE_TRAIN
    te_x, te_y = [], []
    for p in T.read_jsonl(Path(data_dir) / "probes.jsonl"):
        d = domain_of(p["cat"])
        if d:
            te_x.append(last_user_turn(p["turns"]))
            te_y.append(d)
    te_x += CODE_PROMPTS[CODE_TRAIN:]
    te_y += ["код"] * (len(CODE_PROMPTS) - CODE_TRAIN)
    return tr_x, tr_y, te_x, te_y


def cmd_router(args: argparse.Namespace) -> None:
    torch.set_flush_denormal(True)
    model, tok, _ = T.load_chat_model(args.ckpt, torch.device("cpu"), allow_base=False)
    tr_x, tr_y, te_x, te_y = router_data(args.data, args.per_class, args.seed)
    print(f"Примеров для центров: {len(tr_x)}, отложенных для проверки: {len(te_x)}")
    report = {}
    for layer in args.layers:
      etr, ete = embed_texts(model, tok, tr_x, layer=layer), embed_texts(model, tok, te_x, layer=layer)
      for head in args.heads:
        t0 = time.time()
        r = (DomainRouter() if head == "centroid" else LinearRouter()).fit(etr, tr_y)
        pred = r.predict(ete)
        acc = float(np.mean([p == y for p, y in zip(pred, te_y)]))
        per = {c: float(np.mean([p == y for p, y in zip(pred, te_y) if y == c])) for c in r.labels}
        conf = {c: dict(sorted({p2: sum(1 for p, y in zip(pred, te_y) if y == c and p == p2)
                                for p2 in r.labels}.items())) for c in r.labels}
        errors = [{"text": x, "true": y, "pred": p} for x, y, p in zip(te_x, te_y, pred) if p != y][:30]
        report[f"{layer}:{head}"] = {"accuracy": acc, "per_class": per, "confusion": conf, "errors": errors,
                                     "n_test": len(te_x), "seconds": round(time.time() - t0, 1)}
        print(f"  слой {layer}, {'ближайший центр' if head == 'centroid' else 'линейная голова'}: верно {acc:.1%}; "
              f"по темам {json.dumps(per, ensure_ascii=False)}", flush=True)
        if args.save and layer == args.layers[0] and head == "centroid":
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save).write_text(json.dumps({"layer": layer, **r.to_dict()}), encoding="utf-8")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


# =============================================================================
# ВЫБОР МОДУЛЯ ЗНАНИЙ
# =============================================================================

def token_states(model, tok, texts: Sequence[str], layer: str = "core", batch: int = 64) -> List[np.ndarray]:
    """Скрытые состояния каждого токена реплики пользователя (без модулей знаний): [(число токенов, d)]."""
    import self_grow as S
    was = [g.enabled for g in S.grown_layers(model).values()]
    S.set_enabled(model, False)
    model.eval()
    seqs = [[tok.user_id] + tok.encode_line(t)[:200] + [tok.nl_id] for t in texts]
    out: List[Optional[np.ndarray]] = [None] * len(seqs)
    with torch.no_grad():
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            L = max(len(seqs[i]) for i in idx)
            x = torch.full((len(idx), L), tok.nl_id, dtype=torch.long)
            for r, i in enumerate(idx):
                x[r, :len(seqs[i])] = torch.tensor(seqs[i])
            h = model._embed(x)
            for blk in model.encoder:
                h = blk(h)
            if layer != "encoder":
                h, _, _ = model.core(h, "threshold")
            for r, i in enumerate(idx):
                out[i] = h[r, 1:len(seqs[i]) - 1].float().numpy()
    for g, e in zip(S.grown_layers(model).values(), was):
        g.enabled = e
    return out  # type: ignore[return-value]


def _find(seq: Sequence[int], sub: Sequence[int]) -> int:
    for i in range(len(seq) - len(sub), -1, -1):
        if list(seq[i:i + len(sub)]) == list(sub):
            return i
    return -1


class ModuleRouter:
    """
    Какой модуль знаний включить для вопроса. Три способа:
      • "sentence" — ключ слова = средний вектор обучающих вопросов о нём, вопрос — тоже одним вектором;
      • "token" — ключ слова = состояние сети на ПОСЛЕДНЕМ кусочке самого слова (как ключи в GRACE, Hartvigsen и
        др., 2023), вопрос сравнивается с ключом на каждом своём токене (берётся лучший);
      • "lexical" — в вопросе есть само слово (с грубым учётом окончаний);
      • "hybrid" — "lexical" или "token".
    Вопрос включает модуль, если оценка ближайшего ключа ≥ порога (порог подбирается на других вопросах).
    """

    def __init__(self, model, tok, layer: str = "core", kind: str = "token"):
        self.model, self.tok, self.layer, self.kind = model, tok, layer, kind
        self.keys: List[np.ndarray] = []
        self.owner: List[Tuple[str, str]] = []           # (модуль, слово)
        self.center: Optional[np.ndarray] = None
        self.threshold = 0.5

    def add_module(self, tag: str, facts) -> None:
        import self_grow as S
        for f in facts:
            if self.kind == "sentence":
                qs = [q for q, _, _ in S.question_bank(S.Fact(f.word, f.text), train=True)]
                self.keys.append(embed_texts(self.model, self.tok, qs, layer=self.layer).mean(0))
            else:
                w, W = f.word.lower(), C.cap(f.word.lower())
                ctxs = [f"Что такое {w}?", f"Расскажи про слово «{w}».", f"{W} — это что?", f"Ты знаешь, что такое {w}?",
                        f.text]
                vecs = []
                for c, st in zip(ctxs, token_states(self.model, self.tok, ctxs, layer=self.layer)):
                    ids = self.tok.encode_line(c)
                    for form in (w, W):
                        sub = self.tok.encode_line(form)
                        j = _find(ids, sub)
                        if j < 0 and len(sub) > 1:
                            j = _find(ids, sub[1:]) - 1 if _find(ids, sub[1:]) > 0 else -1
                        if j >= 0:
                            vecs.append(st[min(j + len(sub) - 1, len(st) - 1)])
                            break
                self.keys.append(np.mean(vecs, axis=0) if vecs else np.zeros(self.model.config.d_model, np.float32))
            self.owner.append((tag, f.word))

    def set_center(self, texts: Sequence[str]) -> None:
        if self.kind == "sentence":
            self.center = embed_texts(self.model, self.tok, texts, layer=self.layer).mean(0)
        else:
            self.center = np.concatenate(token_states(self.model, self.tok, texts, layer=self.layer)).mean(0)

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        if self.kind == "lexical":
            return np.array([[float(lexical_route(t, [w])) for _, w in self.owner] for t in texts]).reshape(len(texts), -1)
        K = normalize(np.stack(self.keys), self.center)
        if self.kind == "sentence":
            return normalize(embed_texts(self.model, self.tok, texts, layer=self.layer), self.center) @ K.T
        out = np.stack([(normalize(st, self.center) @ K.T).max(0) if len(st) else np.zeros(len(self.keys))
                        for st in token_states(self.model, self.tok, texts, layer=self.layer)])
        if self.kind == "hybrid":
            lex = np.array([[float(lexical_route(t, [w])) for _, w in self.owner] for t in texts]).reshape(len(texts), -1)
            out = np.maximum(out, lex * 10.0)              # слово найдено — включаем при любом пороге
        return out

    def route(self, texts: Sequence[str]) -> List[List[str]]:
        if not self.keys and self.kind != "lexical":
            return [[] for _ in texts]
        if not self.owner:
            return [[] for _ in texts]
        s = self.scores(texts)
        out = []
        for row in s:
            j = int(row.argmax())
            out.append([self.owner[j][0]] if row[j] >= self.threshold else [])
        return out


def lexical_route(text: str, words: Sequence[str]) -> bool:
    """Для сравнения: есть ли в вопросе выученное слово (с грубым учётом окончаний)."""
    t = text.lower().replace("ё", "е")
    for w in words:
        stem = w.lower().replace("ё", "е")
        stem = stem[:-1] if len(stem) > 4 and stem[-1] in "аяьй" else stem
        if re.search(r"\b" + re.escape(stem), t):
            return True
    return False


def ask_routed(model, tok, orch_modules: "ModuleRouter", turns_list, max_new: int = 48) -> Tuple[List[str], List[bool]]:
    """Ответы, где модуль знаний включается только для вопросов, которые роутер к нему отнёс."""
    import self_grow as S
    qs = [last_user_turn(t) for t in turns_list]
    routes = orch_modules.route(qs)
    out: List[Optional[str]] = [None] * len(turns_list)
    for on in (True, False):
        idx = [i for i, r in enumerate(routes) if bool(r) == on]
        if not idx:
            continue
        S.set_active(model, None if on else [])
        for i, a in zip(idx, S.ask(model, tok, [turns_list[i] for i in idx], max_new=max_new)):
            out[i] = a
    S.set_active(model, None)
    return out, [bool(r) for r in routes]  # type: ignore[return-value]


def cmd_modules(args: argparse.Namespace) -> None:
    """
    Модуль знаний через оркестратор. (1) Точность выбора модуля: вопросы о выученных словах (отложенные
    формулировки) против вопросов о невыученных словах и старых задач. Порог подбирается на ДРУГИХ вопросах
    (обучающие формулировки, другие выдуманные слова, обучающие диалоги). (2) Ответы: без модуля, модуль всегда,
    модуль через оркестратор (каждым способом выбора). Ответ на вопрос зависит только от того, включён ли модуль,
    поэтому «через оркестратор» собирается из двух прогонов (без модуля / с модулем) по решениям роутера.
    """
    import self_grow as S
    torch.set_flush_denormal(True)
    rng = random.Random(args.seed)
    model, tok, _ = T.load_chat_model(args.ckpt, torch.device("cpu"), allow_base=False)
    meta = S.load_module(model, args.module)
    facts = [S.Fact(w, t, i) for w, t, i in zip(meta["words"], meta["texts"], meta.get("infos") or [None] * len(meta["words"]))]
    control = meta.get("control_words") or []
    learned = {f.word for f in facts}
    rows = T.read_jsonl(Path(args.data) / "sft_train.jsonl")
    user_turns = [last_user_turn(r["turns"]) for r in rng.sample(rows, 1500)]
    cal_pos = [q for f in facts for q, _, _ in S.question_bank(f, train=True)][:400]
    other = [w for w in C.NONCE_TRAIN if w not in learned][:100]
    cal_neg = [q for w in other for q in (f"Что такое {w}?", f"Какого цвета {w}?")] + user_turns[500:1000]
    test_pos = [(f, q, c) for f in facts for q, c, _ in S.question_bank(f, train=False, seed=args.seed + 1)]
    probes = T.read_jsonl(Path(args.data) / "probes.jsonl")[::args.probes_every]
    ctrl = S.control_questions(control)
    groups = {"выученные слова, отложенные вопросы": ([[("user", q)] for _, q, _ in test_pos], [c for _, _, c in test_pos]),
              "невыученные слова: «не знаю»": ([[("user", q)] for q, _ in ctrl], [c for _, c in ctrl]),
              "старые задачи": ([[tuple(t) for t in p["turns"]] for p in probes], [p["check"] for p in probes])}
    report: dict = {"routers": {}, "systems": {}}
    decisions: Dict[str, Dict[str, List[bool]]] = {}
    for kind in args.kinds:
        t0 = time.time()
        mr = ModuleRouter(model, tok, layer=args.layer, kind=kind)
        if kind != "lexical":
            mr.set_center(user_turns[:500])
        mr.add_module(meta["tag"], facts)
        if kind == "lexical":
            mr.threshold = 0.5
        else:
            sp, sn = mr.scores(cal_pos).max(1), mr.scores(cal_neg).max(1)
            grid = np.unique(np.concatenate([sp, sn]))
            mr.threshold = float(max(grid, key=lambda t: (np.mean(sp >= t) + np.mean(sn < t)) / 2))
        decisions[kind] = {g: [bool(r) for r in mr.route([last_user_turn(t) for t in turns])]
                           for g, (turns, _) in groups.items()}
        d = decisions[kind]
        res = {"порог": mr.threshold,
               "вопросы о выученных (включил)": float(np.mean(d["выученные слова, отложенные вопросы"])),
               "невыученные слова (не включил)": 1 - float(np.mean(d["невыученные слова: «не знаю»"])) if ctrl else None,
               "старые задачи (не включил)": 1 - float(np.mean(d["старые задачи"])),
               "seconds": round(time.time() - t0, 1)}
        report["routers"][kind] = res
        print(f"  выбор модуля «{kind}»: {json.dumps(res, ensure_ascii=False)}", flush=True)
    answers: Dict[str, Dict[str, List[bool]]] = {}
    for name, active in (("без модуля", []), ("модуль всегда", None)):
        t0 = time.time()
        S.set_active(model, active)
        answers[name] = {g: [bool(T.check_reply(t, c)) for t, c in zip(S.ask(model, tok, turns, max_new=64), checks)]
                         for g, (turns, checks) in groups.items()}
        report["systems"][name] = {g: float(np.mean(v)) for g, v in answers[name].items()}
        print(f"  {name}: {json.dumps(report['systems'][name], ensure_ascii=False)} ({time.time() - t0:.0f} c)", flush=True)
    S.set_active(model, None)
    for kind, d in decisions.items():
        name = f"через оркестратор ({kind})"
        report["systems"][name] = {g: float(np.mean([answers["модуль всегда"][g][i] if on else answers["без модуля"][g][i]
                                                     for i, on in enumerate(d[g])])) for g in groups}
        print(f"  {name}: {json.dumps(report['systems'][name], ensure_ascii=False)}", flush=True)
    report["per_item"] = {"answers": answers, "decisions": decisions}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


# =============================================================================
# ГЛУБИНА МЫШЛЕНИЯ
# =============================================================================

def cmd_depth(args: argparse.Namespace) -> None:
    """
    «Много раз пропустить мысль»: та же модель, но ядро мышления крутится 1, 2, 3 (как при обучении), 4, 6...
    кругов при ответе. Задачи с проверкой по темам — помогают ли лишние круги и где.
    """
    import self_grow as S
    torch.set_flush_denormal(True)
    model, tok, _ = T.load_chat_model(args.ckpt, torch.device("cpu"), allow_base=False)
    probes = T.read_jsonl(Path(args.data) / "probes.jsonl")[::args.every]
    probes = [p for p in probes if not args.domains or domain_of(p["cat"]) in args.domains]
    print(f"Задач: {len(probes)}; обучено кругов: {model.config.max_latent_steps}")
    report = {}
    for k in args.steps:
        model.config.core_infer_steps = k
        for mode in args.routing:
            t0 = time.time()
            if mode == "all":
                # все токены проходят все круги: роутер ядра не останавливает мысль
                orig = model.core.router.bias.data.clone()
                model.core.router.bias.data += 1e4
            texts = S.ask(model, tok, [[tuple(t) for t in p["turns"]] for p in probes], max_new=64)
            if mode == "all":
                model.core.router.bias.data.copy_(orig)
            ok = [bool(T.check_reply(t, p["check"])) for t, p in zip(texts, probes)]
            by: Dict[str, List[bool]] = defaultdict(list)
            for o, p in zip(ok, probes):
                by[domain_of(p["cat"]) or "прочее"].append(o)
            res = {"ВСЕГО": float(np.mean(ok)), **{d: float(np.mean(v)) for d, v in sorted(by.items())},
                   "n": len(ok), "seconds": round(time.time() - t0, 1)}
            report[f"{k}:{mode}"] = res
            print(f"  кругов {k} ({mode}): {json.dumps(res, ensure_ascii=False)}", flush=True)
            if args.json:
                Path(args.json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


# =============================================================================
# ОРКЕСТРАТОР В РАЗГОВОРЕ
# =============================================================================

class Orchestrator:
    """Вопрос -> (эксперт, модули знаний, круги мышления) -> ответ. Каждый шаг виден в trace."""

    def __init__(self, model, tok, router=None, depth: Optional[Dict[str, int]] = None,
                 layer: str = "core", module_kind: str = "hybrid"):
        self.model, self.tok = model, tok
        self.router = router
        self.depth = depth or {}
        self.modules = ModuleRouter(model, tok, layer=layer, kind=module_kind)
        self.layer = layer
        self.default_steps = model.config.core_infer_steps

    def answer(self, history: List[Tuple[str, str]]) -> Tuple[str, dict]:
        import self_grow as S
        q = last_user_turn(history)
        trace: dict = {}
        if self.router is not None:
            dom = self.router.predict(embed_texts(self.model, self.tok, [q], layer=self.layer))[0]
            trace["эксперт"] = dom
            if dom == "код":
                return ("Это задача для эксперта по JavaScript, а его ещё нет: пока я только собеседник. "
                        "(PLAN_MULTIMODULE_RU.md, раздел 2.5)"), trace
            self.model.config.core_infer_steps = self.depth.get(dom, self.default_steps)
            trace["кругов мышления"] = self.model.config.infer_steps
        tags = self.modules.route([q])[0]
        S.set_active(self.model, tags)
        trace["модули"] = tags
        reply = S.ask(self.model, self.tok, [history], max_new=64)[0]
        self.model.config.core_infer_steps = self.default_steps
        return reply, trace


def calibrate_threshold(mr: ModuleRouter, pos: Sequence[str], neg: Sequence[str]) -> float:
    """Порог, при котором поровну важны «включил для своих» и «не включил для чужих» (средняя из двух точностей)."""
    sp, sn = mr.scores(pos).max(1), mr.scores(neg).max(1)
    grid = np.unique(np.concatenate([sp, sn]))
    return float(max(grid, key=lambda t: (np.mean(sp >= t) + np.mean(sn < t)) / 2))


def build_orchestrator(model, tok, data_dir: str, modules: Sequence[str], per_class: int = 300,
                       seed: int = 0, layer: str = "core", threshold: Optional[float] = None,
                       module_kind: str = "hybrid") -> Orchestrator:
    """
    Оркестратор для разговора: линейная голова темы и выбор модуля «hybrid» (слово из модуля в вопросе ИЛИ ключ
    слова по состоянию сети — в опыте раздела 2.3 PLAN_MULTIMODULE_RU.md: 100% вопросов о выученном, 0%
    невыученных слов, 2.4% старых задач). Порог ключей подбирается сам: свои вопросы против вопросов о других
    выдуманных словах и обычных реплик (threshold — задать вручную).
    """
    import self_grow as S
    tr_x, tr_y, _, _ = router_data(data_dir, per_class, seed)
    router = LinearRouter().fit(embed_texts(model, tok, tr_x, layer=layer), tr_y)   # 85.5% против 65% у центров
    orch = Orchestrator(model, tok, router, layer=layer, module_kind=module_kind)
    if module_kind != "lexical":
        orch.modules.set_center(tr_x[::3])
    words: List[str] = []
    pos: List[str] = []
    for path in modules:
        meta = S.load_module(model, path)
        facts = [S.Fact(w, t) for w, t in zip(meta["words"], meta["texts"])]
        orch.modules.add_module(meta["tag"], facts)
        words += [f.word for f in facts]
        pos += [q for f in facts for q, _, _ in S.question_bank(f, train=True)]
    if threshold is not None:
        orch.modules.threshold = threshold
    elif pos and module_kind in ("token", "sentence", "hybrid"):
        other = [w for w in C.NONCE_TRAIN if w not in set(words)][:60]
        neg = [q for w in other for q in (f"Что такое {w}?", f"Какого цвета {w}?")] + tr_x[1::7]
        kind = orch.modules.kind
        orch.modules.kind = "token" if kind == "hybrid" else kind      # порог — для нейронной части
        orch.modules.threshold = calibrate_threshold(orch.modules, pos, neg)
        orch.modules.kind = kind
    return orch


def cmd_chat(args: argparse.Namespace) -> None:
    torch.set_flush_denormal(True)
    model, tok, _ = T.load_chat_model(args.ckpt, torch.device("cpu"), allow_base=False)
    orch = build_orchestrator(model, tok, args.data, args.module or [], threshold=args.threshold,
                              module_kind=args.module_kind)
    if args.module:
        print(f"Выбор модуля: {orch.modules.kind}, порог ключей {orch.modules.threshold:.3f}")
    history: List[Tuple[str, str]] = []
    print("Диалог через оркестратор (пустая строка — выход, /новый — заново)")
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
        history.append(("user", text))
        reply, trace = orch.answer(history)
        history.append(("bot", reply))
        print(f"   [{json.dumps(trace, ensure_ascii=False)}]")
        print(f"SLR-Net: {reply}")
        if args.prompt is not None:
            break


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Оркестратор: эксперт, модули знаний, глубина мышления")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("router", help="точность выбора эксперта на отложенных задачах")
    r.add_argument("--ckpt", default="slr_net_chat_ru_mini3.npz")
    r.add_argument("--data", default="data/chat")
    r.add_argument("--per-class", type=int, default=300)
    r.add_argument("--layers", nargs="+", default=["encoder", "core", "final"])
    r.add_argument("--heads", nargs="+", default=["centroid", "linear"])
    r.add_argument("--save", default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--json", default=None)
    d = sub.add_parser("depth", help="помогают ли лишние круги ядра мышления")
    d.add_argument("--ckpt", default="slr_net_chat_ru_mini3.npz")
    d.add_argument("--data", default="data/chat")
    d.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    d.add_argument("--routing", nargs="+", default=["threshold"], choices=["threshold", "all"])
    d.add_argument("--every", type=int, default=2)
    d.add_argument("--domains", nargs="*", default=None)
    d.add_argument("--json", default=None)
    mo = sub.add_parser("modules", help="модуль знаний через оркестратор: точность роутера и ответы")
    mo.add_argument("--ckpt", default="slr_net_chat_ru_mini3.npz")
    mo.add_argument("--data", default="data/chat")
    mo.add_argument("--module", required=True)
    mo.add_argument("--layer", default="core")
    mo.add_argument("--kinds", nargs="+", default=["sentence", "token", "lexical", "hybrid"])
    mo.add_argument("--probes-every", type=int, default=3)
    mo.add_argument("--seed", type=int, default=0)
    mo.add_argument("--json", default=None)
    c = sub.add_parser("chat", help="диалог через оркестратор")
    c.add_argument("--ckpt", default="slr_net_chat_ru_mini3.npz")
    c.add_argument("--data", default="data/chat")
    c.add_argument("--module", nargs="*")
    c.add_argument("--threshold", type=float, default=None, help="порог ключей модулей (по умолчанию — подбирается сам)")
    c.add_argument("--module-kind", default="hybrid", choices=["hybrid", "token", "lexical", "sentence"])
    c.add_argument("--prompt", default=None)
    args = ap.parse_args(argv)
    {"router": cmd_router, "depth": cmd_depth, "modules": cmd_modules, "chat": cmd_chat}[args.cmd](args)


if __name__ == "__main__":
    main()
