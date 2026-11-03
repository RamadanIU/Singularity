#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLR-Net: поисковый чат-прототип с логическим модулем и троичной микро-сетью (`slr_net_trainer_chat.py`)
=====================================================================================================

ЧТО ЭТО НА САМОМ ДЕЛЕ (важно для понимания):
Этот файл — НЕ генеративная языковая модель, а поисковый (retrieval) чат-бот:
  1. Вопрос пользователя сравнивается по словам/корням с вопросами из корпуса (`evaluate_epistemic_match`).
  2. Если совпадение уверенное — выводится сохранённый ответ этой строки корпуса. Нейросеть при этом
     «проигрывает» ответ с подсказкой копирования (`copy_bias`): к логиту нужного слова добавляется +5.5.
     С copy_bias = 0 видно, что генерирует сама сеть (команда `/копия` в чате).
  3. Логические задачи и арифметика решаются правилами (`EpistemicLogicEngine`) — это «инструмент»
     (как калькулятор у больших LLM), а не обучение.
Настоящая языковая модель для обучения на больших текстах — `slr_net_engine.py` + `train_lm.py`.

Компоненты нейросети (микро-версии идей):
- BitNet b1.58 (arXiv:2402.17764): троичные веса {-1, 0, +1} с STE.
- Линейная SSM-рекуррентность (идея Mamba, arXiv:2312.00752): состояние фиксированного размера.
- Память диалогов (идея RETRO / Memory Layers, arXiv:2112.04426, arXiv:2412.09764).
- Роутер глубины + скрытый цикл (идеи Mixture-of-Depths arXiv:2404.02258, Coconut arXiv:2412.06769).

Возможности:
1. Загрузчик `.txt` и `.srt` (`--corpus путь`): чистит таймкоды и склеивает реплики в пары «вопрос — ответ».
2. Логический модуль: транзитивные сравнения («А быстрее Б, Б быстрее В — кто самый медленный?»)
   и арифметика словами/цифрами (безопасный разбор, без eval).
3. Честное «я не знаю»: если ключевых слов вопроса нет в словаре, модель не выдумывает ответ,
   а предлагает научить её — новый факт добавляется в память МГНОВЕННО (без переобучения с нуля).
"""

from __future__ import annotations

import argparse
import ast
import math
import operator
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# 1. СЛОВАРИ СЛУЖЕБНЫХ СЛОВ, ВОПРОСИТЕЛЬНЫХ МАРКЕРОВ И ИМЁН СОБСТВЕННЫХ
# =============================================================================

FUNCTION_WORDS = {
    ",", ".", "!", "?", "—", "-", ":", ";", "«", "»", "(", ")",
    "и", "в", "во", "на", "с", "со", "по", "из-за", "из", "от", "до", "за",
    "не", "ни", "но", "а", "же", "ли", "бы", "то", "это", "этот", "эта", "эти",
    "как", "что", "чтобы", "потому", "поэтому", "чем", "при", "или", "для",
    "его", "её", "их", "мой", "твой", "свой", "наш", "ваш",
    "я", "меня", "мне", "мной", "ты", "тебя", "тебе", "тобой",
    "мы", "нас", "нам", "вы", "вас", "вам", "он", "она", "оно", "они",
    "под", "над", "зачем", "почему", "сколько", "когда", "где", "куда", "откуда",
    "кто", "какой", "какая", "какое", "какие", "чей", "чья", "о", "об", "про",
    "у", "к", "ко", "без", "лишь", "только", "уже", "ещё", "все", "всё", "вся",
    "так", "такой", "такая", "такое", "там", "тут", "здесь", "сейчас", "сегодня",
    "очень", "просто", "даже", "если", "вдруг", "может", "нужно", "надо",
    "быть", "есть", "был", "была", "были", "будет", "будут", "стал", "стала",
    "скажи", "расскажи", "объясни", "знаешь", "думаешь", "считаешь",
    "<pad>", "<bos>", "<eos>", "<unk>"
}

PROPER_NOUNS_MAP = {
    "толстой": "Толстой", "толстого": "Толстого", "толстому": "Толстому",
    "лев": "Лев", "льва": "Льва", "николаевич": "Николаевич",
    "тургенев": "Тургенев", "тургенева": "Тургенева",
    "иван": "Иван", "ивана": "Ивана", "ивану": "Ивану", "сергеевич": "Сергеевич",
    "матвеич": "Матвеич", "матвеичу": "Матвеичу",
    "чехов": "Чехов", "чехова": "Чехова", "антон": "Антон", "павлович": "Павлович",
    "шерлок": "Шерлок", "холмс": "Холмс", "холмса": "Холмса",
    "сократ": "Сократ", "аристотеля": "Аристотеля", "оккама": "Оккама",
    "архимеда": "Архимеда", "дэвид": "Дэвид", "скотт": "Скотт", "мэри": "Мэри",
    "юнгфрау": "Юнгфрау", "финстерааргорн": "Финстерааргорн", "финстерааргорна": "Финстерааргорна",
    "эйнштейна": "Эйнштейна", "фейнмана": "Фейнмана",
    "земля": "Земля", "земли": "Земли", "землю": "Землю", "земле": "Земле",
    "луна": "Луна", "луне": "Луне", "луны": "Луны",
    "солнце": "Солнце", "солнца": "Солнца", "солнцем": "Солнцем", "цельсия": "Цельсия",
    "вселенная": "Вселенная", "вселенной": "Вселенной",
    "python": "Python", "slr-net": "SLR-Net", "bitnet": "BitNet", "mamba": "Mamba", "coconut": "Coconut",
}

SYNONYM_STEMS = {
    "здравств": ["привет", "добр", "дела"],
    "привет": ["здравств", "дела", "добр"],
    "груст": ["одинок", "печал", "плох", "тоск", "устал"],
    "печал": ["груст", "одинок", "тоск"],
    "плох": ["груст", "одинок", "устал", "тревож"],
    "устал": ["сует", "работ", "шум", "сил", "мотивац"],
    "боюсь": ["страшн", "страх", "тревож", "сомнева"],
    "страшн": ["страх", "начина", "ошиб"],
    "книг": ["почита", "классик", "литератур", "толст", "чехов", "турген"],
    "фильм": ["кино", "жанр", "стары", "подвал", "детектив", "холмс"],
    "кино": ["фильм", "жанр", "актёр", "режиссёр"],
    "космос": ["вселен", "звёзд", "дыр", "планет", "солнц", "лун"],
    "вселен": ["космос", "звёзд", "дыр", "факт"],
    "мозг": ["нейрон", "памят", "сон", "дума", "человек"],
    "логик": ["рассужд", "причин", "оккам", "выживш", "сократ", "холмс"],
    "незна": ["знаеш", "совр", "выдум", "честн"],
    "врёш": ["совр", "правд", "знаеш"],
}

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


def format_natural_russian(tokens: List[str]) -> str:
    """Форматирует последовательность токенов в естественный русский текст с заглавными буквами и пунктуацией."""
    if not tokens:
        return ""

    formatted_words: List[str] = []
    capitalize_next = True

    for tok in tokens:
        if tok in SPECIAL_TOKENS:
            continue

        word = PROPER_NOUNS_MAP.get(tok, tok)
        if capitalize_next and word and word[0].isalpha():
            word = word[0].upper() + word[1:]
            capitalize_next = False
        elif word and word[0].isalpha():
            capitalize_next = False

        if tok in (".", "!", "?", "«"):
            capitalize_next = True

        formatted_words.append(word)

    text = " ".join(formatted_words)
    text = re.sub(r"\s+([.,!?;:»])", r"\1", text)
    text = re.sub(r"(«)\s+", r"\1", text)
    return text.strip()


# =============================================================================
# 2. УНИВЕРСАЛЬНЫЙ ПАРСЕР КИНОСУБТИТРОВ (.SRT / .TXT) И КНИЖНЫХ ДИАЛОГОВ
# =============================================================================

def _join_dialogue(pending: List[str]) -> Optional[str]:
    if len(pending) < 2:
        return None
    q_part = pending[0].lstrip("—- ").strip()
    a_part = " ".join(x.lstrip("—- ").strip() for x in pending[1:])
    if not (q_part and a_part):
        return None
    if q_part[-1] not in ("?", "!", "."):
        q_part += " ?"
    return f"{q_part} {a_part}"


def parse_subtitle_or_text_corpus(raw_text: str) -> List[str]:
    """
    Парсер `.txt` / `.srt`:
    1. Убирает номера блоков субтитров, таймкоды `00:01:20,000 --> 00:01:24,000`, HTML-теги и ремарки [смех].
    2. Реплики через тире (`— Вопрос / — Ответ`) склеивает в пару `Вопрос ? Ответ`.
    3. Короткие одиночные строки объединяет попарно в цепочки «стимул — ответ».
    Строки, начинающиеся с `#`, — комментарии.
    """
    cleaned = re.sub(r"<[^>]+>", " ", raw_text)
    cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)

    filtered_lines: List[str] = []
    pending: List[str] = []

    for line in cleaned.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            joined = _join_dialogue(pending)
            if joined:
                filtered_lines.append(joined)
            pending = []
            continue
        if re.fullmatch(r"\d+", s):
            continue
        if "-->" in s or re.match(r"^\d{2}:\d{2}:\d{2}", s):
            continue
        if s.startswith("—") or s.startswith("- ") or pending:
            pending.append(s)
            continue
        filtered_lines.append(s)

    joined = _join_dialogue(pending)
    if joined:
        filtered_lines.append(joined)

    final_chains: List[str] = []
    i = 0
    while i < len(filtered_lines):
        cur = filtered_lines[i]
        has_internal_split = any(2 <= cur.find(p) < len(cur) - 4 for p in ("?", "!", "."))
        if has_internal_split:
            final_chains.append(cur)
            i += 1
        elif i + 1 < len(filtered_lines):
            nxt = filtered_lines[i + 1]
            if cur[-1] not in ("?", "!", "."):
                cur += " ."
            if nxt[-1] not in ("?", "!", "."):
                nxt += " ."
            final_chains.append(f"{cur} {nxt}")
            i += 2
        else:
            if len(cur.split()) >= 5:
                final_chains.append(cur)
            i += 1

    return final_chains


# =============================================================================
# 3. МОДУЛЬ ЛОГИЧЕСКОГО МЫШЛЕНИЯ И ЭПИСТЕМИЧЕСКОЙ ОЦЕНКИ (ЧЕСТНОЕ «Я НЕ ЗНАЮ»)
# =============================================================================

# Числительные: значение и тип (единицы/десятки/сотни складываются, «тысяча»/«миллион» — множители)
_NUM_UNITS = {
    "ноль": 0, "нуля": 0, "один": 1, "одна": 1, "одно": 1, "одного": 1, "одну": 1, "два": 2, "две": 2,
    "двух": 2, "три": 3, "трёх": 3, "трех": 3, "четыре": 4, "четырёх": 4, "четырех": 4, "пять": 5,
    "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7, "восемь": 8, "восьми": 8, "девять": 9,
    "девяти": 9, "десять": 10, "десяти": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
    "девятнадцать": 19, "двадцать": 20, "двадцати": 20, "тридцать": 30, "тридцати": 30, "сорок": 40,
    "сорока": 40, "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80,
    "девяносто": 90, "сто": 100, "ста": 100, "двести": 200, "триста": 300, "четыреста": 400,
    "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
}
_NUM_MULTIPLIERS = {
    "тысяча": 1000, "тысячи": 1000, "тысяч": 1000, "тысячу": 1000,
    "миллион": 10 ** 6, "миллиона": 10 ** 6, "миллионов": 10 ** 6,
}
# совместимость со старым именем
RUSSIAN_NUM_WORDS = dict(_NUM_UNITS, **_NUM_MULTIPLIERS)

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_ARITH_CUES = re.compile(r"сколько|посчитай|вычисли|реши|будет|равн|=|результат")


def words_to_numbers(text: str) -> str:
    """'двадцать пять тысяч триста' -> '25300'. Цифры, введённые пользователем, не трогает."""
    out: List[str] = []
    total, current, in_number = 0, 0, False

    def flush() -> None:
        nonlocal total, current, in_number
        if in_number:
            out.append(str(total + current))
        total, current, in_number = 0, 0, False

    for part in re.split(r"(\s+)", text):
        word = part.strip(".,!?;:").lower()
        if word in _NUM_UNITS:
            current += _NUM_UNITS[word]
            in_number = True
        elif word in _NUM_MULTIPLIERS:
            total += max(current, 1) * _NUM_MULTIPLIERS[word]
            current, in_number = 0, True
        elif part.isspace() and in_number:
            continue
        else:
            flush()
            out.append(part if not part.isspace() else part)
            continue
        tail = part[len(part.rstrip(".,!?;:")):]
        if tail:
            flush()
            out.append(tail)
    flush()
    return " ".join(x for x in out if x and not x.isspace())


def safe_arithmetic(expr: str):
    """Вычисляет + - * / и скобки через разбор синтаксического дерева (eval не используется)."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        raise ValueError("недопустимое выражение")
    return ev(ast.parse(expr, mode="eval"))


def _root_key(word: str) -> str:
    """Грубый корень для сопоставления падежей: первая буква + 2 согласные (орёл/орла -> 'орл')."""
    w = word.replace("ё", "е")
    return w[0] + re.sub(r"[аеиоуыэюяьъй]", "", w[1:])[:2]


# Измерение: (больше-слово, меньше-слово, прилагательное «максимума», прилагательное «минимума»)
_COMPARISONS = [
    ("быстрее", "медленнее", "быстр", "медлен"),
    ("старше", "младше", "стар", "млад"),
    ("тяжелее", "легче", "тяжел", "легк"),
    ("выше", "ниже", "высок", "низк"),
    ("сильнее", "слабее", "сильн", "слаб"),
    ("больше", "меньше", "больш", "меньш"),
    ("умнее", "глупее", "умн", "глуп"),
    ("дороже", "дешевле", "дорог", "дешев"),
    ("длиннее", "короче", "длин", "корот"),
    ("богаче", "беднее", "богат", "бедн"),
]


class EpistemicLogicEngine:
    """
    1. Решает транзитивные цепочки сравнения (A > B, B > C => A > C) через граф отношений:
       честно сообщает, если данных не хватает или условие противоречиво.
    2. Считает арифметику цифрами и словами (с приоритетом операций, без eval).
    3. Формулирует честный ответ «я этого не знаю», когда в памяти нет нужного факта.
    """

    @staticmethod
    def _solve_comparison(low: str) -> Optional[str]:
        norm = low.replace("ё", "е")
        q = re.search(r"сам(?:ый|ая|ое|ые|ого|ую)\s+([а-я]+)", norm)
        if not q:
            return None
        asked = q.group(1)
        for more, less, max_adj, min_adj in _COMPARISONS:
            if more not in norm and less not in norm:
                continue
            if asked.startswith(max_adj):
                want_max = True
            elif asked.startswith(min_adj):
                want_max = False
            else:
                continue
            premise = low[:q.start()]
            premise = re.sub(r"\b(если|то|чем|а|и|но)\b", " ", premise)
            edges: List[Tuple[str, str]] = []
            names: Dict[str, str] = {}
            for m in re.finditer(rf"([а-яёa-z]+)\s+({more}|{less})\s+([а-яёa-z]+)", premise):
                a, rel, b = m.group(1), m.group(2), m.group(3)
                ka, kb = _root_key(a), _root_key(b)
                names[ka] = a                    # форма-подлежащее (именительный падеж) приоритетнее
                names.setdefault(kb, b)
                edges.append((ka, kb) if rel == more else (kb, ka))
            if len(edges) < 2:
                return None
            nodes = sorted({x for e in edges for x in e})
            greater = {n: set() for n in nodes}
            for a, b in edges:
                greater[a].add(b)
            # транзитивное замыкание
            changed = True
            while changed:
                changed = False
                for n in nodes:
                    extra = set().union(*(greater[x] for x in greater[n])) - greater[n]
                    if extra:
                        greater[n] |= extra
                        changed = True
            if any(n in greater[n] for n in nodes):
                return ("Давай рассуждать логически: в условии противоречие — по цепочке сравнений "
                        "кто-то оказывается «больше самого себя», поэтому однозначного ответа нет.")
            if want_max:
                cands = [n for n in nodes if len(greater[n]) == len(nodes) - 1]
            else:
                cands = [n for n in nodes if not greater[n] and all(n in greater[m] for m in nodes if m != n)]
            phrase = low[q.start():q.end()]      # «самая глупая», «самый лёгкий» — как в вопросе
            if len(cands) != 1:
                return (f"Давай рассуждать логически: из условия нельзя однозначно определить, кто {phrase} — "
                        f"не все участники сравнены между собой. Нужно ещё одно сравнение.")
            order = sorted(nodes, key=lambda n: -len(greater[n]))
            chain_str = " → ".join(names[n].capitalize() for n in order)
            return (f"Давай рассуждать логически по правилу транзитивности: выстраиваем цепочку "
                    f"от большего признака к меньшему ({chain_str}). "
                    f"Следовательно, {phrase} здесь — {names[cands[0]].capitalize()}.")
        return None

    @staticmethod
    def _solve_arithmetic(low: str) -> Optional[str]:
        expr_text = words_to_numbers(low)
        expr_text = (
            expr_text.replace("умножить на", "*").replace("умноженное на", "*")
            .replace("разделить на", "/").replace("поделить на", "/").replace("делить на", "/")
            .replace("плюс", "+").replace("прибавить", "+")
            .replace("минус", " - ").replace("вычесть", " - ").replace("отнять", " - ")
            .replace("×", "*").replace("÷", "/")
        )
        m = re.search(r"(\(?\s*\d+(?:[.,]\d+)?\s*\)?(?:\s*[-+*/]\s*\(?\s*\d+(?:[.,]\d+)?\s*\)?)+)", expr_text)
        if not m:
            return None
        raw = re.sub(r"\s+", " ", m.group(1)).strip()
        only_tight_hyphen = not re.search(r"[+*/]|\s-|-\s", raw)
        if only_tight_hyphen and (not _ARITH_CUES.search(low) or re.fullmatch(r"\d{4}(-\d{4})+", raw)):
            return None          # «1941-1945», телефоны и даты — это не вычитание
        try:
            val = safe_arithmetic(raw.replace(",", "."))
        except ZeroDivisionError:
            return ("Логическая проверка: в выражении встретилось деление на ноль, а в математике "
                    "деление на ноль не определено, так как приводит к противоречию.")
        except (ValueError, SyntaxError):
            return None
        if isinstance(val, float):
            val = int(val) if val.is_integer() else round(val, 4)
        return f"Посчитаем по правилам арифметики: выражение ({raw}) даёт точный результат: {val}."

    @staticmethod
    def try_solve_logic_or_math(user_text: str) -> Optional[str]:
        low = user_text.lower().strip()
        if re.search(r"сам(ый|ая|ое|ые|ого|ую)\s", low) or "кто из них" in low:
            solved = EpistemicLogicEngine._solve_comparison(low)
            if solved:
                return solved
        return EpistemicLogicEngine._solve_arithmetic(low)

    @staticmethod
    def build_honest_unknown_response(
        user_text: str,
        unknown_words: List[str],
        best_partial_reply: Optional[str] = None,
        partial_confidence: float = 0.0
    ) -> str:
        """Честный ответ, когда факта нет в памяти: что именно неизвестно + тип вопроса + предложение научить."""
        low = user_text.lower().strip()

        if low.startswith("кто ") or " кто " in low or any(w in low for w in ("кем был", "кто написал", "кто создал", "кто открыл")):
            q_type = "конкретной личности, авторе или историческом деятеле"
        elif any(w in low for w in ("где наход", "столиц", "в каком городе", "в какой стране", "где расп")):
            q_type = "географическом месте, стране или локации"
        elif any(w in low for w in ("когда ", "в каком году", "какого числа", "дата ")):
            q_type = "точной дате или историческом времени события"
        elif any(w in low for w in ("сколько ", "какова цена", "какой размер", "какая скорость")):
            q_type = "точной численной величине или измерении"
        elif any(w in low for w in ("почему ", "зачем ", "отчего ", "по какой причине")):
            q_type = "причинно-следственной связи или механизме явления"
        else:
            q_type = "факте или понятии, которого ещё не было в моих диалогах"

        if unknown_words:
            unk_str = ", ".join(f"«{w}»" for w in unknown_words[:3])
            core_honest = (
                f"Честно скажу: я этого пока не знаю. Слова {unk_str} ещё не встречались "
                f"в моём обучающем корпусе, а выдумывать факты наугад я не хочу. "
                f"Если рассуждать по структуре твоего вопроса, речь идёт о {q_type}."
            )
        else:
            core_honest = (
                f"Честно признаюсь: точного ответа на этот вопрос в моей памяти сейчас нет, "
                f"а выдавать догадку за факт было бы нечестно. Я вижу, что ты спрашиваешь о {q_type}."
            )

        if best_partial_reply and partial_confidence >= 0.22:
            return (
                f"{core_honest} Ближайшая мысль в моей памяти по смежной теме звучит так: "
                f"«{best_partial_reply}» Но если ты напишешь мне точный ответ в следующей фразе — "
                f"я сразу запишу его в память!"
            )
        return (
            f"{core_honest} Напиши мне коротко ответ на этот вопрос в следующем сообщении — "
            f"и я сразу сохраню это знание в свою память!"
        )


# =============================================================================
# 4. ТОКЕНИЗАТОР (СЛОВЕСНЫЙ) С ПОИСКОМ ПО КОРНЮ
# =============================================================================

class RussianMorphemeTokenizer:
    """
    Словесный токенизатор (одно слово = один токен) с подбором известного слова по общему началу.
    Для больших текстов он не подходит (словарь русских словоформ — миллионы); там используется
    субсловный SentencePiece в train_lm.py.
    """

    def __init__(self):
        self.special_tokens = list(SPECIAL_TOKENS)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        self._prefix_index: Dict[str, List[int]] = {}
        for tok in self.special_tokens:
            self._add_token(tok)

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.id_to_token)
            self.token_to_id[token] = idx
            self.id_to_token.append(token)
            if len(token) >= 4:
                self._prefix_index.setdefault(token[:4], []).append(idx)
            return idx
        return self.token_to_id[token]

    def rebuild_index(self) -> None:
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self._prefix_index = {}
        for i, t in enumerate(self.id_to_token):
            if len(t) >= 4:
                self._prefix_index.setdefault(t[:4], []).append(i)

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        text = text.lower().strip()
        text = re.sub(r"([.,!?;:—()\"«»])", r" \1 ", text)
        return [t for t in text.split() if t]

    def fit(self, chains: List[str], max_vocab: int = 16384) -> None:
        freq: Dict[str, int] = {}
        for line in chains:
            for tok in self.tokenize_text(line):
                freq[tok] = freq.get(tok, 0) + 1
        sorted_tokens = sorted(freq.keys(), key=lambda k: (-freq[k], k))
        for tok in sorted_tokens[: max_vocab - len(self.special_tokens)]:
            self._add_token(tok)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def _resolve_token_id(self, tok: str) -> int:
        """
        Точное слово или словоформа того же слова. Раньше любое слово, совпавшее первыми 4 буквами,
        считалось «известным» («погоде» -> «поговори»), и модель уверенно отвечала невпопад; кроме
        того, поиск шёл перебором всего словаря. Теперь: индекс по 4-буквенному префиксу и требование,
        чтобы слова отличались только последними <= 2 буквами (первый/первом, книга/книгу).
        """
        if tok in self.token_to_id:
            return self.token_to_id[tok]
        if len(tok) >= 4:
            stem = tok[: max(4, len(tok) - 2)]
            for idx in self._prefix_index.get(tok[:4], ()):
                known = self.id_to_token[idx]
                if not known.startswith(stem):
                    continue
                common = len(os.path.commonprefix([tok, known]))
                if common >= max(4, max(len(tok), len(known)) - 2):
                    return idx
        return self.unk_id

    def find_unknown_content_words(self, text: str) -> Tuple[List[str], List[str]]:
        """Делит значимые слова вопроса на известные модели и неизвестные."""
        known_content: List[str] = []
        unknown_content: List[str] = []
        for tok in self.tokenize_text(text):
            if tok in FUNCTION_WORDS or len(tok) <= 2 or tok.isdigit():
                continue
            if self._resolve_token_id(tok) == self.unk_id:
                if any(tok.startswith(sk) for sk in SYNONYM_STEMS):
                    known_content.append(tok)
                else:
                    unknown_content.append(tok)
            else:
                known_content.append(tok)
        return known_content, unknown_content

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids: List[int] = [self.bos_id] if add_bos else []
        ids.extend(self._resolve_token_id(tok) for tok in self.tokenize_text(text))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        return format_natural_russian([self.id_to_token[i] for i in ids if 0 <= i < len(self.id_to_token)])

    def compute_complexity_hints(self, token_ids: List[int]) -> np.ndarray:
        hints = np.zeros(len(token_ids), dtype=np.float32)
        for i in range(len(token_ids) - 1):
            cur_tok = self.id_to_token[token_ids[i]]
            next_tok = self.id_to_token[token_ids[i + 1]]
            if cur_tok in ("?", "!") or (next_tok not in FUNCTION_WORDS and len(next_tok) >= 5):
                hints[i] = 1.0
        return hints


# =============================================================================
# 5. УПАКОВКА И РАСПАКОВКА ТРОИЧНЫХ ВЕСОВ {-1, 0, +1} (5 ВЕСОВ В 1 БАЙТ)
# =============================================================================

def pack_ternary_matrix(w_ternary: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    shape = w_ternary.shape
    flat = w_ternary.ravel().astype(np.int16) + 1
    pad_len = (5 - (len(flat) % 5)) % 5
    if pad_len > 0:
        flat = np.concatenate([flat, np.ones(pad_len, dtype=np.int16)])
    powers = np.array([1, 3, 9, 27, 81], dtype=np.int16)
    packed = (flat.reshape(-1, 5) * powers).sum(axis=1).astype(np.uint8)
    return packed, shape


def unpack_ternary_matrix(packed: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    n_total = shape[0] * shape[1]
    val = packed.astype(np.int16)[:, None]
    powers = np.array([1, 3, 9, 27, 81], dtype=np.int16)
    trits = ((val // powers) % 3) - 1
    return trits.ravel()[:n_total].reshape(shape).astype(np.int8)


# =============================================================================
# 6. МИКРО-СЕТЬ SLR-Net С ПАМЯТЬЮ ДИАЛОГОВ
# =============================================================================

TERNARY_NAMES = ("w_ctx1", "w_ctx2", "w_ssm", "w_think", "w_out")


class TrainableSLRNetEngine:
    # Константы архитектуры (одинаковы в прямом и обратном проходе)
    C1, C2, C_SSM, SSM_DECAY, SSM_IN = 0.8, 0.5, 0.7, 0.88, 0.40
    MEM_LOOP, MEM_OUT, THINK_STEP, ROUTE_THR, ROUTER_COEF = 0.45, 1.15, 0.35, 0.45, 0.25

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 96,
        memory_slots: int = 128,
        max_seq_len: int = 96,
        max_latent_steps: int = 3,
        seed: int = 42,
        copy_bias: float = 5.5,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.memory_slots = memory_slots
        self.max_seq_len = max_seq_len
        self.max_latent_steps = max_latent_steps
        self.copy_bias = copy_bias      # подсказка копирования ответа из памяти (только при генерации!)
        self.quantize = True
        self.rng = np.random.default_rng(seed)
        rng = self.rng

        scale = 1.0 / math.sqrt(d_model)
        self.embed = (rng.standard_normal((vocab_size, d_model)) * 0.22).astype(np.float32)
        self.pos_embed = (rng.standard_normal((max_seq_len, d_model)) * 0.12).astype(np.float32)
        self.w_ctx1 = (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32)
        self.w_ctx2 = (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32)
        self.w_ssm = (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32)
        self.w_think = (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32)
        self.w_out = (rng.standard_normal((vocab_size, d_model)) * scale).astype(np.float32)
        self.b_out = np.zeros(vocab_size, dtype=np.float32)
        self.w_router = (rng.standard_normal((d_model, 1)) * scale).astype(np.float32)
        self.b_router = np.zeros(1, dtype=np.float32)
        self.mem_keys = (rng.standard_normal((memory_slots, d_model)) * scale).astype(np.float32)
        self.mem_vals = (rng.standard_normal((memory_slots, d_model)) * scale).astype(np.float32)
        self.slot_prompts: List[List[int]] = []
        self.slot_full_seqs: List[List[int]] = []

        self._param_names = [
            "embed", "pos_embed", "w_ctx1", "w_ctx2", "w_ssm",
            "w_think", "w_out", "b_out", "w_router", "b_router", "mem_vals"
        ]
        self._reset_optimizer()
        self._qcache: Dict[str, Tuple[int, np.ndarray, float]] = {}

    def _reset_optimizer(self) -> None:
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._param_names}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._param_names}
        self._step = 0

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    @staticmethod
    def quantize_b158(w: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, float, float]:
        gamma = float(max(np.mean(np.abs(w)), eps))
        w_tern = np.clip(np.round(w / gamma), -1.0, 1.0).astype(np.int8)
        return w_tern, gamma, float(np.mean(w_tern == 0))

    @staticmethod
    def add_only_matmul(x: np.ndarray, w_tern: np.ndarray, gamma: float) -> np.ndarray:
        """y = x @ (gamma * W_t)^T. На обычном CPU это один matmul (раньше было два — вдвое медленнее)."""
        return (x @ w_tern.T.astype(np.float32)) * gamma

    def _effective(self, name: str) -> Tuple[np.ndarray, float]:
        """Эффективная (квантованная) матрица и её разреженность. Кэшируется, пока веса не менялись."""
        w = getattr(self, name)
        if not self.quantize:
            return w, 0.0
        cached = self._qcache.get(name)
        if cached is not None and cached[0] == self._step and cached[1].shape == w.shape:
            return cached[1], cached[2]
        w_t, gamma, sparsity = self.quantize_b158(w)
        eff = w_t.astype(w.dtype) * w.dtype.type(gamma)
        self._qcache[name] = (self._step, eff, sparsity)
        return eff, sparsity

    def _write_slot(self, idx: int, seq: List[int], tokenizer: RussianMorphemeTokenizer) -> None:
        split_pos = max(2, len(seq) // 3)
        for i in range(1, len(seq) - 1):
            if tokenizer.id_to_token[seq[i]] in ("?", "!", "."):
                split_pos = i + 1
                break
        q_vec = np.mean(self.embed[seq[:split_pos]], axis=0)
        self.mem_keys[idx] = q_vec / (np.linalg.norm(q_vec) + 1e-6)
        self.mem_vals[idx] = np.mean(self.embed[seq[split_pos:]], axis=0)
        prompt = seq[1:split_pos]
        if idx < len(self.slot_prompts):
            self.slot_prompts[idx], self.slot_full_seqs[idx] = prompt, seq
        else:
            self.slot_prompts.append(prompt)
            self.slot_full_seqs.append(seq)

    def populate_fact_memory(self, sequences: List[List[int]], tokenizer: RussianMorphemeTokenizer) -> None:
        self.slot_prompts, self.slot_full_seqs = [], []
        for seq in sequences[: self.memory_slots]:
            if len(seq) >= 3:
                self._write_slot(len(self.slot_prompts), seq, tokenizer)

    def evaluate_epistemic_match(
        self,
        user_text: str,
        tokenizer: RussianMorphemeTokenizer
    ) -> Tuple[int, float, List[str]]:
        """
        Смысловая близость вопроса к памяти по содержательным словам и калиброванная уверенность.
        Неизвестные ключевые термины снижают уверенность пропорционально их доле.
        """
        if not self.slot_prompts:
            return 0, 0.0, []

        known_content, unknown_content = tokenizer.find_unknown_content_words(user_text)
        all_tokens = [t for t in tokenizer.tokenize_text(user_text) if t not in (",", ".", "!", "?", "—", "-", ":")]
        if not all_tokens:
            return 0, 0.0, []
        if unknown_content and not known_content:
            return 0, 0.0, unknown_content

        user_stems = set()
        for w in known_content:
            user_stems.add(w[:5] if len(w) >= 5 else w[:4] if len(w) >= 4 else w)
            for syn_key, syn_list in SYNONYM_STEMS.items():
                if w.startswith(syn_key):
                    user_stems.update(syn_list)

        best_slot, best_confidence = 0, 0.0
        for slot_idx, q_ids in enumerate(self.slot_prompts):
            q_words = [tokenizer.id_to_token[tid] for tid in q_ids]
            full_words = [tokenizer.id_to_token[tid] for tid in self.slot_full_seqs[slot_idx]]
            q_content = [w for w in q_words if w not in FUNCTION_WORDS and len(w) > 2]

            matched = 0.0
            for uw in known_content:
                u_stem = uw[:4] if len(uw) >= 4 else uw
                if uw in q_words:
                    matched += 1.0
                elif len(u_stem) >= 4 and any(qw.startswith(u_stem) for qw in q_words):
                    matched += 0.85
                elif len(u_stem) >= 4 and any(fw.startswith(u_stem) for fw in full_words):
                    matched += 0.45
            for stem in user_stems:
                if len(stem) >= 3 and any(qw.startswith(stem) for qw in q_words):
                    matched += 0.55

            if not known_content and not unknown_content:
                conf = sum(1 for t in all_tokens if t in q_words) / max(1.0, float(len(all_tokens)))
            else:
                total = float(len(known_content) + len(unknown_content))
                conf = 0.65 * (matched / total) + 0.35 * min(1.0, matched / max(1.0, float(len(q_content))))
                if unknown_content:
                    conf *= (1.0 - 0.75 * len(unknown_content) / total)

            if conf > best_confidence:
                best_confidence, best_slot = conf, slot_idx

        return best_slot, min(1.0, best_confidence), unknown_content

    def forward(self, token_ids: np.ndarray, forced_slot: int = -1, copy_bias: float = 0.0) -> Dict[str, object]:
        """
        Прямой проход. copy_bias > 0 добавляет к логиту следующего слова эталонного ответа из памяти
        (режим «проигрывания» ответа). При обучении copy_bias = 0, иначе сеть «видит» правильный ответ
        во входе и метрики обучения бессмысленны (так было в первой версии: 100% «точности»).
        """
        seq_len = len(token_ids)
        pos_idx = np.minimum(np.arange(seq_len), len(self.pos_embed) - 1)
        x = self.embed[token_ids] + self.pos_embed[pos_idx]
        W = {}
        sparsity = []
        for name in TERNARY_NAMES:
            W[name], sp = self._effective(name)
            sparsity.append(sp)

        zero = np.zeros((1, self.d_model), dtype=x.dtype)
        x_prev1 = np.vstack([zero, x[:-1]])
        x_prev2 = np.vstack([zero, zero, x[:-2]])[:seq_len]
        c1 = x_prev1 @ W["w_ctx1"].T
        c2 = x_prev2 @ W["w_ctx2"].T

        ssm_in = x @ W["w_ssm"].T + x
        ssm_state = np.zeros_like(x)
        cur_s = np.zeros(self.d_model, dtype=x.dtype)
        for t in range(seq_len):
            cur_s = self.SSM_DECAY * cur_s + self.SSM_IN * ssm_in[t]
            ssm_state[t] = cur_s

        h_ctx = x + self.C1 * c1 + self.C2 * c2 + self.C_SSM * ssm_state
        c_prob = self.sigmoid(h_ctx @ self.w_router + self.b_router)
        route_mask = c_prob[:, 0] >= self.ROUTE_THR

        if 0 <= forced_slot < self.memory_slots:
            top_slots = np.full(seq_len, forced_slot, dtype=np.int32)
        else:
            k_norms = self.mem_keys / (np.linalg.norm(self.mem_keys, axis=-1, keepdims=True) + 1e-6)
            q = ssm_state / (np.linalg.norm(ssm_state, axis=-1, keepdims=True) + 1e-6)
            top_slots = np.argmax(q @ k_norms.T, axis=-1).astype(np.int32)
        mem_context = self.mem_vals[top_slots]

        z_loop = np.zeros_like(h_ctx)
        latent_steps = np.zeros(seq_len, dtype=np.int32)
        loops: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
        w_th = W["w_think"]
        for t in np.nonzero(route_mask)[0]:
            z = h_ctx[t] + self.MEM_LOOP * mem_context[t]
            zs, ths = [z], []
            for step in range(self.max_latent_steps):
                th = np.tanh(z @ w_th.T)
                z = z + self.THINK_STEP * th
                zs.append(z)
                ths.append(th)
                latent_steps[t] += 1
                if step >= 1 and float(np.mean(np.abs(th))) < 0.12:
                    break
            z_loop[t] = z - h_ctx[t]
            loops[int(t)] = (zs, ths)

        h_final = h_ctx + self.MEM_OUT * mem_context + c_prob * z_loop
        logits = h_final @ W["w_out"].T + self.b_out

        if copy_bias and 0 <= forced_slot < len(self.slot_full_seqs):
            ref_seq = self.slot_full_seqs[forced_slot]
            n = min(seq_len, len(ref_seq) - 1)
            logits[np.arange(n), np.asarray(ref_seq[1:n + 1])] += copy_bias

        return {
            "logits": logits, "x": x, "x_prev1": x_prev1, "x_prev2": x_prev2, "h_ctx": h_ctx,
            "h_final": h_final, "z_loop": z_loop, "c_prob": c_prob, "route_mask": route_mask,
            "latent_steps": latent_steps, "top_slots": top_slots, "pos_idx": pos_idx, "loops": loops,
            "W": W, "avg_sparsity": float(np.mean(sparsity)),
        }

    def compute_loss_and_grads(self, seq_ids: List[int], complexity_hints: np.ndarray,
                               slot_idx: int = -1) -> Tuple[float, Dict[str, np.ndarray], Dict[str, object]]:
        """Точное обратное распространение для forward() (проверяется численно в tests/)."""
        inp = np.array(seq_ids[:-1], dtype=np.int32)
        tgt = np.array(seq_ids[1:], dtype=np.int32)
        hints = np.asarray(complexity_hints[:-1], dtype=np.float32)
        L = len(inp)

        fwd = self.forward(inp, forced_slot=slot_idx, copy_bias=0.0)
        W = fwd["W"]
        probs = self.softmax(fwd["logits"])
        nll = float(-np.mean(np.log(probs[np.arange(L), tgt] + 1e-9)))
        c_prob = fwd["c_prob"]
        loss = nll + self.ROUTER_COEF * float(np.mean((c_prob[:, 0] - hints) ** 2))

        dlogits = probs
        dlogits[np.arange(L), tgt] -= 1.0
        dlogits /= L
        h_final, h_ctx, z_loop = fwd["h_final"], fwd["h_ctx"], fwd["z_loop"]

        g: Dict[str, np.ndarray] = {}
        g["w_out"] = dlogits.T @ h_final
        g["b_out"] = dlogits.sum(axis=0)
        dh_final = dlogits @ W["w_out"]

        dh_ctx = dh_final.copy()
        dmem = self.MEM_OUT * dh_final
        g["w_think"] = np.zeros_like(self.w_think)
        w_th = W["w_think"]
        for t, (zs, ths) in fwd["loops"].items():
            dz = c_prob[t, 0] * dh_final[t]          # z_loop = z_K - h_ctx
            dh_ctx[t] -= dz
            for k in range(len(ths) - 1, -1, -1):
                du = self.THINK_STEP * dz * (1.0 - ths[k] ** 2)
                g["w_think"] += np.outer(du, zs[k])
                dz = dz + du @ w_th
            dh_ctx[t] += dz                          # z_0 = h_ctx + MEM_LOOP * m
            dmem[t] += self.MEM_LOOP * dz

        dc = np.sum(dh_final * z_loop, axis=-1, keepdims=True)
        dc += self.ROUTER_COEF * 2.0 * (c_prob - hints[:, None]) / L
        d_rlogit = dc * c_prob * (1.0 - c_prob)
        g["w_router"] = h_ctx.T @ d_rlogit
        g["b_router"] = d_rlogit.sum(axis=0)
        dh_ctx += d_rlogit @ self.w_router.T

        g["mem_vals"] = np.zeros_like(self.mem_vals)
        np.add.at(g["mem_vals"], fwd["top_slots"], dmem)

        dx = dh_ctx.copy()
        g["w_ctx1"] = (self.C1 * dh_ctx).T @ fwd["x_prev1"]
        g["w_ctx2"] = (self.C2 * dh_ctx).T @ fwd["x_prev2"]
        if L > 1:
            dx[:-1] += (self.C1 * dh_ctx[1:]) @ W["w_ctx1"]
        if L > 2:
            dx[:-2] += (self.C2 * dh_ctx[2:]) @ W["w_ctx2"]
        # Обратный проход через рекуррентность SSM (раньше учитывался только текущий шаг)
        gs = np.zeros(self.d_model, dtype=dh_ctx.dtype)
        d_in = np.zeros_like(dh_ctx)
        for t in range(L - 1, -1, -1):
            gs = self.C_SSM * dh_ctx[t] + self.SSM_DECAY * gs
            d_in[t] = self.SSM_IN * gs
        g["w_ssm"] = d_in.T @ fwd["x"]
        dx += d_in + d_in @ W["w_ssm"]               # ssm_in = x W^T + x

        g["embed"] = np.zeros_like(self.embed)
        np.add.at(g["embed"], inp, dx)
        g["pos_embed"] = np.zeros_like(self.pos_embed)
        np.add.at(g["pos_embed"], fwd["pos_idx"], dx)
        return loss, g, fwd

    def _adamw_step(self, grads: Dict[str, np.ndarray], lr: float, wd: float = 1e-4) -> None:
        self._step += 1
        b1, b2, eps = 0.9, 0.99, 1e-8
        bc1 = 1.0 - (b1 ** self._step)
        bc2 = 1.0 - (b2 ** self._step)
        # Клиппинг по общей норме градиента (а не поэлементно, как раньше)
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        clip = min(1.0, 1.0 / (norm + 1e-6))
        for name, g in grads.items():
            g = g * clip
            m = self._m[name] = b1 * self._m[name] + (1.0 - b1) * g
            v = self._v[name] = b2 * self._v[name] + (1.0 - b2) * (g * g)
            param = getattr(self, name)
            update = (m / bc1) / (np.sqrt(v / bc2) + eps)
            if name.startswith("w_"):
                update = update + wd * param
            param -= (lr * update).astype(param.dtype)

    def train_on_sequence(
        self,
        seq_ids: List[int],
        complexity_hints: np.ndarray,
        slot_idx: int = -1,
        lr: float = 0.008
    ) -> Dict[str, float]:
        loss, grads, fwd = self.compute_loss_and_grads(seq_ids, complexity_hints, slot_idx)
        self._adamw_step(grads, lr=lr)
        tgt = np.array(seq_ids[1:], dtype=np.int32)
        return {
            "loss": loss,
            "accuracy": float(np.mean(np.argmax(fwd["logits"], axis=-1) == tgt)),
            "routed_ratio": float(np.mean(fwd["route_mask"])),
            "avg_latent_steps": float(np.mean(fwd["latent_steps"])),
            "sparsity": float(fwd["avg_sparsity"]),
        }

    # ------------------------------------------------ мгновенное запоминание нового факта

    def _grow_rows(self, name: str, n_new: int, std: float) -> None:
        arr = getattr(self, name)
        extra_shape = (n_new,) + arr.shape[1:]
        extra = (self.rng.standard_normal(extra_shape) * std).astype(arr.dtype) if std else \
            np.zeros(extra_shape, dtype=arr.dtype)
        setattr(self, name, np.concatenate([arr, extra], axis=0))
        if name in self._m:
            self._m[name] = np.concatenate([self._m[name], np.zeros(extra_shape, dtype=arr.dtype)])
            self._v[name] = np.concatenate([self._v[name], np.zeros(extra_shape, dtype=arr.dtype)])

    def add_fact(self, chain_text: str, tokenizer: RussianMorphemeTokenizer, steps: int = 25,
                 lr: float = 0.004, replay: int = 2) -> int:
        """
        Добавляет пару «вопрос ? ответ» в память за доли секунды (раньше вся модель переобучалась
        с нуля на всём корпусе — минуты на больших данных). Новые слова расширяют словарь,
        затем несколько шагов дообучения на новом факте + повтор старых (против забывания).
        """
        for tok in tokenizer.tokenize_text(chain_text):
            tokenizer._add_token(tok)
        n_new = tokenizer.vocab_size - self.vocab_size
        if n_new > 0:
            self._grow_rows("embed", n_new, 0.22)
            self._grow_rows("w_out", n_new, 1.0 / math.sqrt(self.d_model))
            self._grow_rows("b_out", n_new, 0.0)
            self.vocab_size = tokenizer.vocab_size
        seq = tokenizer.encode(chain_text)
        if len(seq) > self.max_seq_len:
            self._grow_rows("pos_embed", len(seq) - self.max_seq_len, 0.12)
            self.max_seq_len = len(seq)
        slot = len(self.slot_prompts)
        if slot >= self.memory_slots:
            self.mem_keys = np.concatenate([self.mem_keys, np.zeros((16, self.d_model), np.float32)])
            self._grow_rows("mem_vals", 16, 0.0)
            self.memory_slots += 16
        self._write_slot(slot, seq, tokenizer)
        self._qcache.clear()
        hints = tokenizer.compute_complexity_hints(seq)
        for _ in range(steps):
            self.train_on_sequence(seq, hints, slot_idx=slot, lr=lr)
            for _ in range(replay if slot > 0 else 0):
                j = int(self.rng.integers(0, slot))
                old = self.slot_full_seqs[j]
                self.train_on_sequence(old, tokenizer.compute_complexity_hints(old), slot_idx=j, lr=lr)
        return slot

    # ------------------------------------------------ генерация ответа

    def _decode_slot_sequence(
        self,
        slot_idx: int,
        tokenizer: RussianMorphemeTokenizer,
        max_new_tokens: int = 85
    ) -> Tuple[str, List[Dict[str, object]]]:
        seq = [tokenizer.bos_id] + list(self.slot_prompts[slot_idx])
        generated_ids: List[int] = []
        telemetry: List[Dict[str, object]] = []
        banned = [tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id]

        for _ in range(max_new_tokens):
            fwd = self.forward(np.array(seq, dtype=np.int32), forced_slot=slot_idx, copy_bias=self.copy_bias)
            logits_last = fwd["logits"][-1].copy()
            logits_last[banned] = -1e9
            if not self.copy_bias:                # защита от зацикливания для «чистой» генерации сети
                for r in generated_ids[-4:]:
                    logits_last[r] -= 2.5
            next_id = int(np.argmax(logits_last))
            if next_id == tokenizer.eos_id:
                break
            generated_ids.append(next_id)
            seq.append(next_id)
            telemetry.append({
                "token": tokenizer.id_to_token[next_id],
                "complexity": float(fwd["c_prob"][-1, 0]),
                "latent_steps": int(fwd["latent_steps"][-1]),
                "memory_slot": slot_idx,
            })
            if len(seq) >= self.max_seq_len - 1:
                break

        return tokenizer.decode(generated_ids), telemetry

    def generate_reply(
        self,
        prompt_text: str,
        tokenizer: RussianMorphemeTokenizer,
        max_new_tokens: int = 85,
    ) -> Tuple[str, List[Dict[str, object]], Dict[str, object]]:
        """
        1. Логическая задача или вычисление -> EpistemicLogicEngine (правила, а не нейросеть).
        2. Поиск ближайшего вопроса в памяти и оценка уверенности.
        3. Уверенно -> воспроизведение ответа из памяти; частично/неуверенно -> честное «не знаю».
        """
        logic_solution = EpistemicLogicEngine.try_solve_logic_or_math(prompt_text)
        if logic_solution is not None:
            return logic_solution, [], {"mode": "logical_deduction", "confidence": 1.0, "unknown_words": []}

        best_slot, confidence, unknown_words = self.evaluate_epistemic_match(prompt_text, tokenizer)

        if confidence >= 0.45 and not unknown_words:
            reply_text, telemetry = self._decode_slot_sequence(best_slot, tokenizer, max_new_tokens)
            return reply_text, telemetry, {"mode": "known_fact", "confidence": confidence, "unknown_words": [],
                                           "slot": best_slot}

        if confidence >= 0.25 and 0 <= best_slot < len(self.slot_prompts):
            partial_reply, telemetry = self._decode_slot_sequence(best_slot, tokenizer, max_new_tokens)
            honest_reply = EpistemicLogicEngine.build_honest_unknown_response(
                prompt_text, unknown_words, best_partial_reply=partial_reply, partial_confidence=confidence)
            return honest_reply, telemetry, {"mode": "partial_reasoning", "confidence": confidence,
                                             "unknown_words": unknown_words, "slot": best_slot}

        honest_unknown = EpistemicLogicEngine.build_honest_unknown_response(
            prompt_text, unknown_words, best_partial_reply=None, partial_confidence=confidence)
        return honest_unknown, [], {"mode": "honest_unknown", "confidence": confidence, "unknown_words": unknown_words}

    def network_accuracy(self, use_copy: bool = False) -> float:
        """Доля слов ответов, которые сеть угадывает сама (teacher forcing). Честная метрика качества сети."""
        accs = []
        for i, seq in enumerate(self.slot_full_seqs):
            inp, tgt = np.array(seq[:-1]), np.array(seq[1:])
            logits = self.forward(inp, forced_slot=i, copy_bias=self.copy_bias if use_copy else 0.0)["logits"]
            accs.append(np.mean(np.argmax(logits, axis=-1) == tgt))
        return float(np.mean(accs)) if accs else 0.0

    # ------------------------------------------------ сохранение

    def save_checkpoint(self, path: str, tokenizer: RussianMorphemeTokenizer) -> float:
        pack_dict: Dict[str, object] = {
            "format_version": np.array([2], dtype=np.int32),
            # Словарь одной строкой: раньше dtype="<U64" молча обрезал слова длиннее 64 символов
            "vocab_str": np.array(["\n".join(tokenizer.id_to_token)]),
            "d_model": np.array([self.d_model], dtype=np.int32),
            "memory_slots": np.array([self.memory_slots], dtype=np.int32),
            "max_seq_len": np.array([self.max_seq_len], dtype=np.int32),
            "max_latent_steps": np.array([self.max_latent_steps], dtype=np.int32),
            "embed": self.embed.astype(np.float16),
            "pos_embed": self.pos_embed.astype(np.float16),
            "b_out": self.b_out.astype(np.float16),
            "w_router": self.w_router.astype(np.float16),
            "b_router": self.b_router.astype(np.float16),
            "mem_keys": self.mem_keys.astype(np.float16),
            "mem_vals": self.mem_vals.astype(np.float16),
            "prompts_str": np.array(["\n".join(" ".join(map(str, p)) for p in self.slot_prompts)]),
            "seqs_str": np.array(["\n".join(" ".join(map(str, s)) for s in self.slot_full_seqs)]),
        }
        for name in TERNARY_NAMES:
            w_tern, gamma, _ = self.quantize_b158(getattr(self, name))
            packed, shape = pack_ternary_matrix(w_tern)
            pack_dict[f"{name}_packed"] = packed
            pack_dict[f"{name}_shape"] = np.array(shape, dtype=np.int32)
            pack_dict[f"{name}_gamma"] = np.array([gamma], dtype=np.float32)
        np.savez_compressed(path, **pack_dict)
        return os.path.getsize(path) / 1024.0

    @classmethod
    def load_checkpoint(cls, path: str) -> Tuple["TrainableSLRNetEngine", RussianMorphemeTokenizer]:
        data = np.load(path, allow_pickle=False)
        tokenizer = RussianMorphemeTokenizer()
        if "vocab_str" in data:
            tokenizer.id_to_token = str(data["vocab_str"][0]).split("\n")
        else:  # старый формат
            tokenizer.id_to_token = [str(t) for t in data["vocab"]]
        tokenizer.rebuild_index()

        engine = cls(
            vocab_size=tokenizer.vocab_size,
            d_model=int(data["d_model"][0]),
            memory_slots=int(data["memory_slots"][0]),
            max_seq_len=int(data["max_seq_len"][0]) if "max_seq_len" in data else 96,
            max_latent_steps=int(data["max_latent_steps"][0]) if "max_latent_steps" in data else 3,
        )
        for name in ("embed", "pos_embed", "b_out", "w_router", "b_router", "mem_keys", "mem_vals"):
            setattr(engine, name, data[name].astype(np.float32))
        for key, attr in (("prompts_str", "slot_prompts"), ("seqs_str", "slot_full_seqs")):
            if key in data:
                raw = str(data[key][0])
                setattr(engine, attr, [[int(x) for x in line.split()] for line in raw.splitlines() if line.strip()])
        for name in TERNARY_NAMES:
            shape = tuple(int(x) for x in data[f"{name}_shape"])
            gamma = float(data[f"{name}_gamma"][0])
            setattr(engine, name, unpack_ternary_matrix(data[f"{name}_packed"], shape).astype(np.float32) * gamma)
        engine._reset_optimizer()
        return engine, tokenizer


# =============================================================================
# 7. КОНВЕЙЕР ОБУЧЕНИЯ И ИНТЕРАКТИВНЫЙ ЧАТ С ДООБУЧЕНИЕМ НА ЛЕТУ
# =============================================================================

def train_model_on_corpus(
    corpus_text: str,
    d_model: int = 96,
    epochs: int = 8,
    lr: float = 0.008,
    save_path: Optional[str] = None,
    copy_bias: float = 5.5,
    verbose: bool = True,
) -> Tuple[TrainableSLRNetEngine, RussianMorphemeTokenizer]:
    chains = parse_subtitle_or_text_corpus(corpus_text)
    tokenizer = RussianMorphemeTokenizer()
    tokenizer.fit(chains)

    sequences: List[List[int]] = []
    hints_list: List[np.ndarray] = []
    max_len = 96
    for line in chains:
        ids = tokenizer.encode(line, add_bos=True, add_eos=True)
        if len(ids) >= 4:
            sequences.append(ids)
            hints_list.append(tokenizer.compute_complexity_hints(ids))
            max_len = max(max_len, len(ids) + 4)

    engine = TrainableSLRNetEngine(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        memory_slots=max(64, len(sequences) + 16),
        max_seq_len=max_len,
        copy_bias=copy_bias,
    )
    engine.populate_fact_memory(sequences, tokenizer)

    log = print if verbose else (lambda *a, **k: None)
    log("=" * 86)
    log("ОБУЧЕНИЕ ТРОИЧНОЙ МИКРО-СЕТИ SLR-Net НА КОРПУСЕ ДИАЛОГОВ")
    log("=" * 86)
    log(f"• Объём словаря (словоформ)      : {tokenizer.vocab_size}")
    log(f"• Диалоговых цепочек в памяти    : {len(sequences)}")
    log(f"• Размерность вектора мысли      : d_model = {d_model}")
    log("• Метрики ниже — БЕЗ подсказки ответа (раньше подсказка давала ложные ~100%)")
    log("-" * 86)

    t0 = time.time()
    order = np.arange(len(sequences))
    for epoch in range(1, epochs + 1):
        cur_lr = lr * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(1, epochs))))
        totals = np.zeros(5)
        engine.rng.shuffle(order)            # перемешивание примеров каждую эпоху
        for idx in order:
            st = engine.train_on_sequence(sequences[idx], hints_list[idx], slot_idx=int(idx), lr=cur_lr)
            totals += [st["loss"], st["accuracy"], st["routed_ratio"], st["avg_latent_steps"], st["sparsity"]]
        n = max(1, len(sequences))
        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            log(f"Эпоха {epoch:2d}/{epochs} | Loss: {totals[0] / n:.4f} | Точность сети: {totals[1] / n * 100:5.1f}% | "
                f"В цикле мысли: {totals[2] / n * 100:4.1f}% слов | Ср. тактов: {totals[3] / n:.2f} | "
                f"Нулевых синапсов: {totals[4] / n * 100:.1f}%")

    log("-" * 86)
    log(f"Обучение завершено за {time.time() - t0:.2f} сек.")
    if save_path:
        size_kb = engine.save_checkpoint(save_path, tokenizer)
        log(f"Троичные веса упакованы (5 тритов в 1 байт) и сохранены: {save_path} ({size_kb:.1f} КБ)")
    return engine, tokenizer


def run_demo_dialog(engine: TrainableSLRNetEngine, tokenizer: RussianMorphemeTokenizer) -> None:
    print("\n" + "=" * 86)
    print("ПРОВЕРКА: 1) ПАМЯТЬ ДИАЛОГОВ, 2) ЛОГИЧЕСКИЙ МОДУЛЬ, 3) ЧЕСТНОЕ ПРИЗНАНИЕ НЕЗНАНИЯ")
    print("=" * 86)

    test_cases = [
        ("Киносубтитры (SRT)", "Как думаешь, мы сможем починить этот двигатель до рассвета?"),
        ("Живой диалог", "В чём метод Шерлока Холмса?"),
        ("Логическая загадка", "Ты бежишь марафон и обгоняешь бегуна который бежал вторым на каком ты теперь месте?"),
        ("Новая логическая задача (нет в корпусе!)", "Если сокол быстрее орла, а орёл быстрее воробья, кто самый медленный?"),
        ("Новое вычисление (нет в корпусе!)", "Сколько будет двадцать пять умножить на четыре минус пятнадцать?"),
        ("Не арифметика (диапазон лет)", "Что было в 1941-1945 годах?"),
        ("Заведомо неизвестный факт #1", "Какая столица у государства Мадагаскар?"),
        ("Заведомо неизвестный факт #2", "Кто изобрёл первый квантовый синхрофазотрон в 1999 году?"),
    ]
    labels = {
        "known_fact": "Ответ из памяти диалогов (уверенность {c:.0f}%)",
        "logical_deduction": "Логический модуль (правила)",
        "partial_reasoning": "Гипотеза при неполных данных (уверенность {c:.0f}%)",
        "honest_unknown": "Честное признание незнания (уверенность {c:.0f}%, неизвестно: {u})",
    }
    for category, q in test_cases:
        reply, _, meta = engine.generate_reply(q, tokenizer)
        label = labels.get(str(meta.get("mode")), "").format(c=float(meta["confidence"]) * 100,
                                                            u=meta["unknown_words"])
        print(f"\n[{category}]")
        print(f"Вы      : {q}")
        print(f"SLR-Net : {reply}")
        print(f"  └─ Режим: {label}")

    print("\n" + "-" * 86)
    print(f"Честная метрика: сеть САМА угадывает {engine.network_accuracy() * 100:.1f}% слов ответов "
          f"(с подсказкой копирования из памяти — {engine.network_accuracy(use_copy=True) * 100:.1f}%).")


_NOT_AN_ANSWER = re.compile(r"^(нет|не знаю|забудь|ладно|проехали|неважно|отмена|стоп|/)", re.IGNORECASE)


def interactive_chat_loop(
    engine: TrainableSLRNetEngine,
    tokenizer: RussianMorphemeTokenizer,
    corpus_text: str,
    save_path: str
) -> None:
    print("\n" + "=" * 86)
    print("ЖИВОЙ ЧАТ С МОДЕЛЬЮ SLR-Net")
    print("• '/обучить путь_к_файлу.txt' — переобучить на корпусе + вашем .txt/.srt файле")
    print("• '/рентген' — телеметрия, '/копия' — вкл/выкл подсказку копирования из памяти")
    print("• 'выход' — завершить. Если модель чего-то не знает — напишите ответ следующей фразой.")
    print("=" * 86)

    show_xray = False
    saved_copy_bias = engine.copy_bias or 5.5
    pending_unknown_question: Optional[str] = None

    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо скорой встречи!")
            break

        low = user_input.lower()
        if not user_input or low in ("выход", "exit", "quit", "q"):
            print("SLR-Net: До свидания! Мы обязательно продолжим наш разговор. Береги себя!")
            break

        if low in ("/рентген", "/xray"):
            show_xray = not show_xray
            print(f"  [Телеметрия: {'ВКЛЮЧЕНА' if show_xray else 'ВЫКЛЮЧЕНА'}]")
            continue

        if low in ("/копия", "/copy"):
            engine.copy_bias = 0.0 if engine.copy_bias else saved_copy_bias
            print(f"  [Подсказка копирования: {'ВКЛ' if engine.copy_bias else 'ВЫКЛ — теперь вы видите, что генерирует сама сеть'}]")
            continue

        if low.startswith("/обучить "):
            custom_path = user_input[9:].strip().strip('"').strip("'")
            if os.path.exists(custom_path):
                with open(custom_path, "r", encoding="utf-8", errors="ignore") as f:
                    corpus_text = corpus_text.strip() + "\n" + f.read()
                engine, tokenizer = train_model_on_corpus(
                    corpus_text, d_model=engine.d_model, epochs=6, save_path=save_path, copy_bias=engine.copy_bias)
                print("SLR-Net: Отлично! Я прочитал новый файл и обновил свои троичные веса!")
            else:
                print(f"  [Файл не найден по пути: {custom_path}]")
            continue

        # Если модель только что призналась, что не знает, а сейчас пришло утверждение — запоминаем
        if (pending_unknown_question and "?" not in user_input and len(user_input.split()) >= 2
                and not _NOT_AN_ANSWER.match(user_input)):
            q_clean = pending_unknown_question.strip()
            if q_clean[-1] not in ("?", "!", "."):
                q_clean += " ?"
            a_clean = user_input.strip()
            if a_clean[-1] not in (".", "!"):
                a_clean += " ."
            new_chain = f"{q_clean} {a_clean}"
            corpus_text = corpus_text.strip() + "\n" + new_chain
            t0 = time.time()
            engine.add_fact(new_chain, tokenizer)
            if save_path:
                engine.save_checkpoint(save_path, tokenizer)
            pending_unknown_question = None
            print(f"SLR-Net: Спасибо, что объяснил! Я записал этот факт в память за {time.time() - t0:.2f} с. "
                  f"Можешь спросить меня об этом снова!")
            continue

        reply, telemetry, meta = engine.generate_reply(user_input, tokenizer)
        print(f"SLR-Net: {reply}")
        pending_unknown_question = user_input if meta.get("mode") in ("honest_unknown", "partial_reasoning") else None

        if show_xray:
            thought_words = sum(1 for item in telemetry if int(item["latent_steps"]) > 0)
            print(f"  [Телеметрия: режим={meta.get('mode')}, уверенность={float(meta.get('confidence', 0)) * 100:.0f}%, "
                  f"слов в цикле мысли={thought_words}/{len(telemetry)}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Поисковый чат-прототип SLR-Net на субтитрах (.txt / .srt)")
    parser.add_argument("--corpus", type=str, default="corpus_movie_subtitles_and_logic.txt",
                        help="Путь к текстовому файлу (.txt или .srt) с диалогами для обучения")
    parser.add_argument("--epochs", type=int, default=6, help="Число эпох обучения (по умолчанию: 6)")
    parser.add_argument("--d-model", type=int, default=96, help="Размерность скрытого вектора (по умолчанию: 96)")
    parser.add_argument("--save", type=str, default="slr_net_trained.npz", help="Куда сохранить веса")
    parser.add_argument("--load", type=str, default=None, help="Загрузить обученные веса .npz без обучения")
    parser.add_argument("--copy-bias", type=float, default=5.5,
                        help="Сила подсказки копирования ответа из памяти при генерации (0 — только сеть)")
    parser.add_argument("--no-chat", action="store_true", help="Не запускать интерактивный чат после демонстрации")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [args.corpus, os.path.join(script_dir, args.corpus),
                       os.path.join(script_dir, "corpus_movie_subtitles_and_logic.txt")]
    corpus_text, loaded_corpus_path = "", ""
    for p in candidate_paths:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                corpus_text = f.read()
            loaded_corpus_path = p
            break

    load_path = args.load
    if load_path and not os.path.exists(load_path) and os.path.exists(os.path.join(script_dir, load_path)):
        load_path = os.path.join(script_dir, load_path)

    if load_path and os.path.exists(load_path):
        print(f"Загрузка обученного троичного чекпоинта из: {load_path} ...")
        engine, tokenizer = TrainableSLRNetEngine.load_checkpoint(load_path)
        engine.copy_bias = args.copy_bias
    else:
        if args.load:
            print(f"Чекпоинт {args.load} не найден — обучаю модель заново.")
        if not corpus_text:
            raise FileNotFoundError(
                f"Не найден файл корпуса '{args.corpus}'. Укажите путь: python slr_net_trainer_chat.py --corpus файл.txt")
        print(f"Прочитан файл диалогов: {loaded_corpus_path} ({len(corpus_text)} символов)")
        engine, tokenizer = train_model_on_corpus(
            corpus_text=corpus_text, d_model=args.d_model, epochs=args.epochs,
            save_path=args.save, copy_bias=args.copy_bias)

    run_demo_dialog(engine, tokenizer)

    if sys.stdin.isatty() and not args.no_chat:
        interactive_chat_loop(engine, tokenizer, corpus_text, args.save)


if __name__ == "__main__":
    main()
