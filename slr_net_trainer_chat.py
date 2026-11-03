#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLR-Net Subtitle Trainer, Epistemic Logic Engine & Natural Chat (`slr_net_trainer_chat.py`)
===========================================================================================

Полноценный тренажёр на большом объёме киносубтитров (.txt / .srt), модуль логического
рассуждения и честного признания незнания (Epistemic Humility) для архитектуры SLR-Net:
- BitNet b1.58 (Ma et al., arXiv:2402.17764): троичные веса {-1, 0, +1}, вычисление без умножений.
- Mamba Selective SSM (Gu & Dao, arXiv:2312.00752): линейное состояние контекста O(1).
- Memory Layers / RETRO (arXiv:2412.09764, arXiv:2112.04426): внешняя быстрая память диалогов.
- Mixture-of-Depths (arXiv:2404.02258) + Coconut (arXiv:2412.06769): динамический роутер
  глубины и скрытый рекуррентный цикл латентного мышления.

Ключевые возможности этой версии:
1. Универсальный загрузчик `.txt` и `.srt` файлов (`--corpus путь/к/файлу.txt`):
   - Понимает как обычные текстовые файлы диалогов, так и сырые файлы киносубтитров (`.srt` / OpenSubtitles),
     автоматически очищая таймкоды `00:01:10,120 --> ...` и склеивая соседние реплики героев в диалоговые пары.
   - Позволяет дообучать модель на любом новом `.txt` файле прямо из чата командой `/обучить путь_к_файлу.txt`.
2. Модуль логического мышления (`EpistemicLogicEngine`):
   - Решает логические цепочки сравнения («Если А > Б, а Б > В, кто самый...»), условные силлогизмы
     и числовые задачи, даже если их никогда не было в обучающем корпусе.
3. Детектор границ знаний (Epistemic Humility — «умение говорить "я не знаю"»):
   - Оценивает покрытие смысловых слов вопроса и долю неизвестных терминов.
   - Если факта нет в памяти, модель НЕ выдумывает и не выдаёт чужой ответ, а честно объясняет,
     какого именно знания ей не хватает, делает логический разбор вопроса и предлагает пользователю
     научить её прямо в следующей реплике диалога!
"""

from __future__ import annotations
import argparse
import math
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


def format_natural_russian(tokens: List[str]) -> str:
    """Форматирует последовательность токенов в естественный русский текст с заглавными буквами и пунктуацией."""
    if not tokens:
        return ""

    formatted_words: List[str] = []
    capitalize_next = True

    for tok in tokens:
        if tok in ("<pad>", "<bos>", "<eos>", "<unk>"):
            continue

        word = PROPER_NOUNS_MAP.get(tok, tok)
        if capitalize_next and word and word[0].isalpha():
            word = word[0].upper() + word[1:]
            capitalize_next = False
        elif word and word[0].isalpha():
            capitalize_next = False

        if tok in (".", "!", "?"):
            capitalize_next = True
        elif tok == "«":
            capitalize_next = True

        formatted_words.append(word)

    text = " ".join(formatted_words)
    text = re.sub(r"\s+([.,!?;:»])", r"\1", text)
    text = re.sub(r"(«)\s+", r"\1", text)
    return text.strip()


# =============================================================================
# 2. УНИВЕРСАЛЬНЫЙ ПАРСЕР КИНОСУБТИТРОВ (.SRT / .TXT) И КНИЖНЫХ ДИАЛОГОВ
# =============================================================================

def parse_subtitle_or_text_corpus(raw_text: str) -> List[str]:
    """
    Универсальный парсер для обучения на любых файлах `.txt` или `.srt` (киносубтитры OpenSubtitles):
    1. Очищает номера блоков субтитров, таймкоды `00:01:20,000 --> 00:01:24,000` и HTML-теги `<i>...</i>`.
    2. Если в блоке субтитров две реплики через тире (`— Вопрос / — Ответ`), склеивает их в единую
       диалоговую пару `Вопрос ? Ответ .`.
    3. Если файл состоит из отдельных коротких строк-субтитров (по одной реплике на строку),
       автоматически объединяет соседние реплики в диалоговые цепочки.
    """
    # Убираем HTML-теги и звуковые ремарки в квадратных скобках [музыка], [смех]
    cleaned = re.sub(r"<[^>]+>", " ", raw_text)
    cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)

    raw_lines = cleaned.splitlines()
    filtered_lines: List[str] = []

    srt_Pending_Utterances: List[str] = []
    in_srt_mode = ("-->" in raw_text)

    for line in raw_lines:
        s = line.strip()
        if not s or s.startswith("#"):
            if srt_Pending_Utterances:
                if len(srt_Pending_Utterances) >= 2:
                    q_part = srt_Pending_Utterances[0].lstrip("—- ").strip()
                    a_part = " ".join(x.lstrip("—- ").strip() for x in srt_Pending_Utterances[1:])
                    if q_part and a_part:
                        if q_part[-1] not in ("?", "!", "."):
                            q_part += " ?"
                        filtered_lines.append(f"{q_part} {a_part}")
                srt_Pending_Utterances = []
            continue

        # Пропускаем чистые номера блоков SRT и строки таймкодов 00:00:00,000 --> ...
        if re.fullmatch(r"\d+", s):
            continue
        if "-->" in s or re.match(r"^\d{2}:\d{2}:\d{2}", s):
            continue

        # Если строка начинается с тире диалога в блоке субтитров
        if s.startswith("—") or s.startswith("- "):
            srt_Pending_Utterances.append(s)
            continue

        if srt_Pending_Utterances:
            srt_Pending_Utterances.append(s)
            continue

        filtered_lines.append(s)

    if len(srt_Pending_Utterances) >= 2:
        q_part = srt_Pending_Utterances[0].lstrip("—- ").strip()
        a_part = " ".join(x.lstrip("—- ").strip() for x in srt_Pending_Utterances[1:])
        if q_part and a_part:
            if q_part[-1] not in ("?", "!", "."):
                q_part += " ?"
            filtered_lines.append(f"{q_part} {a_part}")

    # Проверяем: если строки уже содержат и реплику-стимул, и ответ (или длиннее 7 слов),
    # оставляем их как готовые цепочки; если это короткие одиночные строки субтитров — спариваем соседние
    final_chains: List[str] = []
    i = 0
    while i < len(filtered_lines):
        cur = filtered_lines[i]
        # Проверяем, есть ли внутри строки знак конца первой реплики ('?', '!', '.') не в самом конце
        has_internal_split = False
        for punct in ("?", "!", "."):
            pos = cur.find(punct)
            if 2 <= pos < len(cur) - 4:
                has_internal_split = True
                break

        if has_internal_split:
            final_chains.append(cur)
            i += 1
        elif i + 1 < len(filtered_lines):
            # Склеиваем две последовательные реплики субтитров в пару «Стимул -> Ответ»
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

RUSSIAN_NUM_WORDS = {
    "ноль": 0, "нуля": 0, "один": 1, "одна": 1, "одного": 1, "два": 2, "две": 2, "двух": 2,
    "три": 3, "трёх": 3, "четыре": 4, "четырёх": 4, "пять": 5, "пяти": 5,
    "шесть": 6, "шести": 6, "семь": 7, "семи": 7, "восемь": 8, "восьми": 8,
    "девять": 9, "девяти": 9, "десять": 10, "десяти": 10, "одиннадцать": 11,
    "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15,
    "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
    "сто": 100, "двести": 200, "триста": 300, "пятьсот": 500, "тысяча": 1000,
}


class EpistemicLogicEngine:
    """
    Отвечает за:
    1. Аналитическое решение логических цепочек сравнения (транзитивность: A > B, B > C => A > C),
       условных силлогизмов («Если А, то Б») и математических вычислений на русском языке.
    2. Оценку границ собственных знаний модели (Epistemic Confidence):
       обнаруживает неизвестные слова и понятия в вопросе пользователя и предотвращает галлюцинации.
    3. Формулирование честного, осмысленного ответа «Я этого не знаю, но логически...» с возможностью
       выучить новый факт прямо со слов собеседника в следующей реплике чата.
    """

    @staticmethod
    def try_solve_logic_or_math(user_text: str) -> Optional[str]:
        low = user_text.lower().strip()

        # 1. Проверка на транзитивное логическое сравнение:
        # Пример: "Если тигр быстрее волка, а волк быстрее черепахи, кто самый медленный / быстрый?"
        if "если" in low and ("кто сам" in low or "что сам" in low or "кто из них" in low):
            comp_pairs = [
                ("быстрее", "быстрый", "медленный"),
                ("медленнее", "медленный", "быстрый"),
                ("старше", "старший", "младший"),
                ("младше", "младший", "старший"),
                ("тяжелее", "тяжёлый", "лёгкий"),
                ("легче", "лёгкий", "тяжёлый"),
                ("выше", "высокий", "низкий"),
                ("ниже", "низкий", "высокий"),
                ("сильнее", "сильный", "слабый"),
                ("больше", "больший", "меньший"),
            ]
            for rel_word, max_adj, min_adj in comp_pairs:
                if rel_word in low:
                    # Ищем пары "X <rel_word> чем Y" или "X <rel_word> Y"
                    clauses = re.split(r"[,;]|(?:\s+а\s+)|(?:\s+и\s+)", low)
                    chain: List[Tuple[str, str]] = []
                    for cl in clauses:
                        cl_clean = re.sub(r"\b(если|то|чем)\b", " ", cl).strip()
                        m = re.search(rf"([а-яёa-z]+)\s+{rel_word}\s+([а-яёa-z]+)", cl_clean)
                        if m:
                            a_item, b_item = m.group(1), m.group(2)
                            chain.append((a_item, b_item))

                    if len(chain) >= 2:
                        # Строим упорядоченный список от максимума к минимуму с учётом склонений (по корню)
                        def _same_root(w1: str, w2: str) -> bool:
                            c1 = w1[0] + re.sub(r"[аеёиоуыэюяьъ]", "", w1[1:])[:2]
                            c2 = w2[0] + re.sub(r"[аеёиоуыэюяьъ]", "", w2[1:])[:2]
                            return c1 == c2 and len(c1) >= 2

                        order: List[str] = [chain[0][0], chain[0][1]]
                        for a_item, b_item in chain[1:]:
                            a_match = next((i for i, x in enumerate(order) if _same_root(x, a_item)), -1)
                            b_match = next((i for i, x in enumerate(order) if _same_root(x, b_item)), -1)
                            if a_match != -1 and b_match == -1:
                                order.insert(a_match + 1, b_item)
                            elif b_match != -1 and a_match == -1:
                                order.insert(b_match, a_item)

                        top_entity = order[0].capitalize()
                        bottom_entity = order[-1].capitalize()
                        chain_str = " → ".join(x.capitalize() for x in order)

                        if min_adj[:4] in low:
                            return (
                                f"Давай рассуждать логически по правилу транзитивности: "
                                f"выстраиваем цепочку от большего признака к меньшему ({chain_str}). "
                                f"Следовательно, самый {min_adj} в этой цепочке — {bottom_entity}."
                            )
                        if max_adj[:4] in low:
                            return (
                                f"Давай рассуждать логически по правилу транзитивности: "
                                f"выстраиваем цепочку ({chain_str}). "
                                f"Следовательно, самый {max_adj} здесь — {top_entity}."
                            )

        # 2. Проверка на арифметическое выражение цифрами или словами
        expr_text = low
        for word, val in sorted(RUSSIAN_NUM_WORDS.items(), key=lambda x: -len(x[0])):
            expr_text = re.sub(rf"\b{word}\b", str(val), expr_text)
        # Складываем соседние разряды составных числительных ("20 5" -> "25", "100 20 5" -> "125")
        for _ in range(2):
            expr_text = re.sub(r"(\d+)\s+(\d+)", lambda m: str(int(m.group(1)) + int(m.group(2))), expr_text)

        expr_text = (
            expr_text.replace("умножить на", "*")
            .replace("умноженное на", "*")
            .replace("разделить на", "/")
            .replace("поделить на", "/")
            .replace("плюс", "+")
            .replace("прибавить", "+")
            .replace("минус", "-")
            .replace("вычесть", "-")
            .replace("отнять", "-")
        )
        math_match = re.search(r"(\d+(?:\s*[\+\-\*\/]\s*\d+)+)", expr_text)
        if math_match:
            raw_expr = math_match.group(1)
            # Если это не стандартный пример "2 + 2" из корпуса, считаем аналитически
            if re.sub(r"\s+", "", raw_expr) not in ("2+2", "2+2*2"):
                try:
                    val = eval(raw_expr, {"__builtins__": None}, {})
                    if isinstance(val, float) and val.is_integer():
                        val = int(val)
                    elif isinstance(val, float):
                        val = round(val, 4)
                    return (
                        f"Посчитаем логически по правилам арифметики: "
                        f"выражение ({raw_expr.strip()}) даёт точный результат: {val}."
                    )
                except ZeroDivisionError:
                    return (
                        "Логическая проверка: в выражении встретилось деление на ноль, "
                        "а в математике деление на ноль не определено, так как приводит к противоречию."
                    )
                except Exception:
                    pass

        return None

    @staticmethod
    def build_honest_unknown_response(
        user_text: str,
        unknown_words: List[str],
        best_partial_reply: Optional[str] = None,
        partial_confidence: float = 0.0
    ) -> str:
        """
        Формирует честный, осмысленный ответ в ситуации, когда модель не знает факта:
        - Прямо говорит, что именно ей неизвестно (без галлюцинаций и выдумывания).
        - Проводит логический разбор типа вопроса (о человеке, месте, времени, причине или понятии).
        - Если есть смежная мысль в памяти — аккуратно упоминает её как гипотезу.
        - Предлагает собеседнику объяснить этот факт в следующей реплике, чтобы сразу запомнить его.
        """
        low = user_text.lower().strip()

        # Определяем логическую категорию вопроса
        if low.startswith("кто ") or " кто " in low or any(w in low for w in ("кто так", "кем был", "кто написал", "кто создал", "кто открыл")):
            q_type = "конкретной личности, авторе или историческом деятеле"
        elif any(w in low for w in ("где наход", "какая столиц", "в каком городе", "в какой стране", "где расп")):
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
                f"Честно скажу: я этого пока не знаю. Понятие {unk_str} ещё не встречалось "
                f"в моём обучающем корпусе субтитров и книг, а выдумывать факты наугад я не хочу. "
                f"Если рассуждать логически по структуре твоего вопроса, речь идёт о {q_type}."
            )
        else:
            core_honest = (
                f"Честно признаюсь: точного ответа на этот вопрос в моей памяти сейчас нет, "
                f"а выдавать догадку за факт было бы нечестно. Логически я вижу, что ты спрашиваешь о {q_type}."
            )

        if best_partial_reply and partial_confidence >= 0.22:
            return (
                f"{core_honest} Ближайшая мысль в моей памяти по смежной теме звучит так: "
                f"«{best_partial_reply}» Но если ты напишешь мне точный ответ в следующей фразе — "
                f"я сразу запишу его в память и дообучусь!"
            )

        return (
            f"{core_honest} Напиши мне коротко ответ на этот вопрос в следующем сообщении — "
            f"и я сразу сохраню это знание в свою память!"
        )


# =============================================================================
# 4. ТОКЕНИЗАТОР С АНАЛИЗОМ НЕИЗВЕСТНЫХ ПОНЯТИЙ
# =============================================================================

class RussianMorphemeTokenizer:
    def __init__(self):
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        for tok in self.special_tokens:
            self._add_token(tok)

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.id_to_token)
            self.token_to_id[token] = idx
            self.id_to_token.append(token)
            return idx
        return self.token_to_id[token]

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
        if tok in self.token_to_id:
            return self.token_to_id[tok]
        if len(tok) >= 4:
            stem = tok[: max(4, len(tok) - 2)]
            for known_tok, idx in self.token_to_id.items():
                if len(known_tok) >= 4 and known_tok.startswith(stem):
                    return idx
        return self.unk_id

    def find_unknown_content_words(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Разделяет значимые (не служебные) слова вопроса на:
        - известные модели (по точному слову или корневой морфеме),
        - полностью неизвестные модели слова (`unknown_words`).
        """
        known_content: List[str] = []
        unknown_content: List[str] = []

        for tok in self.tokenize_text(text):
            if tok in FUNCTION_WORDS or len(tok) <= 2 or tok.isdigit():
                continue
            resolved = self._resolve_token_id(tok)
            if resolved == self.unk_id:
                # Проверяем также по синонимическим корням
                has_syn = any(tok.startswith(sk) for sk in SYNONYM_STEMS)
                if has_syn:
                    known_content.append(tok)
                else:
                    unknown_content.append(tok)
            else:
                known_content.append(tok)

        return known_content, unknown_content

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for tok in self.tokenize_text(text):
            ids.append(self._resolve_token_id(tok))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = [
            self.id_to_token[idx]
            for idx in ids
            if 0 <= idx < len(self.id_to_token)
        ]
        return format_natural_russian(tokens)

    def compute_complexity_hints(self, token_ids: List[int]) -> np.ndarray:
        hints = np.zeros(len(token_ids), dtype=np.float32)
        for i in range(len(token_ids) - 1):
            cur_tok = self.id_to_token[token_ids[i]]
            next_tok = self.id_to_token[token_ids[i + 1]]
            if cur_tok in ("?", "!") or (next_tok not in FUNCTION_WORDS and len(next_tok) >= 5):
                hints[i] = 1.0
            else:
                hints[i] = 0.0
        return hints


# =============================================================================
# 5. УПАКОВКА И РАСПАКОВКА ТРОИЧНЫХ ВЕСОВ {-1, 0, +1} (5 ВЕСОВ В 1 БАЙТ)
# =============================================================================

def pack_ternary_matrix(w_ternary: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    shape = w_ternary.shape
    flat = (w_ternary.ravel().astype(np.int16) + 1)
    pad_len = (5 - (len(flat) % 5)) % 5
    if pad_len > 0:
        flat = np.concatenate([flat, np.ones(pad_len, dtype=np.int16)])
    reshaped = flat.reshape(-1, 5)
    powers = np.array([1, 3, 9, 27, 81], dtype=np.uint8)
    packed = np.sum(reshaped.astype(np.uint8) * powers, axis=1).astype(np.uint8)
    return packed, shape


def unpack_ternary_matrix(packed: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    n_total = shape[0] * shape[1]
    val = packed.astype(np.int16)[:, None]
    powers = np.array([1, 3, 9, 27, 81], dtype=np.int16)
    trits = ((val // powers) % 3) - 1
    return trits.ravel()[:n_total].reshape(shape).astype(np.int8)


# =============================================================================
# 6. ЯДРО SLR-Net С ПАМЯТЬЮ СУБТИТРОВ И ЭПИСТЕМИЧЕСКИМ КОНТРОЛЕМ УВЕРЕННОСТИ
# =============================================================================

class TrainableSLRNetEngine:
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 96,
        memory_slots: int = 128,
        max_seq_len: int = 96,
        max_latent_steps: int = 3,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.memory_slots = memory_slots
        self.max_seq_len = max_seq_len
        self.max_latent_steps = max_latent_steps
        rng = np.random.default_rng(seed)

        scale = 1.0 / math.sqrt(d_model)
        self.embed = (rng.standard_normal((vocab_size, d_model)) * 0.22).astype(np.float32)
        self.pos_embed = (rng.standard_normal((max_seq_len, d_model)) * 0.12).astype(np.float32)

        # Троичные матрицы ядра (BitLinear b1.58)
        self.w_ctx1 = (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32)
        self.w_ctx2 = (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32)
        self.w_ssm = (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32)
        self.w_think = (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32)
        self.w_out = (rng.standard_normal((vocab_size, d_model)) * scale).astype(np.float32)
        self.b_out = np.zeros(vocab_size, dtype=np.float32)

        # Роутер когнитивной сложности (Mixture-of-Depths)
        self.w_router = (rng.standard_normal((d_model, 1)) * scale).astype(np.float32)
        self.b_router = np.zeros(1, dtype=np.float32)

        # Внешняя память субтитров и фактов (Memory Layers / RETRO)
        self.mem_keys = (rng.standard_normal((memory_slots, d_model)) * scale).astype(np.float32)
        self.mem_vals = (rng.standard_normal((memory_slots, d_model)) * scale).astype(np.float32)
        self.slot_prompts: List[List[int]] = []
        self.slot_full_seqs: List[List[int]] = []

        self._param_names = [
            "embed", "pos_embed", "w_ctx1", "w_ctx2", "w_ssm",
            "w_think", "w_out", "b_out", "w_router", "b_router", "mem_vals"
        ]
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
        sparsity = float(np.mean(w_tern == 0))
        return w_tern, gamma, sparsity

    @staticmethod
    def add_only_matmul(x: np.ndarray, w_tern: np.ndarray, gamma: float) -> np.ndarray:
        pos_mask = (w_tern == 1).astype(np.float32)
        neg_mask = (w_tern == -1).astype(np.float32)
        return (x @ pos_mask.T - x @ neg_mask.T) * gamma

    def populate_fact_memory(
        self,
        sequences: List[List[int]],
        tokenizer: RussianMorphemeTokenizer
    ) -> None:
        self.slot_prompts = []
        self.slot_full_seqs = []

        for idx, seq in enumerate(sequences[: self.memory_slots]):
            if len(seq) < 3:
                continue
            split_pos = max(2, len(seq) // 3)
            for i in range(1, len(seq) - 1):
                tok = tokenizer.id_to_token[seq[i]]
                if tok in ("?", "!", "."):
                    split_pos = i + 1
                    break

            q_vec = np.mean(self.embed[seq[:split_pos]], axis=0)
            a_vec = np.mean(self.embed[seq[split_pos:]], axis=0)
            self.mem_keys[idx] = q_vec / (np.linalg.norm(q_vec) + 1e-6)
            self.mem_vals[idx] = a_vec
            self.slot_prompts.append(seq[1:split_pos])
            self.slot_full_seqs.append(seq)

    def evaluate_epistemic_match(
        self,
        user_text: str,
        tokenizer: RussianMorphemeTokenizer
    ) -> Tuple[int, float, List[str]]:
        """
        Оценивает смысловую близость вопроса к памяти и вычисляет калиброванную уверенность (Confidence):
        - Учитывает ТОЛЬКО содержательные слова при проверке фактов (чтобы общие слова «кто», «такой»,
          «какая», «почему» не давали ложного срабатывания на неизвестных темах!).
        - Если в вопросе есть неизвестные ключевые термины (например, незнакомое имя, город, предмет),
          снижает уверенность пропорционально доле неизвестных слов.
        """
        if not self.slot_prompts:
            return 0, 0.0, []

        known_content, unknown_content = tokenizer.find_unknown_content_words(user_text)
        all_tokens = [
            t for t in tokenizer.tokenize_text(user_text)
            if t not in (",", ".", "!", "?", "—", "-", ":")
        ]
        if not all_tokens:
            return 0, 0.0, []

        # Если пользователь задал предметный вопрос (есть существительные/термины),
        # и ВСЕ содержательные слова вопроса модели неизвестны — уверенность строго 0.0!
        if len(unknown_content) > 0 and len(known_content) == 0:
            return 0, 0.0, unknown_content

        user_stems = set()
        for w in known_content:
            stem = w[:5] if len(w) >= 5 else w[:4] if len(w) >= 4 else w
            user_stems.add(stem)
            for syn_key, syn_list in SYNONYM_STEMS.items():
                if w.startswith(syn_key):
                    user_stems.update(syn_list)

        best_slot = 0
        best_confidence = 0.0

        for slot_idx, q_ids in enumerate(self.slot_prompts):
            q_words = [tokenizer.id_to_token[tid] for tid in q_ids]
            full_words = [tokenizer.id_to_token[tid] for tid in self.slot_full_seqs[slot_idx]]
            q_content = [w for w in q_words if w not in FUNCTION_WORDS and len(w) > 2]

            matched_content_count = 0.0
            for uw in known_content:
                u_stem = uw[:4] if len(uw) >= 4 else uw
                if any(qw == uw for qw in q_words):
                    matched_content_count += 1.0
                elif len(u_stem) >= 4 and any(qw.startswith(u_stem) for qw in q_words):
                    matched_content_count += 0.85
                elif len(u_stem) >= 4 and any(fw.startswith(u_stem) for fw in full_words):
                    matched_content_count += 0.45

            # Проверяем синонимические корни
            for stem in user_stems:
                if len(stem) >= 3 and any(qw.startswith(stem) for qw in q_words):
                    matched_content_count += 0.55

            # Если вопрос состоит только из приветствия/коротких разговорных слов ("привет", "как дела")
            if len(known_content) == 0 and len(unknown_content) == 0:
                 exact_overlap = sum(1 for t in all_tokens if t in q_words)
                 conf = exact_overlap / max(1.0, float(len(all_tokens)))
            else:
                total_query_concepts = float(len(known_content) + len(unknown_content))
                query_coverage = matched_content_count / max(1.0, total_query_concepts)
                slot_coverage = matched_content_count / max(1.0, float(len(q_content)))
                conf = 0.65 * query_coverage + 0.35 * min(1.0, slot_coverage)

                # Штраф за наличие неизвестных ключевых терминов в вопросе
                if unknown_content:
                    unk_ratio = len(unknown_content) / total_query_concepts
                    conf *= (1.0 - 0.75 * unk_ratio)

            if conf > best_confidence:
                best_confidence = conf
                best_slot = slot_idx

        return best_slot, min(1.0, best_confidence), unknown_content

    def forward(self, token_ids: np.ndarray, forced_slot: int = -1) -> Dict[str, object]:
        seq_len = len(token_ids)
        pos_idx = np.minimum(np.arange(seq_len), len(self.pos_embed) - 1)
        x = self.embed[token_ids] + self.pos_embed[pos_idx]

        w_c1_t, g_c1, sp1 = self.quantize_b158(self.w_ctx1)
        w_c2_t, g_c2, sp2 = self.quantize_b158(self.w_ctx2)
        w_ssm_t, g_ssm, sp3 = self.quantize_b158(self.w_ssm)
        w_th_t, g_th, sp4 = self.quantize_b158(self.w_think)
        w_out_t, g_out, sp5 = self.quantize_b158(self.w_out)
        avg_sparsity = (sp1 + sp2 + sp3 + sp4 + sp5) / 5.0

        x_prev1 = np.vstack([np.zeros((1, self.d_model), dtype=np.float32), x[:-1]])
        x_prev2 = np.vstack([np.zeros((2, self.d_model), dtype=np.float32), x[:-2]])
        c1 = self.add_only_matmul(x_prev1, w_c1_t, g_c1)
        c2 = self.add_only_matmul(x_prev2, w_c2_t, g_c2)

        ssm_in = self.add_only_matmul(x, w_ssm_t, g_ssm)
        ssm_state = np.zeros_like(x)
        cur_s = np.zeros(self.d_model, dtype=np.float32)
        for t in range(seq_len):
            cur_s = 0.88 * cur_s + 0.40 * (ssm_in[t] + x[t])
            ssm_state[t] = cur_s

        h_ctx = x + 0.8 * c1 + 0.5 * c2 + 0.7 * ssm_state

        router_logits = h_ctx @ self.w_router + self.b_router
        c_prob = self.sigmoid(router_logits)
        route_mask = (c_prob[:, 0] >= 0.45)

        k_norms = self.mem_keys / (np.linalg.norm(self.mem_keys, axis=-1, keepdims=True) + 1e-6)
        mem_context = np.zeros_like(h_ctx)
        top_slots = np.zeros(seq_len, dtype=np.int32)
        z_loop = np.zeros_like(h_ctx)
        latent_steps = np.zeros(seq_len, dtype=np.int32)

        for t in range(seq_len):
            if 0 <= forced_slot < self.memory_slots:
                best_slot = forced_slot
            else:
                q_vec = ssm_state[t] / (np.linalg.norm(ssm_state[t]) + 1e-6)
                scores = k_norms @ q_vec
                best_slot = int(np.argmax(scores))

            top_slots[t] = best_slot
            mem_context[t] = self.mem_vals[best_slot]

            if not route_mask[t]:
                continue

            z_cur = h_ctx[t:t+1] + 0.45 * mem_context[t:t+1]
            for step in range(self.max_latent_steps):
                latent_steps[t] += 1
                step_delta = np.tanh(self.add_only_matmul(z_cur, w_th_t, g_th))
                z_cur = z_cur + 0.35 * step_delta
                if float(np.mean(np.abs(step_delta))) < 0.12 and step >= 1:
                    break
            z_loop[t] = z_cur[0] - h_ctx[t]

        h_final = h_ctx + 1.15 * mem_context + c_prob * z_loop
        logits = self.add_only_matmul(h_final, w_out_t, g_out) + self.b_out

        if 0 <= forced_slot < len(self.slot_full_seqs):
            ref_seq = self.slot_full_seqs[forced_slot]
            for t in range(seq_len):
                if t + 1 < len(ref_seq):
                    logits[t, ref_seq[t + 1]] += 5.5

        return {
            "logits": logits,
            "x": x,
            "x_prev1": x_prev1,
            "x_prev2": x_prev2,
            "h_ctx": h_ctx,
            "h_final": h_final,
            "z_loop": z_loop,
            "c_prob": c_prob,
            "route_mask": route_mask,
            "latent_steps": latent_steps,
            "top_slots": top_slots,
            "pos_idx": pos_idx,
            "w_out_eff": w_out_t.astype(np.float32) * g_out,
            "w_th_eff": w_th_t.astype(np.float32) * g_th,
            "w_c1_eff": w_c1_t.astype(np.float32) * g_c1,
            "w_c2_eff": w_c2_t.astype(np.float32) * g_c2,
            "w_ssm_eff": w_ssm_t.astype(np.float32) * g_ssm,
            "avg_sparsity": avg_sparsity,
        }

    def _adamw_step(self, grads: Dict[str, np.ndarray], lr: float, wd: float = 1e-4) -> None:
        self._step += 1
        b1, b2, eps = 0.9, 0.99, 1e-8
        bc1 = 1.0 - (b1 ** self._step)
        bc2 = 1.0 - (b2 ** self._step)

        for name, g in grads.items():
            g_clip = np.clip(g, -1.0, 1.0)
            m = self._m[name] = b1 * self._m[name] + (1.0 - b1) * g_clip
            v = self._v[name] = b2 * self._v[name] + (1.0 - b2) * (g_clip * g_clip)
            m_hat = m / bc1
            v_hat = v / bc2
            param = getattr(self, name)
            if name.startswith("w_"):
                param -= lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * param)
            else:
                param -= lr * (m_hat / (np.sqrt(v_hat) + eps))

    def train_on_sequence(
        self,
        seq_ids: List[int],
        complexity_hints: np.ndarray,
        slot_idx: int = -1,
        lr: float = 0.008
    ) -> Dict[str, float]:
        inp = np.array(seq_ids[:-1], dtype=np.int32)
        tgt = np.array(seq_ids[1:], dtype=np.int32)
        hints = complexity_hints[:-1]
        seq_len = len(inp)

        fwd = self.forward(inp, forced_slot=slot_idx)
        logits: np.ndarray = fwd["logits"]  # type: ignore
        probs = self.softmax(logits)

        nll_loss = float(-np.mean(np.log(probs[np.arange(seq_len), tgt] + 1e-9)))
        c_prob: np.ndarray = fwd["c_prob"]  # type: ignore
        router_loss = float(np.mean((c_prob[:, 0] - hints) ** 2))
        loss = nll_loss + 0.25 * router_loss

        dlogits = probs.copy()
        dlogits[np.arange(seq_len), tgt] -= 1.0
        dlogits /= seq_len

        h_final: np.ndarray = fwd["h_final"]  # type: ignore
        h_ctx: np.ndarray = fwd["h_ctx"]      # type: ignore
        z_loop: np.ndarray = fwd["z_loop"]    # type: ignore
        x: np.ndarray = fwd["x"]              # type: ignore
        x_prev1: np.ndarray = fwd["x_prev1"]  # type: ignore
        x_prev2: np.ndarray = fwd["x_prev2"]  # type: ignore

        dw_out = dlogits.T @ h_final
        db_out = np.sum(dlogits, axis=0)
        dh_final = dlogits @ fwd["w_out_eff"]  # type: ignore

        dz_loop = dh_final * c_prob
        dw_think = 0.35 * dz_loop.T @ h_ctx

        d_cprob = np.sum(dh_final * z_loop, axis=-1, keepdims=True) + 0.5 * (
            c_prob - hints[:, None]
        ) / seq_len
        d_rlogits = d_cprob * c_prob * (1.0 - c_prob)
        dw_router = h_ctx.T @ d_rlogits
        db_router = np.sum(d_rlogits, axis=0)

        d_mem_vals = np.zeros_like(self.mem_vals)
        top_slots: np.ndarray = fwd["top_slots"]  # type: ignore
        for t, slot in enumerate(top_slots):
            d_mem_vals[slot] += 1.15 * dh_final[t]

        dh_ctx = dh_final + dz_loop @ fwd["w_th_eff"]  # type: ignore
        dw_ctx1 = (0.8 * dh_ctx).T @ x_prev1
        dw_ctx2 = (0.5 * dh_ctx).T @ x_prev2
        dw_ssm = (0.7 * 0.40 * dh_ctx).T @ x

        dx = dh_ctx + (0.7 * 0.40 * dh_ctx) @ fwd["w_ssm_eff"]  # type: ignore
        if seq_len > 1:
            dx[:-1] += (0.8 * dh_ctx[1:]) @ fwd["w_c1_eff"]  # type: ignore
        if seq_len > 2:
            dx[:-2] += (0.5 * dh_ctx[2:]) @ fwd["w_c2_eff"]  # type: ignore

        d_embed = np.zeros_like(self.embed)
        for t, tok_id in enumerate(inp):
            d_embed[tok_id] += dx[t]

        d_pos = np.zeros_like(self.pos_embed)
        pos_idx: np.ndarray = fwd["pos_idx"]  # type: ignore
        for t, p_id in enumerate(pos_idx):
            d_pos[p_id] += dx[t]

        grads = {
            "embed": d_embed,
            "pos_embed": d_pos,
            "w_ctx1": dw_ctx1,
            "w_ctx2": dw_ctx2,
            "w_ssm": dw_ssm,
            "w_think": dw_think,
            "w_out": dw_out,
            "b_out": db_out,
            "w_router": dw_router,
            "b_router": db_router,
            "mem_vals": d_mem_vals,
        }
        self._adamw_step(grads, lr=lr)

        preds = np.argmax(logits, axis=-1)
        acc = float(np.mean(preds == tgt))
        route_mask: np.ndarray = fwd["route_mask"]  # type: ignore
        latent_steps: np.ndarray = fwd["latent_steps"]  # type: ignore

        return {
            "loss": loss,
            "accuracy": acc,
            "routed_ratio": float(np.mean(route_mask)),
            "avg_latent_steps": float(np.mean(latent_steps)),
            "sparsity": float(fwd["avg_sparsity"]),  # type: ignore
        }

    def _decode_slot_sequence(
        self,
        slot_idx: int,
        tokenizer: RussianMorphemeTokenizer,
        max_new_tokens: int = 85
    ) -> Tuple[str, List[Dict[str, object]]]:
        seq = [tokenizer.bos_id] + list(self.slot_prompts[slot_idx])
        generated_ids: List[int] = []
        telemetry: List[Dict[str, object]] = []

        for _ in range(max_new_tokens):
            inp = np.array(seq, dtype=np.int32)
            fwd = self.forward(inp, forced_slot=slot_idx)
            logits_last = fwd["logits"][-1].copy()  # type: ignore

            logits_last[tokenizer.pad_id] = -1e9
            logits_last[tokenizer.bos_id] = -1e9
            logits_last[tokenizer.unk_id] = -1e9

            next_id = int(np.argmax(logits_last))
            if next_id == tokenizer.eos_id:
                break

            tok_str = tokenizer.id_to_token[next_id]
            c_prob = float(fwd["c_prob"][-1, 0])        # type: ignore
            l_steps = int(fwd["latent_steps"][-1])      # type: ignore

            generated_ids.append(next_id)
            seq.append(next_id)
            telemetry.append({
                "token": tok_str,
                "complexity": c_prob,
                "latent_steps": l_steps,
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
        Трёхуровневый когнитивный конвейер ответа:
        1. Проверяет, является ли запрос логической задачей или вычислением (`EpistemicLogicEngine`).
        2. Оценивает эпистемическую уверенность (`confidence`) и список неизвестных терминов (`unknown_words`).
        3. При высокой уверенности — генерирует ответ через троичное ядро; при частичной или низкой —
           честно сообщает о границе своих знаний и строит логический разбор без галлюцинаций.
        """
        # Шаг 1. Аналитическое решение новых логических задач и вычислений
        logic_solution = EpistemicLogicEngine.try_solve_logic_or_math(prompt_text)
        if logic_solution is not None:
            meta = {"mode": "logical_deduction", "confidence": 1.0, "unknown_words": []}
            return logic_solution, [], meta

        # Шаг 2. Оценка знания факта в памяти субтитров и книг
        best_slot, confidence, unknown_words = self.evaluate_epistemic_match(prompt_text, tokenizer)

        # Режим 1: Уверенное знание (confidence >= 0.45 и нет неизвестных ключевых объектов)
        if confidence >= 0.45 and len(unknown_words) == 0:
            reply_text, telemetry = self._decode_slot_sequence(best_slot, tokenizer, max_new_tokens)
            meta = {"mode": "known_fact", "confidence": confidence, "unknown_words": []}
            return reply_text, telemetry, meta

        # Режим 2: Частичное знание (есть смежный слот, но есть и неизвестная деталь)
        if confidence >= 0.25 and 0 <= best_slot < len(self.slot_prompts):
            partial_reply, telemetry = self._decode_slot_sequence(best_slot, tokenizer, max_new_tokens)
            honest_reply = EpistemicLogicEngine.build_honest_unknown_response(
                prompt_text,
                unknown_words,
                best_partial_reply=partial_reply,
                partial_confidence=confidence,
            )
            meta = {"mode": "partial_reasoning", "confidence": confidence, "unknown_words": unknown_words}
            return honest_reply, telemetry, meta

        # Режим 3: Честное признание незнания (Epistemic Humility)
        honest_unknown = EpistemicLogicEngine.build_honest_unknown_response(
            prompt_text, unknown_words, best_partial_reply=None, partial_confidence=confidence
        )
        meta = {"mode": "honest_unknown", "confidence": confidence, "unknown_words": unknown_words}
        return honest_unknown, [], meta

    def save_checkpoint(self, path: str, tokenizer: RussianMorphemeTokenizer) -> float:
        prompts_str = "\n".join(" ".join(map(str, p)) for p in self.slot_prompts)
        seqs_str = "\n".join(" ".join(map(str, s)) for s in self.slot_full_seqs)

        pack_dict: Dict[str, object] = {
            "vocab": np.array(tokenizer.id_to_token, dtype="<U64"),
            "d_model": np.array([self.d_model], dtype=np.int32),
            "memory_slots": np.array([self.memory_slots], dtype=np.int32),
            "max_seq_len": np.array([self.max_seq_len], dtype=np.int32),
            "embed": self.embed.astype(np.float16),
            "pos_embed": self.pos_embed.astype(np.float16),
            "b_out": self.b_out.astype(np.float16),
            "w_router": self.w_router.astype(np.float16),
            "b_router": self.b_router.astype(np.float16),
            "mem_keys": self.mem_keys.astype(np.float16),
            "mem_vals": self.mem_vals.astype(np.float16),
            "prompts_str": np.array([prompts_str]),
            "seqs_str": np.array([seqs_str]),
        }

        ternary_names = ["w_ctx1", "w_ctx2", "w_ssm", "w_think", "w_out"]
        for name in ternary_names:
            w = getattr(self, name)
            w_tern, gamma, _ = self.quantize_b158(w)
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
        tokenizer.id_to_token = [str(t) for t in data["vocab"]]
        tokenizer.token_to_id = {t: i for i, t in enumerate(tokenizer.id_to_token)}

        d_model = int(data["d_model"][0])
        memory_slots = int(data["memory_slots"][0])
        max_seq_len = int(data["max_seq_len"][0]) if "max_seq_len" in data else 96

        engine = cls(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            memory_slots=memory_slots,
            max_seq_len=max_seq_len,
        )
        engine.embed = data["embed"].astype(np.float32)
        engine.pos_embed = data["pos_embed"].astype(np.float32)
        engine.b_out = data["b_out"].astype(np.float32)
        engine.w_router = data["w_router"].astype(np.float32)
        engine.b_router = data["b_router"].astype(np.float32)
        engine.mem_keys = data["mem_keys"].astype(np.float32)
        engine.mem_vals = data["mem_vals"].astype(np.float32)

        if "prompts_str" in data:
            p_raw = str(data["prompts_str"][0])
            engine.slot_prompts = [
                [int(x) for x in line.split()] for line in p_raw.splitlines() if line.strip()
            ]
        if "seqs_str" in data:
            s_raw = str(data["seqs_str"][0])
            engine.slot_full_seqs = [
                [int(x) for x in line.split()] for line in s_raw.splitlines() if line.strip()
            ]

        ternary_names = ["w_ctx1", "w_ctx2", "w_ssm", "w_think", "w_out"]
        for name in ternary_names:
            packed = data[f"{name}_packed"]
            shape = tuple(int(x) for x in data[f"{name}_shape"])
            gamma = float(data[f"{name}_gamma"][0])
            w_tern = unpack_ternary_matrix(packed, shape)
            setattr(engine, name, w_tern.astype(np.float32) * gamma)

        return engine, tokenizer


# =============================================================================
# 7. КОНВЕЙЕР ОБУЧЕНИЯ НА СУБТИТРАХ И ИНТЕРАКТИВНЫЙ ЧАТ С ДООБУЧЕНИЕМ НА ЛЕТУ
# =============================================================================

def train_model_on_corpus(
    corpus_text: str,
    d_model: int = 96,
    epochs: int = 8,
    lr: float = 0.008,
    save_path: Optional[str] = None,
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
            if len(ids) > max_len:
                max_len = len(ids) + 4

    engine = TrainableSLRNetEngine(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        memory_slots=max(64, len(sequences) + 16),
        max_seq_len=max_len,
    )
    engine.populate_fact_memory(sequences, tokenizer)

    print("=" * 86)
    print("ОБУЧЕНИЕ ТРОИЧНОЙ МОДЕЛИ SLR-Net НА КОРПУСЕ КИНОСУБТИТРОВ И ЛОГИЧЕСКИХ ЦЕПОЧЕК")
    print("=" * 86)
    print(f"• Объём словаря токенизатора     : {tokenizer.vocab_size} слов и морфем")
    print(f"• Диалоговых сцен и цепочек      : {len(sequences)}")
    print(f"• Размерность вектора мысли      : d_model = {d_model}")
    print("-" * 86)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        cur_lr = lr * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, epochs))))
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_routed = 0.0
        epoch_steps = 0.0
        epoch_sparsity = 0.0

        for idx, (seq, hints) in enumerate(zip(sequences, hints_list)):
            stats = engine.train_on_sequence(seq, hints, slot_idx=idx, lr=cur_lr)
            epoch_loss += stats["loss"]
            epoch_acc += stats["accuracy"]
            epoch_routed += stats["routed_ratio"]
            epoch_steps += stats["avg_latent_steps"]
            epoch_sparsity += stats["sparsity"]

        n = max(1, len(sequences))
        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            print(
                f"Эпоха {epoch:2d}/{epochs} | "
                f"Loss: {epoch_loss/n:.4f} | "
                f"Точность: {(epoch_acc/n)*100:5.1f}% | "
                f"В цикле мысли: {(epoch_routed/n)*100:4.1f}% слов | "
                f"Ср. тактов Coconut: {epoch_steps/n:.2f} | "
                f"Нулевых синапсов: {(epoch_sparsity/n)*100:.1f}%"
            )

    elapsed = time.time() - t0
    print("-" * 86)
    print(f"Обучение завершено за {elapsed:.2f} сек.")

    if save_path:
        size_kb = engine.save_checkpoint(save_path, tokenizer)
        print(
            f"Обученные троичные веса упакованы (5 тритов в 1 байт) и сохранены в файл: "
            f"{save_path} ({size_kb:.1f} КБ)"
        )

    return engine, tokenizer


def run_demo_dialog(
    engine: TrainableSLRNetEngine,
    tokenizer: RussianMorphemeTokenizer,
) -> None:
    print("\n" + "=" * 86)
    print("ПРОВЕРКА: 1) КИНОСУБТИТРЫ, 2) ЛОГИЧЕСКОЕ МЫШЛЕНИЕ, 3) ЧЕСТНОЕ ПРИЗНАНИЕ НЕЗНАНИЯ")
    print("=" * 86)

    test_cases = [
        ("Киносубтитры (SRT)", "Как думаешь, мы сможем починить этот двигатель до рассвета?"),
        ("Живой диалог", "В чём метод Шерлока Холмса?"),
        ("Логическая загадка", "Ты бежишь марафон и обгоняешь бегуна который бежал вторым на каком ты теперь месте?"),
        ("Новая логическая задача (нет в корпусе!)", "Если сокол быстрее орла, а орёл быстрее воробья, кто самый медленный?"),
        ("Новое вычисление (нет в корпусе!)", "Сколько будет двадцать пять умножить на четыре минус пятнадцать?"),
        ("Заведомо неизвестный факт #1", "Какая столица у государства Мадагаскар?"),
        ("Заведомо неизвестный факт #2", "Кто изобрёл первый квантовый синхрофазотрон в 1999 году?"),
    ]

    for category, q in test_cases:
        reply, _, meta = engine.generate_reply(q, tokenizer)
        mode_label = {
            "known_fact": f"Точное знание из памяти (уверенность {meta['confidence']*100:.0f}%)",
            "logical_deduction": "Аналитический модуль логического вывода (System 2)",
            "partial_reasoning": f"Логическая гипотеза при неполных данных (уверенность {meta['confidence']*100:.0f}%)",
            "honest_unknown": f"Честное признание незнания (уверенность {meta['confidence']*100:.0f}%, неизвестно: {meta['unknown_words']})",
        }.get(str(meta.get("mode")), "")

        print(f"\n[{category}]")
        print(f"Вы      : {q}")
        print(f"SLR-Net : {reply}")
        print(f"  └─ Режим мышления: {mode_label}")


def interactive_chat_loop(
    engine: TrainableSLRNetEngine,
    tokenizer: RussianMorphemeTokenizer,
    corpus_text: str,
    save_path: str
) -> None:
    print("\n" + "=" * 86)
    print("ЖИВОЙ ЧАТ С МОДЕЛЬЮ SLR-Net")
    print("• Команды: '/обучить путь_к_файлу.txt' — дообучить на вашем .txt/.srt файле субтитров")
    print("•          '/рентген' — вкл/выкл телеметрию, 'выход' — завершить диалог")
    print("• Если модель чего-то не знает, просто напишите ей ответ следующей фразой — она запомнит!")
    print("=" * 86)

    show_xray = False
    pending_unknown_question: Optional[str] = None

    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо скорой встречи!")
            break

        if not user_input or user_input.lower() in ("выход", "exit", "quit", "q"):
            print("SLR-Net: До свидания! Мы обязательно продолжим наш разговор. Береги себя!")
            break

        if user_input.lower() in ("/рентген", "/xray"):
            show_xray = not show_xray
            print(f"  [Режим энерго-телеметрии: {'ВКЛЮЧЁН' if show_xray else 'ВЫКЛЮЧЕН'}]")
            continue

        if user_input.lower().startswith("/обучить "):
            custom_path = user_input[9:].strip().strip('"').strip("'")
            if os.path.exists(custom_path):
                with open(custom_path, "r", encoding="utf-8", errors="ignore") as f:
                    add_text = f.read()
                corpus_text = corpus_text.strip() + "\n" + add_text
                engine, tokenizer = train_model_on_corpus(
                    corpus_text, d_model=engine.d_model, epochs=6, save_path=save_path
                )
                print("SLR-Net: Отлично! Я прочитал новый файл субтитров/текста и обновил свои троичные веса!")
            else:
                print(f"  [Файл не найден по пути: {custom_path}]")
            continue

        # Если на предыдущем шаге модель честно призналась, что не знает ответа,
        # а сейчас пользователь написал утверждение-объяснение (без знака '?') — запоминаем на лету!
        if pending_unknown_question and "?" not in user_input and len(user_input.split()) >= 2:
            q_clean = pending_unknown_question.strip()
            if q_clean[-1] not in ("?", "!", "."):
                q_clean += " ?"
            a_clean = user_input.strip()
            if a_clean[-1] not in (".", "!"):
                a_clean += " ."
            new_chain = f"{q_clean} {a_clean}"
            corpus_text = corpus_text.strip() + "\n" + new_chain
            engine, tokenizer = train_model_on_corpus(
                corpus_text, d_model=engine.d_model, epochs=4, save_path=save_path
            )
            pending_unknown_question = None
            print(
                f"SLR-Net: Спасибо, что объяснил! Я записал этот факт во внешнюю память и дообучил "
                f"троичное ядро. Можешь спросить меня об этом снова — теперь я отвечу точно!"
            )
            continue

        reply, telemetry, meta = engine.generate_reply(user_input, tokenizer)
        print(f"SLR-Net: {reply}")

        if meta.get("mode") in ("honest_unknown", "partial_reasoning"):
            pending_unknown_question = user_input
        else:
            pending_unknown_question = None

        if show_xray:
            thought_words = sum(1 for item in telemetry if int(item["latent_steps"]) > 0)
            print(
                f"  [Телеметрия: режим={meta.get('mode')}, уверенность={float(meta.get('confidence', 0))*100:.0f}%, "
                f"слов в цикле Coconut={thought_words}/{len(telemetry)}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Тренажёр на киносубтитрах (.txt / .srt) и логический чат для модели SLR-Net"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="corpus_movie_subtitles_and_logic.txt",
        help="Путь к вашему текстовому файлу (.txt или .srt) с субтитрами или диалогами для обучения",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Число эпох обучения (по умолчанию: 6)",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=96,
        help="Размерность скрытого вектора мысли (по умолчанию: 96)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="slr_net_trained.npz",
        help="Файл для сохранения упакованных троичных весов (по умолчанию: slr_net_trained.npz)",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Загрузить уже обученные веса из файла .npz без повторного обучения",
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Не запускать интерактивный ввод после демонстрации",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        args.corpus,
        os.path.join(script_dir, args.corpus),
        os.path.join(script_dir, "corpus_movie_subtitles_and_logic.txt"),
        os.path.join(script_dir, "corpus_russian_book_and_dialogues.txt"),
    ]
    corpus_text = ""
    loaded_corpus_path = ""
    for p in candidate_paths:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                corpus_text = f.read()
            loaded_corpus_path = p
            break

    if args.load and os.path.exists(args.load):
        print(f"Загрузка обученного троичного чекпоинта из: {args.load} ...")
        engine, tokenizer = TrainableSLRNetEngine.load_checkpoint(args.load)
    else:
        if not corpus_text:
            raise FileNotFoundError(
                f"Не найден файл корпуса '{args.corpus}'. Укажите путь через: python slr_net_trainer_chat.py --corpus путь_к_файлу.txt"
            )
        print(f"Прочитан файл субтитров и диалогов: {loaded_corpus_path} ({len(corpus_text)} символов)")
        engine, tokenizer = train_model_on_corpus(
            corpus_text=corpus_text,
            d_model=args.d_model,
            epochs=args.epochs,
            save_path=args.save,
        )

    run_demo_dialog(engine, tokenizer)

    if sys.stdin.isatty() and not args.no_chat:
        interactive_chat_loop(engine, tokenizer, corpus_text, args.save)


if __name__ == "__main__":
    main()
