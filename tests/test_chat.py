# -*- coding: utf-8 -*-
"""Проверки поискового чат-прототипа: логика, токенизатор, точность градиентов, память фактов."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import slr_net_trainer_chat as T  # noqa: E402

solve = T.EpistemicLogicEngine.try_solve_logic_or_math


@pytest.mark.parametrize("question,expected", [
    ("Сколько будет двадцать пять умножить на четыре минус пятнадцать?", "85"),
    ("Сколько будет пять тысяч плюс триста?", "5300"),
    ("сколько будет 10-3?", "7"),
    ("2+2*2", "6"),
    ("(2+3)*4 = ?", "20"),
])
def test_arithmetic(question, expected):
    ans = solve(question)
    assert ans is not None and ans.rstrip(".").endswith(expected)


@pytest.mark.parametrize("text", ["Что было в 1941-1945 годах?", "Сколько лет длилась война 1941-1945?",
                                  "Встреча в 10:30?", "позвони 8-800-555-35-35", "Привет, как дела?"])
def test_not_arithmetic(text):
    assert solve(text) is None


def test_division_by_zero_is_explained():
    assert "деление на ноль" in solve("сколько будет 7/0")


@pytest.mark.parametrize("question,answer", [
    ("Если сокол быстрее орла, а орёл быстрее воробья, кто самый медленный?", "Воробья"),
    ("Сокол быстрее орла. Орёл быстрее воробья. Кто самый быстрый?", "Сокол"),
    ("Если кот тяжелее мыши, а слон тяжелее кота, кто самый лёгкий?", "Мыши"),
    ("Если Петя старше чем Вася, а Вася старше чем Коля, кто самый младший?", "Коля"),
    ("Если Петя младше Васи, а Вася младше Коли, кто самый старший?", "Коли"),
])
def test_transitive_reasoning(question, answer):
    ans = solve(question)
    assert ans is not None and ans.rstrip(".").endswith(answer), ans


def test_transitive_reasoning_admits_missing_information():
    ans = solve("Если Петя старше Васи, а Коля старше Васи, кто самый старший?")
    assert "нельзя однозначно" in ans


def test_tokenizer_does_not_confuse_unrelated_words():
    tok = T.RussianMorphemeTokenizer()
    tok.fit(["мне грустно поговори со мной", "первом месте книга"])
    assert tok._resolve_token_id("погоде") == tok.unk_id          # раньше -> «поговори»
    assert tok._resolve_token_id("первый") == tok.token_to_id["первом"]
    assert tok._resolve_token_id("книгу") == tok.token_to_id["книга"]


def _small_engine(dtype=np.float64):
    chains = ["привет ! как дела ? Отлично , спасибо .", "почему небо синее ? Атмосфера рассеивает синий свет ."]
    tok = T.RussianMorphemeTokenizer()
    tok.fit(chains)
    eng = T.TrainableSLRNetEngine(vocab_size=tok.vocab_size, d_model=10, memory_slots=8, max_seq_len=24, seed=3)
    seqs = [tok.encode(c) for c in chains]
    eng.populate_fact_memory(seqs, tok)
    for name in eng._param_names + ["mem_keys"]:
        setattr(eng, name, getattr(eng, name).astype(dtype))
    return eng, tok, seqs


def test_chat_network_gradients_are_exact():
    eng, tok, seqs = _small_engine()
    eng.quantize = False
    eng.b_router[:] = 2.0                       # все токены в цикле мысли -> проверяем и его градиент
    seq = seqs[1]
    hints = tok.compute_complexity_hints(seq)
    _, grads, _ = eng.compute_loss_and_grads(seq, hints, slot_idx=1)
    rng = np.random.default_rng(0)
    eps, worst = 1e-6, 0.0
    for name in eng._param_names:
        p = getattr(eng, name)
        for _ in range(5):
            idx = tuple(int(rng.integers(0, s)) for s in p.shape)
            if name in ("embed", "pos_embed"):
                idx = (int(rng.choice(seq[:-1])) if name == "embed" else int(rng.integers(0, len(seq) - 1)),) + idx[1:]
            if name == "mem_vals":
                idx = (1,) + idx[1:]
            old = p[idx]
            p[idx] = old + eps
            lp = eng.compute_loss_and_grads(seq, hints, slot_idx=1)[0]
            p[idx] = old - eps
            lm = eng.compute_loss_and_grads(seq, hints, slot_idx=1)[0]
            p[idx] = old
            num, ana = (lp - lm) / (2 * eps), grads[name][idx]
            if abs(num) + abs(ana) > 1e-9:
                worst = max(worst, abs(num - ana) / (abs(num) + abs(ana)))
    assert worst < 1e-4, worst


def test_training_has_no_answer_leak_and_learns():
    text = open(os.path.join(ROOT, "corpus_movie_subtitles_and_logic.txt"), encoding="utf-8").read()
    eng, tok = T.train_model_on_corpus(text, epochs=3, verbose=False)
    assert eng.network_accuracy(use_copy=False) > 0.5


def test_add_fact_is_instant_and_keeps_old_knowledge(tmp_path):
    eng, tok, _ = _small_engine(np.float32)
    q = "Какая столица Мадагаскара?"
    assert eng.generate_reply(q, tok)[2]["mode"] == "honest_unknown"
    eng.add_fact("какая столица мадагаскара ? Столица Мадагаскара — Антананариву .", tok)
    reply, _, meta = eng.generate_reply(q, tok)
    assert meta["mode"] == "known_fact" and "антананариву" in reply.lower()
    assert "отлично" in eng.generate_reply("привет ! как дела ?", tok)[0].lower()
    path = str(tmp_path / "ckpt.npz")
    eng.save_checkpoint(path, tok)
    eng2, tok2 = T.TrainableSLRNetEngine.load_checkpoint(path)
    assert tok2.id_to_token == tok.id_to_token
    assert "антананариву" in eng2.generate_reply(q, tok2)[0].lower()


def test_bundled_checkpoint_loads():
    eng, tok = T.TrainableSLRNetEngine.load_checkpoint(os.path.join(ROOT, "slr_net_trained.npz"))
    reply, _, meta = eng.generate_reply("В чём метод Шерлока Холмса?", tok)
    assert meta["mode"] == "known_fact" and "дедукции" in reply.lower()


# =============================================================================
# Навыки «несколько вопросов в одном сообщении» и «контекст» (chat_corpus.py, раздел 3б;
# TRAINING_MINI2_CONTEXT_RU.md): формат, проверки, разделение обучения и проверки, нет утечек.
# =============================================================================

import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402

import chat_corpus as C  # noqa: E402

FILLER = [[("user", "Как дела?"), ("bot", "Отлично!")], [("user", "Что делаешь?"), ("bot", "Разговариваю с тобой.")]]


def _check_reply():
    pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    import train_chat as TC
    return TC.check_reply


def _context_rows(train: bool, n: int = 150, seed: int = 0):
    g = C.ContextGen(random.Random(seed), FILLER)
    for name, n_train, n_test in C.CONTEXT_PLAN:
        if train and n_train == 0:
            continue
        for _ in range(n):
            yield name, getattr(g, name)(train)


def test_context_skills_format_and_reference_passes_check():
    """Каждый диалог: реплики по очереди, первая — пользователя, последняя — бота; эталонный ответ проходит
    свою проверку (иначе проверка или данные неверны)."""
    check = _check_reply()
    cats = set()
    for train in (True, False):
        for name, r in _context_rows(train):
            roles = [t[0] for t in r["turns"]]
            assert roles[0] == "user" and roles[-1] == "bot", r
            assert all(a != b for a, b in zip(roles, roles[1:])), r
            assert all(t[1].strip() for t in r["turns"]), r
            assert r["src"].startswith(C.CONTEXT_SRC) and r.get("check"), r
            assert check(r["turns"][-1][1], r["check"]), r
            cats.add(r["src"])
            if r["src"].startswith("навык:несколько вопросов: два") or r["src"].endswith(": три"):
                assert r["check"].get("order") and len(r["check"]["all"]) == (2 if "два" in r["src"] else 3), r
            if r["src"].endswith("один вопрос"):
                assert r["check"].get("max_words"), r
    for c in ("навык:несколько вопросов: два", "навык:несколько вопросов: три", "навык:несколько вопросов: один вопрос",
              "навык:несколько вопросов: с фактом из разговора", "навык:контекст: произведения",
              "навык:контекст: бот назвал", "навык:контекст: еда, питомцы, места", "навык:контекст: цепочка",
              "навык:контекст: а ты?", "навык:контекст: смена темы", "навык:контекст: без темы",
              "навык:контекст: длинные названия"):
        assert c in cats, c


def test_composite_check_needs_all_parts_in_order_and_no_echo():
    check = _check_reply()
    mq = C.chk_all(C.chk("slr-net"), C.chk("7", "семь"), C.chk("Париж"), order=True)
    assert check("Меня зовут SLR-Net. 3 + 4 = 7. Столица Франции — Париж.", mq)
    assert not check("Меня зовут SLR-Net.", mq)                              # ответил только на первый вопрос
    assert not check("Столица Франции — Париж. Меня зовут SLR-Net. 3 + 4 = 7.", mq)   # не по порядку
    assert not check("Меня зовут SLR-Net. 3 + 4 = 8. Столица Франции — Париж.", mq)   # одна часть неверна
    # пример владельца: «А что в нём интересного?» — нужно название и факт из разговора, эхо вопроса — провал
    ctx = C.chk_all(C.chk("Синий ветер"), C.chk("мальчик", "брат"), none=["что в нем интересного"])
    assert check("В фильме «Синий ветер» интересно то, что это история про мальчика, который ищет брата.", ctx)
    assert not check("В чём интересного?", ctx)
    assert not check("Фильм «Синий ветер» хороший.", ctx)                   # нет ни одного факта из разговора
    assert not check("Это история про мальчика, который ищет брата.", ctx)   # не назвал, о чём речь
    # один вопрос — один ответ: лишние выдуманные ответы не проходят
    one = C.chk_all(C.chk("7", "семь"), max_words=8)
    assert check("Привет! 5 + 2 = 7.", one)
    assert not check("Привет! 5 + 2 = 7. Столица Франции — Париж. Меня зовут SLR-Net.", one)
    # прежние проверки работают как раньше: «none» с «tail» смотрит только на вывод
    c = {"any": ["Аня"], "none": ["Боря"], "tail": "значит"}
    assert check("Рассуждаю: по росту Аня > Боря. Значит, выше всех Аня.", c)


def test_old_part_and_old_probes_unchanged_by_context_skills():
    """Новые навыки собираются отдельным генератором ПОСЛЕ старой части: старые диалоги и прежние задачи — те же."""
    kw = dict(seed=5, n_compose=200, scale=0.04)
    old = C.build_dataset([], context_skills=False, **kw)
    new = C.build_dataset([], **kw)
    for k in ("sft_val", "choice", "open_eval"):
        assert new[k] == old[k], k
    assert new["probes"][:len(old["probes"])] == old["probes"]
    assert all(p["cat"].startswith(C.CONTEXT_SRC) for p in new["probes"][len(old["probes"]):])
    dump = lambda rows: sorted(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows)  # noqa: E731
    old_part = [r for r in new["sft_train"] if not r["src"].startswith(C.CONTEXT_SRC)]
    assert dump(old_part) == dump(old["sft_train"])
    ctx = [r for r in new["sft_train"] if r["src"].startswith(C.CONTEXT_SRC)]
    assert 0.04 < len(ctx) / len(new["sft_train"]) < 0.4      # заметная, но не подавляющая доля (в полном наборе 15%)
    assert new["sft_val_context"] and all(r["src"].startswith(C.CONTEXT_SRC) for r in new["sft_val_context"])
    # прежние отложенные вопросы не появились в новых обучающих диалогах
    new_user = {t[1] for r in ctx for t in r["turns"] if t[0] == "user"}
    assert not {p["turns"][-1][1] for p in old["probes"]} & new_user


def _test_phrases():
    """Отложенные формулировки вопросов-продолжений во всех родах и числах."""
    out = set()
    for g in C.PRONOUNS:
        for kind in C.WORK_KINDS.values():
            fill = C.ContextGen._fill(g, kind)
            for table in (C.FOLLOWUP_Q, C.PET_Q):
                for tr, te in table.values():
                    out |= {q.format(**fill) for q in te}
            out |= {q.format(**fill) for tr, te in C.FOOD_Q.values() for q in te}
    for tr, te in list(C.CITY_Q.values()) + list(C.VILLAGE_Q.values()):
        out |= set(te)
    out |= set(C.CITY_ASK[1]) | {a for tr, te in C.FAMOUS_ASK.values() for a in te}
    out |= set(C.UNCLEAR_Q_TEST) | set(C.YOU_TOO_ASK_TEST)
    out |= {q for b in C.BOT_PARTS for q in b[1]}
    return out


def test_context_skills_held_out_from_training():
    data = C.build_dataset([], seed=6, n_compose=200, scale=0.04, context_scale=2.0)
    ctx = [r for r in data["sft_train"] if r["src"].startswith(C.CONTEXT_SRC)]
    text = json.dumps(data["sft_train"], ensure_ascii=False).lower()
    ctx_text = json.dumps(ctx, ensure_ascii=False).lower()
    held = [t for cat in C.WORK_TITLES.values() for t in cat[1]] + C.WORK_TITLES_LONG_TEST
    held += [p[0] for p in C.PLOTS_TEST + C.GAMEPLAY_TEST + C.SONG_PLOTS_TEST + C.FEATURES_TEST]
    held += [f[1] for f in C.FAMOUS_WORKS_TEST] + [f[0] for f in C.FOODS_CTX_TEST]
    held += [f[0] for f in C.FILLINGS_SWEET_TEST + C.FILLINGS_SAVORY_TEST] + C.VILLAGES_TEST
    held += [c[5] for c in C.CITIES_CTX_TEST] + [c[6] for c in C.CITIES_CTX_TEST]       # факты о городах
    held += [r[0] for v in C.YOU_TOO_TEST.values() for x in v for r in x[4]]
    for h in held:
        assert h.lower() not in text, h                                     # нигде в обучении
    # имена героев, клички, способы склейки вопросов и отложенные города — нигде в новых диалогах
    for h in (C.HERO_NAMES_M_TEST + C.HERO_NAMES_F_TEST + C.HERO_NAMES_LONG_TEST + C.PET_NAMES_TEST +
              [c[0] for c in C.CITIES_CTX_TEST] + ["кроме того,", "во-первых,", "ещё вопрос:"] +
              [x.strip().lower() for x in C.MQ_LEADS_TEST]):
        assert not re.search(r"(?<!\w)" + re.escape(h.lower()) + r"(?!\w)", ctx_text), h
    user_turns = {t[1] for r in data["sft_train"] for t in r["turns"] if t[0] == "user"}
    ctx_user = " | ".join(t[1] for r in ctx for t in r["turns"] if t[0] == "user")
    for q in _test_phrases():
        assert q not in user_turns, q                                       # отложенная формулировка не учится
        assert q not in ctx_user, q
    new_probes = [p for p in data["probes"] if p["cat"].startswith(C.CONTEXT_SRC)]
    assert len(new_probes) > 300
    assert not {p["turns"][-1][1] for p in new_probes} & user_turns
    # обучающие и отложенные списки не пересекаются
    for tr, te in C.WORK_TITLES.values():
        assert not set(tr) & set(te)
    assert not {p[0] for p in C.PLOTS_TRAIN} & {p[0] for p in C.PLOTS_TEST}
    assert not set(C.HERO_NAMES_M_TEST + C.HERO_NAMES_F_TEST) & set(C.NAMES_TRAIN)
    assert set(C.HERO_NAMES_M_TEST + C.HERO_NAMES_F_TEST) <= set(C.NAMES_TEST)
    assert not {f[1] for f in C.FAMOUS_WORKS_TRAIN} & {f[1] for f in C.FAMOUS_WORKS_TEST}


def test_multi_question_more_single_than_multi_and_answers_short():
    """Контраст: примеров с одним вопросом больше, чем с несколькими; ответ на 3 вопроса — короткий."""
    data = C.build_dataset([], seed=7, n_compose=200, scale=0.04)
    rows = data["sft_train"]
    multi = [r for r in rows if r["src"].startswith("навык:несколько вопросов") and not r["src"].endswith("один вопрос")]
    assert multi and len(multi) * 3 < len(rows) - len(multi)
    for r in multi:
        assert len(r["turns"][-1][1].split()) <= 40, r


def test_context_dialogues_fit_mini2_window():
    """Диалог целиком влезает в 256 токенов (иначе начало обрежется и пропадёт то, о чём спрашивают),
    ответ — в лимит генерации 64 токена."""
    pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    import train_chat as TC
    path = os.path.join(ROOT, "slr_net_subtitles_ru_mini2.npz")
    if not os.path.exists(path):
        pytest.skip("нет весов mini2")
    with np.load(path) as z:
        tok = TC.ChatTokenizer(model_proto=z["tokenizer_proto"].tobytes())
    for train in (True, False):
        for name, r in _context_rows(train, n=120, seed=1):
            ids, _ = TC.encode_dialogue(tok, r["turns"])
            assert len(ids) <= 257, (len(ids), r)
            assert len(tok.encode_line(r["turns"][-1][1])) < 64, r
    for dialog in C.LIVE_DIALOGUES:
        assert 2 <= len(dialog) <= 4 and all(isinstance(u, str) and u for u in dialog)
