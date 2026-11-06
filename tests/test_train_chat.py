# -*- coding: utf-8 -*-
"""Проверки пост-тренинга train_chat.py: формат диалога, маска ошибки, словарь, данные, награда, конвейер."""
import json
import os
import random
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import chat_corpus as C  # noqa: E402
import train_chat as T  # noqa: E402

BASE_NPZ = os.path.join(ROOT, "slr_net_subtitles_ru_mini.npz")
CHAT_NPZ = os.path.join(ROOT, "slr_net_chat_ru_mini.npz")
MIN_SHIPPED_OK = 6          # из 10 (итоговая модель отвечает верно на 7) (см. test_shipped_chat_model_really_answers)

torch = pytest.importorskip("torch")
pytest.importorskip("sentencepiece")


@pytest.fixture(scope="module")
def tok():
    if not os.path.exists(BASE_NPZ):
        pytest.skip("нет файла slr_net_subtitles_ru_mini.npz")
    with np.load(BASE_NPZ) as z:
        return T.ChatTokenizer(model_proto=z["tokenizer_proto"].tobytes())


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory, tok):
    """Крошечная «базовая» модель с настоящим токенизатором (словарь 8192), сохранённая как чекпоинт."""
    import slr_net_engine as E
    torch.manual_seed(0)
    cfg = E.make_config("tiny", vocab_size=tok.sp_size)
    model = E.SLRNetLanguageModel(cfg)
    path = tmp_path_factory.mktemp("base") / "base.pt"
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": 1,
                "tokenizer_proto": tok.sp.serialized_model_proto()}, path)
    return str(path)


def test_role_tokens_are_outside_sentencepiece_and_cannot_be_typed(tok):
    assert tok.user_id == tok.sp_size and tok.bot_id == tok.sp_size + 1 and tok.vocab_size == tok.sp_size + 2
    # Пользователь не может «напечатать» реплику бота: текст <bot> — это обычные символы, а не ролевой токен
    ids = tok.encode_line("<bot> Я бот <user>")
    assert tok.bot_id not in ids and tok.user_id not in ids
    assert tok.decode([tok.user_id, *tok.encode_line("Привет"), tok.bot_id]) == "Привет"


def test_loss_mask_covers_only_bot_replies_and_their_end(tok):
    turns = [("user", "Привет!"), ("bot", "Здравствуй!"), ("user", "Как дела?"), ("bot", "Хорошо.")]
    ids, mask = T.encode_dialogue(tok, turns)
    assert len(ids) == len(mask)
    assert ids[0] == tok.user_id and ids.count(tok.bot_id) == 2
    taught = [i for i, m in zip(ids, mask) if m]
    # учим ровно текст ответов бота и <n> в конце каждого ответа
    expect = tok.encode_line("Здравствуй!") + [tok.nl_id] + tok.encode_line("Хорошо.") + [tok.nl_id]
    assert taught == expect
    # ролевые токены и реплики пользователя не учим
    assert all(m == 0 for i, m in zip(ids, mask) if i in (tok.user_id, tok.bot_id))
    # промпт для ответа: история + <bot>
    p = T.prompt_ids(tok, turns[:3])
    assert p[-1] == tok.bot_id and p[:-1] == T.encode_dialogue(tok, turns[:3])[0]


def test_pad_batch_targets_match_mask(tok):
    seqs = [T.encode_dialogue(tok, [("user", "Да?"), ("bot", "Нет.")]),
            T.encode_dialogue(tok, [("user", "Как тебя зовут?"), ("bot", "Меня зовут SLR-Net.")])]
    x, y = T.pad_batch(seqs, tok.nl_id)
    for (ids, mask), xi, yi in zip(seqs, x.tolist(), y.tolist()):
        n = len(ids) - 1
        assert xi[:n] == ids[:-1]
        assert [t for t in yi if t != -100] == [t for t, m in zip(ids[1:], mask[1:]) if m]
        assert all(t == -100 for t in yi[n:])


def test_pack_rows_keeps_dialogues_whole(tok):
    rng = random.Random(0)
    ex = [T.encode_dialogue(tok, [("user", f"Вопрос {i}?"), ("bot", f"Ответ {i}.")]) for i in range(40)]
    rows = T.pack_rows(ex, 64, tok.nl_id, rng, tok.user_id)
    assert all(len(r[0]) == 64 == len(r[1]) for r in rows)
    assert all(r[0][0] == tok.user_id for r in rows)                  # строка начинается с начала диалога
    assert sum(sum(r[1]) for r in rows) == sum(sum(m) for _, m in ex)  # ни один ответ не потерян


def test_extend_vocab_keeps_old_tokens_exactly(tiny_base, tok):
    model, _, chat = T.load_chat_model(tiny_base, torch.device("cpu"))
    assert not chat
    big = T.extend_vocab(model, tok.vocab_size, init_from=tok.nl_id)
    assert big.config.vocab_size == tok.vocab_size
    x = torch.tensor([tok.encode_line("Привет, как дела?")])
    with torch.no_grad():
        a, _, _ = model(x, routing="threshold")
        b, _, _ = big(x, routing="threshold")
    assert torch.allclose(a, b[..., :tok.sp_size], atol=1e-5)          # старые токены — без изменений
    assert big.lm_head.weight.data_ptr() == big.tok_embeddings.weight.data_ptr()   # выход связан с эмбеддингом


def test_batched_generation_equals_one_by_one(tiny_base, tok):
    """Ответы модели на «свои» вопросы считаются батчем — результат тот же, что и по одному."""
    model, _, _ = T.load_chat_model(tiny_base, torch.device("cpu"))
    model = T.extend_vocab(model, tok.vocab_size, init_from=tok.nl_id)
    turns = [[("user", "Привет!")], [("user", "Пока!")], [("user", "Ага.")]]
    ctxs = [T.prompt_ids(tok, t) for t in turns]
    assert len({len(c) for c in ctxs}) == 1
    batched = T.generate_batch(model, tok, ctxs, n=2, max_new_tokens=6, temperature=0)
    for c, b in zip(ctxs, batched):
        single = T.generate_replies(model, tok, c, n=1, max_new_tokens=6, temperature=0)[0]
        assert b[0] == single and b[1] == single


def test_reply_check_matches_whole_numbers_and_word_starts():
    assert T.check_reply("Солнце жёлтое.", {"any": ["желт"]})
    assert not T.check_reply("Давай поговорим.", {"any": ["да"]})        # «да» — только целым словом
    assert T.check_reply("Да, конечно!", {"any": ["да"]})
    assert not T.check_reply("После 7 идёт 89.", {"any": ["идет 8"]})    # числа — целиком
    assert T.check_reply("Меня зовут SLR-Net.", {"any": ["slr-net"]})
    assert not T.check_reply("Тебя зовут Аня.", {"any": ["Оля"], "none": ["Аня"]})
    assert T.check_reply("Тебя зовут Оля.", {"any": ["Оля"], "none": ["Аня"]})


def test_defects_reward_catches_bad_replies(tok):
    nl = tok.nl_id
    assert T.reply_defects("Привет!", [5, nl], tok, "Привет", 40) == []
    assert "не закончил реплику" in T.reply_defects("Привет", [5], tok, "x", 40)
    assert "не по-русски" in T.reply_defects("Hello my friend", [5, nl], tok, "x", 40)
    assert "повторы" in T.reply_defects("да да да да да", [5, nl], tok, "x", 40)
    assert "повторил вопрос" in T.reply_defects("Как у тебя дела?", [5, nl], tok, "Как у тебя дела?", 40)


def test_subtitle_pair_filter():
    assert T.good_subtitle_pair("Ты голоден?", "Нет, спасибо.")
    assert not T.good_subtitle_pair("Почему?", "Потому что.")               # вопрос из одного слова
    assert not T.good_subtitle_pair("Как тебя зовут?", "Джон.")              # вопрос о личности бота
    assert not T.good_subtitle_pair("Ты где?", "Я у Джона дома.")            # имя персонажа
    assert not T.good_subtitle_pair("Ты устала?", "Да, я очень устала.")     # женский род о себе
    assert not T.good_subtitle_pair("Что это?", "Это пистолет.")             # насилие
    assert T.good_subtitle_pair("Как поворачивать?", "Так, назад, на дорогу.")   # «назад» — не «ад»


def test_seed_files_parse_and_hold_out_one_wording():
    singles, multis = C.load_seeds()
    assert len(singles) > 150 and len(multis) > 30
    for b in singles:
        assert b.questions and b.answers, b.questions
        assert all(len(a.split()) <= 40 for a in b.answers), b.answers
    data = C.build_dataset([], seed=1, n_compose=200, scale=0.05)
    train_q = {r["turns"][0][1] for r in data["sft_train"]}
    held = [p["turns"][-1][1] for p in data["probes"] if p["cat"].startswith("ручные")]
    assert held and not (set(held) & train_q)                             # отложенные формулировки не учим
    # отложенные имена/города не попадают в обучение
    text = json.dumps(data["sft_train"], ensure_ascii=False)
    for name in C.NAMES_TEST[:10] + [c[0] for c in C.CITIES_TEST[:5]]:
        assert name not in text, name


def test_grammar_helpers():
    assert C.v_prep("Владивостоке") == "во" and C.v_prep("Волгограде") == "в" and C.v_prep("Москве") == "в"
    assert [C.years_word(n) for n in (1, 2, 5, 11, 21, 22, 25, 112)] == ["год", "года", "лет", "лет", "год", "года",
                                                                        "лет", "лет"]
    assert C.num_word(42) == "сорок два" and C.num_word(20) == "двадцать" and C.num_word(7) == "семь"


def test_reply_check_tail_looks_only_at_the_conclusion():
    c = {"any": ["Аня"], "none": ["Боря", "Вова"], "tail": "значит"}
    assert T.check_reply("Рассуждаю: по росту Аня > Боря > Вова. Значит, выше всех Аня.", c)
    assert not T.check_reply("Рассуждаю: по росту Аня > Боря > Вова. Значит, выше всех Вова.", c)
    assert not T.check_reply("Выше всех Аня.", c)                         # нет вывода — не засчитываем
    assert T.check_reply("Было 7, съел 3: 7 - 3 = 4. Значит, осталось 4 конфеты.", {"any": ["4"], "tail": "значит"})
    assert not T.check_reply("Было 7, съел 3: 7 - 3 = 4. Значит, осталось 5 конфет.", {"any": ["4"], "tail": "значит"})


def test_nonce_words_are_new_and_held_out():
    assert len(C.NONCE_TRAIN) >= 300 and len(C.NONCE_TEST) >= 50
    assert not set(C.NONCE_TRAIN) & set(C.NONCE_TEST)
    known = {d.split("|")[0] for d in C.DEFINITIONS.strip().splitlines()}
    assert not set(C.NONCE_TRAIN + C.NONCE_TEST) & known                  # не путаются с «настоящими» словами
    assert C.nonce_plural("клуфа") == "клуфы" and C.nonce_plural("бармук") == "бармуки"
    assert C.nonce_plural("стипос") == "стипосы" and C.nonce_gen("стипос") == "стипоса"
    assert [C.plural(n, ("яблоко", "яблока", "яблок")) for n in (1, 3, 5, 11, 21, 24)] == \
        ["яблоко", "яблока", "яблок", "яблок", "яблоко", "яблока"]


def test_new_skills_templates_answer_from_context():
    """Ответ в шаблонах новых навыков действительно следует из разговора (а не из памяти модели)."""
    rng = random.Random(3)
    g = C.Gen(rng, [[("user", "Как дела?"), ("bot", "Отлично!")]])
    for train in (True, False):
        words = C.NONCE_TRAIN if train else C.NONCE_TEST
        for _ in range(200):
            r = g.new_word(train)
            ctx = " ".join(t[1] for t in r["turns"][:-1])
            assert any(w in ctx.lower() for w in words)                     # слово объяснено в разговоре
            assert T.check_reply(r["turns"][-1][1], r["check"]), r          # эталон проходит свою проверку
            r = g.new_word_unknown(train)
            assert T.check_reply(r["turns"][-1][1], r["check"]), r
            for fn in (g.new_word_two, g.logic_chain, g.syllogism, g.word_problem):
                r = fn(train)
                assert T.check_reply(r["turns"][-1][1], r["check"]), r
            r = g.new_word_reference(train, C._table(C.DEFINITIONS))
            assert r["turns"][-1][0] == "bot" and r["turns"][-2][1].startswith("Справка: ")
            assert T.check_reply(r["turns"][-1][1], r["check"]), r
    # задача со счётом: ответ — верная арифметика по условию
    for _ in range(100):
        r = g.word_problem(True)
        ans = r["turns"][-1][1]
        for a, op, b, res in re.findall(r"(\d+) ([-+]) (\d+) = (\d+)", ans):
            assert int(res) == (int(a) + int(b) if op == "+" else int(a) - int(b))


def test_new_skills_held_out_from_training():
    data = C.build_dataset([], seed=4, n_compose=200, scale=0.05)
    text = json.dumps(data["sft_train"], ensure_ascii=False).lower()
    for w in C.NONCE_TEST:
        assert w not in text, w                                           # отложенные выдуманные слова не учим
    cats = {p["cat"] for p in data["probes"]}
    for cat in ("шаблон:новое слово", "шаблон:новое слово: вывод", "шаблон:новое слово: не знаю",
                "шаблон:два новых слова", "шаблон:логика: цепочка", "шаблон:логика: силлогизм", "шаблон:задачи",
                "шаблон:справка"):
        assert cat in cats, cat
    assert not any(p.get("from", "").startswith(C.NEW_SKILL_SRC) for p in data["probes"])


def test_copy_mechanism_continues_context_words(tok):
    ctx = T.prompt_ids(tok, [("user", "Я живу в Костроме, мою кошку зовут Ватрушка."), ("bot", "Здорово!"),
                             ("user", "Где я живу?")])
    words = T.context_words(tok, ctx)
    kostroma = tok.sp.encode(" Костроме")
    assert kostroma in words                                       # многокусочковые слова разговора найдены
    assert T.copy_bonus(tok, words, kostroma[:1]) == {kostroma[1]: 1.0}       # «▁Ко» → следующий кусочек
    assert T.copy_bonus(tok, words, kostroma[:2]) == {kostroma[2]: 1.0}
    assert T.copy_bonus(tok, words, tok.sp.encode(" Ты живёшь в")) == {}      # нет начатого слова из разговора
    assert T.copy_bonus(tok, words, kostroma + tok.sp.encode(".")) == {}      # слово закончено


def test_external_dictionary_lookup(tmp_path):
    path = tmp_path / "dict.txt"
    path.write_text("# комментарий\nбарометр — прибор, который измеряет давление воздуха\n"
                    "фуникулёр | это вагончик на канате.\nмаяк — это башня с огнём.\n", encoding="utf-8")
    d = T.load_dictionary(str(path))
    assert d["барометр"] == ("барометр", "это прибор, который измеряет давление воздуха.")
    assert T.dictionary_hit("Где используют барометры?", d)[0] == "барометр"      # падеж/число
    assert T.dictionary_hit("Что такое фуникулер?", d)[0] == "фуникулёр"          # ё/е
    assert T.dictionary_hit("Я видел маяк", d)[0] == "маяк"
    assert T.dictionary_hit("Как дела?", d) is None
    shipped = T.load_dictionary(os.path.join(ROOT, "dictionary_example.txt"))
    known = {r.split("|")[0] for r in C.DEFINITIONS.strip().splitlines()}
    assert len(shipped) >= 20 and not set(shipped) & known                       # слова, которых модель не учила
    assert C.reference_turn("барометр", "это прибор.", "Что это?") == "Справка: Барометр — это прибор. Что это?"


def test_tiny_pipeline_sft_align_export_chat(tmp_path, tiny_base, tok, capsys):
    data = C.build_dataset([], seed=2, n_compose=100, scale=0.02)
    ddir = tmp_path / "data"
    ddir.mkdir()
    for k, rows in data.items():
        T.write_jsonl(ddir / f"{k}.jsonl", rows[:300] if k == "sft_train" else rows[:40])
    sft = tmp_path / "sft"
    T.main(["sft", "--base", tiny_base, "--data", str(ddir), "--out", str(sft), "--seq-len", "64",
            "--batch-size", "4", "--max-steps", "6", "--warmup", "1", "--eval-every", "3", "--log-every", "2",
            "--sample-every", "0"])
    assert (sft / "best.pt").exists() and (sft / "last.pt").exists()
    model, ctok, chat = T.load_chat_model(str(sft / "best.pt"), torch.device("cpu"))
    assert chat and model.config.vocab_size == tok.vocab_size

    al = tmp_path / "align"
    T.main(["align", "--ckpt", str(sft / "best.pt"), "--data", str(ddir), "--out", str(al), "--rounds", "1",
            "--steps-per-round", "3", "--batch-size", "2", "--checked-items", "20", "--open-items", "10",
            "--sample-items", "4", "--samples", "2", "--max-new", "6", "--log-every", "1"])
    out = capsys.readouterr().out
    assert "верный порядок" in out and (al / "last.pt").exists()

    npz = tmp_path / "chat.npz"
    T.main(["export", "--ckpt", str(al / "last.pt"), "--out", str(npz)])
    with np.load(npz) as z:
        assert str(z["meta.chat_format"]) == T.CHAT_FORMAT and int(z["meta.bot_id"][0]) == tok.bot_id
    m2, t2, chat2 = T.load_chat_model(str(npz), torch.device("cpu"))
    assert chat2
    ctx = T.prompt_ids(t2, [("user", "Привет!")])
    r = T.generate_replies(m2, t2, ctx, n=2, max_new_tokens=8, temperature=0.9)
    assert len(r) == 2 and all(0 < len(x) <= 8 for x in r)
    assert all(t not in t2.banned_ids for x in r for t in x)              # ролевые/служебные токены не пишет
    # «удержание темы» (guidance) работает и ничего не ломает
    g = T.generate_replies(m2, t2, ctx, n=1, max_new_tokens=8, temperature=0, guidance=1.5,
                           uncond_ids=T.prompt_ids(t2, []))
    assert len(g[0]) > 0

    T.main(["eval", "--ckpt", str(npz), "--data", str(ddir), "--choice-items", "10", "--open-items", "5",
            "--probes", "8", "--showcase", "2", "--print-showcase", "1", "--json", str(tmp_path / "eval.json")])
    rep = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))
    assert "chat.npz" in rep and "ВСЕГО" in rep["chat.npz"]["probes"]


SHIPPED_CHATS = [(CHAT_NPZ, BASE_NPZ),                                        # mini
                 (os.path.join(ROOT, "slr_net_chat_ru_mini2.npz"),
                  os.path.join(ROOT, "slr_net_subtitles_ru_mini2.npz"))]      # mini2 (по умолчанию в браузере)


@pytest.mark.parametrize("chat_npz,base_npz", SHIPPED_CHATS, ids=["mini", "mini2"])
def test_shipped_chat_model_really_answers(tok, chat_npz, base_npz):
    """Готовая чат-модель из репозитория отвечает по смыслу на вопросы, которых не было в обучении."""
    if not (os.path.exists(chat_npz) and os.path.exists(base_npz)):
        pytest.skip("нет обученной чат-модели")
    model, ctok, chat = T.load_chat_model(chat_npz, torch.device("cpu"))
    assert chat
    probes = [                                     # 10 вопросов разной трудности, заданы заранее
        ([("user", "Меня зовут Тимофей."), ("bot", "Приятно познакомиться, Тимофей!"), ("user", "Как меня зовут?")],
         ["Тимофей"]),                             # имени нет в обучении
        ([("user", "Здравствуй, меня зовут Всеволод."), ("bot", "Приятно познакомиться, Всеволод!"),
          ("user", "Скажи моё имя.")], ["Всеволод"]),  # трудное: редкое имя + новая формулировка
        ([("user", "Я живу в Костроме."), ("bot", "Кострома — хороший город!"), ("user", "Напомни, где я живу.")],
         ["Костроме", "Кострома"]),                # трудное: город и формулировка не из обучения
        ([("user", "Я живу в Казани."), ("bot", "Казань — хороший город!"), ("user", "Где я живу?")], ["Казани"]),
        ([("user", "Ты кто такой?")], ["slr-net", "модель", "нейросеть"]),
        ([("user", "Какая столица Японии?")], ["Токио"]),
        ([("user", "Hello!")], ["русск"]),
        ([("user", "Сколько будет 2 + 3?")], ["5", "пять"]),
        ([("user", "Какой день идёт после пятницы?")], ["суббота"]),
        ([("user", "Какая сегодня погода?")], ["не знаю"]),
    ]
    ok = 0
    for turns, keys in probes:
        reply = T.answer(model, ctok, True, turns)
        ok += bool(T.check_reply(reply, {"any": keys}))
    assert ok >= MIN_SHIPPED_OK, ok
    # базовая модель (до пост-тренинга) с теми же вопросами справляется хуже
    base, btok, _ = T.load_chat_model(base_npz, torch.device("cpu"))
    base_ok = sum(bool(T.check_reply(T.answer(base, btok, False, t), {"any": k})) for t, k in probes)
    assert base_ok <= 2 and ok >= base_ok + 5, (ok, base_ok)
