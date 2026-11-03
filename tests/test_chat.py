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
