# -*- coding: utf-8 -*-
"""Проверки конвейера обучения train_lm.py: данные, кодировки, чекпоинты, экспорт и загрузка весов."""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train_lm  # noqa: E402

SRT = "1\n00:00:01,000 --> 00:00:02,000\n<i>- Привет, как дела? Ёлка!</i>\n- Отлично.\n\n"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1251", "utf-16"])
def test_subtitles_in_any_common_encoding_are_read(tmp_path, encoding):
    """Раньше всё читалось как UTF-8 с errors="ignore": .srt в Windows-1251 молча становился пустым."""
    path = tmp_path / "film.srt"
    path.write_bytes(SRT.encode(encoding))
    assert list(train_lm.iter_clean_lines(path)) == ["- Привет, как дела? Ёлка!", "- Отлично."]


def test_full_pipeline_prepare_train_export_sample(tmp_path, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    corpus = tmp_path / "texts"
    corpus.mkdir()
    lines = ["Привет! Как дела?", "Хорошо, спасибо. А у тебя?", "Куда ты идёшь?", "Домой, уже поздно.",
             "Пойдём в кино завтра вечером.", "Я не знаю, что сказать.", "Это было давно, в 1999 году."]
    (corpus / "subs.txt").write_text("\n".join(lines * 60), encoding="utf-8")
    data, run = tmp_path / "data", tmp_path / "run"

    train_lm.main(["prepare", "--input", str(corpus), "--out", str(data), "--vocab-size", "400"])
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_tokens"] > 1000 and meta["vocab_size"] <= 400

    train_lm.main(["train", "--data", str(data), "--out", str(run), "--preset", "tiny",
                   "--seq-len", "32", "--batch-size", "4", "--max-steps", "4", "--warmup", "1",
                   "--eval-every", "2", "--eval-batches", "2", "--save-every", "2", "--log-every", "1",
                   "--sample-prompt", ""])
    assert (run / "best.pt").exists() and (run / "last.pt").exists()
    log = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["kind"] for r in log} == {"train", "val"}

    # Продолжение обучения с чекпоинта
    train_lm.main(["train", "--data", str(data), "--out", str(run), "--preset", "tiny",
                   "--seq-len", "32", "--batch-size", "4", "--max-steps", "6", "--warmup", "1",
                   "--eval-every", "100", "--eval-batches", "2", "--save-every", "100", "--log-every", "1",
                   "--sample-prompt", "", "--resume", str(run / "last.pt")])
    assert "шаг      4" in capsys.readouterr().out

    report = tmp_path / "eval.json"
    train_lm.main(["eval", "--ckpt", str(run / "best.pt"), "--data", str(data), "--seq-len", "32",
                   "--batch-size", "2", "--batches", "2", "--json", str(report)])
    rows = json.loads(report.read_text(encoding="utf-8"))
    assert [r["model"].split(":")[0] for r in rows] == ["Частоты токенов (униграммы)", "Обученная модель",
                                                         "Та же сеть со случайными весами"]

    packed = tmp_path / "model.npz"
    train_lm.main(["export", "--ckpt", str(run / "best.pt"), "--out", str(packed)])
    # Экспорт самодостаточен: токенизатор внутри файла, исходные данные больше не нужны
    shutil.rmtree(data)
    with np.load(packed) as z:
        assert "tokenizer_proto" in z.files and int(z["meta.step"][0]) > 0
    model, tok = train_lm.load_model(str(packed), __import__("torch").device("cpu"))
    text = train_lm.sample_text(model, tok, "Привет!", 10, 0.8, 20, 0.95, 1.1, __import__("torch").device("cpu"))
    assert isinstance(text, str)
    # Упакованная модель = чекпоинт (троичные веса те же, остальное в float16)
    import torch
    ref, _ = train_lm.load_model(str(run / "best.pt"), torch.device("cpu"))
    x = torch.tensor([tok.encode("Привет! Как дела?")])
    with torch.no_grad():
        a, _, _ = ref(x, routing="all")
        b, _, _ = model(x, routing="all")
    assert (a - b).abs().max() < 0.05 * a.abs().max()


def test_init_drop_memory_and_compress_embedding_then_copytest(tmp_path, capsys):
    """Готовую модель можно дообучать без памяти фактов и со сжатым при обучении эмбеддингом (--init + --set),
    упаковать (эмбеддинг — полбайта на число) и проверить дословное копирование (copytest)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    corpus = tmp_path / "subs.txt"
    lines = ["Привет! Как дела?", "Хорошо, спасибо.", "Куда ты идёшь?", "Домой, уже поздно.",
             "Пойдём в кино завтра.", "Я не знаю, что сказать."]
    corpus.write_text("\n".join(lines * 80), encoding="utf-8")
    data, base, ft = tmp_path / "data", tmp_path / "base", tmp_path / "ft"
    train_lm.main(["prepare", "--input", str(corpus), "--out", str(data), "--vocab-size", "300"])
    common = ["--data", str(data), "--seq-len", "32", "--batch-size", "4", "--warmup", "1", "--eval-every", "100",
              "--eval-batches", "2", "--save-every", "100", "--log-every", "1", "--sample-prompt", ""]
    train_lm.main(["train", "--out", str(base), "--preset", "tiny", "--max-steps", "3"] + common)
    train_lm.main(["train", "--out", str(ft), "--init", str(base / "last.pt"), "--max-steps", "3",
                   "--set", "memory_keys=0", "embed_bits=4", "embed_group=32"] + common)
    assert "изменено: {'memory_keys': 0, 'embed_bits': 4, 'embed_group': 32}" in capsys.readouterr().out
    ref, _ = train_lm.load_model(str(base / "last.pt"), torch.device("cpu"))
    new, _ = train_lm.load_model(str(ft / "last.pt"), torch.device("cpu"))
    assert new.core.memory is None and new.config.embed_bits == 4
    assert ref.core.memory is not None
    packed = tmp_path / "ft.npz"
    train_lm.main(["export", "--ckpt", str(ft / "last.pt"), "--out", str(packed)])
    with np.load(packed) as z:
        assert not any(k.startswith("core.memory") for k in z.files)
        assert "tok_embeddings.weight" not in z.files
        assert z["tok_embeddings.weight.qdata"].nbytes == ref.config.vocab_size * ref.config.d_model // 2
    m2, _ = train_lm.load_model(str(packed), torch.device("cpu"))
    with torch.no_grad():
        assert torch.equal(m2.embedding_weight(), new.embedding_weight())
    # без дообучения: та же модель, просто без памяти и с округлённым эмбеддингом
    posthoc = tmp_path / "posthoc.npz"
    train_lm.main(["export", "--ckpt", str(base / "last.pt"), "--out", str(posthoc), "--set", "memory_keys=0",
                   "embed_bits=4", "embed_group=32"])
    assert posthoc.stat().st_size < (tmp_path / "ft.npz").stat().st_size * 1.2
    report = tmp_path / "copy.json"
    train_lm.main(["copytest", "--ckpt", str(packed), str(posthoc), "--data", str(data), "--span", "8",
                   "--gaps", "0", "4", "--n", "6", "--json", str(report)])
    res = json.loads(report.read_text(encoding="utf-8"))
    assert len(res["results"]) == 2 and set(res["results"][0]["by_distance"]) == {"8", "12"}
    assert all(0.0 <= v["acc"] <= 1.0 for r in res["results"] for v in r["by_distance"].values())


def test_tokenizer_proto_roundtrip(tmp_path):
    pytest.importorskip("sentencepiece")
    import sentencepiece as spm
    src = tmp_path / "t.txt"
    src.write_text("\n".join(["мама мыла раму", "папа читал книгу", "кошка спит на окне"] * 50), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(src), model_prefix=str(tmp_path / "tok"), vocab_size=60,
                                   user_defined_symbols=["<n>"], hard_vocab_limit=False, minloglevel=2)
    a = train_lm.Tokenizer(str(tmp_path / "tok.model"))
    b = train_lm.Tokenizer(model_proto=Path(tmp_path / "tok.model").read_bytes())
    s = "мама читала книгу\nкошка"
    assert a.encode(s) == b.encode(s) and b.decode(b.encode(s)) == s


SHIPPED_MODELS = [Path(ROOT) / "slr_net_subtitles_ru_mini.npz", Path(ROOT) / "slr_net_subtitles_ru_mini2.npz"]
DIALOGUE = ("- Привет! Как дела?\n- Хорошо, спасибо. А у тебя?\n- Тоже неплохо. Куда ты идёшь?\n"
            "- Домой, уже поздно.\n- Давай я тебя провожу.\n- Не надо, я сама дойду.\n")


@pytest.mark.parametrize("shipped", SHIPPED_MODELS, ids=["mini", "mini2"])
def test_shipped_subtitles_model_really_learned(shipped):
    """Веса из репозитория обучены: обычный диалог предсказывают намного лучше случайных весов
    и того же набора слов в случайном порядке (то есть выучили не только частоты, но и порядок слов)."""
    if not shipped.exists():
        pytest.skip("нет файла обученных весов")
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    import random
    import torch.nn.functional as F
    import slr_net_engine as E

    model, tok = train_lm.load_model(str(shipped), torch.device("cpu"))

    def loss(m, text):
        ids = torch.tensor([tok.encode(text)])
        with torch.no_grad():
            logits, _, _ = m(ids[:, :-1], routing="threshold")
        return F.cross_entropy(logits[0], ids[0, 1:]).item()

    words = DIALOGUE.split()
    random.Random(0).shuffle(words)
    torch.manual_seed(0)
    untrained = E.SLRNetLanguageModel(model.config).eval()

    natural = loss(model, DIALOGUE)
    assert natural < 4.0                                   # на отложенных субтитрах ≈ 3.5
    assert loss(model, " ".join(words)) > natural + 2.0    # перемешанные слова — «не по-русски»
    assert loss(untrained, DIALOGUE) > 8.0                 # случайные веса: ≈ ln 8192 = 9.0

    torch.manual_seed(0)
    text = train_lm.sample_text(model, tok, "Привет! Как дела?", 30, 0.8, 50, 0.95, 1.1, torch.device("cpu"))
    letters = [c for c in text if c.isalpha()]
    assert len(letters) > 20 and sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in letters) > 0.9 * len(letters)
