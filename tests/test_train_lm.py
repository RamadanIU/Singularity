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


SHIPPED = Path(ROOT) / "slr_net_subtitles_ru_mini.npz"
DIALOGUE = ("- Привет! Как дела?\n- Хорошо, спасибо. А у тебя?\n- Тоже неплохо. Куда ты идёшь?\n"
            "- Домой, уже поздно.\n- Давай я тебя провожу.\n- Не надо, я сама дойду.\n")


@pytest.mark.skipif(not SHIPPED.exists(), reason="нет файла обученных весов")
def test_shipped_subtitles_model_really_learned():
    """Веса из репозитория обучены: обычный диалог предсказывают намного лучше случайных весов
    и того же набора слов в случайном порядке (то есть выучили не только частоты, но и порядок слов)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    import random
    import torch.nn.functional as F
    import slr_net_engine as E

    model, tok = train_lm.load_model(str(SHIPPED), torch.device("cpu"))

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
