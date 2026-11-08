# -*- coding: utf-8 -*-
"""
Проверки многомодульной части (PLAN_MULTIMODULE_RU.md): рост нейронов и модули знаний (self_grow.py),
оркестратор (orchestrator.py), награда за целостность игры (game_reward.py).
Все проверки идут на крошечной модели (пресет tiny) — секунды на CPU.
"""

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

import orchestrator as O         # noqa: E402
import self_grow as S            # noqa: E402
import slr_net_engine as E       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny_model(seed=0):
    torch.manual_seed(seed)
    model = E.SLRNetLanguageModel(E.make_config("tiny"))
    model.eval()
    return model


def logits_of(model, x):
    with torch.no_grad():
        return model(x, routing="threshold")[0]


# ------------------------------------------------------------------ новые нейроны

def test_attach_and_grow_preserve_function():
    """Вставка нейронов и каждый рост не меняют ответ модели: выход новых нейронов в начале — нули."""
    model = tiny_model()
    x = torch.randint(0, 4096, (2, 40))
    before = logits_of(model, x)
    layers = S.attach(model)
    assert len(layers) == len(S.mlp_sites(model)) >= 2
    assert torch.equal(logits_of(model, x), before)
    S.grow_all(layers, 8, -0.5, seed=1)
    S.grow_all(layers, 16, -0.5, seed=2)
    assert torch.allclose(logits_of(model, x), before, atol=1e-6)
    assert all(g.n_neurons == 24 for g in layers.values())
    d = model.config.d_model
    assert S.n_new_params(model) == len(layers) * 24 * (2 * d + 1)


def test_only_new_neurons_learn_and_old_weights_stay():
    model = tiny_model()
    old = {k: v.clone() for k, v in model.state_dict().items()}
    layers = S.attach(model)
    S.freeze_base(model)
    params = S.grow_all(layers, 8, -0.5, seed=0)
    assert all(p.requires_grad for p in params)
    assert not any(p.requires_grad for n, p in model.named_parameters() if ".chunks." not in n)
    opt = torch.optim.AdamW(params, lr=1e-2)
    x = torch.randint(0, 4096, (4, 24))
    y = torch.randint(0, 4096, (4, 24))
    for _ in range(3):
        logits = model(x, routing="threshold")[0]
        loss = S.masked_ce(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    new_state = model.state_dict()
    for k, v in old.items():
        k2 = k.replace(".mlp.", ".mlp.base.") if k not in new_state else k
        assert torch.equal(new_state[k2], v), k
    assert any(float(ch.down.detach().abs().sum()) > 0 for g in layers.values() for ch in g.chunks)


def test_frozen_bitlinear_uses_cache_and_matches_ste():
    """Замороженный троичный слой при включённом градиенте берёт квантованные веса из кэша — значения те же."""
    lin = E.BitLinear158(32, 16)
    x = torch.randn(3, 32)
    ste = lin(x)
    lin.weight.requires_grad_(False)
    cached = lin(x)
    assert torch.allclose(ste, cached, atol=1e-6)
    assert lin._wq_cache is not None


def test_enable_disable_and_active_tags():
    model = tiny_model()
    x = torch.randint(0, 4096, (1, 20))
    base = logits_of(model, x)
    layers = S.attach(model)
    for name, g in layers.items():
        a = g.grow(4, 0.0, seed=1)
        a.tag = "A"
        b = g.grow(4, 0.0, seed=2)
        b.tag = "B"
        with torch.no_grad():
            a.down.normal_(0, 0.5)
            b.down.normal_(0, 0.5)
    both = logits_of(model, x)
    assert not torch.allclose(both, base)
    S.set_active(model, [])
    assert torch.equal(logits_of(model, x), base)
    S.set_active(model, ["A"])
    only_a = logits_of(model, x)
    assert not torch.allclose(only_a, base) and not torch.allclose(only_a, both)
    S.set_active(model, None)
    S.set_enabled(model, False)
    assert torch.equal(logits_of(model, x), base)


def test_module_save_load_roundtrip(tmp_path):
    model = tiny_model()
    x = torch.randint(0, 4096, (2, 30))
    layers = S.attach(model)
    S.grow_all(layers, 6, -0.3, seed=0)
    S.grow_all(layers, 6, -0.3, seed=1)
    with torch.no_grad():
        for g in layers.values():
            for ch in g.chunks:
                ch.down.normal_(0, 0.1)
                for p in ch.parameters():
                    p.copy_(p.half().float())          # в файле — float16: сравниваем точно
    grown = logits_of(model, x)
    path = str(tmp_path / "mod.npz")
    S.save_module(path, model, {"words": ["курдяпа"], "texts": ["Курдяпа — это птица."]})
    meta, arrays = S.read_module(path)
    assert meta["words"] == ["курдяпа"] and meta["format"] == S.MODULE_FORMAT
    fresh = tiny_model()
    assert not torch.allclose(logits_of(fresh, x), grown)
    meta2 = S.load_module(fresh, path)
    assert meta2["tag"] == "mod"
    assert torch.allclose(logits_of(fresh, x), grown, atol=1e-5)


def test_topk_kl_zero_for_same_model():
    logits = torch.randn(2, 5, 50)
    y = torch.full((2, 5), -100)
    y[0, 2:] = 1
    y[1, :3] = 2
    targets = []
    lp = torch.log_softmax(logits, -1)
    for r in range(2):
        top = lp[r][y[r] != -100].topk(50, dim=-1)
        targets.append((top.indices, top.values))
    assert abs(float(S.topk_kl(logits, y, targets))) < 1e-5
    assert float(S.topk_kl(torch.randn(2, 5, 50), y, targets)) > 0.01


# ------------------------------------------------------------------ знания и вопросы

def test_facts_from_text_and_questions():
    facts = S.facts_from_text("Курдяпа — это маленькая птица синего цвета. Она живёт в лесу. "
                              "Бармук — это инструмент. Он лежит в ящике.")
    assert [f.word for f in facts] == ["курдяпа", "бармук"]
    assert "живёт в лесу" in facts[0].text
    qs = S.text_questions(facts[0].word, facts[0].text)
    kinds = {k for _, k in qs}
    assert {"что", "цвет", "место", "размер"} <= kinds
    assert any("Где живёт курдяпа?" == q for q, _ in qs)
    assert {k for _, k in S.text_questions(facts[1].word, facts[1].text)} == {"что", "место"}


def test_question_bank_train_and_heldout_differ():
    f = S.make_facts(["зурмоп"], seed=3)[0]
    train = {q for q, _, _ in S.question_bank(f, train=True)}
    test = {q for q, _, _ in S.question_bank(f, train=False)}
    assert len(train) >= 10 and len(test) >= 4
    assert not (train & test), "отложенные формулировки не должны попадать в обучение"
    assert all(c is not None for _, c, _ in S.question_bank(f, train=False))


def test_open_recall_weights_by_count():
    kinds = {"что": 1.0, "цвет": 0.0, "выбор": 1.0}
    assert S.open_recall(kinds, {"что": 3, "цвет": 1, "выбор": 10}) == pytest.approx(0.75)


# ------------------------------------------------------------------ оркестратор

def test_domain_of_mapping():
    assert O.domain_of("шаблон:счёт") == "счёт"
    assert O.domain_of("склейка:шаблон:логика: цепочка") == "логика"
    assert O.domain_of("ручные:знания") == "знания"
    assert O.domain_of("ручные:личность") == "разговор"
    assert O.domain_of("навык:контекст: цепочка") is None
    assert len(O.CODE_PROMPTS) > O.CODE_TRAIN + 10


def test_domain_router_nearest_centroid():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 16)) * 3
    xs, ys = [], []
    for i, c in enumerate(centers):
        xs.append(c + rng.normal(size=(30, 16)))
        ys += [f"c{i}"] * 30
    r = O.DomainRouter().fit(np.concatenate(xs), ys)
    test = np.stack([c + rng.normal(size=16) for c in centers])
    assert r.predict(test) == ["c0", "c1", "c2"]


def test_lexical_route():
    assert O.lexical_route("Напомни, что такое курдяпа?", ["курдяпа"])
    assert O.lexical_route("Какого цвета курдяпы?", ["курдяпа"])
    assert not O.lexical_route("Что такое кошка?", ["курдяпа"])


def test_embed_texts_shape_and_modules_off():
    """Вектор вопроса считается без модулей знаний, и включённость модулей после этого восстанавливается."""
    import train_chat as T
    model = tiny_model()
    tok = T.ChatTokenizer.__new__(T.ChatTokenizer)       # без SentencePiece: свой простой токенизатор
    tok.user_id, tok.bot_id, tok.nl_id = 4094, 4095, 3
    tok.encode_line = lambda t: [10 + (ord(ch) % 4000) for ch in t][:50]
    v1 = O.embed_texts(model, tok, ["привет", "как дела у тебя"], layer="core")
    layers = S.attach(model)
    S.grow_all(layers, 4, 0.0, seed=0)
    with torch.no_grad():
        for g in layers.values():
            g.chunks[0].down.normal_(0, 1.0)
    v2 = O.embed_texts(model, tok, ["привет", "как дела у тебя"], layer="core")
    assert v1.shape == (2, model.config.d_model)
    assert np.allclose(v1, v2, atol=1e-5)
    assert all(g.enabled for g in layers.values())


# ------------------------------------------------------------------ награда за игру

def test_png_decoder_matches_pillow():
    PIL = pytest.importorskip("PIL.Image")
    import io
    import game_reward as G
    rng = np.random.default_rng(0)
    img = (rng.random((23, 31, 3)) * 255).astype(np.uint8)
    img[5:15, 4:20] = [10, 200, 30]
    buf = io.BytesIO()
    PIL.fromarray(img).save(buf, format="PNG")
    ours = G.decode_png_gray(buf.getvalue())
    assert np.allclose(ours, img.astype(np.float32).mean(-1), atol=1e-4)


def test_strip_comments():
    import game_reward as G
    code = G.strip_comments("let a = 1; // счёт\n/* game over */ fetch('http://x');")
    assert "счёт" not in code and "game over" not in code and "http://x" in code


def _have_browser():
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        return False
    import game_reward as G
    return G._browser_path() is not None or os.environ.get("PLAYWRIGHT_BROWSERS_PATH")


@pytest.mark.skipif(not _have_browser(), reason="нужны playwright и Chromium")
def test_game_reward_orders_games():
    import game_reward as G
    d = os.path.join(ROOT, "training", "game_reward")
    full = G.evaluate_game(open(os.path.join(d, "tanks_full.html"), encoding="utf-8").read())
    bare = G.evaluate_game(open(os.path.join(d, "tanks_bare.html"), encoding="utf-8").read())
    broken = G.evaluate_game(open(os.path.join(d, "tanks_broken.html"), encoding="utf-8").read())
    assert full["reward"] >= 0.9, full
    assert all(v == 1.0 for v in full["checks"].values()), full["checks"]
    assert bare["reward"] < 0.4 and bare["checks"]["управление"] == 0.0 and bare["checks"]["масштаб"] == 0.0
    assert bare["details"]["страница прокручивается клавишами"]
    assert broken["reward"] <= 0.1 and broken["checks"]["запускается"] == 0.0


# ------------------------------------------------------------------ весь цикл самообучения (крошечная модель)

def _chat_tokenizer():
    import train_chat as T
    with np.load(os.path.join(ROOT, "slr_net_chat_ru_mini2.npz"), allow_pickle=False) as z:
        proto = z["tokenizer_proto"].tobytes()
    return T.ChatTokenizer(model_proto=proto)


@pytest.fixture
def one_thread():
    """Крошечной модели хватает одного потока; так проверка не тормозит, если процессор занят обучением."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def test_self_learn_grows_until_check_and_keeps_base(one_thread):
    """Полный цикл на крошечной модели: учитель отвечает, нейроны растут раундами (порог недостижим — растёт
    до предела раундов), учатся только новые нейроны, отчёт с самопроверками по раундам."""
    pytest.importorskip("sentencepiece")
    tok = _chat_tokenizer()
    torch.manual_seed(0)
    model = E.SLRNetLanguageModel(E.make_config("tiny", vocab_size=tok.vocab_size, memory_keys=0)).eval()
    old = {k: v.clone() for k, v in model.state_dict().items()}
    facts = S.make_facts(["зурмоп", "прермуф"], seed=0)
    cfg = S.GrowConfig(steps=3, k_init=2, max_rounds=2, batch=4, replay_batch=2, negative_words=3,
                       target=2.0, bf16=False, seed=1)
    rep = S.self_learn(model, tok, facts, cfg, data_dir=None, log=lambda *a, **k: None)
    assert rep["neurons_per_layer"] == 4 and len(rep["rounds"]) == 2
    assert [r["neurons_per_layer"] for r in rep["rounds"]] == [2, 4]
    assert all(0.0 <= r["self_check_heldout"] <= 1.0 and 0.0 <= r["keep"] <= 1.0 for r in rep["rounds"])
    d = model.config.d_model
    assert rep["new_params"] == len(S.mlp_sites(model)) * 4 * (2 * d + 1)
    new_state = model.state_dict()
    for k, v in old.items():
        k2 = k if k in new_state else k.replace(".mlp.", ".mlp.base.")
        assert torch.equal(new_state[k2], v), k
    assert not any(p.requires_grad for p in model.parameters())


def test_module_router_kinds_on_tiny_model(one_thread):
    """Все способы выбора модуля считаются; «по словам» включает модуль ровно для вопросов с выученным словом."""
    pytest.importorskip("sentencepiece")
    tok = _chat_tokenizer()
    torch.manual_seed(0)
    model = E.SLRNetLanguageModel(E.make_config("tiny", vocab_size=tok.vocab_size, memory_keys=0)).eval()
    facts = [S.Fact("зурмоп", "Зурмоп — это маленькая птица синего цвета.")]
    qs = ["Какого цвета зурмоп?", "Как тебя зовут?", "Что такое клыдел?"]
    assert O._find([5, 1, 2, 3, 1, 2], [1, 2]) == 4 and O._find([1, 2], [3]) == -1
    for kind in ("sentence", "token", "lexical", "hybrid"):
        mr = O.ModuleRouter(model, tok, kind=kind)
        if kind != "lexical":
            mr.set_center(["Привет!", "Сколько будет два плюс два?"])
        mr.add_module("m", facts)
        sc = mr.scores(qs)
        assert sc.shape == (3, 1) and np.isfinite(sc).all()
        if kind in ("lexical", "hybrid"):
            mr.threshold = 0.5 if kind == "lexical" else 2.0
            assert mr.route(qs) == [["m"], [], []]


# ------------------------------------------------------------------ облегчённое самообучение (как в браузере)

def test_sampled_logz_is_exact_when_sample_covers_the_rest():
    """Оценка log Σ exp по выборке совпадает с точной, если случайные слова — ровно все невыбранные."""
    import selflearn_tail as SL
    g = torch.Generator().manual_seed(0)
    V = 50
    logits = torch.randn(3, V, generator=g) * 3
    cand = torch.tensor([[4, 7, 4, 9], [0, 1, 2, 3], [10, 11, 12, 10]])
    for r in range(3):
        rest = torch.tensor([v for v in range(V) if v not in set(cand[r].tolist())])
        idx = torch.cat([cand[r], rest])
        lc = logits[r, idx][None]
        got = SL.sampled_logz(lc, cand[r:r + 1], rest, V)
        assert torch.allclose(got, torch.logsumexp(logits[r], 0)[None], atol=1e-5)


def test_feature_quantization_roundtrip():
    import selflearn_tail as SL
    X = torch.randn(20, 64) * torch.rand(20, 1) * 5
    q, s = SL.quantize_rows(X)
    assert q.dtype == np.int8 and s.dtype == np.float16
    Xr = SL.dequantize_rows(q, s)
    assert (Xr - X).abs().max(1).values.le(X.abs().max(1).values / 127 + 1e-3).all()


def test_self_learn_tail_on_tiny_model(one_thread):
    """Облегчённое самообучение целиком на крошечной модели: данные «не менять старое», рост нейронов только в
    последнем слое, учатся только новые нейроны, выученное ложится в GrownMLP последнего слоя."""
    pytest.importorskip("sentencepiece")
    import selflearn_tail as SL
    tok = _chat_tokenizer()
    torch.manual_seed(0)
    model = E.SLRNetLanguageModel(E.make_config("tiny", vocab_size=tok.vocab_size, memory_keys=0)).eval()
    old = {k: v.clone() for k, v in model.state_dict().items()}
    pack = SL.build_pack(model, tok, None, n_old=10, n_neg=40, log=lambda *a, **k: None)
    assert pack["selflearn.replay_h"].dtype == np.int8 and pack["selflearn.replay_neg"].all()
    facts = S.facts_from_text("Курдяпа — это маленькая птица синего цвета. Она живёт в лесу.")
    rep = SL.self_learn_tail(model, tok, facts, SL.load_pack(pack),
                             {"steps": 3, "k_init": 2, "max_rounds": 2, "batch": 8, "replay_batch": 8, "target": 2.0,
                              "keep_items": 20}, log=lambda *a, **k: None, tag="kurd")
    assert [r["neurons"] for r in rep["rounds"]] == [2, 4] and rep["new_params"] == 4 * (2 * model.config.d_model + 1)
    g = SL.tail_block(model).mlp
    assert isinstance(g, S.GrownMLP) and [c.tag for c in g.chunks] == ["kurd", "kurd"]
    assert list(S.grown_layers(model)) == [S.mlp_sites(model)[-1][0]]           # только последний слой
    new_state = model.state_dict()
    for k, v in old.items():
        k2 = k if k in new_state else k.replace(".mlp.", ".mlp.base.")
        assert torch.equal(new_state[k2], v), k


def test_near_misses_keep_ending_and_differ():
    """Похожие несуществующие слова: другая основа, то же окончание (падежная форма выученного слова — не «чужое»)."""
    import selflearn_tail as SL
    rng = random.Random(0)
    for w in ("курдяпа", "бармук", "залкута"):
        vs = SL.near_misses(w, 6, rng)
        assert len(vs) == 6 and len(set(vs)) == 6
        assert all(v != w and len(v) == len(w) and v[-2:] == w[-2:] for v in vs)
        assert all(sum(a != b for a, b in zip(v, w)) in (1, 2) for v in vs)
