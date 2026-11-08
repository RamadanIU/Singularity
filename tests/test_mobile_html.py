# -*- coding: utf-8 -*-
"""
Проверка мобильного чата в настоящем браузере (headless Chromium через Playwright).
Интернет на странице отключён — так заодно проверяется, что всё работает офлайн.
Запуск: pip install playwright && playwright install chromium
"""
import os
import sys

import numpy as np
import pytest

sync_api = pytest.importorskip("playwright.sync_api")

HTML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slr_net_mobile_chat.html")


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            # CHROMIUM_PATH — путь к уже установленному Chrome/Chromium, если не ставили его через playwright
            path = os.environ.get("CHROMIUM_PATH")
            br = p.chromium.launch(executable_path=path) if path else p.chromium.launch()
        except Exception as exc:  # браузер не установлен
            pytest.skip(f"Chromium недоступен: {exc}")
        yield br
        br.close()


def _open_page(browser):
    pg = browser.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.route("http*://**", lambda route: route.abort())       # офлайн-режим
    pg.goto("file://" + HTML)
    pg.wait_for_function("() => appState && appState.engine !== null && !appState.training", timeout=60000)
    pg.errors = errors
    return pg


@pytest.fixture(scope="module")
def page(browser):
    pg = _open_page(browser)
    yield pg
    pg.close()


@pytest.fixture()
def fresh_page(browser):
    """Отдельная страница: тесты загрузки файлов меняют корпус и не должны влиять на остальные."""
    pg = _open_page(browser)
    yield pg
    pg.close()


def test_page_loads_offline_without_js_errors(page):
    assert page.errors == []
    assert page.evaluate("appState.tokenizer.idToToken.length") > 100


@pytest.mark.parametrize("question,expected", [
    ("Сколько будет двадцать пять умножить на четыре минус пятнадцать?", "85."),
    ("(2+3)*4 = ?", "20."),
    ("Что было в 1941-1945 годах?", None),
    ("Встреча в 10:30?", None),
    ("Если сокол быстрее орла, а орёл быстрее воробья, кто самый медленный?", "Воробья."),
    ("Если Петя старше чем Вася, а Вася старше чем Коля, кто самый младший?", "Коля."),
    ("Если Петя старше Васи, а Коля старше Васи, кто самый старший?", "Нужно ещё одно сравнение."),
])
def test_logic_module_matches_python_version(page, question, expected):
    ans = page.evaluate("q => trySolveLogicOrMathJS(q)", question)
    if expected is None:
        assert ans is None
    else:
        assert ans is not None and ans.endswith(expected), ans


def test_tokenizer_does_not_confuse_unrelated_words(page):
    res = page.evaluate("""() => {
        const t = new MorphemeTokenizerJS();
        t.fit(["мне грустно поговори со мной", "первом месте"]);
        return [t.resolveId("погоде") === t.tokenToId.get("<unk>"), t.resolveId("первый") === t.tokenToId.get("первом")];
    }""")
    assert res == [True, True]


def test_js_gradients_match_numerical_derivative(page):
    worst = page.evaluate("""() => {
        const tok = new MorphemeTokenizerJS();
        const chains = ["привет ! как дела ? отлично , спасибо .", "почему небо синее ? атмосфера рассеивает свет ."];
        tok.fit(chains);
        const seqs = chains.map(c => tok.encode(c));
        const eng = new SLRNetBrowserEngine(tok.idToToken.length, 8, 4);
        eng.populateFactMemory(seqs, chains, tok);
        // Непрерывная версия сети (без квантования), чтобы производная существовала
        eng.quantizeB158 = w => ({ wTern: w, gamma: 1.0, sparsity: 0 });
        eng.addOnlyMatVec = (x, w, rows, cols, g) => {
            const out = new Float32Array(rows);
            for (let r = 0; r < rows; r++) { let a = 0; for (let c = 0; c < cols; c++) a += w[r * cols + c] * x[c]; out[r] = a * g; }
            return out;
        };
        eng.b_router[0] = 2.0;            // все токены идут в цикл мысли
        const seq = seqs[1];
        const hints = tok.computeComplexityHints(seq);
        const lossOf = () => {
            eng._qCache = null;           // веса меняли «вручную» — сбрасываем кэш квантования
            const L = seq.length - 1;
            const fwd = eng.forwardSequence(seq.slice(0, L), 1, 0);
            let loss = 0;
            for (let t = 0; t < L; t++) {
                const lg = fwd.logitsList[t]; let mx = -1e9; for (const v of lg) mx = Math.max(mx, v);
                let s = 0; for (const v of lg) s += Math.exp(v - mx);
                loss += -(lg[seq[t + 1]] - mx - Math.log(s)) / L;
                loss += 0.25 * (fwd.cProbList[t] - hints[t]) ** 2 / L;
            }
            return loss;
        };
        let grads = null;
        eng._adamwStep = g => { grads = g; };
        eng.trainSequenceStep(seq, hints, 0.0, 1);
        let worst = 0;
        const checks = [["w_out", 5], ["w_think", 7], ["w_ssm", 3], ["w_ctx1", 9], ["w_ctx2", 11], ["w_router", 2],
                        ["b_router", 0], ["memVals", 8 + 3], ["embed", seq[2] * 8 + 1], ["b_out", seq[3]]];
        for (const [name, idx] of checks) {
            const p = eng[name]; const old = p[idx]; const eps = 1e-2;
            p[idx] = old + eps; const lp = lossOf();
            p[idx] = old - eps; const lm = lossOf();
            p[idx] = old;
            const num = (lp - lm) / (2 * eps), ana = grads[name][idx];
            if (Math.abs(num) + Math.abs(ana) > 1e-5) worst = Math.max(worst, Math.abs(num - ana) / (Math.abs(num) + Math.abs(ana)));
        }
        return worst;
    }""")
    assert worst < 2e-2, worst


def test_chat_answers_from_memory_and_admits_unknown(page):
    out = page.evaluate("""() => {
        const a = appState.engine.generateReply("В чём метод Шерлока Холмса?", appState.tokenizer);
        const b = appState.engine.generateReply("Какая столица у государства Мадагаскар?", appState.tokenizer);
        return [a.replyText, b.epistemicMode];
    }""")
    assert "дедукции" in out[0].lower()
    assert out[1] == "honest_unknown"


def test_xray_mode_is_safe_from_html_injection(page):
    result = page.evaluate("""() => {
        appState.xrayEnabled = true;
        appendChatMessage("assistant", "x", { telemetry: [{ token: "<img src=x onerror=window.__pwned=1>", latentSteps: 2 }], elapsedMs: 1 });
        appState.xrayEnabled = false;
        const last = document.getElementById("chatMessagesContainer").lastElementChild;
        return [last.querySelector("img") === null, last.textContent.includes("<img")];
    }""")
    page.wait_for_timeout(100)
    assert result == [True, True]
    assert page.evaluate("window.__pwned === undefined")


# ---------------------------------------------------------------------------------------------
# Большие файлы субтитров: раньше файл на сотни МБ читался целиком, разбирался одним синхронным
# куском, вставлялся в textarea — страница «замерзала» без прогресс-бара и падала.
# ---------------------------------------------------------------------------------------------

# Текст как в больших дампах субтитров: реплики через тире и БЕЗ пустых строк между ними
DIALOGUE_NO_BLANK_LINES = """- А вы выбрали жаркий денек, мистер.
- Я всегда так делаю.
- Полный бак?
- Еще бы!
А у вас мощная тачка, как я погляжу.
- Могу продать ее вам.
- Нет, спасибо!
Вы знаете, не проживает тут поблизости парень по имени Кэмерон?
Бэн Кэмерон?
Конечно!
- Ты куда
идёшь так поздно?
- Домой.
"""

SRT_SAMPLE = """1
00:00:01,000 --> 00:00:02,500
<i>- Привет, как дела?</i>
- Отлично, спасибо.

2
00:00:03,000 --> 00:00:04,000
[музыка] Куда ты идёшь сегодня вечером?

3
00:00:05,000 --> 00:00:06,000
В кино, на новый фильм.

"""


def test_stream_parser_matches_python_and_does_not_glue_whole_file(page):
    sys.path.insert(0, os.path.dirname(HTML))
    import slr_net_trainer_chat as T
    for text in (DIALOGUE_NO_BLANK_LINES * 40, SRT_SAMPLE * 5):
        js = page.evaluate("t => parseSubtitleOrTextCorpusJS(t)", text)
        assert js == T.parse_subtitle_or_text_corpus(text)
    chains = page.evaluate("t => parseSubtitleOrTextCorpusJS(t)", DIALOGUE_NO_BLANK_LINES * 40)
    # Раньше после первой реплики с тире весь остаток текста склеивался в одну «реплику»
    assert len(chains) >= 150 and max(len(c) for c in chains) < 200
    srt = page.evaluate("t => parseSubtitleOrTextCorpusJS(t)", SRT_SAMPLE)
    assert srt[0] == "Привет, как дела? Отлично, спасибо."
    assert not any("-->" in c or "<i>" in c or "музыка" in c for c in srt)


def _start_heartbeat(pg):
    pg.evaluate("""() => {
        window.__maxGap = 0; window.__last = performance.now();
        window.__beat = setInterval(() => {
            const n = performance.now();
            window.__maxGap = Math.max(window.__maxGap, n - window.__last);
            window.__last = n;
        }, 20);
    }""")


def test_large_file_is_streamed_with_progress_and_page_stays_responsive(fresh_page, tmp_path):
    pg = fresh_page
    big = tmp_path / "big_subtitles.txt"
    with open(big, "w", encoding="utf-8") as f:
        for i in range(25000):     # ~13 МБ, без пустых строк
            f.write(DIALOGUE_NO_BLANK_LINES.replace("Кэмерон", f"Кэмерон-{i % 997}"))
    pg.evaluate("() => { BROWSER_LIMITS.maxFileChains = 80; }")   # чтобы тест обучался быстро
    _start_heartbeat(pg)
    pg.click("#navTabTrain")
    pg.set_input_files("#subtitleFileInput", str(big))
    # Прогресс появляется сразу, а не после многоминутной «заморозки»
    pg.wait_for_function("() => !document.getElementById('trainProgressBox').classList.contains('hidden')"
                         " && document.getElementById('trainProgressPhase').textContent.includes('Чтение файла')",
                         timeout=5000)
    pg.wait_for_function("() => !appState.training && document.getElementById('trainProgressBox').classList.contains('hidden')",
                         timeout=240000)
    gap, note, n_mem, n_starter, text_len = pg.evaluate("""() => [window.__maxGap,
        document.getElementById('trainResultNote').textContent, appState.engine.questionPrompts.length,
        starterChainCount(), document.getElementById('corpusTextarea').value.length]""")
    assert pg.errors == []
    assert gap < 2000, f"главный поток был занят {gap:.0f} мс подряд"
    assert "случайная выборка 80" in note and "кодировка utf-8" in note
    assert n_mem <= n_starter + 80                # стартовый корпус + выборка, а не все 75 000 диалогов
    assert text_len < 100000                      # в поле ввода — выборка, а не 13 МБ текста
    reply = pg.evaluate("() => appState.engine.generateReply('В чём метод Шерлока Холмса?', appState.tokenizer).replyText")
    assert "дедукц" in reply.lower()              # стартовые знания при этом не потерялись


def test_cp1251_subtitles_are_decoded(fresh_page, tmp_path):
    pg = fresh_page
    srt = tmp_path / "old_windows.srt"
    srt.write_bytes(SRT_SAMPLE.encode("cp1251"))
    pg.click("#navTabTrain")
    pg.set_input_files("#subtitleFileInput", str(srt))
    pg.wait_for_function("() => document.getElementById('trainResultNote').textContent.includes('old_windows.srt')"
                         " && !appState.training", timeout=60000)
    corpus, note = pg.evaluate("() => [appState.corpusText, document.getElementById('trainResultNote').textContent]")
    assert "Привет, как дела? Отлично, спасибо." in corpus
    assert "windows-1251" in note


def test_cancel_button_stops_long_training(fresh_page):
    pg = fresh_page
    pg.click("#navTabTrain")
    pg.evaluate("() => { window.__done = null; trainEngineFromCorpus(appState.corpusText, 500).then(ok => window.__done = ok); }")
    pg.wait_for_function("() => document.getElementById('trainProgressPhase').textContent.startsWith('Обучение')",
                         timeout=30000)
    pg.click("#trainCancelBtn")
    pg.wait_for_function("() => window.__done !== null", timeout=15000)
    assert pg.evaluate("() => [window.__done, appState.lastTrainStopped, appState.training]") == [True, True, False]
    assert pg.evaluate("() => document.getElementById('globalProgress').classList.contains('hidden')")


# ------------------------------------------------------------------ режим mini: обученная LLM в браузере
ROOT = os.path.dirname(HTML)
MINI_NPZ = os.path.join(ROOT, "slr_net_subtitles_ru_mini.npz")
CHAT_NPZ = os.path.join(ROOT, "slr_net_chat_ru_mini3.npz")          # собеседник по умолчанию (mini3 = mini2 + самообучение)
PREV_CHAT_NPZ = os.path.join(ROOT, "slr_net_chat_ru_mini2.npz")     # прототип mini2 (без самообучения)
OLD_CHAT_NPZ = os.path.join(ROOT, "slr_net_chat_ru_mini.npz")       # прежний собеседник (mini)
MINI_DIALOGUE = ("- Привет! Как дела?\n- Хорошо, спасибо. А у тебя?\n- Тоже неплохо. Куда ты идёшь?\n"
                 "- Домой, уже поздно.\nДоктор, что с ним?\n- С ним всё будет хорошо.\n")
MINI_STRINGS = ["Привет! Как дела?", "  много   пробелов  ", "Ёлка и ёж", "№ 5… «цитата» — тире", "Hello, world 1999!",
                "emoji 😀 тест", "\tтаб", "", "ДА НЕТ", "1234567890", "x²+y² ½", "Café naïve", "日本語", "- Что?",
                "А-а-а!!!", "привет\nмир\n", "Мне было 17.", "Ты можешь взять их на 40%.", "O'Нил, сэр!"]


def _load_mini(pg):
    pg.click("#modeMiniBtn")
    pg.set_input_files("#miniFileInput", MINI_NPZ)
    pg.wait_for_function("() => miniState.bundle !== null && !miniState.loading", timeout=60000)


@pytest.fixture(scope="module")
def mini_page(browser):
    if not os.path.exists(MINI_NPZ):
        pytest.skip("нет файла весов slr_net_subtitles_ru_mini.npz")
    pg = _open_page(browser)
    _load_mini(pg)
    yield pg
    pg.close()


def test_mini_in_browser_matches_pytorch(mini_page):
    """JS-движок mini даёт те же токены и те же логиты, что slr_net_engine.py (PyTorch) по тому же файлу весов."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    import base64

    import numpy as np
    sys.path.insert(0, ROOT)
    import train_lm

    model, tok = train_lm.load_model(MINI_NPZ, torch.device("cpu"))
    ids = tok.encode(MINI_DIALOGUE)
    cache = model.init_cache(1)
    with torch.no_grad():
        ref = torch.stack([model.step(torch.tensor([t]), cache)[0] for t in ids]).float().numpy()

    js = mini_page.evaluate("""([strings, ids]) => {
        const { model, tokenizer } = miniState.bundle;
        const st = model.newState(), lg = new Float32Array(model.V), all = new Float32Array(ids.length * model.V);
        const depths = ids.map((t, j) => { const d = model.step(t, st, lg); all.set(lg, j * model.V); return d; });
        let bin = ""; const u8 = new Uint8Array(all.buffer);
        for (let i = 0; i < u8.length; i += 0x8000) bin += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
        return { enc: strings.map(s => tokenizer.encode(s)), logits: btoa(bin), depths };
    }""", [MINI_STRINGS, ids])

    assert js["enc"] == [tok.encode(s) for s in MINI_STRINGS]            # токенизатор = SentencePiece
    assert mini_page.evaluate("(e) => e.map(x => miniState.bundle.tokenizer.decode(x))", js["enc"]) == \
        [tok.decode(e) for e in js["enc"]]
    got = np.frombuffer(base64.b64decode(js["logits"]), dtype=np.float32).reshape(ref.shape)
    assert (got.argmax(1) == ref.argmax(1)).all()                          # тот же самый вероятный токен везде
    assert np.abs(got - ref).max() < 0.25 and np.abs(got - ref).mean() < 0.01, np.abs(got - ref).max()
    assert sum(js["depths"]) > 0 and max(js["depths"]) <= 3                # роутер ядра мышления работает


def test_mini_chat_replies_without_freezing_and_can_stop(mini_page):
    pg = mini_page
    pg.evaluate("() => { miniState.rng = SLRMini.mulberry32(1); miniState.replyLines = 2; }")
    _start_heartbeat(pg)
    n0 = pg.evaluate("() => document.querySelectorAll('[data-mini-reply] .mini-meta').length")
    pg.fill("#chatInputField", "Привет! Как дела?")
    pg.click("#sendMsgBtn")
    pg.wait_for_function(f"() => !miniState.busy && document.querySelectorAll('[data-mini-reply] .mini-meta').length === {n0 + 1}",
                         timeout=60000)
    reply = pg.evaluate("() => [...document.querySelectorAll('[data-mini-reply] .mini-reply-body')].pop().innerText")
    letters = [c for c in reply if c.isalpha()]
    assert len(letters) >= 3 and sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in letters) > 0.8 * len(letters), reply
    assert 1 <= len([ln for ln in reply.split("\n") if ln.strip()]) <= 2
    assert pg.evaluate("() => window.__maxGap") < 1000, "главный поток был занят слишком долго"
    assert pg.evaluate("() => miniState.lastSpeed") > 5                     # токенов в секунду

    # Кнопка «Стоп» прерывает длинный ответ
    pg.evaluate("() => { miniState.replyLines = 99; MINI_LIMITS.maxNewTokens = 180; }")
    pg.fill("#chatInputField", "Расскажи мне, что случилось.")
    pg.click("#sendMsgBtn")
    pg.wait_for_function("() => miniState.busy && document.getElementById('sendMsgBtn').innerText.includes('Стоп')", timeout=10000)
    pg.click("#sendMsgBtn")
    pg.wait_for_function("() => !miniState.busy", timeout=10000)
    assert "остановлено" in pg.evaluate("() => [...document.querySelectorAll('[data-mini-reply] .mini-meta')].pop().innerText")
    pg.evaluate("() => { miniState.replyLines = 2; MINI_LIMITS.maxNewTokens = 64; }")

    # «Рентген»: видно, сколько шагов ядра мышления получил каждый кусочек слова
    pg.click("#xrayToggleBtn")
    pg.fill("#chatInputField", "Кто здесь?")
    pg.click("#sendMsgBtn")
    pg.wait_for_function("() => !miniState.busy && [...document.querySelectorAll('[data-mini-reply]')].pop().querySelector('.mini-meta')",
                         timeout=60000)
    sups = pg.evaluate("() => [...[...document.querySelectorAll('[data-mini-reply]')].pop().querySelectorAll('sup')].map(s => s.textContent)")
    pg.click("#xrayToggleBtn")
    assert sups and all(s in ("[1т]", "[2т]", "[3т]") for s in sups)
    assert pg.errors == []


def test_switching_back_to_regular_model_keeps_old_behaviour(mini_page):
    pg = mini_page
    pg.click("#modeRegularBtn")
    assert pg.evaluate("() => appState.modelMode") == "regular"
    assert pg.evaluate("() => document.getElementById('miniPanel').classList.contains('hidden')")
    pg.fill("#chatInputField", "В чём метод Шерлока Холмса?")
    pg.click("#sendMsgBtn")
    last = pg.evaluate("() => [...document.querySelectorAll('#chatMessagesContainer > div')].pop().innerText")
    assert "дедукц" in last.lower()
    pg.click("#modeMiniBtn")                                               # веса остаются загруженными
    assert pg.evaluate("() => miniState.bundle !== null && appState.modelMode === 'mini'")


def test_mini_weights_are_cached_and_mode_is_remembered(browser):
    if not os.path.exists(MINI_NPZ):
        pytest.skip("нет файла весов slr_net_subtitles_ru_mini.npz")
    pg = _open_page(browser)
    try:
        _load_mini(pg)
        pg.reload()
        pg.wait_for_function("() => miniState.bundle !== null", timeout=60000)
        assert pg.evaluate("() => [appState.modelMode, miniState.source]") == ["mini", "из памяти браузера"]
        pg.click("#miniForgetBtn")
        pg.wait_for_function("() => miniState.bundle === null", timeout=10000)
        pg.reload()
        pg.wait_for_function("() => appState.engine !== null", timeout=60000)
        pg.wait_for_timeout(500)
        assert pg.evaluate("() => miniState.bundle === null && !document.getElementById('miniLoadBox').classList.contains('hidden')")
        assert pg.errors == []
    finally:
        pg.close()


def test_mini_download_button_fetches_weights_from_server(browser):
    """Страница, открытая с сервера (например, GitHub Pages), скачивает веса собеседника, лежащие рядом с ней."""
    if not os.path.exists(CHAT_NPZ):
        pytest.skip("нет файла весов slr_net_chat_ru_mini3.npz")
    import functools
    import http.server
    import threading

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=ROOT))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    pg = browser.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.route("http*://**", lambda route: route.continue_() if "127.0.0.1" in route.request.url else route.abort())
    try:
        pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/slr_net_mobile_chat.html")
        pg.wait_for_function("() => appState.engine !== null", timeout=60000)
        pg.click("#modeMiniBtn")
        pg.evaluate("""() => { window.__texts = new Set();
            new MutationObserver(() => window.__texts.add(document.getElementById('miniLoadText').textContent))
              .observe(document.getElementById('miniLoadText'), { childList: true, characterData: true, subtree: true }); }""")
        pg.click("#miniDownloadBtn")
        pg.wait_for_function("() => miniState.bundle !== null && !miniState.loading", timeout=60000)
        assert pg.evaluate("() => miniState.source") == "с сервера"
        assert pg.evaluate("() => miniState.bundle.chat !== null")              # по умолчанию — собеседник
        assert pg.evaluate("() => [...window.__texts].some(t => t.startsWith('Скачано'))")
        assert errors == []
    finally:
        pg.close()
        srv.shutdown()


# ------------------------------------------------------------------ mini-собеседник (train_chat.py)

@pytest.fixture(scope="module")
def chat_page(browser):
    if not os.path.exists(CHAT_NPZ):
        pytest.skip("нет файла весов slr_net_chat_ru_mini3.npz")
    pg = _open_page(browser)
    pg.click("#modeMiniBtn")
    pg.set_input_files("#miniFileInput", CHAT_NPZ)
    pg.wait_for_function("() => miniState.bundle !== null && !miniState.loading", timeout=60000)
    yield pg
    pg.close()


CHAT_PROMPTS = [
    [("user", "Привет! Как тебя зовут?")],
    [("user", "Меня зовут Всеволод."), ("bot", "Приятно познакомиться, Всеволод!"), ("user", "Как меня зовут?")],
    [("user", "Какая столица Франции?")],
    [("user", "Hello!")],
]


def test_chat_model_in_browser_matches_pytorch(chat_page):
    """Жадный ответ собеседника в браузере совпадает с PyTorch токен в токен (тот же формат диалога)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    sys.path.insert(0, ROOT)
    import train_chat

    model, tok, chat = train_chat.load_chat_model(CHAT_NPZ, torch.device("cpu"))
    assert chat
    assert chat_page.evaluate("() => [miniState.bundle.chat.userId, miniState.bundle.chat.botId]") == [tok.user_id, tok.bot_id]
    for turns in CHAT_PROMPTS:
        ctx = train_chat.prompt_ids(tok, turns)
        ref = train_chat.generate_replies(model, tok, ctx, n=1, max_new_tokens=24, temperature=0,
                                          repetition_penalty=1.0)[0]
        js = chat_page.evaluate("""([ctx, maxNew]) => {
            const { model, tokenizer: tok, chat } = miniState.bundle;
            const st = model.newState(), lg = new Float32Array(model.V);
            for (const t of ctx) model.step(t, st, lg);
            const banned = [tok.eosId, tok.bosId, tok.unkId, chat.userId, chat.botId], out = [];
            for (let n = 0; n < maxNew; n++) {
                const t = SLRMini.sampleToken(lg, { temperature: 0, repetitionPenalty: 1,
                                                    banned: n === 0 ? [...banned, tok.nlId] : banned });
                out.push(t);
                if (t === tok.nlId) break;
                model.step(t, st, lg);
            }
            return out;
        }""", [ctx, 24])
        assert js == ref, (turns, tok.decode(js), tok.decode(ref))


def test_current_chat_weights_need_no_update_and_long_replies_fit(chat_page):
    """Страница знает, какие веса собеседника актуальны (иначе предлагает «Обновить»), и даёт ответу 64 токена:
    рассуждение «Рассуждаю: … Значит, …» длиннее прежних 48."""
    with np.load(CHAT_NPZ) as z:
        step, best = int(z["meta.step"][0]), float(z["meta.best_val"][0])
    assert chat_page.evaluate("() => [MINI_CHAT_VERSION.step, MINI_CHAT_VERSION.bestVal]") == [step, pytest.approx(best, abs=1e-3)]
    assert chat_page.evaluate("() => miniChatOutdated(miniState.bundle)") is False
    assert chat_page.evaluate("() => document.getElementById('miniUpdateBox').classList.contains('hidden')")
    assert chat_page.evaluate("() => MINI_LIMITS.chatMaxNewTokens") >= 64


CONTEXT_PROMPTS = [                     # навыки «несколько вопросов в одном сообщении» и «контекст»
    [("user", "Как тебя зовут? Сколько будет 3 + 4? Какая столица Франции?")],
    [("user", "Какой твой любимый фильм?"), ("bot", "Мне нравится «Иван Васильевич меняет профессию»."),
     ("user", "А что в нём интересного?")],
    [("user", "Я сейчас читаю книгу «Тёмная вода». Она про водолаза, который нашёл затонувший корабль."),
     ("bot", "Звучит интересно!"), ("user", "Помнишь, о чём она?")],
]


def test_previous_mini2_weights_offer_update(chat_page):
    """У кого в браузере сохранена прежняя mini2 (до навыков «несколько вопросов» и «контекст»: шаг 2 700,
    ошибка 1.1897) или нынешняя mini2 без самообучения, тому страница предлагает «⬇ Обновить»; mini3 — актуальна."""
    sl = "{meta: {version: 1}}"
    assert chat_page.evaluate(f"() => miniChatOutdated({{chat: true, selflearn: {sl}, meta: {{step: 2700, bestVal: 1.1897}}}})") is True
    with np.load(CHAT_NPZ) as z:
        step, best = int(z["meta.step"][0]), float(z["meta.best_val"][0])
    assert chat_page.evaluate(f"() => miniChatOutdated({{chat: true, selflearn: {sl}, meta: {{step: {step}, bestVal: {best}}}}})") is False
    assert chat_page.evaluate(f"() => miniChatOutdated({{chat: true, meta: {{step: {step}, bestVal: {best}}}}})") is True


def test_context_skills_in_browser_match_pytorch(chat_page):
    """Ответы на несколько вопросов и вопросы-продолжения в браузере совпадают с PyTorch. Расхождение допустимо
    только на «ничьей»: PyTorch считает оба токена почти равновероятными (порядок сложения float32 в JS и PyTorch
    разный), после чего ответы просто идут разными, одинаково возможными путями."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    sys.path.insert(0, ROOT)
    import train_chat
    model, tok, chat = train_chat.load_chat_model(CHAT_NPZ, torch.device("cpu"))
    for turns in CONTEXT_PROMPTS:
        ctx = train_chat.prompt_ids(tok, turns)
        ref = train_chat.generate_replies(model, tok, ctx, n=1, max_new_tokens=40, temperature=0,
                                          repetition_penalty=1.0)[0]
        js = chat_page.evaluate("""([ctx, maxNew]) => {
            const { model, tokenizer: tok, chat } = miniState.bundle;
            const st = model.newState(), lg = new Float32Array(model.V);
            for (const t of ctx) model.step(t, st, lg);
            const banned = [tok.eosId, tok.bosId, tok.unkId, chat.userId, chat.botId], out = [];
            for (let n = 0; n < maxNew; n++) {
                const t = SLRMini.sampleToken(lg, { temperature: 0, repetitionPenalty: 1,
                                                    banned: n === 0 ? [...banned, tok.nlId] : banned });
                out.push(t);
                if (t === tok.nlId) break;
                model.step(t, st, lg);
            }
            return out;
        }""", [ctx, 40])
        if js != ref:
            i = next(k for k, (a, b) in enumerate(zip(js, ref)) if a != b)
            with torch.no_grad():
                logits = model(torch.tensor([ctx + ref[:i]]))[0][0, -1].float()
            gap = float(logits[ref[i]] - logits[js[i]])
            assert 0 <= gap < 0.1, (turns, i, gap, tok.decode(js), tok.decode(ref))
        else:
            assert tok.decode(js).strip(), turns


def test_old_chat_weights_offer_update(browser):
    """Прежний собеседник (mini, с памятью фактов и эмбеддингом float16) по-прежнему работает в браузере,
    но страница предлагает обновить его до mini2."""
    if not os.path.exists(OLD_CHAT_NPZ):
        pytest.skip("нет файла весов slr_net_chat_ru_mini.npz")
    pg = _open_page(browser)
    try:
        pg.click("#modeMiniBtn")
        pg.set_input_files("#miniFileInput", OLD_CHAT_NPZ)
        pg.wait_for_function("() => miniState.bundle !== null && !miniState.loading", timeout=60000)
        assert pg.evaluate("() => miniChatOutdated(miniState.bundle)") is True
        assert pg.evaluate("() => !document.getElementById('miniUpdateBox').classList.contains('hidden')")
        assert pg.errors == []
    finally:
        pg.close()


def test_chat_mode_keeps_dialogue_and_can_start_over(chat_page):
    pg = chat_page
    assert pg.evaluate("() => !document.getElementById('miniChatPromptsBar').classList.contains('hidden')")
    assert pg.evaluate("() => document.getElementById('miniQuickPromptsBar').classList.contains('hidden')")
    assert pg.evaluate("() => document.getElementById('miniLinesLabel').classList.contains('hidden')")
    assert pg.evaluate("() => document.getElementById('miniUpgradeBox').classList.contains('hidden')")
    pg.evaluate("() => { miniResetContext(); miniState.temperature = 0; }")      # жадный выбор: ответ предсказуем
    _start_heartbeat(pg)

    def say(text):
        n0 = pg.evaluate("() => document.querySelectorAll('[data-mini-reply] .mini-meta').length")
        pg.fill("#chatInputField", text)
        pg.click("#sendMsgBtn")
        pg.wait_for_function(f"() => !miniState.busy && document.querySelectorAll('[data-mini-reply] .mini-meta').length === {n0 + 1}",
                             timeout=60000)
        return pg.evaluate("() => [...document.querySelectorAll('[data-mini-reply] .mini-reply-body')].pop().innerText")

    say("Меня зовут Тимофей.")                                               # имени нет в обучающих данных
    reply = say("Как меня зовут?")
    assert "Тимофей" in reply, reply                                        # помнит, что сказали раньше
    assert len([ln for ln in reply.split("\n") if ln.strip()]) == 1         # ответ — одна реплика
    ids = pg.evaluate("() => miniState.ctxIds")
    user_id, bot_id, nl = pg.evaluate("() => [miniState.bundle.chat.userId, miniState.bundle.chat.botId, miniState.bundle.tokenizer.nlId]")
    assert ids[0] == user_id and ids.count(user_id) == 2 and ids.count(bot_id) == 2 and ids[-1] == nl
    assert pg.evaluate("() => window.__maxGap") < 1000, "главный поток был занят слишком долго"

    pg.click("#miniNewChatBtn")                                              # новый разговор: всё забыто
    assert pg.evaluate("() => miniState.ctxIds.length") == 0
    pg.evaluate("() => { miniState.temperature = 0.8; }")
    assert pg.errors == []


def test_old_subtitle_model_still_works_and_offers_upgrade(browser):
    if not os.path.exists(MINI_NPZ):
        pytest.skip("нет файла весов slr_net_subtitles_ru_mini.npz")
    pg = _open_page(browser)
    try:
        _load_mini(pg)
        assert pg.evaluate("() => miniState.bundle.chat") is None
        assert pg.evaluate("() => !document.getElementById('miniUpgradeBox').classList.contains('hidden')")
        assert pg.evaluate("() => !document.getElementById('miniQuickPromptsBar').classList.contains('hidden')")
        assert pg.evaluate("() => document.getElementById('miniChatPromptsBar').classList.contains('hidden')")
        assert pg.errors == []
    finally:
        pg.close()


# ------------------------------------------------------------------ новые модели: без памяти, сжатый эмбеддинг

@pytest.mark.parametrize("bits", [4, 6])
def test_model_without_memory_and_compressed_embedding_matches_pytorch(browser, tmp_path, bits):
    """Модель нового вида (memory_keys = 0, эмбеддинг сжат до 4 или 6 бит, у каждой головы внимания свои
    ключи, маленькое окно — чтобы кольцевой буфер ключей в браузере успел провернуться) считается в браузере
    так же, как в PyTorch: распаковка эмбеддинга, выход по сжатой таблице, ядро мышления без памяти."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    if not os.path.exists(MINI_NPZ):
        pytest.skip("нет файла весов slr_net_subtitles_ru_mini.npz (нужен токенизатор)")
    import base64
    import json
    sys.path.insert(0, ROOT)
    import slr_net_engine as E
    import train_lm

    torch.manual_seed(bits)
    cfg = E.make_config("tiny", vocab_size=8192, d_model=64, n_heads=4, n_kv_heads=4, window_size=12,
                        ssm_headdim=32, d_state=16, memory_keys=0, embed_bits=bits, embed_group=32)
    model = E.SLRNetLanguageModel(cfg).eval()
    with torch.no_grad():                                   # «обученные» веса с разбросом, роутер пропускает токены
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
        model.core.router.bias.fill_(0.5)
    path = tmp_path / f"new_{bits}.npz"
    with np.load(MINI_NPZ) as z:
        proto = z["tokenizer_proto"]
    np.savez_compressed(path, config=json.dumps(cfg.to_dict()), tokenizer_proto=proto, **model.export_packed())
    ref_model, tok = train_lm.load_model(str(path), torch.device("cpu"))
    assert ref_model.core.memory is None
    ids = tok.encode(MINI_DIALOGUE)[:40]
    cache = ref_model.init_cache(1)
    with torch.no_grad():
        ref = torch.stack([ref_model.step(torch.tensor([t]), cache)[0] for t in ids]).float().numpy()

    pg = _open_page(browser)
    try:
        pg.click("#modeMiniBtn")
        pg.set_input_files("#miniFileInput", str(path))
        pg.wait_for_function("() => miniState.bundle !== null && !miniState.loading", timeout=60000)
        js = pg.evaluate("""(ids) => {
            const { model } = miniState.bundle;
            const st = model.newState(), lg = new Float32Array(model.V), all = new Float32Array(ids.length * model.V);
            const depths = ids.map((t, j) => { const d = model.step(t, st, lg); all.set(lg, j * model.V); return d; });
            let bin = ""; const u8 = new Uint8Array(all.buffer);
            for (let i = 0; i < u8.length; i += 0x8000) bin += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
            return { logits: btoa(bin), depths, memory: model.memory, emb: Array.from(model.emb.subarray(0, 64)) };
        }""", ids)
        assert js["memory"] is None
        assert np.array_equal(np.float32(js["emb"]), ref_model.tok_embeddings.weight[0].detach().numpy())
        got = np.frombuffer(base64.b64decode(js["logits"]), dtype=np.float32).reshape(ref.shape)
        assert (got.argmax(1) == ref.argmax(1)).mean() > 0.95
        assert np.abs(got - ref).max() < 0.05 * np.abs(ref).max() and np.abs(got - ref).mean() < 0.01
        assert sum(js["depths"]) > 0
        assert pg.errors == []
    finally:
        pg.close()


# ------------------------------------------------------------------ самообучение mini3 (selflearn_tail.py)

def _b64_f32(pg, expr):
    import base64
    return np.frombuffer(base64.b64decode(pg.evaluate(f"""() => {{
        const a = {expr}; let bin = ""; const u8 = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
        for (let i = 0; i < u8.length; i += 0x8000) bin += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
        return btoa(bin); }}""")), dtype=np.float32)


def test_mini3_file_is_mini2_weights_plus_selflearn_data():
    """mini3 — те же веса, что у mini2 (собеседник не менялся), плюс данные самообучения и их описание."""
    if not (os.path.exists(CHAT_NPZ) and os.path.exists(PREV_CHAT_NPZ)):
        pytest.skip("нет файлов mini2/mini3")
    import json
    sys.path.insert(0, ROOT)
    import selflearn_tail as SL
    with np.load(CHAT_NPZ) as z3, np.load(PREV_CHAT_NPZ) as z2:
        assert set(z2.files) <= set(z3.files)
        for k in z2.files:
            assert np.array_equal(z2[k], z3[k]), k
        extra = sorted(set(z3.files) - set(z2.files))
        assert extra == sorted(["meta.selflearn", "selflearn.replay_h", "selflearn.replay_h_scale", "selflearn.replay_y",
                                "selflearn.replay_y_scale", "selflearn.replay_top_idx", "selflearn.replay_top_lp",
                                "selflearn.replay_neg"])
        meta = json.loads(str(z3["meta.selflearn"]))
        n, d = z3["selflearn.replay_h"].shape
        assert meta["version"] == 1 and meta["site"] == "decoder.3" and meta["settings"] == SL.SETTINGS
        assert d == 512 and z3["selflearn.replay_top_idx"].shape == (n, 32) and 0 < z3["selflearn.replay_neg"].sum() < n


def test_prototype_mini2_still_works_and_offers_mini3(browser):
    """mini2 осталась прототипом: работает, но учиться не умеет, и страница предлагает обновиться до mini3."""
    if not os.path.exists(PREV_CHAT_NPZ):
        pytest.skip("нет файла весов slr_net_chat_ru_mini2.npz")
    pg = _open_page(browser)
    try:
        pg.click("#modeMiniBtn")
        pg.set_input_files("#miniFileInput", PREV_CHAT_NPZ)
        pg.wait_for_function("() => miniState.bundle !== null && !miniState.loading", timeout=60000)
        assert pg.evaluate("() => miniState.bundle.selflearn") is None
        assert pg.evaluate("() => miniChatOutdated(miniState.bundle)") is True
        assert pg.evaluate("() => !document.getElementById('miniUpdateBox').classList.contains('hidden')")
        assert pg.evaluate("() => document.getElementById('miniLearnBox').classList.contains('hidden')")
        assert pg.evaluate("async () => await miniLearn('Курдяпа — это птица.')") is None
        assert pg.errors == []
    finally:
        pg.close()


SELF_TEXTS = [
    "Курдяпа — это маленькая птица синего цвета. Она живёт в лесу.",
    "Бармук — это большой инструмент серого цвета. Он лежит в гараже.  Глимпа — это напиток.\nЕго пьют летом.",
    "Слово без определения. Зурмоп - это огромный овощ красного цвета, растёт на грядке.",
    "Ничего определённого тут нет.",
]


def test_selflearn_question_tool_matches_python(chat_page):
    """Новые слова, вопросы о них и выбор модуля «по словам» в браузере — те же, что в self_grow.py/orchestrator.py."""
    sys.path.insert(0, ROOT)
    pytest.importorskip("torch")
    import orchestrator as O
    import self_grow as S
    for text in SELF_TEXTS:
        js = chat_page.evaluate("(t) => SLRSelf.factsFromText(t).map(f => [f.word, f.text, SLRSelf.textQuestions(f.word, f.text)])", text)
        py = [[f.word, f.text, [list(q) for q in S.text_questions(f.word, f.text)]] for f in S.facts_from_text(" ".join(text.split()))]
        assert js == py, text
    cases = [("Напомни, что такое курдяпа?", ["курдяпа"]), ("Какого цвета курдяпы?", ["курдяпа"]),
             ("Что такое кошка?", ["курдяпа"]), ("А бармук где лежит?", ["курдяпа", "бармук"]), ("Скурдяпа", ["курдяпа"])]
    for text, words in cases:
        assert chat_page.evaluate("([t, w]) => SLRSelf.mentions(t, w)", [text, words]) == O.lexical_route(text, words), text
    vs = chat_page.evaluate("() => SLRSelf.nearMisses('курдяпа', 6, SLRMini.mulberry32(3))")
    assert len(set(vs)) == 6 and all(v != "курдяпа" and v[-2:] == "па" and len(v) == 7 for v in vs)


def test_selflearn_features_and_neurons_match_pytorch(chat_page):
    """Признаки последнего слоя, которые страница собирает для обучения, и ответ модели с новыми нейронами — те же,
    что в PyTorch (selflearn_tail.capture_tail и GrownMLP последнего слоя)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    sys.path.insert(0, ROOT)
    import self_grow as S
    import selflearn_tail as SL
    import train_chat
    model, tok, _ = train_chat.load_chat_model(CHAT_NPZ, torch.device("cpu"))
    q, a = "Что такое курдяпа?", "Курдяпа — это маленькая птица синего цвета."
    q_ids, a_ids = tok.encode_line(q), tok.encode_line(a)
    H, Y, tg, _ = SL.capture_tail(model, tok, [train_chat.encode_dialogue(tok, [("user", q), ("bot", a)])])
    js = chat_page.evaluate("async ([q, a]) => { const c = await SLRSelf.captureReply(miniState.bundle, q, a); window.__cap = c; return c.targets; }",
                            [q_ids, a_ids])
    assert js == tg.tolist()
    h = _b64_f32(chat_page, "window.__cap.H").reshape(H.shape)
    y = _b64_f32(chat_page, "window.__cap.Y").reshape(Y.shape)
    assert np.abs(h - H.numpy()).max() < 0.05 * np.abs(H.numpy()).max()
    assert np.abs(y - Y.numpy()).max() < 0.05 * np.abs(Y.numpy()).max()
    # случайный «выученный» кусок нейронов — в PyTorch и в браузере
    g = torch.Generator().manual_seed(0)
    k, D = 6, model.config.d_model
    up, bias, down = torch.randn(k, D, generator=g) * 0.3, torch.randn(k, generator=g) * 0.1, torch.randn(D, k, generator=g) * 0.05
    block = SL.tail_block(model)
    block.mlp = S.GrownMLP(block.mlp)
    ch = block.mlp.grow(k, 0.0, seed=0)
    with torch.no_grad():
        ch.up.copy_(up); ch.bias.copy_(bias); ch.down.copy_(down)
    ctx = train_chat.prompt_ids(tok, [("user", q)])
    with torch.no_grad():
        cache = model.init_cache(1)
        ref = torch.stack([model.step(torch.tensor([t]), cache)[0] for t in ctx]).float().numpy()
    got = chat_page.evaluate("""([ctx, up, bias, down, k]) => {
        const m = miniState.bundle.model;
        m.tail.extra = { modules: [{ id: "t", chunks: [{ k, up: Float32Array.from(up), bias: Float32Array.from(bias), down: Float32Array.from(down) }] }], active: null };
        const st = m.newState(), lg = new Float32Array(m.V), out = [];
        try { for (const t of ctx) { m.step(t, st, lg); out.push(Array.from(lg)); } }
        finally { miniAttachModules(); }
        return out;
    }""", [ctx, up.flatten().tolist(), bias.tolist(), down.flatten().tolist(), k])
    got = np.array(got, dtype=np.float32)
    assert (got.argmax(1) == ref.argmax(1)).all()
    # кусок нарочно сильный (сдвигает логиты на единицы) — допуск чуть шире, чем у модели без него
    assert np.abs(got - ref).max() < 0.25 and np.abs(got - ref).mean() < 0.03


def test_selflearn_train_step_matches_python(chat_page):
    """Шаг обучения новых нейронов в браузере (ручной обратный проход, ошибка по выборке словаря, Adam) даёт те же
    веса, что эталон на PyTorch с автоматическими производными (selflearn_tail.train_step)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    sys.path.insert(0, ROOT)
    import selflearn_tail as SL
    import train_chat
    model, tok, _ = train_chat.load_chat_model(CHAT_NPZ, torch.device("cpu"))
    Ew, fw, eps = SL.head_weights(model)
    V, D = Ew.shape
    g = torch.Generator().manual_seed(1)
    ns, nr, kc, kr = 5, 4, 6, 5
    Hs, Ys = torch.randn(ns, D, generator=g), torch.randn(ns, D, generator=g) * 3
    Hr, Yr = torch.randn(nr, D, generator=g), torch.randn(nr, D, generator=g) * 3
    cand = torch.randint(0, V, (ns, kc), generator=g)
    cand[0, 3] = cand[0, 0]                                      # нужное слово ещё и в топе — не считать дважды
    ridx = torch.stack([torch.randperm(V, generator=g)[:kr] for _ in range(nr)])
    rlp = torch.log_softmax(torch.randn(nr, kr, generator=g), -1)
    rand = torch.randint(0, V, (7,), generator=g)
    rand[0] = cand[1, 2]                                         # случайное слово совпало с выбранным
    k = 3
    chunk = {"up": torch.randn(k, D, generator=g) * 0.05, "bias": torch.full((k,), 0.2),
             "down": torch.randn(D, k, generator=g) * 0.01}
    init = {n: v.clone() for n, v in chunk.items()}
    ce_ref = float(SL.batch_loss([chunk], Ew, fw, eps, Hs, Ys, cand, rand, V, "ce"))
    kl_ref = float(SL.batch_loss([chunk], Ew, fw, eps, Hr, Yr, ridx, rand, V, "kl", rlp))
    opt = SL.Adam([chunk["up"], chunk["bias"], chunk["down"]])
    SL.train_step([chunk], [chunk], Ew, fw, eps, V, (Hs, Ys, cand), (Hr, Yr, ridx, rlp), opt, 0.01, 0.7, rand)
    js = chat_page.evaluate("""(a) => {
        const F = x => Float32Array.from(x), I = x => Int32Array.from(x);
        const hd = SLRSelf.headOf(miniState.bundle);
        const ch = { k: a.k, up: F(a.up), bias: F(a.bias), down: F(a.down) };
        const stud = { H: F(a.Hs), Y: F(a.Ys), cand: I(a.cand), Kc: a.kc };
        const rep = { H: F(a.Hr), Y: F(a.Yr), idx: I(a.ridx), lp: F(a.rlp), K: a.kr };
        const adam = new SLRSelf.Adam([ch.up, ch.bias, ch.down]);
        const loss = SLRSelf.trainStep(hd, [ch], stud, [...Array(a.ns).keys()], rep, [...Array(a.nr).keys()], I(a.rand), adam, 0.01, 0.7);
        return { loss, up: Array.from(ch.up), bias: Array.from(ch.bias), down: Array.from(ch.down) };
    }""", {"k": k, "up": init["up"].flatten().tolist(), "bias": init["bias"].tolist(), "down": init["down"].flatten().tolist(),
           "Hs": Hs.flatten().tolist(), "Ys": Ys.flatten().tolist(), "cand": cand.flatten().tolist(), "kc": kc,
           "Hr": Hr.flatten().tolist(), "Yr": Yr.flatten().tolist(), "ridx": ridx.flatten().tolist(),
           "rlp": rlp.flatten().tolist(), "kr": kr, "rand": rand.tolist(), "ns": ns, "nr": nr})
    assert js["loss"]["ce"] == pytest.approx(ce_ref, rel=1e-4, abs=1e-5)
    assert js["loss"]["kl"] == pytest.approx(kl_ref, rel=1e-3, abs=1e-5)
    for name in ("up", "bias", "down"):
        got = np.array(js[name], dtype=np.float32).reshape(chunk[name].shape)
        moved = (chunk[name] - init[name]).abs().max().item()
        assert moved > 0
        assert np.abs(got - chunk[name].numpy()).max() < 1e-3 * max(moved, 1e-3) + 1e-6, name


def test_selflearn_end_to_end_in_browser(browser):
    """Полный цикл в браузере: «/запомни …» в чате -> модель сама учится (растут новые нейроны) -> в новом
    разговоре отвечает о выученном слове, про невыученное говорит «не знаю»; выученное переживает перезагрузку
    страницы (IndexedDB) и удаляется кнопкой ✕."""
    if not os.path.exists(CHAT_NPZ):
        pytest.skip("нет файла весов slr_net_chat_ru_mini3.npz")
    pg = _open_page(browser)
    try:
        pg.click("#modeMiniBtn")
        pg.set_input_files("#miniFileInput", CHAT_NPZ)
        pg.wait_for_function("() => miniState.bundle !== null && !miniState.loading", timeout=60000)
        assert pg.evaluate("() => !document.getElementById('miniLearnBox').classList.contains('hidden')")
        _start_heartbeat(pg)
        pg.fill("#chatInputField", "/запомни Курдяпа — это маленькая птица синего цвета. Она живёт в лесу.")
        pg.click("#sendMsgBtn")
        pg.wait_for_function("() => !miniState.busy && document.querySelector('[data-mini-learn] .learn-result')", timeout=600000)
        assert pg.evaluate("() => window.__maxGap") < 1500, "главный поток был занят слишком долго"
        result = pg.evaluate("() => document.querySelector('[data-mini-learn] .learn-result').textContent")
        assert "Запомнил: курдяпа" in result, result
        mods = pg.evaluate("() => miniState.modules.map(m => [m.words, m.report.neurons, m.report.rounds.length])")
        assert len(mods) == 1 and mods[0][0] == ["курдяпа"] and mods[0][1] >= 4

        def ask(q):
            return pg.evaluate("""async (q) => {
                miniResetContext(); miniState.temperature = 0; miniState.lastUserText = "";
                const r = await miniGenerateChatReply(q, () => {});
                return [miniState.bundle.tokenizer.decode(r.ids).trim(), r.knowledge];
            }""", q)

        a, kn = ask("Что такое курдяпа?")
        assert "птиц" in a.lower() and kn == ["курдяпа"], a
        a, kn = ask("Что такое клыдел?")
        assert kn == [] and ("незнаком" in a or "не знаю" in a.lower()), a
        assert "курдяпа" in pg.evaluate("() => document.getElementById('miniModulesList').textContent")
        pg.reload()                                                           # выученное сохранено в браузере
        pg.wait_for_function("() => miniState.bundle !== null && !miniState.loading && miniState.modules.length === 1",
                             timeout=120000)
        a, kn = ask("Напомни, что такое курдяпа.")
        assert "птиц" in a.lower() and kn == ["курдяпа"], a
        pg.click("#miniModulesList .learned-chip button")                     # забыть
        pg.wait_for_function("() => miniState.modules.length === 0", timeout=10000)
        a, kn = ask("Что такое курдяпа?")
        assert kn == [] and "птиц" not in a.lower(), a
        assert pg.errors == []
    finally:
        pg.close()
