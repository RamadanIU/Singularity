# -*- coding: utf-8 -*-
"""
Проверка мобильного чата в настоящем браузере (headless Chromium через Playwright).
Интернет на странице отключён — так заодно проверяется, что всё работает офлайн.
Запуск: pip install playwright && playwright install chromium
"""
import os
import sys

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
    """Страница, открытая с сервера (например, GitHub Pages), скачивает веса, лежащие рядом с ней."""
    if not os.path.exists(MINI_NPZ):
        pytest.skip("нет файла весов slr_net_subtitles_ru_mini.npz")
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
        assert pg.evaluate("() => [...window.__texts].some(t => t.startsWith('Скачано'))")
        assert errors == []
    finally:
        pg.close()
        srv.shutdown()
