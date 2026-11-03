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
