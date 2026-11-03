# -*- coding: utf-8 -*-
"""
Проверка мобильного чата в настоящем браузере (headless Chromium через Playwright).
Интернет на странице отключён — так заодно проверяется, что всё работает офлайн.
Запуск: pip install playwright && playwright install chromium
"""
import os

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

HTML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slr_net_mobile_chat.html")


@pytest.fixture(scope="module")
def page():
    with sync_api.sync_playwright() as p:
        try:
            # CHROMIUM_PATH — путь к уже установленному Chrome/Chromium, если не ставили его через playwright
            path = os.environ.get("CHROMIUM_PATH")
            browser = p.chromium.launch(executable_path=path) if path else p.chromium.launch()
        except Exception as exc:  # браузер не установлен
            pytest.skip(f"Chromium недоступен: {exc}")
        pg = browser.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.route("http*://**", lambda route: route.abort())       # офлайн-режим
        pg.goto("file://" + HTML)
        pg.wait_for_function("() => appState && appState.engine !== null", timeout=60000)
        pg.errors = errors
        yield pg
        browser.close()


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
