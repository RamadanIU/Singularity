#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Награда за целостный проект (идея 3 из PLAN_MULTIMODULE_RU.md): проверщик браузерной игры.
==========================================================================================

Чтобы поощрять модель «за завершённый, продуманный проект», награду нужно считать автоматически и честно.
Этот файл запускает HTML-игру (один файл: HTML + JavaScript) в настоящем браузере без окна (Chromium через
Playwright) и проходит чек-лист — как придирчивый тестировщик:

  запускается   — нет ошибок JavaScript при загрузке и в первые секунды игры;
  рисует        — на экране что-то есть (не пустой холст);
  живёт         — картинка меняется сама (игровой цикл), даже без нажатий;
  управление    — стрелки / WASD меняют картинку иначе, чем без нажатий (сравнение с контрольным запуском);
  действие      — пробел (выстрел, прыжок, поворот фигуры) тоже что-то меняет;
  масштаб       — на экране телефона (400×700) поле помещается целиком и подстраивается под размер окна;
  касания       — есть управление пальцем (touch / pointer) — для телефона;
  счёт          — игра выводит счёт (текстом на странице или на холсте);
  конец и рестарт — есть конец игры и способ начать заново.

Награда = взвешенная доля пройденных пунктов (0…1). Если игра падает с ошибкой при загрузке, награда не больше
0.1: недоделанный проект не должен выигрывать у простого, но рабочего. Случайность в игре (Math.random)
фиксируется, чтобы сравнение «с нажатиями / без нажатий» было честным.

Именно такую награду предлагается использовать при обучении с подкреплением (GRPO): модель пишет игру по
просьбе («сделай танчики»), проверщик ставит оценку, модель поощряется за более целостные проекты.

  python game_reward.py training/game_reward/tanks_full.html training/game_reward/tanks_bare.html
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Вес каждого пункта: важное для «настоящей игры» (работает, управляется) весит больше
WEIGHTS: Dict[str, float] = {
    "запускается": 2.0, "рисует": 1.0, "живёт": 1.0, "управление": 2.0, "действие": 1.0,
    "масштаб": 1.5, "касания": 0.5, "счёт": 1.0, "конец и рестарт": 1.0,
}

# Вставляется в страницу до её кода: фиксированный Math.random, ВИРТУАЛЬНОЕ время (кадры requestAnimationFrame и
# таймеры двигает проверщик через window.__advance(мс) — два запуска совпадают до пикселя, и любое отличие —
# реакция на нажатие), запись слушателей событий и текста, который игра пишет на холсте.
INIT_SCRIPT = r"""
(() => {
  let s = 12345;
  Math.random = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; };
  let now = 0, nextId = 1, raf = [], timers = [];
  const t0 = Date.now();
  window.__errors = [];
  const safe = (cb, args) => { try { cb(...args); } catch (e) { window.__errors.push(String(e && e.message || e)); } };
  window.requestAnimationFrame = cb => { const id = nextId++; raf.push({ id, cb }); return id; };
  window.cancelAnimationFrame = id => { raf = raf.filter(r => r.id !== id); };
  window.setTimeout = (cb, ms, ...args) => { const id = nextId++;
    if (typeof cb === 'function') timers.push({ id, t: now + (+ms || 0), cb, args, every: 0 }); return id; };
  window.setInterval = (cb, ms, ...args) => { const id = nextId++; const every = Math.max(1, +ms || 1);
    if (typeof cb === 'function') timers.push({ id, t: now + every, cb, args, every }); return id; };
  window.clearTimeout = window.clearInterval = id => { timers = timers.filter(t => t.id !== id); };
  performance.now = () => now;
  Date.now = () => t0 + now;
  window.__advance = ms => {
    const end = now + ms;
    while (now < end) {
      now = Math.min(end, now + 1000 / 60);
      for (;;) {
        timers.sort((a, b) => a.t - b.t);
        const t = timers[0];
        if (!t || t.t > now) break;
        if (t.every) t.t += t.every; else timers.shift();
        safe(t.cb, t.args);
      }
      const q = raf; raf = [];
      for (const r of q) safe(r.cb, [now]);
    }
  };
  window.__listeners = [];
  const add = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function (type, fn, opt) {
    window.__listeners.push(type);
    return add.call(this, type, fn, opt);
  };
  for (const p of ['onkeydown', 'onkeyup', 'ontouchstart', 'onpointerdown', 'onresize', 'onclick']) {
    for (const target of [window, document]) {
      let v = null;
      try {
        Object.defineProperty(target, p, { configurable: true, get: () => v,
          set: f => { v = f; window.__listeners.push(p.slice(2)); add.call(target, p.slice(2), e => v && v(e)); } });
      } catch (e) {}
    }
  }
  window.__texts = [];
  for (const m of ['fillText', 'strokeText']) {
    const orig = CanvasRenderingContext2D.prototype[m];
    CanvasRenderingContext2D.prototype[m] = function (t, ...a) {
      if (window.__texts.length < 2000) window.__texts.push(String(t));
      return orig.call(this, t, ...a);
    };
  }
})();
"""

SCORE_WORDS = re.compile(r"(сч[её]т|очк|score|points|рекорд|record|уров|level|жизн|lives)", re.I)
OVER_WORDS = re.compile(r"(game\s*over|игра\s+окончена|конец\s+игры|вы\s+проиграли|проигрыш|победа|you\s+win|"
                        r"you\s+lose|gameover|game_over)", re.I)
RESTART_WORDS = re.compile(r"(restart|рестарт|заново|снова|ещё\s+раз|еще\s+раз|новая\s+игра|play\s+again|"
                           r"location\.reload|reset\w*\s*\()", re.I)

def strip_comments(html: str) -> str:
    """Код без комментариев: признаки «счёт», «конец игры» ищем в коде, а не в пояснениях к нему."""
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.S)
    html = re.sub(r"/\*.*?\*/", " ", html, flags=re.S)
    return re.sub(r"(^|[^:\\'\"])//[^\n]*", r"\1", html)


def _browser_path() -> Optional[str]:
    path = os.environ.get("CHROMIUM_PATH")
    if path:
        return path
    return "/opt/pw-browsers/chromium" if os.path.exists("/opt/pw-browsers/chromium") else None


def decode_png_gray(png: bytes):
    """PNG (8 бит, RGB/RGBA/серый, без чересстрочности — так снимает Playwright) -> яркость (h, w) float32.
    Свой маленький декодер, чтобы не требовать Pillow."""
    import struct
    import zlib
    import numpy as np
    pos, idat, w = 8, b"", 0
    h = color = 0
    while pos < len(png):
        n, kind = struct.unpack(">I4s", png[pos:pos + 8])
        data = png[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, color = struct.unpack(">IIBB", data[:10])
            if depth != 8 or data[12] != 0:
                raise ValueError("поддерживаются только 8-битные PNG без чересстрочности")
        elif kind == b"IDAT":
            idat += data
        pos += 12 + n
    ch = {0: 1, 2: 3, 4: 2, 6: 4}[color]
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, 1 + w * ch)
    filters, rows = raw[:, 0], raw[:, 1:].astype(np.int32)
    out = np.zeros((h, w * ch), dtype=np.int32)
    prev = np.zeros(w * ch, dtype=np.int32)
    for y in range(h):
        line, f = rows[y].copy(), filters[y]
        if f == 1:                               # Sub: накопленная сумма слева по модулю 256
            line = (line.reshape(w, ch).cumsum(0) & 255).reshape(-1)
        elif f == 2:                             # Up
            line = (line + prev) & 255
        elif f == 3:                             # Average
            line = line.reshape(w, ch)
            pr = prev.reshape(w, ch)
            for x in range(w):
                left = line[x - 1] if x else 0
                line[x] = (line[x] + ((left + pr[x]) >> 1)) & 255
            line = line.reshape(-1)
        elif f == 4:                             # Paeth
            line = line.reshape(w, ch)
            pr = prev.reshape(w, ch)
            for x in range(w):
                a = line[x - 1] if x else np.zeros(ch, dtype=np.int32)
                b, c = pr[x], (pr[x - 1] if x else np.zeros(ch, dtype=np.int32))
                pa, pb, pc = np.abs(b - c), np.abs(a - c), np.abs(a + b - 2 * c)
                pred = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
                line[x] = (line[x] + pred) & 255
            line = line.reshape(-1)
        out[y] = line
        prev = line
    img = out.reshape(h, w, ch).astype(np.float32)
    return img[..., :3].mean(-1) if ch >= 3 else img[..., 0]


def _frame_stats(png: bytes):
    try:
        import io
        import numpy as np
        from PIL import Image
        arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32).mean(-1)
    except ImportError:
        arr = decode_png_gray(png)
    return arr, float(arr.std())


def _diff(a, b) -> float:
    """Сколько точек экрана заметно изменилось (яркость больше чем на 8 из 255)."""
    import numpy as np
    if a.shape != b.shape:
        return float(a.size)
    return float((np.abs(a - b) > 8).sum())


class _Run:
    """Один запуск игры в свежей странице: та же случайность, то же виртуальное время."""

    def __init__(self, browser, html: str, width: int = 800, height: int = 600):
        self.page = browser.new_page(viewport={"width": width, "height": height})
        self.errors: List[str] = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.on("console", lambda m: self.errors.append(m.text) if m.type == "error" else None)
        self.page.add_init_script(INIT_SCRIPT)
        # Игра отдаётся по выдуманному адресу (а не set_content): так скрипт выше точно выполняется до её кода
        self.page.route("http://game.local/**", lambda route: route.fulfill(
            status=200, content_type="text/html; charset=utf-8", body=html))
        self.page.goto("http://game.local/index.html", wait_until="load")

    def advance(self, ms: int) -> None:
        self.page.evaluate(f"window.__advance && window.__advance({int(ms)})")

    def start(self) -> None:
        """Многие игры ждут нажатия: щелчок по центру и Enter (но не пробел и не стрелки — их проверяем отдельно)."""
        vp = self.page.viewport_size
        self.advance(100)
        self.page.mouse.click(vp["width"] // 2, vp["height"] // 2)
        self.page.keyboard.press("Enter")
        self.advance(300)

    def shot(self):
        return _frame_stats(self.page.screenshot())[0]

    def hold(self, key: str, ms: int) -> None:
        self.page.keyboard.down(key)
        self.advance(ms)
        self.page.keyboard.up(key)
        self.advance(50)

    def scrolled(self) -> bool:
        return bool(self.page.evaluate("scrollX !== 0 || scrollY !== 0"))

    def all_errors(self) -> List[str]:
        return self.errors + list(self.page.evaluate("window.__errors || []"))

    def close(self) -> None:
        self.page.close()


FIT_JS = """() => {
    const el = document.querySelector('canvas') || document.body.firstElementChild || document.body;
    const r = el.getBoundingClientRect();
    return {w: Math.round(r.width), h: Math.round(r.height), right: Math.round(r.right), bottom: Math.round(r.bottom),
            sw: document.documentElement.scrollWidth, iw: innerWidth, ih: innerHeight};
}"""


def evaluate_game(html: str, headless: bool = True) -> dict:
    """Чек-лист для одной игры. Возвращает {"reward": 0..1, "checks": {...}, "details": {...}}."""
    from playwright.sync_api import sync_playwright
    checks: Dict[str, float] = {}
    details: Dict[str, object] = {}
    eps = 12                   # точек; время виртуальное: одинаковые запуски совпадают, любое отличие — отклик
    with sync_playwright() as p:
        path = _browser_path()
        browser = p.chromium.launch(executable_path=path, headless=headless) if path else \
            p.chromium.launch(headless=headless)
        try:
            # --- контрольный запуск: без нажатий
            a = _Run(browser, html)
            a.start()
            f0 = a.shot()
            a.advance(700)
            f1 = a.shot()
            listeners = set(a.page.evaluate("window.__listeners || []"))
            texts = a.page.evaluate("(window.__texts || []).join(' ')")
            dom_text = a.page.evaluate("document.body ? document.body.innerText : ''")
            errors = a.all_errors()
            a.close()
            checks["запускается"] = 0.0 if errors else 1.0
            details["ошибки"] = errors[:5]
            std = float(f1.std())
            checks["рисует"] = 1.0 if std > 3.0 else 0.0
            live = _diff(f0, f1)
            checks["живёт"] = 1.0 if live > eps else 0.0
            details["разброс яркости"] = round(std, 2)
            details["изменение без нажатий (точек)"] = int(live)

            def respond(keys: List[str], ms: int, presses: int = 1) -> Tuple[float, bool, List[str]]:
                r = _Run(browser, html)
                r.start()
                for _ in range(presses):
                    for key in keys:
                        r.hold(key, ms)
                r.advance(700 - presses * len(keys) * (ms + 50))
                d, sc, err = _diff(f1, r.shot()), r.scrolled(), r.all_errors()
                d = 0.0 if sc else d                 # картинка сдвинулась из-за прокрутки страницы, а не игры
                r.close()
                return d, sc, err

            # --- управление: стрелки (если не слушает стрелки — WASD)
            key_listen = bool(listeners & {"keydown", "keyup", "keypress"})
            resp, scrolled, err = respond(["ArrowRight", "ArrowUp"], 300)
            if resp <= eps or scrolled:
                resp2, scrolled2, err2 = respond(["d", "w"], 300)
                if resp2 > resp:
                    resp, scrolled, err = resp2, scrolled2, err2
            moved = resp > eps and not scrolled
            checks["управление"] = 0.5 * key_listen + 0.5 * moved
            details["отклик на стрелки (точек)"] = int(resp)
            details["страница прокручивается клавишами"] = scrolled
            errors += err

            # --- пробел: выстрел / прыжок / поворот
            resp_space, scrolled_s, err = respond([" "], 80, presses=3)
            checks["действие"] = 1.0 if resp_space > eps and not scrolled_s else 0.0
            details["отклик на пробел (точек)"] = int(resp_space)
            errors += err

            # --- телефон: поле помещается в экран 400×700 и подстраивается под размер окна
            m = _Run(browser, html, 400, 700)
            m.start()
            fit = m.page.evaluate(FIT_JS)
            m.page.set_viewport_size({"width": 700, "height": 500})
            m.page.wait_for_timeout(250)          # событие resize браузер присылает в своём (настоящем) кадре
            m.advance(200)
            fit2 = m.page.evaluate(FIT_JS)
            m.close()
            fits = all(f["right"] <= f["iw"] + 1 and f["bottom"] <= f["ih"] + 1 and f["sw"] <= f["iw"] + 1
                       for f in (fit, fit2))
            adapts = abs(fit["w"] - fit2["w"]) > 1 or abs(fit["h"] - fit2["h"]) > 1
            checks["масштаб"] = 0.5 * fits + 0.5 * adapts
            details["телефон 400×700"] = fit
            details["окно 700×500"] = fit2

            checks["касания"] = 1.0 if listeners & {"touchstart", "touchmove", "pointerdown", "pointermove"} else 0.0
            shown = texts + " " + dom_text
            code = strip_comments(html)
            checks["счёт"] = 1.0 if (SCORE_WORDS.search(shown) and re.search(r"\d", shown)) else (
                0.5 if SCORE_WORDS.search(code) else 0.0)
            over = bool(OVER_WORDS.search(code)) or bool(OVER_WORDS.search(shown))
            restart = bool(RESTART_WORDS.search(code))
            checks["конец и рестарт"] = 0.5 * over + 0.5 * restart
            if errors and checks["запускается"] == 1.0:      # падает уже во время игры (от нажатий)
                checks["запускается"] = 0.5
                details["ошибки"] = errors[:5]
            details["слушатели событий"] = sorted(listeners)
            details["текст на холсте"] = texts[:200]
        finally:
            browser.close()
    total = sum(WEIGHTS.values())
    reward = sum(WEIGHTS[k] * checks.get(k, 0.0) for k in WEIGHTS) / total
    if checks.get("запускается", 0.0) == 0.0:
        reward = min(reward, 0.1)
    return {"reward": round(reward, 4), "checks": checks, "details": details}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Награда за целостность браузерной игры")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    out = {}
    for f in args.files:
        res = evaluate_game(Path(f).read_text(encoding="utf-8"))
        out[f] = res
        print(f"\n{f}: награда {res['reward']:.2f}")
        for k, v in res["checks"].items():
            print(f"  {'✓' if v >= 1 else ('½' if v > 0 else '✗')} {k}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
