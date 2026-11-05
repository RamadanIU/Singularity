#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
«Рентген» весов SLR-Net: что изменилось в файле весов и как это сказалось на модели.
=====================================================================================

Прежде чем править веса вручную (или верить, что правка помогла), её нужно ИЗМЕРИТЬ. Скрипт делает три вещи:

  1) diff  — сравнивает два файла .npz по каждой матрице: сколько тритов {-1, 0, +1} поменялось, насколько
             изменились обычные (float16) веса, какие значения стали «круглыми» (признак ручной правки).
  2) ablate — выключает части модели и смотрит, как растёт ошибка на отложенных диалогах:
             память фактов, ядро мышления (0 шагов), «все токены думают 3 шага», эмбеддинг в 4 битах вместо 16.
             Так видно, какая часть модели реально работает, а какая только занимает место.
  3) probes — точность на задачах с проверкой (то же, что train_chat.py eval, но на случайной выборке, быстрее).

  python weights_xray.py --ckpt мои_веса.npz --ref slr_net_chat_ru_mini.npz            # всё сразу
  python weights_xray.py --ckpt мои_веса.npz --no-ablate --probes 0                     # только diff

Ошибка считается на отложенных диалогах, которые собирает chat_corpus.py (секунды, без субтитров).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from typing import Dict, List, Optional

import numpy as np

import chat_corpus as C
import slr_net_engine as E
import train_chat as T


# =============================================================================
# 1) DIFF: ЧТО ИЗМЕНИЛОСЬ
# =============================================================================

def tensor_table(path: str) -> Dict[str, np.ndarray]:
    """Все веса файла: троичные матрицы — как триты (int8), остальное — float32."""
    out: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as z:
        if "tok_embeddings.weight.qdata" in z.files:   # эмбеддинг, сжатый при обучении: целые * масштаб группы
            cfg = json.loads(str(z["config"]))
            shape = tuple(int(s) for s in z["tok_embeddings.weight.shape"])
            q = E.unpack_int_matrix(z["tok_embeddings.weight.qdata"], shape, cfg["embed_bits"])
            scale = np.asarray(z["tok_embeddings.weight.qscale"], dtype=np.float32)
            out["tok_embeddings.weight"] = (q.astype(np.float32).reshape(shape[0], -1, cfg["embed_group"])
                                            * scale[..., None]).reshape(shape)
        for k in z.files:
            if k.startswith("tok_embeddings.weight."):
                continue
            if k.endswith(".weight.packed"):
                name = k[: -len(".packed")]
                out[name] = E.unpack_ternary_matrix(z[k], tuple(int(s) for s in z[name + ".shape"]))
                out[name + ".gamma"] = np.asarray(z[name + ".gamma"], dtype=np.float32)
            elif not k.endswith((".shape", ".gamma")) and not k.startswith(("meta.", "config", "tokenizer")):
                out[k] = np.asarray(z[k], dtype=np.float32)
    return out


def diff_report(ckpt: str, ref: str, top: int = 25) -> List[dict]:
    a, b = tensor_table(ckpt), tensor_table(ref)
    rows = []
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if x.shape != y.shape:
            rows.append({"name": k, "kind": "форма", "change": float("inf"), "note": f"{y.shape} → {x.shape}"})
            continue
        if x.dtype == np.int8:
            ch = float((x != y).mean())
            rows.append({"name": k, "kind": "триты", "change": ch, "size": int(x.size),
                         "note": f"изменено {100 * ch:.1f}% тритов, нулей {100 * (x == 0).mean():.0f}%"})
        else:
            denom = float(np.abs(y).mean()) or 1.0
            ch = float(np.abs(x - y).mean() / denom)
            note = f"среднее |w| {np.abs(y).mean():.4g} → {np.abs(x).mean():.4g}"
            if x.ndim == 2 and x.shape[0] > 8:
                zero_rows = int((np.abs(x).sum(1) == 0).sum())
                if zero_rows:
                    note += f", нулевых строк {zero_rows} из {x.shape[0]}"
            if x.size >= 8 and len(np.unique(x)) <= 16:
                note += f", всего {len(np.unique(x))} разных значений"
            elif x.size <= 4 and np.allclose(x, np.round(x, 2)):
                note += f", значение {x.ravel().tolist()}"
            rows.append({"name": k, "kind": "float", "change": ch, "size": int(x.size), "note": note})
    rows.sort(key=lambda r: -r["change"])
    print(f"\nЧто изменилось в {ckpt} относительно {ref} (сверху — сильнее всего):")
    print(f"{'тензор':45s} {'тип':6s} {'изменение':>10s}  подробности")
    for r in rows[:top]:
        print(f"{r['name']:45s} {r['kind']:6s} {r['change']:10.3f}  {r['note']}")
    tri = [r for r in rows if r["kind"] == "триты"]
    if tri:
        tot = sum(r["size"] for r in tri)
        print(f"Троичные матрицы: в среднем изменено {100 * sum(r['change'] * r['size'] for r in tri) / tot:.2f}% тритов")
    return rows


# =============================================================================
# 2) ABLATE: КАКАЯ ЧАСТЬ МОДЕЛИ РАБОТАЕТ
# =============================================================================

def val_loss(model, enc, pad_id: int, routing: str = "threshold", batch: int = 32):
    """Средняя ошибка (nats на токен ответа бота) и средняя глубина «мышления» на отложенных диалогах."""
    import torch
    model.eval()
    tot = cnt = 0.0
    depth = []
    with torch.no_grad():
        for i in range(0, len(enc), batch):
            x, y = T.pad_batch(enc[i:i + batch], pad_id)
            logits, _, st = model(x, routing=routing)
            lp = torch.log_softmax(logits.float(), -1)
            v = y != -100
            tot += float(-(lp.gather(-1, y.clamp(min=0)[..., None])[..., 0] * v).sum())
            cnt += int(v.sum())
            depth.append(float(st["core_depth_mean"]))
    return tot / max(cnt, 1), float(np.mean(depth))


def memory_off(model):
    m = copy.deepcopy(model)
    if m.core.memory is not None:                 # в новых моделях (memory_keys = 0) памяти нет вовсе
        m.core.memory.gate.data.fill_(0.0)        # tanh(0) = 0: память ничего не добавляет
    return m


def thinking_off(model):
    m = copy.deepcopy(model)
    m.core.router.bias.data.fill_(-1e4)           # роутер никого не пускает в ядро мышления
    return m


def memory_usage(model, enc, pad_id: int) -> dict:
    """Сколько ячеек памяти реально используется и насколько сильно память меняет поток."""
    import torch
    if model.core.memory is None:
        return {}
    with torch.no_grad():
        x, _ = T.pad_batch(enc[:64], pad_id)
        h = model._embed(x)
        for layer in model.encoder:
            h = layer(h)
        z = (h + model.core.step_emb[0]).reshape(-1, h.shape[-1])
        idx, w = model.core.memory.lookup(z)
        vals = model.core.memory.values.weight
        mem = torch.tanh(model.core.memory.gate) * (vals[idx] * w[..., None].to(vals.dtype)).sum(1)
        ratio = (mem.norm(dim=-1) / z.norm(dim=-1).clamp(min=1e-6)).median()
        nonzero = vals.norm(dim=1) > 0
    return {"used_cells": len(set(idx.flatten().tolist())), "cells": int(vals.shape[0]),
            "hits_nonzero_cells": float(nonzero[idx].float().mean()),
            "memory_to_stream_norm": float(ratio), "top1_weight": float(w.max(-1).values.mean())}


def embeddings_quantized(model, bits: int = 4):
    """Эмбеддинг (он же выходной слой) сжат до bits бит на число, масштаб — свой у каждой строки.
    Сейчас эмбеддинг хранится в float16 и занимает больше половины файла: проверяем, нужна ли такая точность."""
    m = copy.deepcopy(model)
    w = m.tok_embeddings.weight.data
    if m.config.embed_bits:                       # эмбеддинг уже сжат при обучении (embed_bits бит)
        m.config.embed_bits = min(m.config.embed_bits, bits)
        return m
    q = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / q
    w.copy_((w / scale).round().clamp(-q, q) * scale)
    return m


def probe_accuracy(model, tok, probes) -> float:
    return float(np.mean([bool(T.check_reply(T.answer(model, tok, True, p["turns"], max_new=64), p["check"]))
                          for p in probes]))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Рентген весов SLR-Net: что изменилось и что реально работает")
    ap.add_argument("--ckpt", required=True, help="проверяемые веса (.npz или .pt)")
    ap.add_argument("--ref", default="slr_net_chat_ru_mini.npz", help="с чем сравнивать")
    ap.add_argument("--no-ablate", action="store_true", help="не выключать части модели")
    ap.add_argument("--probes", type=int, default=200, help="сколько задач с проверкой (0 — не проверять)")
    ap.add_argument("--json", default=None, help="сохранить результаты в JSON")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    torch.set_flush_denormal(True)
    res: dict = {}
    if args.ref and args.ref.endswith(".npz") and args.ckpt.endswith(".npz"):
        res["diff"] = diff_report(args.ckpt, args.ref)
    data = C.build_dataset([], seed=11)
    rng = np.random.default_rng(args.seed)
    probes = data["probes"]
    probes = [probes[i] for i in sorted(rng.choice(len(probes), size=min(args.probes, len(probes)), replace=False))]
    res["models"] = {}
    for path in [args.ckpt] + ([args.ref] if args.ref else []):
        model, tok, chat = T.load_chat_model(path, torch.device("cpu"), allow_base=False)
        enc = [T.encode_dialogue(tok, r["turns"]) for r in data["sft_val"]]
        variants = {"как есть": (model, "threshold")}
        if not args.no_ablate:
            if model.core.memory is not None:
                variants["память фактов выключена"] = (memory_off(model), "threshold")
            variants.update({"ядро мышления выключено (0 шагов)": (thinking_off(model), "threshold"),
                             "все токены думают 3 шага": (model, "all")})
            if not model.config.embed_bits or model.config.embed_bits > 4:
                variants["эмбеддинг сжат до 4 бит"] = (embeddings_quantized(model, 4), "threshold")
        print(f"\n=== {path}")
        out = {}
        for name, (m, routing) in variants.items():
            t0 = time.time()
            loss, depth = val_loss(m, enc, tok.nl_id, routing)
            acc = probe_accuracy(m, tok, probes) if probes and (name == "как есть" or args.probes) else None
            out[name] = {"val_loss": loss, "core_depth": depth, "probe_acc": acc}
            print(f"  {name:36s} ошибка {loss:.4f} | шагов мышления на токен {depth:.2f}"
                  + (f" | задачи {100 * acc:.1f}% (из {len(probes)})" if acc is not None else "")
                  + f"  ({time.time() - t0:.0f} c)", flush=True)
        mu = memory_usage(model, enc, tok.nl_id)
        out["память"] = mu
        if mu:
            print(f"  память фактов: используется {mu['used_cells']} ячеек из {mu['cells']}, "
                  f"доля обращений в ненулевые ячейки {mu['hits_nonzero_cells']:.2f}, "
                  f"вклад памяти / поток = {mu['memory_to_stream_norm']:.2f}")
        else:
            print("  памяти фактов в этой модели нет (memory_keys = 0)")
        res["models"][path] = out
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    sys.exit(main())
