#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение настоящей языковой модели SLR-Net на больших текстах (PyTorch).
=======================================================================

Пять команд:

  1) Подготовка данных: обучает субсловный токенизатор (SentencePiece Unigram) и переводит
     все тексты в компактный бинарный файл токенов (uint16, читается через memmap — подходит
     для корпусов на десятки гигабайт, в оперативную память целиком не грузится).

       python train_lm.py prepare --input книги/ субтитры.srt --out data --vocab-size 16384

  2) Обучение:
       python train_lm.py train --data data --out runs/small --preset small --max-steps 20000

  3) Генерация текста:
       python train_lm.py sample --ckpt runs/small/best.pt --prompt "Почему небо синее?"

  4) Проверка, что модель действительно обучилась (против случайных весов и частотной модели):
       python train_lm.py eval --ckpt runs/small/best.pt --data data

  5) Экспорт троичных весов (5 тритов в байт, токенизатор внутри) и отчёт о размере:
       python train_lm.py export --ckpt runs/small/best.pt --out runs/small/model_packed.npz
     Упакованный файл тоже подходит для sample и eval:
       python train_lm.py sample --ckpt runs/small/model_packed.npz --prompt "Привет!"

Зависимости: torch, numpy, sentencepiece  (pip install -r requirements.txt)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

import slr_net_engine as E

TEXT_SUFFIXES = {".txt", ".srt", ".md"}


# =============================================================================
# ДАННЫЕ И ТОКЕНИЗАТОР
# =============================================================================

def iter_input_files(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(f for f in path.rglob("*") if f.suffix.lower() in TEXT_SUFFIXES))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Не найден файл или папка: {p}")
    if not files:
        raise FileNotFoundError("Не найдено ни одного текстового файла (.txt/.srt/.md)")
    return files


_SRT_TIME = re.compile(r"^\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->")


def detect_text_encoding(path: Path, sample_bytes: int = 1 << 20) -> str:
    """
    UTF-8 (с BOM или без), UTF-16 по BOM, иначе Windows-1251 — в ней сохранены многие русские .srt.
    Раньше всё читалось как UTF-8 с errors="ignore", и файл в Windows-1251 молча превращался в пустоту.
    """
    with open(path, "rb") as f:
        head = f.read(sample_bytes)
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        head.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError as err:
        if err.start >= len(head) - 3 and len(head) == sample_bytes:   # символ разрезан границей куска
            return "utf-8"
        return "cp1251"


def iter_clean_lines(path: Path, keep_hash_lines: bool = False) -> Iterator[str]:
    """Строки файла без мусора: таймкоды и номера субтитров, HTML-теги, ремарки [музыка]."""
    is_srt = path.suffix.lower() == ".srt"
    with open(path, "r", encoding=detect_text_encoding(path), errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if not keep_hash_lines and s.startswith("#"):
                continue
            if _SRT_TIME.match(s) or (is_srt and s.isdigit()):
                continue
            s = re.sub(r"<[^>]+>", " ", s)
            s = re.sub(r"\[[^\]]+\]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            if s:
                yield s


class Tokenizer:
    """Обёртка над SentencePiece: перевод строки кодируется отдельным токеном <n>."""

    NEWLINE = "<n>"

    def __init__(self, model_path: Optional[str] = None, model_proto: Optional[bytes] = None):
        import sentencepiece as spm
        if model_proto is not None:
            # Токенизатор, сохранённый внутри чекпоинта/экспорта: веса работают на любом компьютере
            self.sp = spm.SentencePieceProcessor(model_proto=model_proto)
        else:
            self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.nl_id = self.sp.piece_to_id(self.NEWLINE)
        self.eos_id = self.sp.eos_id()
        self.bos_id = self.sp.bos_id()

    def encode_lines(self, lines: List[str]) -> List[List[int]]:
        return self.sp.encode(lines)

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        lines = text.split("\n")
        for i, chunk in enumerate(self.sp.encode(lines)):
            ids.extend(chunk)
            if i < len(lines) - 1:
                ids.append(self.nl_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        parts: List[str] = []
        cur: List[int] = []
        for t in ids:
            t = int(t)
            if t == self.nl_id:
                parts.append(self.sp.decode(cur))
                parts.append("\n")
                cur = []
            elif t == self.eos_id:
                break
            else:
                cur.append(t)
        parts.append(self.sp.decode(cur))
        return "".join(parts)


def cmd_prepare(args: argparse.Namespace) -> None:
    import sentencepiece as spm

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = iter_input_files(args.input)
    total_bytes = sum(f.stat().st_size for f in files)
    print(f"Файлов: {len(files)}, объём: {total_bytes / 1024 ** 2:.2f} МБ")

    model_prefix = out / "tokenizer"
    if args.tokenizer:
        tok_path = args.tokenizer
        print(f"Используется готовый токенизатор: {tok_path}")
    else:
        def sentences() -> Iterator[str]:
            for f in files:
                yield from iter_clean_lines(f, args.keep_hash_lines)

        print(f"Обучение токенизатора SentencePiece Unigram (vocab={args.vocab_size}) ...")
        spm.SentencePieceTrainer.train(
            sentence_iterator=sentences(),
            model_prefix=str(model_prefix),
            vocab_size=args.vocab_size,
            model_type="unigram",            # Unigram лучше BPE совпадает с морфемами (Bostrom & Durrett, 2020)
            character_coverage=0.9995,
            byte_fallback=True,              # никаких <unk>: редкие символы раскладываются на байты
            split_digits=True,               # каждая цифра — отдельный токен: помогает арифметике
            user_defined_symbols=[Tokenizer.NEWLINE],
            unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
            input_sentence_size=args.tokenizer_sample_lines,
            shuffle_input_sentence=True,
            max_sentence_length=65536,
            hard_vocab_limit=False,          # маленький корпус -> словарь будет меньше, а не ошибка
            num_threads=os.cpu_count() or 1,
            minloglevel=2,
        )
        tok_path = str(model_prefix) + ".model"

    tok = Tokenizer(tok_path)
    if tok.vocab_size > 65535:
        raise ValueError("Словарь > 65535 не помещается в uint16")
    print(f"Токенизатор: {tok_path} (словарь {tok.vocab_size})")

    bin_path = out / "tokens.bin"
    n_tokens, n_bytes = 0, 0
    t0 = time.time()
    with open(bin_path, "wb") as fout:
        for fi, f in enumerate(files):
            batch: List[str] = []

            def flush(lines: List[str]) -> int:
                buf: List[int] = []
                for ids in tok.encode_lines(lines):
                    buf.extend(ids)
                    buf.append(tok.nl_id)
                np.asarray(buf, dtype=np.uint16).tofile(fout)
                return len(buf)

            for line in iter_clean_lines(f, args.keep_hash_lines):
                batch.append(line)
                n_bytes += len(line.encode("utf-8")) + 1
                if len(batch) >= 4096:
                    n_tokens += flush(batch)
                    batch = []
            if batch:
                n_tokens += flush(batch)
            np.asarray([tok.eos_id], dtype=np.uint16).tofile(fout)   # граница документа
            n_tokens += 1
            if (fi + 1) % 50 == 0 or fi + 1 == len(files):
                print(f"  закодировано файлов {fi + 1}/{len(files)}, токенов {n_tokens:,} "
                      f"({time.time() - t0:.0f} c)")

    n_val = max(int(n_tokens * args.val_fraction), min(n_tokens // 10, 4096))
    meta = {
        "tokenizer": os.path.relpath(tok_path, out) if not args.tokenizer else os.path.abspath(tok_path),
        "vocab_size": tok.vocab_size,
        "dtype": "uint16",
        "n_tokens": n_tokens,
        "val_start": n_tokens - n_val,
        "bytes_per_token": n_bytes / max(1, n_tokens),
        "files": [str(f) for f in files],
    }
    with open(out / "meta.json", "w", encoding="utf-8") as fm:
        json.dump(meta, fm, ensure_ascii=False, indent=2)
    print(f"Готово: {n_tokens:,} токенов ({meta['bytes_per_token']:.2f} байт текста на токен), "
          f"из них {n_val:,} — валидация. Файлы в {out}/")


def load_dataset(data_dir: str):
    with open(Path(data_dir) / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    tokens = np.memmap(Path(data_dir) / "tokens.bin", dtype=np.uint16, mode="r")
    tok_path = meta["tokenizer"]
    if not os.path.isabs(tok_path):
        tok_path = str(Path(data_dir) / tok_path)
    return meta, tokens, tok_path


class BatchSampler:
    """Случайные окна (seq_len + 1) из memmap — без загрузки корпуса в память."""

    def __init__(self, tokens: np.ndarray, start: int, end: int, seq_len: int, batch_size: int,
                 device, seed: int):
        if end - start < seq_len + 2:
            raise ValueError(f"Слишком мало токенов ({end - start}) для seq_len={seq_len}. "
                             f"Уменьшите --seq-len или добавьте текстов.")
        self.tokens, self.start, self.end = tokens, start, end
        self.seq_len, self.batch_size, self.device = seq_len, batch_size, device
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Начать выборку окон заново (валидация каждый раз на одних и тех же окнах)."""
        self.rng = np.random.default_rng(self.seed)

    def get(self):
        import torch
        offs = self.rng.integers(self.start, self.end - self.seq_len - 1, size=self.batch_size)
        batch = np.stack([np.asarray(self.tokens[o:o + self.seq_len + 1], dtype=np.int64) for o in offs])
        t = torch.from_numpy(batch)
        if self.device.type == "cuda":
            t = t.pin_memory().to(self.device, non_blocking=True)
        else:
            t = t.to(self.device)
        return t[:, :-1], t[:, 1:]


# =============================================================================
# ОБУЧЕНИЕ
# =============================================================================

def pick_device(name: str):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, args: argparse.Namespace) -> float:
    """Разогрев + cosine или WSD (warmup-stable-decay: удобно «дообучать дальше» без перезапуска)."""
    base, min_lr = args.lr, args.lr * args.min_lr_ratio
    if step < args.warmup:
        return base * (step + 1) / args.warmup
    if args.schedule == "wsd":
        decay_start = int(args.max_steps * (1.0 - args.decay_fraction))
        if step < decay_start:
            return base
        frac = (step - decay_start) / max(1, args.max_steps - decay_start)
        return min_lr + (base - min_lr) * (1.0 - frac)
    frac = (step - args.warmup) / max(1, args.max_steps - args.warmup)
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * min(1.0, frac)))


def build_optimizer(model, args: argparse.Namespace, device):
    import torch
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Не затухаем: нормы, смещения, параметры SSM, шаговые эмбеддинги, память фактов
        # (строки памяти обновляются редко, и weight decay стирал бы выученные факты).
        if p.dim() < 2 or "memory.values" in name or "memory.keys" in name or "step_emb" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    kwargs = {"lr": args.lr, "betas": (0.9, 0.95), "eps": 1e-8}
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


def config_from_args(args: argparse.Namespace, vocab_size: int) -> E.SLRNetConfig:
    overrides: Dict[str, object] = {"vocab_size": vocab_size}
    if args.no_ternary:
        overrides["ternary"] = False
    field_types = {f.name: type(f.default) for f in dataclasses.fields(E.SLRNetConfig)}
    for item in args.set or []:
        key, _, value = item.partition("=")
        if key not in field_types:
            raise ValueError(f"Неизвестный параметр конфигурации: {key}. Доступны: {sorted(field_types)}")
        kind = field_types[key]
        overrides[key] = value.lower() in ("1", "true", "yes") if kind is bool else kind(value)
    return E.make_config(args.preset, **overrides)


def evaluate(model, sampler: BatchSampler, n_batches: int, autocast_ctx) -> Dict[str, float]:
    import torch
    model.eval()
    sampler.reset()
    losses, depth = [], []
    with torch.no_grad(), autocast_ctx:
        for _ in range(n_batches):
            x, y = sampler.get()
            _, loss, stats = model(x, y, routing="threshold")   # так модель будет работать в генерации
            losses.append(float(stats["ce_loss"]))
            depth.append(float(stats["core_depth_mean"]))
    model.train()
    return {"loss": float(np.mean(losses)), "depth": float(np.mean(depth))}


def sample_text(model, tok: Tokenizer, prompt: str, max_new_tokens: int, temperature: float,
                top_k: int, top_p: float, repetition_penalty: float, device) -> str:
    import torch
    ids = tok.encode(prompt) or [tok.nl_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
                         top_p=top_p, repetition_penalty=repetition_penalty, eos_id=tok.eos_id)
    return tok.decode(out[0, len(ids):].tolist())


def cmd_train(args: argparse.Namespace) -> None:
    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # На CPU сверхмалые (денормализованные) числа в softmax и градиентах обрабатываются в десятки раз
    # медленнее обычных; их обнуление ничего не меняет в качестве, а шаг обучения ускоряет на ~25%.
    torch.set_flush_denormal(True)
    device = pick_device(args.device)
    meta, tokens, tok_path = load_dataset(args.data)
    tok = Tokenizer(tok_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    start_step, best_val = 0, float("inf")
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = E.SLRNetConfig(**ckpt["config"])
        start_step, best_val = ckpt["step"], ckpt.get("best_val", float("inf"))
    else:
        cfg = config_from_args(args, meta["vocab_size"])
    model = E.SLRNetLanguageModel(cfg).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])

    if args.lr is None:
        # BitNet-модели учатся при бОльшем LR, чем обычные (см. BitNet b1.58 2B4T)
        args.lr = 1.5e-3 if cfg.ternary else 6e-4
    optimizer = build_optimizer(model, args, device)
    if ckpt is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported() and args.dtype in ("auto", "bf16")
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    train_model = torch.compile(model) if args.compile else model

    val_start = int(meta["val_start"])
    # При продолжении (--resume) сдвигаем зерно: иначе модель заново увидела бы те же самые батчи,
    # что и в начале обучения. Валидация, наоборот, всегда на одних и тех же окнах — так её цифры
    # между запусками можно честно сравнивать.
    train_sampler = BatchSampler(tokens, 0, val_start, args.seq_len, args.batch_size, device,
                                 args.seed + 1_000_003 * start_step)
    try:
        val_sampler = BatchSampler(tokens, val_start, len(tokens), args.seq_len, args.batch_size, device,
                                   args.seed + 1)
    except ValueError:
        print("ВНИМАНИЕ: валидационная часть меньше seq_len — валидация на обучающих данных (не честно!)")
        val_sampler = BatchSampler(tokens, 0, val_start, args.seq_len, args.batch_size, device, args.seed + 1)

    n = model.num_params()
    fp = E.calculate_hardware_footprint(cfg)
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    print("=" * 86)
    print(f"Модель SLR-Net «{args.preset}»: {n['total'] / 1e6:.1f} млн параметров "
          f"(троичных {n['ternary'] / 1e6:.1f}, память фактов {n['memory_values'] / 1e6:.1f}) | "
          f"{'троичные веса' if cfg.ternary else 'FP-веса'}")
    print(f"Упакованный размер ≈ {fp['packed_model_mb']:.1f} МБ | устройство {device} | "
          f"{'bf16' if use_bf16 else 'fp32'} | LR {args.lr:g} | {tokens_per_step:,} токенов/шаг")
    print(f"Данные: {val_start:,} обучающих токенов, {len(tokens) - val_start:,} валидационных; "
          f"{args.max_steps} шагов ≈ {args.max_steps * tokens_per_step / max(1, val_start):.1f} эпох")
    print("=" * 86)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg.to_dict(), "args": vars(args)}, f, ensure_ascii=False, indent=2)

    tok_proto = tok.sp.serialized_model_proto()

    def save(path: Path, step: int) -> None:
        # Сначала во временный файл, потом переименование: если процесс убьют посреди записи,
        # старый чекпоинт останется целым.
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": cfg.to_dict(), "step": step, "best_val": best_val,
                    "tokenizer": os.path.abspath(tok_path), "tokenizer_proto": tok_proto,
                    "bytes_per_token": meta["bytes_per_token"]},
                   tmp)
        os.replace(tmp, path)

    # Журнал метрик (по строке JSON на запись) — по нему удобно строить графики обучения
    metrics_file = open(out / "metrics.jsonl", "a", encoding="utf-8")

    def log_metrics(**kv) -> None:
        metrics_file.write(json.dumps(kv, ensure_ascii=False) + "\n")
        metrics_file.flush()

    # Остановка по Ctrl+C / SIGTERM: дописываем текущий шаг и сохраняем last.pt, чтобы потом
    # продолжить с --resume, не потеряв прогресс.
    stop_requested = {"flag": False}

    def request_stop(signum, frame) -> None:
        stop_requested["flag"] = True
        print(f"\nПолучен сигнал {signum}: сохраняю чекпоинт и останавливаюсь ...", flush=True)

    import signal
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, request_stop)
        except (ValueError, OSError):   # не главный поток / платформа без сигнала
            pass

    model.train()
    t_last = time.time()
    t_start = time.time()
    tokens_seen = start_step * tokens_per_step
    for step in range(start_step, args.max_steps):
        lr = lr_at(step, args)
        for g in optimizer.param_groups:
            g["lr"] = lr
        stats = {}
        for _ in range(args.grad_accum):
            x, y = train_sampler.get()
            with autocast_ctx:
                _, loss, stats = train_model(x, y)
            (loss / args.grad_accum).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_step

        if step % args.log_every == 0 or step == args.max_steps - 1:
            dt = time.time() - t_last
            t_last = time.time()
            steps_done = args.log_every if step > start_step else 1
            tps = tokens_per_step * steps_done / max(dt, 1e-9)
            print(f"шаг {step:6d} | loss {float(stats['ce_loss']):.4f} | aux {float(stats['router_aux']):.3f} | "
                  f"глубина {float(stats['core_depth_mean']):.2f} | lr {lr:.2e} | |g| {float(grad_norm):.2f} | "
                  f"{tps:,.0f} ток/с | {tokens_seen / 1e6:.1f} млн ток", flush=True)
            log_metrics(kind="train", step=step, loss=float(stats["ce_loss"]), lr=lr,
                        grad_norm=float(grad_norm), depth=float(stats["core_depth_mean"]),
                        tok_per_s=tps, tokens=tokens_seen, time=time.time())

        last = step == args.max_steps - 1 or stop_requested["flag"]
        if (step > 0 and step % args.eval_every == 0) or (last and not stop_requested["flag"]):
            ev = evaluate(model, val_sampler, args.eval_batches, autocast_ctx)
            bpb = ev["loss"] / math.log(2) / meta["bytes_per_token"]
            improved = ev["loss"] < best_val
            best_val = min(best_val, ev["loss"])
            print(f"  ВАЛИДАЦИЯ: loss {ev['loss']:.4f} | перплексия {math.exp(min(20, ev['loss'])):.1f} | "
                  f"{bpb:.3f} бит/байт | глубина мышления {ev['depth']:.2f}{' | новый лучший' if improved else ''}",
                  flush=True)
            log_metrics(kind="val", step=step, loss=ev["loss"], ppl=math.exp(min(20, ev["loss"])), bpb=bpb,
                        depth=ev["depth"], tokens=tokens_seen, time=time.time())
            if improved:
                save(out / "best.pt", step + 1)
            if args.sample_prompt:
                txt = sample_text(model, tok, args.sample_prompt, 60, 0.8, 50, 0.95, 1.1, device)
                print(f"  ПРИМЕР: {args.sample_prompt!r} -> {txt!r}", flush=True)
        if (step > 0 and step % args.save_every == 0) or last:
            save(out / "last.pt", step + 1)
        if stop_requested["flag"]:
            print(f"Остановлено на шаге {step + 1}. Продолжить: --resume {out}/last.pt", flush=True)
            break
    metrics_file.close()
    print(f"Готово за {(time.time() - t_start) / 60:.1f} мин. "
          f"Чекпоинты: {out}/best.pt (лучший по валидации), {out}/last.pt")


def _tokenizer_for(ckpt_path: str, proto: Optional[bytes], path: Optional[str]) -> Tokenizer:
    """Токенизатор: из самого файла весов, иначе по сохранённому пути, иначе tokenizer.model рядом."""
    if proto:
        return Tokenizer(model_proto=bytes(proto))
    candidates = [path, str(Path(ckpt_path).with_name("tokenizer.model"))]
    for cand in candidates:
        if cand and os.path.exists(cand):
            return Tokenizer(cand)
    raise FileNotFoundError(f"Не найден токенизатор для {ckpt_path} (искал: {candidates})")


def load_model(ckpt_path: str, device):
    """Модель из чекпоинта обучения (.pt) или из упакованных троичных весов (.npz, см. export)."""
    import torch
    if ckpt_path.endswith(".npz"):
        with np.load(ckpt_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        cfg = E.SLRNetConfig(**json.loads(str(arrays.pop("config"))))
        proto = arrays.pop("tokenizer_proto", None)
        for k in [k for k in arrays if k.startswith("meta.")]:
            arrays.pop(k)
        model = E.SLRNetLanguageModel.from_packed(cfg, arrays)
        tok = _tokenizer_for(ckpt_path, None if proto is None else proto.tobytes(), None)
        return model.to(device).eval(), tok
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = E.SLRNetLanguageModel(E.SLRNetConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), _tokenizer_for(ckpt_path, ckpt.get("tokenizer_proto"), ckpt.get("tokenizer"))


def cmd_sample(args: argparse.Namespace) -> None:
    import torch
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model, tok = load_model(args.ckpt, device)
    prompts = [args.prompt] if args.prompt is not None else []
    if not prompts:
        print("Интерактивный режим (пустая строка — выход)")
    while True:
        if prompts:
            prompt = prompts.pop(0)
        elif args.prompt is None:
            try:
                prompt = input("\nВы: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                break
        else:
            break
        t0 = time.time()
        text = sample_text(model, tok, prompt, args.max_new_tokens, args.temperature, args.top_k,
                           args.top_p, args.repetition_penalty, device)
        print(f"SLR-Net: {text}\n  ({time.time() - t0:.2f} c)")


def cmd_export(args: argparse.Namespace) -> None:
    import torch
    model, tok = load_model(args.ckpt, torch.device("cpu"))
    packed = model.export_packed()
    extra: Dict[str, np.ndarray] = {
        # Токенизатор кладём внутрь: один файл весов = готовая к запуску модель
        "tokenizer_proto": np.frombuffer(tok.sp.serialized_model_proto(), dtype=np.uint8),
    }
    if args.ckpt.endswith(".pt"):
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        extra["meta.step"] = np.array([ckpt.get("step", 0)], dtype=np.int64)
        extra["meta.best_val"] = np.array([ckpt.get("best_val", float("nan"))], dtype=np.float32)
        extra["meta.bytes_per_token"] = np.array([ckpt.get("bytes_per_token", float("nan"))], dtype=np.float32)
    np.savez_compressed(args.out, config=json.dumps(model.config.to_dict()), **extra, **packed)
    size = os.path.getsize(args.out) / 1024 ** 2
    fp32 = sum(p.numel() for p in model.parameters()) * 4 / 1024 ** 2
    print(f"Экспортировано в {args.out}: {size:.2f} МБ (в FP32 было бы {fp32:.1f} МБ, сжатие ×{fp32 / size:.1f})")


def cmd_eval(args: argparse.Namespace) -> None:
    """
    Честная проверка «модель действительно обучилась»: на отложенной (валидационной) части корпуса,
    которую модель никогда не видела, сравниваем
      • обученную модель (одну или несколько),
      • ту же архитектуру со случайными весами (ничего не знает: loss ≈ ln V),
      • частотную модель (униграммы): знает только, какие токены встречаются чаще.
    Модель, которая выучила язык, должна быть заметно лучше обеих.
    """
    import torch
    torch.set_flush_denormal(True)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    meta, tokens, _ = load_dataset(args.data)
    val_start, V = int(meta["val_start"]), int(meta["vocab_size"])
    bpt = float(meta["bytes_per_token"])
    sampler = BatchSampler(tokens, val_start, len(tokens), args.seq_len, args.batch_size, device, args.seed)
    rows = []

    n_uni = min(val_start, args.unigram_tokens)
    counts = np.bincount(np.asarray(tokens[:n_uni], dtype=np.int64), minlength=V).astype(np.float64) + 1.0
    logp = np.log(counts / counts.sum())
    val_tokens = np.asarray(tokens[val_start:], dtype=np.int64)
    rows.append(("Частоты токенов (униграммы)", float(-logp[val_tokens].mean()), None))

    first_cfg = None
    for ckpt in args.ckpt:
        model, tok = load_model(ckpt, device)
        if model.config.vocab_size != V:
            raise ValueError(f"{ckpt}: словарь модели {model.config.vocab_size} != словарь данных {V}")
        first_cfg = first_cfg or model.config
        ev = evaluate(model, sampler, args.batches, nullcontext())
        rows.append((f"Обученная модель: {ckpt}", ev["loss"], ev["depth"]))
    if first_cfg is not None:
        rnd = E.SLRNetLanguageModel(E.SLRNetConfig(**first_cfg.to_dict())).to(device)
        ev = evaluate(rnd, sampler, args.batches, nullcontext())
        rows.append(("Та же сеть со случайными весами", ev["loss"], ev["depth"]))

    n_eval = args.batches * args.batch_size * args.seq_len
    print(f"Валидация: {n_eval:,} токенов из отложенной части корпуса ({len(tokens) - val_start:,} токенов, "
          f"модель их не видела). ln V = {math.log(V):.3f}")
    print(f"{'Модель':<60} {'loss':>7} {'перплексия':>11} {'бит/байт':>9}")
    for name, loss, _ in rows:
        print(f"{name[:60]:<60} {loss:7.3f} {math.exp(min(20, loss)):11.1f} {loss / math.log(2) / bpt:9.3f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([{"model": n, "loss": l, "ppl": math.exp(min(20, l)), "bpb": l / math.log(2) / bpt}
                       for n, l, _ in rows], f, ensure_ascii=False, indent=2)


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Обучение языковой модели SLR-Net на больших текстах")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="токенизатор + бинарный корпус")
    p.add_argument("--input", nargs="+", required=True, help="файлы .txt/.srt/.md или папки с ними")
    p.add_argument("--out", default="data")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--tokenizer", default=None, help="использовать готовый tokenizer.model")
    p.add_argument("--tokenizer-sample-lines", type=int, default=2_000_000)
    p.add_argument("--val-fraction", type=float, default=0.005)
    p.add_argument("--keep-hash-lines", action="store_true", help="не выбрасывать строки, начинающиеся с #")

    t = sub.add_parser("train", help="обучение")
    t.add_argument("--data", default="data")
    t.add_argument("--out", default="runs/slr")
    t.add_argument("--preset", default="small", choices=list(E.PRESETS))
    t.add_argument("--set", nargs="*", help="переопределить конфиг: --set d_model=512 max_latent_steps=3")
    t.add_argument("--no-ternary", action="store_true", help="обычные FP-веса (для сравнения качества)")
    t.add_argument("--seq-len", type=int, default=512)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--grad-accum", type=int, default=1)
    t.add_argument("--max-steps", type=int, default=5000)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--min-lr-ratio", type=float, default=0.1)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
    t.add_argument("--decay-fraction", type=float, default=0.2, help="доля шагов спада для WSD")
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--eval-batches", type=int, default=20)
    t.add_argument("--save-every", type=int, default=1000)
    t.add_argument("--log-every", type=int, default=20)
    t.add_argument("--sample-prompt", default="Привет! Как дела?")
    t.add_argument("--device", default="auto")
    t.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    t.add_argument("--compile", action="store_true", help="torch.compile (быстрее и на GPU, и на CPU: ~1.7x)")
    t.add_argument("--resume", default=None, help="продолжить с чекпоинта")
    t.add_argument("--seed", type=int, default=1337)

    s = sub.add_parser("sample", help="генерация текста")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--prompt", default=None, help="без --prompt — интерактивный режим")
    s.add_argument("--max-new-tokens", type=int, default=150)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=50)
    s.add_argument("--top-p", type=float, default=0.95)
    s.add_argument("--repetition-penalty", type=float, default=1.1)
    s.add_argument("--device", default="auto")
    s.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("eval", help="проверка на отложенных данных против случайной и частотной модели")
    e.add_argument("--ckpt", nargs="+", required=True, help="чекпоинты .pt и/или упакованные .npz")
    e.add_argument("--data", required=True)
    e.add_argument("--seq-len", type=int, default=256)
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--batches", type=int, default=50)
    e.add_argument("--unigram-tokens", type=int, default=50_000_000)
    e.add_argument("--json", default=None, help="сохранить таблицу в JSON")
    e.add_argument("--device", default="auto")
    e.add_argument("--seed", type=int, default=4242)

    x = sub.add_parser("export", help="упаковать троичные веса")
    x.add_argument("--ckpt", required=True)
    x.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    {"prepare": cmd_prepare, "train": cmd_train, "sample": cmd_sample, "eval": cmd_eval,
     "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
