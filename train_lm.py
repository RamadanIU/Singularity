#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение настоящей языковой модели SLR-Net на больших текстах (PyTorch).
=======================================================================

Четыре команды:

  1) Подготовка данных: обучает субсловный токенизатор (SentencePiece Unigram) и переводит
     все тексты в компактный бинарный файл токенов (uint16, читается через memmap — подходит
     для корпусов на десятки гигабайт, в оперативную память целиком не грузится).

       python train_lm.py prepare --input книги/ субтитры.srt --out data --vocab-size 16384

  2) Обучение:
       python train_lm.py train --data data --out runs/small --preset small --max-steps 20000

  3) Генерация текста:
       python train_lm.py sample --ckpt runs/small/best.pt --prompt "Почему небо синее?"

  4) Экспорт троичных весов (5 тритов в байт) и отчёт о размере:
       python train_lm.py export --ckpt runs/small/best.pt --out runs/small/model_packed.npz

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


def iter_clean_lines(path: Path, keep_hash_lines: bool = False) -> Iterator[str]:
    """Строки файла без мусора: таймкоды и номера субтитров, HTML-теги, ремарки [музыка]."""
    is_srt = path.suffix.lower() == ".srt"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
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

    def __init__(self, model_path: str):
        import sentencepiece as spm
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
        self.rng = np.random.default_rng(seed)

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
    train_sampler = BatchSampler(tokens, 0, val_start, args.seq_len, args.batch_size, device, args.seed)
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

    def save(path: Path, step: int) -> None:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": cfg.to_dict(), "step": step, "best_val": best_val,
                    "tokenizer": os.path.abspath(tok_path), "bytes_per_token": meta["bytes_per_token"]},
                   path)

    model.train()
    t_last = time.time()
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

        if step % args.log_every == 0 or step == args.max_steps - 1:
            dt = time.time() - t_last
            t_last = time.time()
            steps_done = args.log_every if step > start_step else 1
            print(f"шаг {step:6d} | loss {float(stats['ce_loss']):.4f} | aux {float(stats['router_aux']):.3f} | "
                  f"глубина {float(stats['core_depth_mean']):.2f} | lr {lr:.2e} | |g| {float(grad_norm):.2f} | "
                  f"{tokens_per_step * steps_done / max(dt, 1e-9):,.0f} ток/с")

        last = step == args.max_steps - 1
        if (step > 0 and step % args.eval_every == 0) or last:
            ev = evaluate(model, val_sampler, args.eval_batches, autocast_ctx)
            bpb = ev["loss"] / math.log(2) / meta["bytes_per_token"]
            improved = ev["loss"] < best_val
            best_val = min(best_val, ev["loss"])
            print(f"  ВАЛИДАЦИЯ: loss {ev['loss']:.4f} | перплексия {math.exp(min(20, ev['loss'])):.1f} | "
                  f"{bpb:.3f} бит/байт | глубина мышления {ev['depth']:.2f}{' | новый лучший' if improved else ''}")
            if improved:
                save(out / "best.pt", step + 1)
            if args.sample_prompt:
                txt = sample_text(model, tok, args.sample_prompt, 60, 0.8, 50, 0.95, 1.1, device)
                print(f"  ПРИМЕР: {args.sample_prompt!r} -> {txt!r}")
        if (step > 0 and step % args.save_every == 0) or last:
            save(out / "last.pt", step + 1)
    print(f"Готово. Чекпоинты: {out}/best.pt (лучший по валидации), {out}/last.pt")


def load_model(ckpt_path: str, device):
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = E.SLRNetLanguageModel(E.SLRNetConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), Tokenizer(ckpt["tokenizer"])


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
    model, _ = load_model(args.ckpt, torch.device("cpu"))
    packed = model.export_packed()
    np.savez_compressed(args.out, config=json.dumps(model.config.to_dict()), **packed)
    size = os.path.getsize(args.out) / 1024 ** 2
    fp32 = sum(p.numel() for p in model.parameters()) * 4 / 1024 ** 2
    print(f"Экспортировано в {args.out}: {size:.2f} МБ (в FP32 было бы {fp32:.1f} МБ, сжатие ×{fp32 / size:.1f})")


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
    t.add_argument("--compile", action="store_true", help="torch.compile (быстрее на GPU)")
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

    x = sub.add_parser("export", help="упаковать троичные веса")
    x.add_argument("--ckpt", required=True)
    x.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    {"prepare": cmd_prepare, "train": cmd_train, "sample": cmd_sample, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
