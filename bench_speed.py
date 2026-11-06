#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Замер скорости обучения пресетов SLR-Net / Singularity: токенов в секунду на случайных данных.

    python bench_speed.py --preset mini2 --batch 16 --seq 256 --dtype fp16          # на видеокарте
    python bench_speed.py --preset s300proxy --batch 4 --seq 1024 --dtype fp16 --compile
    python bench_speed.py --preset mini2 --batch 16 --seq 256 --dtype bf16          # CPU с AMX

Печатает одну строку JSON: токенов/с, «полезные» TFLOPS по правилу 6·N·T, пик памяти видеокарты.
Цифры из PLAN_SINGULARITY_300M_RU.md (раздел 14) получены этим скриптом на Tesla T4 бесплатного Colab.
"""
import argparse
import json
import time
from contextlib import nullcontext

import torch

import slr_net_engine as E


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="mini2", choices=list(E.PRESETS))
    p.add_argument("--set", nargs="*", default=[], help="переопределить конфиг: --set ssm_type=mamba2")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2, help="шагов до замера (и на компиляцию)")
    args = p.parse_args()

    from train_lm import config_overrides
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    cfg = E.make_config(args.preset, **config_overrides(args.set))
    model = E.SLRNetLanguageModel(cfg).to(dev)
    run = torch.compile(model) if args.compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "fp16")
    if args.dtype == "fp32":
        ctx = nullcontext()
    else:
        ctx = torch.autocast(dev.type, dtype=torch.float16 if args.dtype == "fp16" else torch.bfloat16)
    x = torch.randint(0, cfg.vocab_size, (args.batch, args.seq + 1), device=dev)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    t_begin = time.time()
    t0 = None
    for i in range(args.warmup + args.steps):
        if i == args.warmup:
            sync()
            t0 = time.time()
        with ctx:
            _, loss, _ = run(x[:, :-1], x[:, 1:])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    sync()
    dt = (time.time() - t0) / args.steps
    n = sum(q.numel() for q in model.parameters())
    tps = args.batch * args.seq / dt
    print(json.dumps({
        "preset": args.preset, "set": args.set, "device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
        "dtype": args.dtype, "compile": args.compile, "params_m": round(n / 1e6, 1), "batch": args.batch,
        "seq": args.seq, "tok_s": round(tps), "tflops_6nd": round(6 * n * tps / 1e12, 2),
        "warmup_s": round(t0 - t_begin, 1), "loss": round(float(loss.detach()), 3),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev.type == "cuda" else None,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
