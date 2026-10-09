"""Parameterized quadratic-world generator for the failure-boundary ladder.

Each stage isolates one dimension: identity count, c magnitude, T mixture.
Transition: y = x^2 + c (mod N), c fixed per modulus.
"""

from __future__ import annotations

import argparse
import random

from data.counting import write_dataset_config, write_split_files
from data.squaring_mod import (
    VOCAB_SIZE,
    is_probable_prime,
    tokenize_squaring_mod_with_result,
)


def sample_semiprime(rng, bits):
    p_bits = bits // 2
    q_bits = bits - p_bits

    def sample_prime(b):
        while True:
            candidate = rng.randrange(1 << (b - 1), 1 << b) | 1
            if is_probable_prime(candidate):
                return candidate

    while True:
        p = sample_prime(p_bits)
        q = sample_prime(q_bits)
        n = p * q
        if p != q and n.bit_length() == bits:
            return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--moduli", type=int, required=True)
    ap.add_argument("--bits", type=int, default=13)
    ap.add_argument("--c_mode", choices=("zero", "small", "full"), default="full")
    ap.add_argument("--t_set", default="1")
    ap.add_argument("--rows", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    t_values = [int(v) for v in args.t_set.split(",")]

    moduli = []
    seen_n = set()
    while len(moduli) < args.moduli:
        n = sample_semiprime(rng, args.bits)
        if n in seen_n:
            continue
        seen_n.add(n)
        if args.c_mode == "zero":
            c = 0
        elif args.c_mode == "small":
            c = rng.randrange(1, 100)
        else:
            c = rng.randrange(1, n)
        moduli.append((n, c))

    def step(v, n, c):
        return (v * v + c) % n

    records = []
    seen = set()

    def add(count, split, t_pool, fresh_only=False):
        made = 0
        attempts = 0
        while made < count and attempts < count * 50:
            attempts += 1
            n, c = rng.choice(moduli)
            t = rng.choice(t_pool)
            x = rng.randrange(2, n)
            key = (n, x, t)
            if key in seen:
                continue
            seen.add(key)
            y = x % n
            for _ in range(t):
                y = step(y, n, c)
            input_ids, labels = tokenize_squaring_mod_with_result(
                n, x, t, y, separate_input_output=True
            )
            records.append(
                {"input_ids": input_ids, "labels": labels, "split": split}
            )
            made += 1

    add(args.rows, "train", t_values)
    add(min(2000, args.rows // 10), "test", t_values)
    for t in (1, 2, 4):
        add(192, f"depth_t_{t}", [t])

    max_len = max(len(r["input_ids"]) for r in records)
    from pathlib import Path

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_split_files(out, records)
    write_dataset_config(
        out,
        {
            "vocab_size": VOCAB_SIZE,
            "max_seq_len": max_len,
            "generator": "quad_ladder",
            "moduli": args.moduli,
            "bits": args.bits,
            "c_mode": args.c_mode,
            "t_set": t_values,
            "rows": len(records),
        },
    )
    print(f"wrote {len(records)} rows to {out} (max_len {max_len})")


if __name__ == "__main__":
    main()
