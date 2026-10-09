"""Generate an offline replica of the hidden Hard (h1) distribution.

Grounded in loss-channel telemetry from scored Hard runs (2026-08-28/29):
moduli ~24-27 bits (7-9 digits), x runs ~2 digits shorter than N (short-x
on-ramp), T bimodal: heavy T=1 mass plus a deep tail to 64. Reuses the
competition's own tokenizer/transition/split-writer so records are
format-identical to the hosted datasets. Offline experimentation only.

Run from the one-layer-deeper repo root:
    python gen_h1_replica.py --out data/generated/h1_replica_v1 --seed 7
"""

from __future__ import annotations

import argparse
import random

from data.counting import write_dataset_config, write_split_files
from data.squaring_mod import (
    VOCAB_SIZE,
    is_probable_prime,
    tokenize_squaring_mod_with_result,
    trapdoor_squaring_mod,
)


ID_BITS = (24, 25, 26, 27)
OOD_BITS = (28, 29)
N_TRAIN_MODULI = 2000
N_OOD_MODULI = 400
TRAIN_ROWS = 1_000_000
TEST_ROWS = 2000
LADDER = (1, 2, 4, 8, 16, 32, 64)
ID_RUNG_ROWS = 512
OOD_RUNG_ROWS = 256
DEEP_T_CHOICES = tuple(range(3, 65))


def sample_semiprime(rng: random.Random, bits: int) -> tuple[int, int, int]:
    p_bits = bits // 2
    q_bits = bits - p_bits

    def sample_prime(b: int) -> int:
        while True:
            candidate = rng.randrange(1 << (b - 1), 1 << b) | 1
            if is_probable_prime(candidate):
                return candidate

    while True:
        p = sample_prime(p_bits)
        q = sample_prime(q_bits)
        n = p * q
        if p != q and n.bit_length() == bits:
            return n, p, q


def sample_t(rng: random.Random) -> int:
    return rng.choice((4, 8, 16, 32, 64))


def sample_x(rng: random.Random, n: int) -> int:
    if rng.random() < 0.5:
        return rng.randrange(2, n)
    digits = len(str(n))
    length = rng.randint(1, digits)
    lo = max(2, 10 ** (length - 1))
    hi = min(10**length, n)
    if lo >= hi:
        return rng.randrange(2, n)
    return rng.randrange(lo, hi)


def build_record(modulus, p, q, x, t, split):
    result = trapdoor_squaring_mod(x, t, p, q)
    input_ids, labels = tokenize_squaring_mod_with_result(
        modulus, x, t, result, separate_input_output=True
    )
    return {"input_ids": input_ids, "labels": labels, "split": split}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    print("sampling moduli...", flush=True)
    id_moduli = [
        sample_semiprime(rng, rng.choice(ID_BITS)) for _ in range(N_TRAIN_MODULI)
    ]
    ood_moduli = [
        sample_semiprime(rng, rng.choice(OOD_BITS)) for _ in range(N_OOD_MODULI)
    ]

    records = []
    seen: set[tuple[int, int, int]] = set()

    def add_rows(count, moduli, t_sampler, split):
        made = 0
        while made < count:
            n, p, q = rng.choice(moduli)
            t = t_sampler()
            x = sample_x(rng, n)
            key = (n, x, t)
            if key in seen:
                continue
            seen.add(key)
            records.append(build_record(n, p, q, x, t, split))
            made += 1

    print("train rows (chained: x pools reused across T)...", flush=True)
    T_SET = (4, 8, 16, 32, 64)
    pool_size = max(1, TRAIN_ROWS // (len(id_moduli) * len(T_SET)))
    for n, p, q in id_moduli:
        for _ in range(pool_size):
            x = sample_x(rng, n)
            for t in T_SET:
                key = (n, x, t)
                if key in seen:
                    continue
                seen.add(key)
                records.append(build_record(n, p, q, x, t, "train"))
    print("test rows...", flush=True)
    add_rows(TEST_ROWS, id_moduli, lambda: sample_t(rng), "test")
    for t in LADDER:
        add_rows(ID_RUNG_ROWS, id_moduli, lambda t=t: t, f"depth_t_{t}")
        add_rows(OOD_RUNG_ROWS, ood_moduli, lambda t=t: t, f"depth_ood_n_t_{t}")
        print(f"ladder T={t} done", flush=True)

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
            "generator": "h1_replica_v4_deeponly_chained",
            "id_bits": list(ID_BITS),
            "ood_bits": list(OOD_BITS),
            "train_moduli": N_TRAIN_MODULI,
            "rows": len(records),
        },
    )
    print(f"wrote {len(records)} records to {out}, max_seq_len={max_len}")


if __name__ == "__main__":
    main()
