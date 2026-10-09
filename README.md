# LoopSmith

My entry to [One Layer Deeper](https://github.com/tilde-research/one-layer-deeper), the Core Automation × Tilde Research competition on learning iterated computation: given `x`, `N` and `T`, square `x` modulo `N` `T` times, and stay exactly right at depths beyond those seen in training.

LoopSmith is a digit-state recurrent transformer. A tied transformer cell runs a few times per squaring step, for as many steps as the prompt's `T` asks for. Between steps the whole sequence state is compressed through a learned 64-entry codebook (straight-through) and the prompt embedding is re-injected. The idea is that the discrete bottleneck makes the recurrent state an attractor: if the learned one-step map is exact on reachable states, depth extrapolates by induction with no error accumulating. The full design notes are in the docstring at the top of [`submission.py`](submission.py).

It didn't certify on Hard, which required every single-step (`T=1`) example to be correct first, but it made the retrospective's list of [interesting ideas that didn't quite work](https://www.coreauto.com/blog/what-the-models-learned-in-one-layer-deeper).

## What's here

**`submission.py`** is the Hard submission, byte-identical to upload `1f7c46f2-2982-4aff-8eb2-5274e4774c9d` (submitted 2026-09-01 00:01 UTC, SHA-256 `3c7561b6a663d6a5a748755d1a526b6895ae877d2dbd9604ca020ee6b8977e58`), which is archived in [GPUMODE/one-layer-deeper-submissions](https://huggingface.co/datasets/GPUMODE/one-layer-deeper-submissions).

**`experiments/`** holds the other variants, as they were left on disk:

| Variant | Last edited (2026) |
|---|---|
| `loopsmith_v2` | Aug 28 17:25 |
| `loopsmith_sam` | Aug 28 18:38 |
| `loopsmith_diag` | Aug 29 00:42 |
| `loopsmith_x` | Aug 29 15:47 |
| `loopsmith_w` | Aug 29 20:01 |
| `loopsmith_id` | Aug 30 02:15 |
| `loopsmith_j`, `loopsmith_k` | Aug 30 18:20 (identical) |
| `loopsmith_h` | Aug 31 14:17 |

`loopsmith_k/submission_orig_backup.py` (Aug 30 04:18) is an earlier copy of `loopsmith_k`. The submitted version, `submission.py` above, was the plain `loopsmith` variant, last edited Aug 29 00:42.

**`scripts/`** has the local evaluation helpers. `gen_h1_replica*.py` generate datasets shaped like the Hard tier using the competition's public data generators, `gen_quad_ladder.py` builds smaller staged datasets, and `ladder.sh`, `controls.sh`, `run11.sh` and `run13.sh` are the runs I used them for.

**`manifests/`** has the runner manifests those scripts start from.

## Running it

Everything runs against a checkout of the competition repo, which supplies the data generators and the runner:

```bash
git clone https://github.com/tilde-research/one-layer-deeper.git
cd one-layer-deeper
uv venv .venv && source .venv/bin/activate && uv sync
python -m benchmark.runner \
  --manifest benchmark/manifests/smoke_cpu.json \
  --submission-file ../loopsmith/submission.py
```

The run scripts read everything else from this repository, so nothing needs copying. Run them from inside the checkout, or from anywhere with `ONE_LAYER_DEEPER` pointing at it:

```bash
ONE_LAYER_DEEPER=~/one-layer-deeper loopsmith/scripts/ladder.sh
```

Generated data and derived manifests are written into the checkout's `data/generated/` and `benchmark/manifests/`. All of these variables are optional:

| Variable | Default | Purpose |
|---|---|---|
| `ONE_LAYER_DEEPER` | current directory | the competition checkout |
| `PYTHON` | `.venv/bin/python` | an interpreter with the checkout's dependencies |
| `LOOPSMITH_BS` | 512 | training batch size |
| `LOOPSMITH_EVAL_BS` | 1024 | evaluation batch size |
| `LOOPSMITH_DIM` | 512 | model width |
| `LOOPSMITH_R` | 3 | base cell applications per squaring step, scaled up with the modulus's digit count |
| `LOOPSMITH_W` | 6 | squaring steps to backpropagate through |

The `LOOPSMITH_*` variables are read by the submission itself, so they work with any run. The defaults above are the submission's. The run scripts default to smaller batches, and `ladder.sh` to width 256, but setting a variable overrides them; `controls.sh` sets the width itself for each of its runs.

Hard was only evaluated through the hosted service, so a local run can't reproduce the official Hard result.

## License

MIT, for the files in this repository. The competition harness they run inside is Apache-2.0, in [its own repository](https://github.com/tilde-research/one-layer-deeper).
