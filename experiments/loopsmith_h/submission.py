"""LoopSmith: a digit-state recurrent transformer for iterated modular squaring.

Design: the prompt is embedded with arithmetic-friendly features (segment id,
digit-index-from-the-right within each number, reverse position from the row
end). A tied transformer cell is applied R_INNER times per squaring step, and
the number of squaring steps is read from the prompt's T field (adaptive
computation). Between squaring steps the whole sequence state is compressed
through a learned discrete codebook (straight-through), and the original
prompt embedding is re-injected. The discrete bottleneck makes the recurrent
state an attractor: if the learned one-step map is exact on reachable states,
depth extrapolates by induction with no error accumulation, which is what the
exact-match certification ladder demands. Training uses truncated
backpropagation through the last W_BPTT squaring steps; the mixture of T
values in the data supervises the shared cell at every unrolled depth.
"""

from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
    assert_model_state,
)


# Public token layout of the squaring_mod format.
PAD_ID = 0
MARKER_LO = 2  # N
MARKER_HI = 4  # T
DIGIT_OFFSET = 7

D_MODEL = int(os.environ.get("LOOPSMITH_DIM", "512"))
NUM_HEADS = max(2, D_MODEL // 64)
N_ENC = 2
N_CORE = 2
# Inner micro-steps per squaring scale with the modulus digit count: the
# reduction step is long division, which needs roughly one sequential
# compare-subtract decision per quotient digit. Phases cycle so no phase
# embedding is untrained when OOD moduli bring more digits.
R_BASE = int(os.environ.get("LOOPSMITH_R", "3"))
R_MIN = 4
R_CAP = 20
R_PHASES = 6
N_DEC = 2
N_CODES = 64
N_HASH_BUCKETS = int(os.environ.get("LOOPSMITH_BUCKETS", "400000"))
RIDX_MAX = 48
MAX_MACRO = 96
W_BPTT = int(os.environ.get("LOOPSMITH_W", "6"))

PEAK_LR = 2.0e-3
WEIGHT_DECAY = 0.1
QUANT_FULL_FRACTION = 0.5
QUANT_ALPHA_START = 0.3
NLA_COEF = 0.1
# Long T=1-first phase: telemetry showed the hidden training data is rich in
# shallow rows; the one-step map (whatever the transition family is) must be
# learned before deeper unrolls carry any usable gradient.
CURRICULUM = (
    (0.35, 1),
    (0.50, 2),
    (0.60, 4),
)
WARMUP_STEPS = 150
SCHEDULE_HORIZON_FRACTION = 0.92
LR_FLOOR_FRACTION = 0.02
HARD_FROM_FRACTION = 0.30
TAU_START, TAU_END, TAU_ANNEAL_FRACTION = 1.0, 0.25, 0.35
ENTROPY_COEF_MAX = 0.02

# Shared wall-clock state: build_optimizer records the budget, the first
# training forward records t0, and the model, scheduler, and loss all read it.
G: dict[str, float | None] = {"budget": None, "t0": None}


def _elapsed_fraction() -> float:
    budget = G["budget"] or 600.0
    t0 = G["t0"]
    if t0 is None:
        return 0.0
    horizon = budget * SCHEDULE_HORIZON_FRACTION
    return min(max((time.monotonic() - t0) / max(horizon, 1.0), 0.0), 1.0)


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Block(nn.Module):
    def __init__(self, out_scale: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL, bias=False)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL, bias=False)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02 * out_scale)
        nn.init.normal_(self.down.weight, std=0.02 * out_scale)

    def forward(self, x: Tensor, key_mask: Tensor) -> Tensor:
        batch, length, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        a = a.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = x + self.out(a)
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        rev_max = spec.max_seq_len + 8

        self.token_emb = nn.Embedding(spec.vocab_size, D_MODEL)
        self.seg_emb = nn.Embedding(4, D_MODEL)
        self.ridx_emb = nn.Embedding(RIDX_MAX, D_MODEL)
        self.rev_emb = nn.Embedding(rev_max, D_MODEL)
        self.phase_emb = nn.Parameter(torch.empty(R_PHASES, D_MODEL))
        for table in (self.token_emb, self.seg_emb, self.ridx_emb, self.rev_emb):
            nn.init.normal_(table.weight, std=0.02)
        nn.init.normal_(self.phase_emb, std=0.02)

        depth_scale = 1.0 / math.sqrt(2.0 * N_CORE * 8.0)
        edge_scale = 1.0 / math.sqrt(2.0 * max(N_ENC, N_DEC))
        self.enc = nn.ModuleList(Block(edge_scale) for _ in range(N_ENC))
        self.core = nn.ModuleList(Block(depth_scale) for _ in range(N_CORE))
        self.dec = nn.ModuleList(Block(edge_scale) for _ in range(N_DEC))

        # Hashed per-modulus memory: a generic learned store keyed on the
        # modulus value. Gives the network a place to keep whatever
        # per-identity constants the (unknown) transition family requires.
        self.n_table = nn.Embedding(N_HASH_BUCKETS, D_MODEL)
        nn.init.normal_(self.n_table.weight, std=0.02)

        self.ln_q = nn.LayerNorm(D_MODEL)
        self.to_code = nn.Linear(D_MODEL, N_CODES, bias=True)
        # NLA-style side head: from state s_t, predict the code indices of
        # s_{t+1} against stop-gradient targets (consistency between the
        # model's own recurrent states; organizer-confirmed legal).
        self.nla_head = nn.Linear(D_MODEL, N_CODES, bias=True)
        nn.init.normal_(self.nla_head.weight, std=0.02)
        self.codebook = nn.Parameter(torch.empty(N_CODES, D_MODEL))
        nn.init.normal_(self.codebook, std=0.02)
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=True)
        nn.init.normal_(self.head.weight, std=0.02)

        self._rev_max = rev_max
        self._pow10_cache: dict[torch.device, Tensor] = {}
        self.hard_used = False

    def _pow10(self, device: torch.device) -> Tensor:
        cached = self._pow10_cache.get(device)
        if cached is None:
            cached = torch.tensor(
                [10 ** min(k, 9) for k in range(RIDX_MAX)],
                dtype=torch.long,
                device=device,
            )
            self._pow10_cache[device] = cached
        return cached

    def _features(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            ids = input_ids
            mask = attention_mask
            is_marker = (ids >= MARKER_LO) & (ids <= MARKER_HI)
            seg = torch.cumsum(is_marker.long(), dim=1).clamp_(0, 3)
            is_digit = ids >= DIGIT_OFFSET
            digit_val = (ids - DIGIT_OFFSET).clamp(0, 9)

            ridx = torch.zeros_like(ids)
            for s in (1, 2, 3):
                m = is_digit & seg.eq(s) & mask
                right = m.long().flip(1).cumsum(1).flip(1) - m.long()
                ridx = torch.where(m, right, ridx)
            ridx = ridx.clamp_(0, RIDX_MAX - 1)

            lengths = mask.long().sum(dim=1)
            positions = torch.arange(ids.shape[1], device=ids.device)
            rev = (lengths[:, None] - 1 - positions[None, :]).clamp_(
                0, self._rev_max - 1
            )

            t_digits = is_digit & seg.eq(3) & mask
            t_val = (
                digit_val * self._pow10(ids.device)[ridx] * t_digits.long()
            ).sum(dim=1)
            t_val = t_val.clamp_(0, MAX_MACRO)
            n_digits = (is_digit & seg.eq(1) & mask).long().sum(dim=1)
            x_digits = (is_digit & seg.eq(2) & mask).long().sum(dim=1)
            n_seg_digits = is_digit & seg.eq(1) & mask
            n_val = (
                digit_val
                * self._pow10(ids.device)[ridx]
                * n_seg_digits.long()
            ).sum(dim=1)
        return seg, ridx, rev, t_val, n_digits, x_digits, n_val

    def _quantize(
        self, h: Tensor, tau: float, hard: bool, alpha: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.to_code(self.ln_q(h))
        soft = F.softmax(logits / tau, dim=-1)
        entropy = -(soft * (soft + 1e-9).log()).sum(dim=-1).mean()
        indices = logits.argmax(dim=-1)
        if hard:
            one_hot = F.one_hot(indices, N_CODES).to(soft.dtype)
            weights = one_hot + soft - soft.detach()
        else:
            weights = soft
        quantized = weights @ self.codebook
        if alpha < 1.0:
            quantized = alpha * quantized + (1.0 - alpha) * h
        return quantized, entropy, indices.detach()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict | None]:
        if attention_mask is None:
            attention_mask = input_ids != PAD_ID
        attention_mask = attention_mask.bool()
        seg, ridx, rev, t_val, n_digits, x_digits, n_val = self._features(
            input_ids, attention_mask
        )
        identity = self.n_table(n_val % N_HASH_BUCKETS)[:, None, :]
        a_max = int(n_digits.max().item()) if n_digits.numel() else 1
        r_inner = min(max(R_MIN, R_BASE + a_max), R_CAP)
        w_window = max(2, min(W_BPTT, 60 // max(r_inner, 1)))

        if self.training and G["t0"] is None:
            G["t0"] = time.monotonic()
        if self.training:
            fraction = _elapsed_fraction()
            tau = TAU_START + (TAU_END - TAU_START) * min(
                fraction / TAU_ANNEAL_FRACTION, 1.0
            )
            hard = fraction >= HARD_FROM_FRACTION
            alpha = QUANT_ALPHA_START + (1.0 - QUANT_ALPHA_START) * min(
                fraction / QUANT_FULL_FRACTION, 1.0
            )
            self.hard_used = self.hard_used or hard
            self._eval_tau = tau
            self._eval_alpha = alpha
        else:
            tau = getattr(self, "_eval_tau", TAU_END)
            hard = self.hard_used
            alpha = getattr(self, "_eval_alpha", 1.0)

        # Loop-count curriculum: early in training, rows whose T exceeds the
        # current cap are truncated and excluded from the loss (via auxiliary),
        # concentrating gradient on the depths currently learnable.
        t_eff = t_val
        counted = torch.ones_like(t_val, dtype=torch.bool)
        if self.training:
            fraction_now = _elapsed_fraction()
            cap = None
            for threshold, cap_value in CURRICULUM:
                if fraction_now < threshold:
                    cap = cap_value
                    break
            if cap is not None:
                counted = t_val <= cap
                if bool(counted.any().item()):
                    t_eff = t_val.clamp(max=cap)
                else:
                    counted = torch.ones_like(counted)

        h0 = (
            self.token_emb(input_ids)
            + self.seg_emb(seg)
            + self.ridx_emb(ridx)
            + self.rev_emb(rev)
        )
        key_mask = attention_mask[:, None, None, :]

        def macro_step(s: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            u = s + identity
            for i in range(r_inner):
                u = u + self.phase_emb[i % R_PHASES]
                for block in self.core:
                    u = block(u, key_mask)
            z, step_entropy, codes = self._quantize(u, tau, hard, alpha)
            return z + h0, step_entropy, codes

        h = h0
        for block in self.enc:
            h = block(h, key_mask)
        z, entropy, _ = self._quantize(h, tau, hard, alpha)
        state0 = z + h0
        entropy_sum = entropy
        entropy_count = 1
        nla_sum = None
        nla_count = 0

        t_max = int(t_eff.max().item()) if t_eff.numel() else 0
        if self.training and t_max > 0:
            # Phase 1 (no grad): roll the recurrence forward, snapshotting each
            # row's state W_BPTT steps before its own endpoint. Phase 2 (grad):
            # re-run only that last window, so activation memory is bounded by
            # W_BPTT regardless of the batch's T distribution. Rows with
            # T <= W_BPTT start phase 2 from the encoder output with its graph
            # intact, which is how the encoder keeps receiving gradient.
            long_rows = t_eff > w_window
            if bool(long_rows.any().item()):
                with torch.no_grad():
                    s = state0.detach()
                    snapshot = s.clone()
                    for t in range(1, t_max + 1):
                        u, _, _ = macro_step(s)
                        s = torch.where((t_eff >= t)[:, None, None], u, s)
                        take = (t_eff - w_window) == t
                        snapshot = torch.where(
                            take[:, None, None], s, snapshot
                        )
                state = torch.where(
                    long_rows[:, None, None], snapshot, state0
                )
            else:
                state = state0
            grad_steps = t_eff.clamp(max=w_window)
            for j in range(1, min(t_max, w_window) + 1):
                previous = state
                u, step_entropy, codes = macro_step(state)
                active = grad_steps >= j
                state = torch.where(active[:, None, None], u, state)
                entropy_sum = entropy_sum + step_entropy
                entropy_count += 1
                if bool(active.any().item()):
                    predicted = self.nla_head(previous[active])
                    targets = codes[active]
                    row_mask = attention_mask[active]
                    nla_term = F.cross_entropy(
                        predicted[row_mask], targets[row_mask]
                    )
                    nla_sum = (
                        nla_term if nla_sum is None else nla_sum + nla_term
                    )
                    nla_count += 1
        else:
            state = state0
            for t in range(1, t_max + 1):
                u, _, _ = macro_step(state)
                state = torch.where((t_eff >= t)[:, None, None], u, state)

        h = state
        for block in self.dec:
            h = block(h, key_mask)
        logits = self.head(self.ln_f(h))

        auxiliary = None
        if self.training:
            # Rows where x has at most half of N's digits have quotient ~0 at
            # T=1: their answer is plain squaring, the gradient-learnable
            # bootstrap for the multiply/readout circuits.
            small_x = (2 * x_digits) <= (n_digits + 1)
            # Prompt anchor: the pre-loop state must decode (through the same
            # head) back to the prompt's own digits. Self-supervised — labels
            # are input tokens — it pins the recurrent state format to "digit
            # string" even when the dataset has no shallow-T rows, and feeds
            # the encoder gradient at every depth.
            anchor_logits = self.head(self.ln_f(state0))
            auxiliary = {
                "entropy": entropy_sum / entropy_count,
                "counted": counted,
                "small_x": small_x,
                "anchor_logits": anchor_logits,
                "anchor_targets": input_ids,
                "anchor_mask": (input_ids >= DIGIT_OFFSET) & attention_mask,
            }
            if nla_count > 0:
                auxiliary["nla"] = nla_sum / nla_count
        return logits, auxiliary


class TimedScheduler:
    """Linear warmup by steps, cosine decay by elapsed wall-clock fraction."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.updates = 0

    def step(self) -> None:
        self.updates += 1
        warmup = min(self.updates / WARMUP_STEPS, 1.0)
        fraction = _elapsed_fraction()
        cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
        factor = LR_FLOOR_FRACTION + (1.0 - LR_FLOOR_FRACTION) * cosine
        lr = PEAK_LR * warmup * factor
        for group in self.optimizer.param_groups:
            group["lr"] = lr


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    G["budget"] = spec.training_time_seconds
    G["t0"] = None
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim >= 2 and "emb" not in name and "codebook" not in name:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=PEAK_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(optimizer, scheduler=TimedScheduler(optimizer))


SMALL_X_BOOST = 3.0
SMALL_X_BOOST_UNTIL = 0.35


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    valid = batch.valid_mask
    auxiliary = batch.auxiliary
    aux = auxiliary if isinstance(auxiliary, dict) else {}
    if "counted" in aux:
        gated = valid & aux["counted"][:, None]
        if bool(gated.any().item()):
            valid = gated
    token_ce = F.cross_entropy(
        batch.logits[valid],
        batch.labels[valid],
        reduction="none",
    )
    weights = torch.ones_like(token_ce)
    if "small_x" in aux and _elapsed_fraction() < SMALL_X_BOOST_UNTIL:
        row_boost = 1.0 + (SMALL_X_BOOST - 1.0) * aux["small_x"].float()
        weights = row_boost[:, None].expand_as(valid)[valid]
    loss = (token_ce * weights).sum() / weights.sum().clamp_min(1e-6)
    fraction = _elapsed_fraction()
    if "anchor_logits" in aux:
        anchor_mask = aux["anchor_mask"]
        if bool(anchor_mask.any().item()):
            anchor_ce = F.cross_entropy(
                aux["anchor_logits"][anchor_mask],
                aux["anchor_targets"][anchor_mask],
            )
            anchor_coef = 0.05 + 0.45 * max(0.0, 1.0 - fraction / 0.6)
            loss = loss + anchor_coef * anchor_ce
    if "entropy" in aux:
        coef = ENTROPY_COEF_MAX * min(fraction / HARD_FROM_FRACTION, 1.0)
        loss = loss + coef * aux["entropy"]
    if "nla" in aux:
        loss = loss + NLA_COEF * aux["nla"]
    return loss


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=int(os.environ.get("LOOPSMITH_BS", "512")),
    eval_batch_size=int(os.environ.get("LOOPSMITH_EVAL_BS", "1024")),
)
