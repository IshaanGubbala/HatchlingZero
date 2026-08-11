"""HZ-0H: faithful PyTorch port of the official BDH-GPU `train.py`.

NEW 2026-08-11, part of the HZ-0H clean-restart grounding. Written the
same way `reference/hz0h_bdh_torch.py` was rewritten: a byte-faithful
transcription of the real `github.com/pathwaycom/bdh/train.py` (fetched
complete and verbatim, not summarized) below the
`# --- REAL train.py, verbatim ---` marker, with this project's own
reusable extensions added BELOW an explicit end-of-verbatim marker
instead of interleaved into the transcription.

Why this file exists: every H3-T training-rule script this session
independently hand-rolled its own training loop, and every one of them
(except one throwaway redo script) used `model(idx, targets=idx)` --
feeding the SAME sequence as both input and target lets BDH shortcut
through the residual stream instead of doing real next-token
prediction (the identical bug H5 already found and fixed for this exact
class, see `reference/hz0h_bdh_h5_memory_tasks.py`). The real
`get_batch` below shows the actual official convention directly:
`x = data[i:i+T]`, `y = data[i+1:i+1+T]` -- shifted by one position,
never the same sequence. `shifted_target_batch` (in the extension
section) generalizes that exact slicing to arbitrary in-memory integer
tensors (not just a byte-mapped file), so every HZ-0H script rebuilt
against this file gets the correct convention by construction rather
than by remembering to do it right.

`train_step` (also in the extensions) factors the real per-iteration
update logic (forward under `ctx`, `scaler.scale(loss).backward()`,
`scaler.step(optimizer)`, `scaler.update()`, `optimizer.zero_grad()`)
out of the verbatim `__main__` loop into a reusable function, so
scripts get the exact real update rule instead of a hand-rolled
approximation. On non-CUDA devices (this project runs on Mac
MPS/CPU) `ctx` is `nullcontext()` and `GradScaler` is constructed with
`enabled=False`, which upstream's own code already does for anything
that isn't `float16`+CUDA -- so on Mac this collapses to plain
`loss.backward(); optimizer.step(); optimizer.zero_grad()`, exactly as
it should.

This is an ISOLATED ORACLE, same contract as `reference/hz0h_bdh_torch.py`:
it does not touch, call, or depend on any HZ-0A-G mechanism, and nothing
in HZ's canonical backbone depends on this file.
"""
from __future__ import annotations

import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH, BDHConfig

# --- REAL train.py, verbatim (github.com/pathwaycom/bdh, fetched complete ---
# and diffed directly, not summarized). `import bdh` -> `from reference import
# hz0h_bdh_torch as bdh` is the only substitution; everything else below,
# including the real Shakespeare data path and CUDA-only autocast/GradScaler
# logic, is transcribed as-is.

import reference.hz0h_bdh_torch as bdh  # noqa: E402  (matches upstream's `import bdh`)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


# Configuration
BDH_CONFIG = bdh.BDHConfig()
BLOCK_SIZE = 512
BATCH_SIZE = 32
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")


# Fetch the tiny Shakespeare dataset
def fetch_data():
    if not os.path.exists(input_file_path):
        import requests

        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)


def get_batch(split):
    # treat the file as bytes
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)) :]
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64))
            for i in ix
        ]
    )
    if torch.cuda.is_available():
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def eval(model):
    model.eval()


def run_reference_training():
    """The real `if __name__ == "__main__":` body, transcribed verbatim
    as a callable function instead of module-level `__main__` code (so
    it can be used as a genuine end-to-end reproduction check without
    running on import). Trains BDH-GPU on real tinyshakespeare bytes
    with the exact official recipe/hyperparameters above."""
    fetch_data()

    model = bdh.BDH(BDH_CONFIG).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    x, y = get_batch("train")

    loss_acc = 0
    loss_steps = 0
    for step in range(MAX_ITERS):
        with ctx:
            logits, loss = model(x, y)
        x, y = get_batch("train")
        loss_acc += loss
        loss_steps += 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if step % LOG_FREQ == 0:
            print(f"Step: {step}/{MAX_ITERS} loss {loss_acc.item() / loss_steps:.3}")
            loss_acc = 0
            loss_steps = 0
    print("Training done, now generating a sample ")
    model.eval()
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)
    return model

# --- end of verbatim upstream source ---------------------------------------


# --- HZ-0H extension: reusable real-recipe building blocks (NOT in upstream) --
# `get_batch` above is real but Shakespeare-file-specific; every HZ-0H script
# needs the SAME shifted-target slicing over its own in-memory synthetic data
# (passkey sequences, random tokens, multi-hop chains, ...). `shifted_target_batch`
# generalizes `get_batch`'s exact convention (`x = seq[:-1]`, `y = seq[1:]`,
# equivalent to upstream's `data[i:i+T]` / `data[i+1:i+1+T]`) so scripts get
# it right by construction instead of re-deriving it (and risking the
# `targets=idx` bug this file exists to prevent a recurrence of).

def shifted_target_batch(full_sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`full_sequences`: (B, T+1) int tensor. Returns `(x, y)` = `(seq[:, :-1],
    seq[:, 1:])`, each (B, T) -- the real official target convention
    (`get_batch`'s `data[i:i+T]`/`data[i+1:i+1+T]`), generalized to any
    already-materialized integer tensor rather than a byte-mapped file.
    `x` and `y` are NEVER the same tensor -- that degenerate case
    (`model(idx, targets=idx)`) is exactly the bug this function exists
    to make structurally impossible to reintroduce."""
    if full_sequences.shape[1] < 2:
        raise ValueError(f"need at least 2 positions to form a shifted (x, y) pair, got shape {tuple(full_sequences.shape)}")
    # .contiguous(): BDH.forward does targets.view(-1), which requires a
    # contiguous tensor -- full_sequences[:, 1:] is a non-contiguous view
    # (offset stride from the parent tensor), so materialize both slices.
    return full_sequences[:, :-1].contiguous(), full_sequences[:, 1:].contiguous()


def build_optimizer(model: BDH, lr: float = LEARNING_RATE, weight_decay: float = WEIGHT_DECAY) -> torch.optim.AdamW:
    """The real optimizer: plain `AdamW(model.parameters(), lr, weight_decay)`
    over EVERY parameter (`embed`, `encoder`, `encoder_v`, `decoder`,
    `lm_head`) -- upstream has no separate treatment of the shared/tied
    long-term parameters, no parameter groups, no LR schedule."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_step(model: BDH, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor, step_scaler: torch.amp.GradScaler | None = None, step_ctx=None) -> float:
    """One real training iteration, factored out of `run_reference_training`'s
    loop body: forward under `ctx`, `scaler.scale(loss).backward()`,
    `scaler.step(optimizer)`, `scaler.update()`, `optimizer.zero_grad()`.
    Defaults to this module's own `scaler`/`ctx` (CUDA-float16-only
    GradScaler, CUDA-only autocast) -- on Mac/CPU both are no-ops, so this
    collapses to plain `loss.backward(); optimizer.step(); optimizer.zero_grad()`,
    matching what upstream's own code does on non-CUDA devices. Returns the
    scalar loss (detached float) for logging."""
    step_scaler = step_scaler if step_scaler is not None else scaler
    step_ctx = step_ctx if step_ctx is not None else ctx
    with step_ctx:
        _logits, loss = model(x, y)
    step_scaler.scale(loss).backward()
    step_scaler.step(optimizer)
    step_scaler.update()
    optimizer.zero_grad()
    return float(loss.detach())
