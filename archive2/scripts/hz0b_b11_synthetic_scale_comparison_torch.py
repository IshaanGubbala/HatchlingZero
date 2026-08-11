"""HZ-0B B11: large-scale, multi-seed torch/CUDA robustness check for the
core memory-vs-no-memory comparison in
`docs/restart/hz0b_b11_evaluation_results.md`.

That first result (frozen HZ-0A checkpoint, MLX/Mac-only) found memory
0.750 vs. an equal-param no-memory adapter 0.562 vs. a 0.000 floor, on
only 16 held-out examples and with HZ-0B's own number still single-seed.
This script targets those exact two disclosed caveats -- NOT a
replacement for the real-checkpoint result, a complementary one, using a
frozen SYNTHETIC (untrained, non-HZ-0A) backbone
(`reference/hz0b_b11_synthetic_backbone_torch.py`) since the real
checkpoint is MLX/Metal-only and cannot run here. Runs on CUDA for real
scale: many seeds for BOTH conditions (not just the adapter), and a much
larger held-out set (default 256, not 16).

Same task shape as B8 Stage 3 / the MLX B11 script: after FACT_MARKER,
show fact-id A or B; much later, after a read-trigger, the correct
target depends on which fact-id was shown.
"""
from __future__ import annotations

import argparse
import random
import time

import torch

from reference.hz0b_b11_equal_param_adapter_torch import EqualParamAdapter, param_count as adapter_param_count
from reference.hz0b_b11_latent_write_torch import LatentWriteController, param_count as memory_param_count
from reference.hz0b_b11_synthetic_backbone_torch import SyntheticFrozenBackbone

VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS = 8192, 256, 4, 4
KEY_DIM = VALUE_DIM = 32
FACT_MARKER, FACT_A_ID, FACT_B_ID = 8000, 8001, 8002
READ_TRIGGER_A, READ_TRIGGER_B = 8003, 8004
TARGET_A, TARGET_B = 8005, 8006
FACT_POS = 6
MIDDLE_LEN = 24
PROMPT_LEN = FACT_POS + 2 + MIDDLE_LEN + 2


def make_prompts(count: int, rng: random.Random, device) -> tuple[torch.Tensor, torch.Tensor]:
    rows, fact_is_a = [], []
    for _ in range(count):
        prefix = [rng.randint(10, VOCAB_SIZE - 10) for _ in range(FACT_POS)]
        is_a = rng.random() < 0.5
        fact_id = FACT_A_ID if is_a else FACT_B_ID
        middle = [rng.randint(10, VOCAB_SIZE - 10) for _ in range(MIDDLE_LEN)]
        row = prefix + [FACT_MARKER, fact_id] + middle + [READ_TRIGGER_A, READ_TRIGGER_B]
        rows.append(row)
        fact_is_a.append(1.0 if is_a else 0.0)
    return torch.tensor(rows, dtype=torch.long, device=device), torch.tensor(fact_is_a, device=device)


def targets_for(is_a: torch.Tensor) -> torch.Tensor:
    return torch.where(is_a > 0.5, torch.tensor(TARGET_A, device=is_a.device), torch.tensor(TARGET_B, device=is_a.device))


def train_and_eval_adapter(backbone, train_tokens, train_is_a, held_out_tokens, held_out_is_a, *, hidden_dim, seed, steps, lr, device):
    torch.manual_seed(seed)
    adapter = EqualParamAdapter(D_MODEL, hidden_dim, seed=seed).to(device)
    lm_head = torch.nn.Linear(D_MODEL, VOCAB_SIZE, bias=False).to(device)
    with torch.no_grad():
        lm_head.weight.copy_(backbone.embed.weight)
    lm_head.weight.requires_grad_(False)

    optimizer = torch.optim.SGD(adapter.parameters(), lr=lr)
    with torch.no_grad():
        train_hidden = backbone(train_tokens)
        held_out_hidden = backbone(held_out_tokens)
    targets = targets_for(train_is_a)
    for step in range(steps):
        optimizer.zero_grad()
        out = adapter(train_hidden)
        logits = lm_head(out[:, -1, :])
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        out = adapter(held_out_hidden)
        logits = lm_head(out[:, -1, :])
        predicted = torch.argmax(logits, dim=-1)
    return float((predicted == targets_for(held_out_is_a)).float().mean())


def train_and_eval_memory(backbone, train_tokens, train_is_a, held_out_tokens, held_out_is_a, *, seed, steps, lr, lambda_sparse, device):
    torch.manual_seed(seed)
    controller = LatentWriteController(D_MODEL, KEY_DIM, VALUE_DIM, seed=seed).to(device)
    lm_head = torch.nn.Linear(D_MODEL, VOCAB_SIZE, bias=False).to(device)
    with torch.no_grad():
        lm_head.weight.copy_(backbone.embed.weight)
    lm_head.weight.requires_grad_(False)

    optimizer = torch.optim.SGD(controller.parameters(), lr=lr)
    with torch.no_grad():
        train_hidden = backbone(train_tokens)
        held_out_hidden = backbone(held_out_tokens)
    targets = targets_for(train_is_a)
    for step in range(steps):
        optimizer.zero_grad()
        out, gates = controller(train_hidden)
        logits = lm_head(out[:, -1, :])
        task_loss = torch.nn.functional.cross_entropy(logits, targets)
        loss = task_loss + lambda_sparse * gates.mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        out, _ = controller(held_out_hidden)
        logits = lm_head(out[:, -1, :])
        predicted = torch.argmax(logits, dim=-1)
    return float((predicted == targets_for(held_out_is_a)).float().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--held-out-count", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--lambda-sparse", type=float, default=5.0)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--backbone-seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"device={device} train_count={args.train_count} held_out_count={args.held_out_count} steps={args.steps} lr={args.lr} num_seeds={args.num_seeds}")

    backbone = SyntheticFrozenBackbone(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, seed=args.backbone_seed).to(device)
    mem_params = memory_param_count(D_MODEL, KEY_DIM, VALUE_DIM)
    # solve adapter_param_count(D_MODEL, h) = h*(2*D_MODEL+1) + D_MODEL for the closest integer h to mem_params
    adapter_hidden = round((mem_params - D_MODEL) / (2 * D_MODEL + 1))
    adapt_params = adapter_param_count(D_MODEL, adapter_hidden)
    print(f"memory controller params: {mem_params}, adapter hidden={adapter_hidden} adapter params: {adapt_params} (matched to {100*adapt_params/mem_params:.2f}%)")

    rng = random.Random(555)
    train_tokens, train_is_a = make_prompts(args.train_count, rng, device)
    held_out_tokens, held_out_is_a = make_prompts(args.held_out_count, rng, device)

    with torch.no_grad():
        floor_hidden = backbone(held_out_tokens)
        floor_logits = torch.nn.functional.linear(floor_hidden[:, -1, :], backbone.embed.weight)
        floor_pred = torch.argmax(floor_logits, dim=-1)
    floor_acc = float((floor_pred == targets_for(held_out_is_a)).float().mean())
    print(f"\n1. True floor (frozen synthetic backbone, 0 extra params): {floor_acc:.4f}")

    print(f"\n2. Equal-parameter no-memory adapter ({args.num_seeds} seeds):")
    t0 = time.time()
    adapter_accs = []
    for i in range(args.num_seeds):
        acc = train_and_eval_adapter(backbone, train_tokens, train_is_a, held_out_tokens, held_out_is_a, hidden_dim=adapter_hidden, seed=555 + i, steps=args.steps, lr=args.lr, device=device)
        print(f"  seed {555+i}: {acc:.4f}")
        adapter_accs.append(acc)
    print(f"  mean: {sum(adapter_accs)/len(adapter_accs):.4f}  std: {torch.tensor(adapter_accs).std():.4f}  ({time.time()-t0:.1f}s)")

    print(f"\n3. HZ-0B real latent write+read memory ({args.num_seeds} seeds):")
    t0 = time.time()
    memory_accs = []
    for i in range(args.num_seeds):
        acc = train_and_eval_memory(backbone, train_tokens, train_is_a, held_out_tokens, held_out_is_a, seed=555 + i, steps=args.steps, lr=args.lr, lambda_sparse=args.lambda_sparse, device=device)
        print(f"  seed {555+i}: {acc:.4f}")
        memory_accs.append(acc)
    print(f"  mean: {sum(memory_accs)/len(memory_accs):.4f}  std: {torch.tensor(memory_accs).std():.4f}  ({time.time()-t0:.1f}s)")

    print("\n--- Summary (synthetic frozen backbone, NOT the real HZ-0A checkpoint) ---")
    print(f"floor:                        {floor_acc:.4f}")
    print(f"equal-param adapter, no mem:  mean {sum(adapter_accs)/len(adapter_accs):.4f}  std {torch.tensor(adapter_accs).std():.4f}  ({args.num_seeds} seeds, held_out={args.held_out_count})")
    print(f"HZ-0B real memory:            mean {sum(memory_accs)/len(memory_accs):.4f}  std {torch.tensor(memory_accs).std():.4f}  ({args.num_seeds} seeds, held_out={args.held_out_count})")


if __name__ == "__main__":
    main()
