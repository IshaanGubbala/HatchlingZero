"""HZ-0H H8: small causal interpretability probe for BDH-GPU.

This is deliberately a scoped capability test, not a monosemanticity claim.
It trains one tiny BDH oracle on three symbol->value associations, ranks
latent neurons by concept selectivity at the query position, then causally
ablates those neurons and compares against a size-matched random ablation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH, BDHConfig


@dataclass(frozen=True)
class H8Result:
    concept_accuracy: float
    selected_ablation_accuracy: float
    random_ablation_accuracy: float
    selectivity_margin: float
    selected_neurons: tuple[tuple[int, int], ...]


def make_concept_sequence(rng: np.random.Generator, concept: int, *, prefix_len: int = 4, filler_len: int = 8) -> tuple[list[int], int]:
    # Keep concepts/answers/query outside the filler alphabet.
    marker, answer, query = 10 + concept, 20 + concept, 30
    prefix = [int(rng.integers(0, 10)) for _ in range(prefix_len)]
    filler = [int(rng.integers(0, 10)) for _ in range(filler_len)]
    return prefix + [marker, answer] + filler + [query], answer


def train_concept_model(*, steps: int = 600, batch_size: int = 16, seed: int = 0) -> BDH:
    torch.manual_seed(seed)
    model = BDH(BDHConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        rows = []
        for _ in range(batch_size):
            seq, answer = make_concept_sequence(rng, int(rng.integers(3)))
            rows.append(seq + [answer])
        batch = torch.tensor(rows, dtype=torch.long)
        logits, loss = model(batch[:, :-1].contiguous(), targets=batch[:, 1:].contiguous())
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def _query_latents(model: BDH, idx: torch.Tensor) -> torch.Tensor:
    """Return final-layer positive latent activations at the query position."""
    C = model.config; B, T = idx.shape; D = C.n_embd; nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    x = model.ln(model.embed(idx).unsqueeze(1))
    for _ in range(C.n_layer):
        xs = F.relu(x @ model._w(model.encoder))
        ykv = model.attn(Q=xs, K=xs, V=x)
        ykv = model.ln(ykv)
        ys = F.relu(ykv @ model._w(model.encoder_v))
        xy = model.drop(xs * ys)
        y = (xy.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder))
        x = model.ln(x + model.ln(y))
    return xs[0, :, -1, :].detach()  # (head, neuron)


def _examples(seed: int, count_per_concept: int = 24) -> tuple[list[torch.Tensor], list[int]]:
    rng = np.random.default_rng(seed); xs=[]; ys=[]
    for c in range(3):
        for _ in range(count_per_concept):
            seq, _ = make_concept_sequence(rng, c)
            xs.append(torch.tensor([seq], dtype=torch.long)); ys.append(c)
    return xs, ys


def _accuracy(model: BDH, xs: list[torch.Tensor], ys: list[int]) -> float:
    correct=0
    with torch.no_grad():
        for x,c in zip(xs,ys):
            pred=int(model(x)[0][0,-1].argmax())
            correct += pred == 20+c
    return correct / len(xs)


def _ablate(model: BDH, neurons: list[tuple[int,int]]) -> BDH:
    out=copy.deepcopy(model).eval(); D=out.config.n_embd; nh=out.config.n_head
    with torch.no_grad():
        for h,n in neurons:
            out.encoder[h, :, n] = 0
            out.encoder_v[h, :, n] = 0
            out.decoder[h * (out.config.mlp_internal_dim_multiplier * D // nh) + n, :] = 0
    return out


def run_h8_probe(*, seed: int = 0, steps: int = 600, top_k: int = 6) -> H8Result:
    model=train_concept_model(steps=steps, seed=seed)
    xs,ys=_examples(seed+100)
    base=_accuracy(model,xs,ys)
    latents=torch.stack([_query_latents(model,x) for x in xs])  # (examples, head, neuron)
    labels=torch.tensor(ys)
    per_head=latents.shape[2]
    ranked=[]
    for h in range(model.config.n_head):
        for n in range(per_head):
            means=[float(latents[labels==c,h,n].mean()) for c in range(3)]
            score=max(means)-sorted(means)[-2]
            ranked.append((score,h,n))
    # Select the most concept-selective neurons, bounded by top_k.
    ranked=sorted(ranked, reverse=True)
    selected=[(h,n) for _,h,n in ranked[:top_k]]
    rng=np.random.default_rng(seed+999); alln=[(h,n) for h in range(model.config.n_head) for n in range(per_head)]
    random=list(rng.choice(len(alln), size=top_k, replace=False)); random_neurons=[alln[i] for i in random]
    selected_acc=_accuracy(_ablate(model,selected),xs,ys)
    random_acc=_accuracy(_ablate(model,random_neurons),xs,ys)
    return H8Result(base,selected_acc,random_acc,float(np.mean([r[0] for r in ranked[:top_k]])),tuple(selected))


if __name__ == "__main__":
    r=run_h8_probe(); print(r)
