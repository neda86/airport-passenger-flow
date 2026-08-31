"""ST-LLM-style forecaster: partially-frozen GPT-2 backbone over node tokens.

Follows the recipe of ST-LLM / ST-LLM+ (Liu et al., TKDE 2025) adapted to the
airport terminal graph:

  1. Each node's flow history (F features x T steps) is linearly embedded into
     one token; a learnable node-identity embedding is added.
  2. Known-future schedule covariates (k x H) are embedded and added to the
     node token (schedule conditioning — the airport-specific novelty).
  3. One graph-convolution mixing step over the physical adjacency injects
     terminal topology into the token sequence.
  4. The N node tokens pass through GPT-2 with most weights FROZEN; only
     LayerNorms, position embeddings, and the last transformer block train.
     The pretrained sequence prior acts as a general pattern machine.
  5. A regression head maps each node token to its H-step forecast.

Designed for Colab GPU (T4 is plenty: N=24 tokens, gpt2-small).
Runs on CPU too, just slowly — use --epochs 2 --limit 500 for a smoke test.

Usage:
  python st_llm.py --data tensors_1h_h3.npz --epochs 20          # GPU
  python st_llm.py --data tensors_1h_h3.npz --epochs 2 --limit 500  # CPU smoke
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from airport_graph import normalized_adjacency, NODES
from train_models import (chrono_split, normalize_with_train_stats, loaders,
                          train, predict, report, DEVICE)


class STLLM(nn.Module):
    def __init__(self, A, num_nodes, in_features, in_seq, horizon, sched_k=0,
                 llm_name="gpt2", unfreeze_last_n=1):
        super().__init__()
        from transformers import GPT2Model
        self.llm = GPT2Model.from_pretrained(llm_name)
        d = self.llm.config.n_embd

        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.num_nodes, self.horizon = num_nodes, horizon

        self.embed_hist = nn.Linear(in_features * in_seq, d)
        self.embed_node = nn.Embedding(num_nodes, d)
        self.embed_sched = nn.Linear(sched_k * horizon, d) if sched_k else None
        self.graph_mix = nn.Linear(d, d)
        self.head = nn.Linear(d, horizon)

        # Freeze the backbone; unfreeze LayerNorms, position embeddings,
        # and the last n transformer blocks (the ST-LLM recipe).
        for p in self.llm.parameters():
            p.requires_grad = False
        for name, p in self.llm.named_parameters():
            if "ln" in name or "wpe" in name:
                p.requires_grad = True
        for block in self.llm.h[-unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x, s=None):            # x: (B, N, F, T)
        B, N, F, T = x.shape
        tok = self.embed_hist(x.reshape(B, N, F * T))
        tok = tok + self.embed_node.weight.unsqueeze(0)
        if self.embed_sched is not None and s is not None:
            tok = tok + self.embed_sched(s.reshape(B, N, -1))
        tok = tok + torch.relu(self.graph_mix(torch.einsum("ij,bjd->bid", self.A, tok)))
        out = self.llm(inputs_embeds=tok).last_hidden_state   # (B, N, d)
        return self.head(out)                                  # (B, N, H)


def main(data_path, epochs, limit, lr):
    torch.manual_seed(42)
    np.random.seed(42)

    d = np.load(data_path, allow_pickle=True)
    X, Y = d["X"], d["Y"]
    S = d["S"] if "S" in d.files else None
    if limit:
        X, Y = X[:limit], Y[:limit]
        S = None if S is None else S[:limit]
    horizon, in_seq, n_feat = Y.shape[2], X.shape[3], X.shape[2]
    sched_k = 0 if S is None else S.shape[2]
    print(f"X {X.shape} Y {Y.shape} S {None if S is None else S.shape} device={DEVICE}")

    raw = chrono_split(X, S, Y)
    splits, (tmin, tmax) = normalize_with_train_stats(raw)
    train_dl, val_dl, test_dl = loaders(splits, batch=16)
    denorm = lambda a: tmin + a * (tmax - tmin)

    model = STLLM(normalized_adjacency(), len(NODES), n_feat, in_seq, horizon,
                  sched_k=sched_k)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train:,} / {n_total:,} "
          f"({100 * n_train / n_total:.1f}%)")

    model = train(model, train_dl, val_dl, epochs, lr=lr, patience=8)
    y, p = map(denorm, predict(model, test_dl))
    res = report("ST-LLM (GPT-2, schedule-conditioned)", y, p)

    out = data_path.replace(".npz", "_stllm.json")
    json.dump(res, open(out, "w"), indent=2)
    np.savez_compressed(data_path.replace(".npz", "_stllm_preds.npz"),
                        y_true=y, pred_STLLM=p)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N samples (CPU smoke test)")
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()
    main(args.data, args.epochs, args.limit, args.lr)
