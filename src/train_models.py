"""Train & evaluate GCN-LSTM, LSTM, and Moving-Average on the flow tensors.

Fixes vs. the original notebooks (05. GCNLSTM / airport__gcn_lstm.py):
  1. Adjacency: the physically-normalized A_norm from airport_graph (matching
     Eq. 3-4 of the paper), in the SAME node order as the tensors. The old code
     silently replaced it with a cosine-similarity matrix.
  2. Normalization stats (target min-max, feature z-score) fit on TRAIN ONLY.
  3. Chronological 70/15/15 split (the old 90/5/5 test set was ~430 samples).
  4. Early stopping on validation loss; best checkpoint restored. Fresh model
     per run — no cross-dataset weight reuse.
  5. Dropout applied where it works (after GCN + after LSTM output); the old
     nn.LSTM(dropout=..., num_layers=1) was a silent no-op.
  6. Metrics reported overall AND per node type (checkpoints vs gates),
     computed in the original passenger-count scale.

Usage: train_models.py --data tensors_15min.npz [--epochs 150] [--seed 42]
Saves metrics JSON and per-model test predictions (for plotting/analysis).
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from airport_graph import adjacency, NODES, CHECKPOINTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- data
def chrono_split(X, S, Y, train=0.70, val=0.15):
    n = X.shape[0]
    i1, i2 = int(n * train), int(n * (train + val))
    cut = lambda a, lo, hi: None if a is None else a[lo:hi]
    return ((X[:i1], cut(S, 0, i1), Y[:i1]),
            (X[i1:i2], cut(S, i1, i2), Y[i1:i2]),
            (X[i2:], cut(S, i2, None), Y[i2:]))


def normalize_with_train_stats(splits):
    """Feature z-score + target min-max, both fit on the training split only.
    Each split is (X, S, Y); S may be None."""
    (Xtr, Str, Ytr), (Xva, Sva, Yva), (Xte, Ste, Yte) = splits
    mu = Xtr.mean(axis=(0, 1, 3), keepdims=True)
    sd = Xtr.std(axis=(0, 1, 3), keepdims=True)
    sd[sd == 0] = 1.0
    tmin, tmax = float(Ytr.min()), float(Ytr.max())
    scale = (tmax - tmin) or 1.0
    if Str is not None:
        smu = Str.mean(axis=(0, 1, 3), keepdims=True)
        ssd = Str.std(axis=(0, 1, 3), keepdims=True)
        ssd[ssd == 0] = 1.0

    def fx(X): return (X - mu) / sd
    def fs(S): return None if S is None else (S - smu) / ssd
    def fy(Y): return (Y - tmin) / scale
    out = tuple((fx(X), fs(S), fy(Y)) for X, S, Y in splits)
    return out, (tmin, tmax)


def loaders(splits, batch=32):
    out = []
    for i, (X, S, Y) in enumerate(splits):
        if S is None:
            S = np.zeros((X.shape[0], 1), dtype=np.float32)  # placeholder
        ds = TensorDataset(torch.tensor(X), torch.tensor(S), torch.tensor(Y))
        out.append(DataLoader(ds, batch_size=batch, shuffle=(i == 0), drop_last=(i == 0)))
    return out


# ---------------------------------------------------------------- models
def _sym_norm(A):
    """D^-1/2 (A_sym + I) D^-1/2 from a raw (possibly directed) adjacency."""
    A = np.maximum(A, A.T) + np.eye(A.shape[0])
    d = A.sum(axis=1)
    Dis = np.diag(1.0 / np.sqrt(d))
    return Dis @ A @ Dis


class IdentitySpatial(nn.Module):
    """No graph: per-node linear only (the 'none' spatial ablation)."""
    def __init__(self, in_features, out_features, A):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.relu(self.linear(x))


class GraphConvolution(nn.Module):
    """Static-graph GCN layer: neighbor weights fixed by the physical layout."""
    def __init__(self, in_features, out_features, A):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("A", torch.tensor(_sym_norm(A), dtype=torch.float32))

    def forward(self, x):                # x: (B, N, F)
        return torch.relu(torch.einsum("ij,bjf->bif", self.A, self.linear(x)))


class ChebConv(nn.Module):
    """Chebyshev spectral convolution of order K (STGCN-family). Uses the
    rescaled Laplacian L~ = -A_norm (Kipf simplification, lambda_max = 2)."""
    def __init__(self, in_features, out_features, A, K=3):
        super().__init__()
        self.K = K
        self.linears = nn.ModuleList(nn.Linear(in_features, out_features)
                                     for _ in range(K))
        L = -_sym_norm(A)
        self.register_buffer("L", torch.tensor(L, dtype=torch.float32))

    def forward(self, x):                # (B, N, F)
        Tk_prev, Tk = x, torch.einsum("ij,bjf->bif", self.L, x)
        out = self.linears[0](Tk_prev)
        if self.K > 1:
            out = out + self.linears[1](Tk)
        for k in range(2, self.K):
            Tk_next = 2 * torch.einsum("ij,bjf->bif", self.L, Tk) - Tk_prev
            out = out + self.linears[k](Tk_next)
            Tk_prev, Tk = Tk, Tk_next
        return torch.relu(out)


class DiffusionConv(nn.Module):
    """DCRNN-style bidirectional diffusion convolution: K-step random walks
    along and against the DIRECTED passenger-flow edges — the only layer here
    that uses edge direction."""
    def __init__(self, in_features, out_features, A, K=2):
        super().__init__()
        self.K = K
        A = np.asarray(A, dtype=np.float64)
        d_out = A.sum(axis=1); d_out[d_out == 0] = 1
        d_in = A.sum(axis=0); d_in[d_in == 0] = 1
        P_f = A / d_out[:, None]           # forward walk (with the flow)
        P_b = A.T / d_in[:, None]          # backward walk (against the flow)
        self.register_buffer("Pf", torch.tensor(P_f, dtype=torch.float32))
        self.register_buffer("Pb", torch.tensor(P_b, dtype=torch.float32))
        self.lin0 = nn.Linear(in_features, out_features)
        self.lin_f = nn.ModuleList(nn.Linear(in_features, out_features) for _ in range(K))
        self.lin_b = nn.ModuleList(nn.Linear(in_features, out_features) for _ in range(K))

    def forward(self, x):                # (B, N, F)
        out = self.lin0(x)
        xf = xb = x
        for k in range(self.K):
            xf = torch.einsum("ij,bjf->bif", self.Pf, xf)
            xb = torch.einsum("ij,bjf->bif", self.Pb, xb)
            out = out + self.lin_f[k](xf) + self.lin_b[k](xb)
        return torch.relu(out)


class GraphAttention(nn.Module):
    """GAT-style layer: edge weights computed from node states each step, so
    spatial influence varies with the traffic situation (masked to the
    physical topology)."""
    def __init__(self, in_features, out_features, A):
        super().__init__()
        self.W = nn.Linear(in_features, out_features)
        self.a_src = nn.Linear(out_features, 1, bias=False)
        self.a_dst = nn.Linear(out_features, 1, bias=False)
        A = torch.tensor(np.maximum(A, A.T), dtype=torch.float32)
        mask = (A + torch.eye(A.shape[0])) > 0
        self.register_buffer("mask", mask)

    def forward(self, x):                # (B, N, F)
        h = self.W(x)
        e = self.a_src(h) + self.a_dst(h).transpose(1, 2)      # (B, N, N)
        e = torch.nn.functional.leaky_relu(e, 0.2)
        e = e.masked_fill(~self.mask, float("-inf"))
        att = torch.softmax(e, dim=-1)
        return torch.relu(torch.bmm(att, h))


class AdaptiveGraphConv(nn.Module):
    """Graph-WaveNet-style layer: a fully learned adjacency (from node
    embeddings) alongside the physical one — lets the model discover
    functional dependencies the layout graph misses."""
    def __init__(self, in_features, out_features, A, emb_dim=10):
        super().__init__()
        n = A.shape[0]
        self.register_buffer("A", torch.tensor(_sym_norm(A), dtype=torch.float32))
        self.E1 = nn.Parameter(torch.randn(n, emb_dim))
        self.E2 = nn.Parameter(torch.randn(n, emb_dim))
        self.lin_static = nn.Linear(in_features, out_features)
        self.lin_adapt = nn.Linear(in_features, out_features)

    def forward(self, x):                # (B, N, F)
        A_adapt = torch.softmax(torch.relu(self.E1 @ self.E2.T), dim=-1)
        out = (torch.einsum("ij,bjf->bif", self.A, self.lin_static(x))
               + torch.einsum("ij,bjf->bif", A_adapt, self.lin_adapt(x)))
        return torch.relu(out)


SPATIAL_LAYERS = {"none": IdentitySpatial, "gcn": GraphConvolution,
                  "gat": GraphAttention, "adaptive": AdaptiveGraphConv,
                  "cheb": ChebConv, "diffusion": DiffusionConv}


# ---------------------------------------------------------------- temporal
class TCN(nn.Module):
    """Causal dilated temporal convolution stack (WaveNet-style)."""
    def __init__(self, in_size, hidden, levels=3, kernel=3):
        super().__init__()
        layers = []
        c = in_size
        for i in range(levels):
            d = 2 ** i
            layers.append(nn.Conv1d(c, hidden, kernel, dilation=d,
                                    padding=(kernel - 1) * d))
            layers.append(nn.ReLU())
            c = hidden
        self.net = nn.ModuleList(layers)
        self.kernel, self.hidden = kernel, hidden

    def forward(self, seq):              # (B, T, C)
        h = seq.transpose(1, 2)          # (B, C, T)
        T = h.shape[2]
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.Conv1d):
                h = h[..., :T]           # trim causal padding
        return h[..., -1]                # (B, hidden)


class TransformerEnc(nn.Module):
    def __init__(self, in_size, hidden, nhead=4, layers=2, max_len=64):
        super().__init__()
        self.proj = nn.Linear(in_size, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        enc = nn.TransformerEncoderLayer(hidden, nhead, hidden * 2,
                                         dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)

    def forward(self, seq):              # (B, T, C)
        h = self.proj(seq) + self.pos[:, :seq.shape[1]]
        return self.enc(h)[:, -1]        # (B, hidden)


class RNNEnc(nn.Module):
    def __init__(self, in_size, hidden, kind="lstm"):
        super().__init__()
        cls = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = cls(in_size, hidden, batch_first=True)

    def forward(self, seq):
        out, _ = self.rnn(seq)
        return out[:, -1]


def make_temporal(kind, in_size, hidden):
    if kind in ("lstm", "gru"):
        return RNNEnc(in_size, hidden, kind)
    if kind == "tcn":
        return TCN(in_size, hidden)
    if kind == "transformer":
        return TransformerEnc(in_size, hidden)
    raise ValueError(kind)


class ScheduleHead(nn.Module):
    """Residual branch mapping the known-future schedule S (B, N, k, H) to a
    per-node correction on the forecast (B, N, H)."""
    def __init__(self, k, horizon, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k * horizon, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon))

    def forward(self, s):                # (B, N, k, H)
        B, N, k, H = s.shape
        return self.net(s.reshape(B, N, k * H))  # (B, N, H)


class STModel(nn.Module):
    """Any spatial layer x any temporal encoder + optional schedule head.
    spatial: none | gcn | gat | adaptive | cheb | diffusion
    temporal: lstm | gru | tcn | transformer
    GCN-LSTM = STModel('gcn', 'lstm'); the old PlainLSTM = STModel('none', 'lstm')."""
    def __init__(self, A, num_nodes, in_features, horizon,
                 gcn_hidden=16, t_hidden=128, dropout=0.2, sched_k=0,
                 spatial="gcn", temporal="lstm"):
        super().__init__()
        self.num_nodes, self.horizon = num_nodes, horizon
        self.gcn = SPATIAL_LAYERS[spatial](in_features, gcn_hidden, A)
        self.drop = nn.Dropout(dropout)
        self.temporal = make_temporal(temporal, num_nodes * gcn_hidden, t_hidden)
        self.fc = nn.Linear(t_hidden, num_nodes * horizon)
        self.sched = ScheduleHead(sched_k, horizon) if sched_k else None

    def forward(self, x, s=None):        # x: (B, N, F, T)
        B, N, F, T = x.shape
        seq = [self.drop(self.gcn(x[..., t])).reshape(B, -1) for t in range(T)]
        out = self.temporal(torch.stack(seq, dim=1))
        out = self.fc(self.drop(out)).view(B, N, self.horizon)
        if self.sched is not None:
            out = out + self.sched(s)
        return out


# ---------------------------------------------------------------- training
def train(model, train_dl, val_dl, epochs, lr=1e-3, patience=15):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.MSELoss()
    best_val, best_state, bad = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        tr = 0.0
        for xb, sb, yb in train_dl:
            xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb, sb), yb)
            loss.backward()
            opt.step()
            tr += loss.item()
        model.eval()
        with torch.no_grad():
            va = sum(crit(model(xb.to(DEVICE), sb.to(DEVICE)), yb.to(DEVICE)).item()
                     for xb, sb, yb in val_dl) / len(val_dl)
        if va < best_val - 1e-6:
            best_val, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0:
            print(f"  ep {ep:3d}  train {tr / len(train_dl):.5f}  val {va:.5f}"
                  f"{'  *' if bad == 0 else ''}")
        if bad >= patience:
            print(f"  early stop at epoch {ep} (best val {best_val:.5f})")
            break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, dl):
    model.eval()
    ys, ps = [], []
    for xb, sb, yb in dl:
        ys.append(yb.numpy())
        ps.append(model(xb.to(DEVICE), sb.to(DEVICE)).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ---------------------------------------------------------------- metrics
def metrics(y, p, mask=None):
    if mask is not None:
        y, p = y[:, mask, :], p[:, mask, :]
    y, p = y.ravel(), p.ravel()
    err = p - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res, ss_tot = np.sum(err ** 2), np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    nz = y != 0
    mape = float(np.mean(np.abs(err[nz] / y[nz])) * 100) if nz.any() else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE_nonzero": mape}


def report(name, y, p):
    cp_mask = np.array([n in CHECKPOINTS for n in NODES])
    res = {"overall": metrics(y, p),
           "checkpoints": metrics(y, p, cp_mask),
           "gates": metrics(y, p, ~cp_mask)}
    print(f"\n== {name} ==")
    for scope, m in res.items():
        print(f"  {scope:12s} RMSE {m['RMSE']:8.2f}  MAE {m['MAE']:7.2f}  "
              f"R2 {m['R2']:6.3f}  MAPE(nz) {m['MAPE_nonzero']:7.1f}%")
    return res


# ---------------------------------------------------------------- main
def main(data_path, epochs, seed, only_models=None,
         spatials=("gcn", "none"), temporals=("lstm",)):
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = np.load(data_path, allow_pickle=True)
    X, Y = d["X"], d["Y"]
    S = d["S"] if "S" in d.files else None
    stamps = d["timestamps"]
    horizon, n_feat = Y.shape[2], X.shape[2]
    sched_k = 0 if S is None else S.shape[2]
    print(f"{data_path}: X {X.shape}  Y {Y.shape}  "
          f"S {None if S is None else S.shape}  device={DEVICE}  seed={seed}")

    raw_splits = chrono_split(X, S, Y)
    splits, (tmin, tmax) = normalize_with_train_stats(raw_splits)
    train_dl, val_dl, test_dl = loaders(splits)
    denorm = lambda a: tmin + a * (tmax - tmin)

    n = X.shape[0]
    test_stamps = stamps[int(n * 0.85):]  # matches chrono_split's test start

    A = adjacency()   # raw directed adjacency; each layer normalizes its way
    results, preds = {}, {}

    NAME = {"none": "", "gcn": "GCN-", "gat": "GAT-", "adaptive": "AdpGCN-",
            "cheb": "Cheb-", "diffusion": "Diff-"}
    runs = []
    for sp in spatials:
        for tp in temporals:
            name = f"{NAME[sp]}{tp.upper()}"
            ctor = (lambda sp=sp, tp=tp: lambda k: STModel(
                A, len(NODES), n_feat, horizon, sched_k=k,
                spatial=sp, temporal=tp))()
            runs.append((name, ctor, 0))
            if sched_k:
                runs.append((f"{name}+Sched", ctor, sched_k))
    if only_models:
        runs = [r for r in runs if r[0] in only_models]

    y = None
    for name, ctor, k in runs:
        print(f"\n--- {name} ---")
        m = train(ctor(k), train_dl, val_dl, epochs)
        y, p = map(denorm, predict(m, test_dl))
        results[name] = report(name, y, p)
        preds[name] = p

    # Moving average: forecast = mean of the input window's Flow_In (feature 0),
    # computed on the raw test split (no normalization involved).
    (Xte, Ste, Yte) = raw_splits[2]
    p_ma = np.repeat(Xte[:, :, 0, :].mean(axis=2, keepdims=True), horizon, axis=2)
    results["MovingAverage"] = report("MovingAverage", Yte, p_ma)
    preds["MovingAverage"] = p_ma

    suffix = "" if seed == 42 else f"_seed{seed}"
    out = data_path.replace(".npz", f"_results{suffix}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(
        data_path.replace(".npz", f"_preds{suffix}.npz"),
        y_true=Yte, timestamps=test_stamps[: Yte.shape[0]],
        **{f"pred_{k}": v for k, v in preds.items()},
    )
    print(f"\nsaved -> {out} (+ predictions npz)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tensors_15min.npz")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default=None,
                    help="comma-separated subset, e.g. 'GCN-LSTM+Sched,GAT-LSTM+Sched'")
    ap.add_argument("--spatials", default="gcn,none",
                    help="comma list from: none,gcn,gat,adaptive,cheb,diffusion")
    ap.add_argument("--temporals", default="lstm",
                    help="comma list from: lstm,gru,tcn,transformer")
    args = ap.parse_args()
    main(args.data, args.epochs, args.seed,
         only_models=set(args.models.split(",")) if args.models else None,
         spatials=args.spatials.split(","), temporals=args.temporals.split(","))
