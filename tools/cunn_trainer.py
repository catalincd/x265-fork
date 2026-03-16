#!/usr/bin/env python3
"""
tools/cunn_trainer.py  —  Train the x265 CU split-decision MLP (cunn.h).

Architecture : 8 inputs → 16 hidden (ReLU) → 1 output (sigmoid)
Weight file  : W1[16][8], b1[16], W2[16], b2   (161 × float32 = 644 bytes)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLLECTING REAL TRAINING DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Rebuild with the collection flag:

     cmake <src_dir> -DCMAKE_CXX_FLAGS="-DX265_CUNN_COLLECT" ...
     make -j$(nproc)

2. Encode with recursion-skip disabled (so both split/no-split are always
   evaluated — unbiased ground-truth labels):

     X265_CUNN_LOG=cu_splits.csv  \\
       x265 input.yuv -o /dev/null --preset medium --rskip 0

   Encode several diverse clips to get a balanced dataset (~500 K rows
   recommended, covering varied content: sports, animation, screen content).

CSV columns (raw, un-normalised):
  cu_variance, depth, qp, slice_type, luma_mean, mad,
  min_temp_depth, sa8d_cost, split

  split = 1  → the encoder chose to split this CU (ground-truth label)
  split = 0  → the encoder chose not to split

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python3 tools/cunn_trainer.py train --data cu_splits.csv --out weights.bin

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEPLOYMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  X265_NN_WEIGHTS=weights.bin  x265 input.yuv -o out.hevc --rskip 3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUBCOMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  train   Train from a CSV data file, export binary weights.
  eval    Evaluate an existing weight file on a held-out CSV.
  synth   Generate synthetic data (useful for smoke-testing the pipeline).
  verify  Print a forward-pass table for a weight file (compare to C++ output).
  export  Convert numpy .npz checkpoint → binary weight file.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Architecture constants — must match cunn.h exactly
# ─────────────────────────────────────────────────────────────────────────────

N_INPUTS  = 8
N_HIDDEN  = 16
N_WEIGHTS = N_HIDDEN * N_INPUTS + N_HIDDEN + N_HIDDEN + 1  # 161

FEATURE_NAMES = [
    "cu_variance",    # [0] raw uint32 → / 5000, clip 0-1
    "depth",          # [1] 0-3        → / 3
    "qp",             # [2] 12-51      → (qp-12)/39, clip 0-1
    "slice_type",     # [3] 0/1/2      → / 2
    "luma_mean",      # [4] 0-maxpix   → / maxpix
    "mad",            # [5] uint32     → / (luma_mean+1)   KEY feature
    "min_temp_depth", # [6] 0-3        → / 3, clip 0-1
    "sa8d_cost",      # [7] uint64     → log(1+x)/30, clip 0-1
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature normalisation  (mirrors Analysis::nnSkipRecursion in analysis.cpp)
# ─────────────────────────────────────────────────────────────────────────────

def normalise(df, x265_depth: int = 8) -> np.ndarray:
    """
    Convert a DataFrame with raw CSV columns into a float32 array (N, 8).
    The normalisation is identical to the C++ nnSkipRecursion() feature
    extraction so that trained weights transfer directly.
    """
    max_pix = float((1 << x265_depth) - 1)
    N = len(df)
    X = np.empty((N, N_INPUTS), dtype=np.float32)

    X[:, 0] = np.minimum(df["cu_variance"].to_numpy(float) / 5000.0, 1.0)
    X[:, 1] = df["depth"].to_numpy(float) / 3.0
    X[:, 2] = np.clip((df["qp"].to_numpy(float) - 12.0) / 39.0, 0.0, 1.0)
    X[:, 3] = df["slice_type"].to_numpy(float) / 2.0
    X[:, 4] = df["luma_mean"].to_numpy(float) / max_pix
    X[:, 5] = df["mad"].to_numpy(float) / (df["luma_mean"].to_numpy(float) + 1.0)
    X[:, 6] = np.minimum(df["min_temp_depth"].to_numpy(float) / 3.0, 1.0)
    X[:, 7] = np.minimum(np.log1p(df["sa8d_cost"].to_numpy(float)) / 30.0, 1.0)

    return X


# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy forward pass  (mirrors cunnPredict in cunn.h)
# ─────────────────────────────────────────────────────────────────────────────

def forward(X: np.ndarray, W1, b1, W2, b2) -> np.ndarray:
    """X: (N, 8) → probabilities: (N,)"""
    H = np.maximum(0.0, X @ W1.T + b1)   # ReLU hidden layer
    return 1.0 / (1.0 + np.exp(-(H @ W2 + b2)))


# ─────────────────────────────────────────────────────────────────────────────
# Weight file I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_weights(path: str | Path):
    """Load a binary weight file → (W1, b1, W2, b2)."""
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.float32)
    if len(data) != N_WEIGHTS:
        raise ValueError(
            f"{path}: expected {N_WEIGHTS} float32 values ({N_WEIGHTS*4} bytes), "
            f"got {len(data)} ({len(data)*4} bytes)."
        )
    W1 = data[: N_HIDDEN * N_INPUTS].reshape(N_HIDDEN, N_INPUTS).copy()
    b1 = data[N_HIDDEN * N_INPUTS : N_HIDDEN * N_INPUTS + N_HIDDEN].copy()
    W2 = data[N_HIDDEN * N_INPUTS + N_HIDDEN : N_HIDDEN * N_INPUTS + 2 * N_HIDDEN].copy()
    b2 = float(data[-1])
    return W1, b1, W2, b2


def save_weights(path: str | Path, W1, b1, W2, b2):
    """
    Write weights in the binary format expected by cunnLoadWeights():
      W1[16][8]  b1[16]  W2[16]  b2    (161 float32 LE = 644 bytes)
    """
    with open(path, "wb") as f:
        np.asarray(W1, dtype=np.float32).flatten().tofile(f)   # 128
        np.asarray(b1, dtype=np.float32).tofile(f)             # 16
        np.asarray(W2, dtype=np.float32).flatten().tofile(f)   # 16
        np.array([b2], dtype=np.float32).tofile(f)             # 1
    print(f"Weights saved → {path}  ({N_WEIGHTS * 4} bytes, {N_WEIGHTS} float32)")


def _default_weights():
    """Return the same initial weights as cunn.h defaults."""
    W1 = np.zeros((N_HIDDEN, N_INPUTS), dtype=np.float32)
    b1 = np.zeros(N_HIDDEN, dtype=np.float32)
    W2 = np.zeros(N_HIDDEN, dtype=np.float32)
    b2 = np.float32(-1.0)
    # Neuron 0: homo/mean threshold at ~0.09  (see cunn.h for derivation)
    W1[0, 5] = -50.0
    b1[0]    =   5.0
    W2[0]    =   2.0
    # Neurons 1-15: small diverse seeds
    rng = np.random.default_rng(1)
    for i in range(1, N_HIDDEN):
        W1[i] = rng.choice([-0.02, -0.01, 0.01, 0.02], size=N_INPUTS)
    return W1, b1, W2, b2


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic(n: int = 50_000, seed: int = 42, noise: float = 0.06):
    """
    Generate synthetic CU split labels.

    Ground truth = sigmoid(-20 * (mad_ratio + 0.05*var_norm - 0.02*depth - 0.10))
    i.e., split when block is complex (high MAD, high variance, shallow depth).
    This mimics the RDCOST_BASED_RSKIP heuristic with added depth/QP sensitivity
    to give the NN something to learn beyond the simple threshold.
    """
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas required: pip install pandas")

    rng = np.random.default_rng(seed)

    depth       = rng.integers(0, 4, n)
    qp          = rng.integers(18, 42, n)
    slice_type  = rng.choice([0, 1, 2], n, p=[0.05, 0.40, 0.55])
    cu_variance = rng.exponential(800, n).clip(0, 20_000).astype(np.int32)
    luma_mean   = rng.integers(16, 230, n).astype(np.int32)
    mad_raw     = (rng.exponential(1.0, n) * luma_mean * 0.15).clip(0, luma_mean)
    min_td      = rng.integers(0, 4, n)
    sa8d_cost   = rng.exponential(50_000, n).clip(0, 1e8).astype(np.int64)

    mad_ratio = mad_raw / (luma_mean.astype(float) + 1.0)
    var_norm  = np.minimum(cu_variance / 5000.0, 1.0)

    # Probabilistic ground truth
    complexity = mad_ratio + 0.05 * var_norm - 0.02 * depth
    p_split    = 1.0 / (1.0 + np.exp(-20.0 * (complexity - 0.10)))
    p_split    = (p_split + noise * rng.standard_normal(n)).clip(0.0, 1.0)
    split      = (rng.random(n) < p_split).astype(np.int32)

    df = pd.DataFrame({
        "cu_variance":    cu_variance,
        "depth":          depth,
        "qp":             qp,
        "slice_type":     slice_type,
        "luma_mean":      luma_mean,
        "mad":            mad_raw.astype(np.int32),
        "min_temp_depth": min_td,
        "sa8d_cost":      sa8d_cost,
        "split":          split,
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(X: np.ndarray, y: np.ndarray, W1, b1, W2, b2, label: str = ""):
    prob  = forward(X, W1, b1, W2, b2)
    y_hat = (prob > 0.5).astype(np.int32)
    yi    = y.astype(np.int32)

    acc = (y_hat == yi).mean()
    tp  = int(((y_hat == 1) & (yi == 1)).sum())
    fp  = int(((y_hat == 1) & (yi == 0)).sum())
    fn  = int(((y_hat == 0) & (yi == 1)).sum())
    tn  = int(((y_hat == 0) & (yi == 0)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2.0 * prec * rec / (prec + rec + 1e-9)

    hdr = f"  [{label}]" if label else ""
    print(f"\nEvaluation{hdr}  ({len(X):,} samples, split rate {y.mean():.3f})")
    print(f"  Accuracy   {acc:.4f}")
    print(f"  Precision  {prec:.4f}  (of predicted-skip, fraction that truly skip)")
    print(f"  Recall     {rec:.4f}  (of true-skip, fraction we predict)")
    print(f"  F1         {f1:.4f}")
    print(f"  Confusion  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print()
    print("  FP = CU predicted skip but should split  → quality loss")
    print("  FN = CU predicted split but should skip  → speed loss")
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: str):
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas required: pip install pandas")
    df = pd.read_csv(path)
    missing = set(FEATURE_NAMES + ["split"]) - set(df.columns)
    if missing:
        sys.exit(f"CSV {path} missing columns: {missing}")
    return df


def train(
    data_path: str,
    out_path: str,
    epochs: int = 60,
    lr: float = 1e-2,
    batch_size: int = 2048,
    val_frac: float = 0.15,
    seed: int = 0,
    device: str = "cpu",
    pos_weight: float | None = None,
    x265_depth: int = 8,
    save_npz: bool = False,
):
    """Train the MLP; try PyTorch first, fall back to numpy SGD."""
    df    = _load_csv(data_path)
    X_all = normalise(df, x265_depth=x265_depth)
    y_all = df["split"].to_numpy(np.float32)

    print(f"Loaded {len(df):,} samples from {data_path}  |  split rate {y_all.mean():.3f}")

    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(df))
    n_val = max(1, int(len(df) * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    try:
        import torch
        W1, b1, W2, b2 = _train_torch(
            X_all, y_all, tr_idx, val_idx,
            epochs=epochs, lr=lr, batch_size=batch_size,
            seed=seed, device=device, pos_weight=pos_weight,
        )
    except ImportError:
        print("PyTorch not found — using numpy SGD.", file=sys.stderr)
        W1, b1, W2, b2 = _train_numpy(
            X_all, y_all, tr_idx, val_idx,
            epochs=epochs, lr=lr, batch_size=batch_size, seed=seed,
        )

    save_weights(out_path, W1, b1, W2, b2)

    if save_npz:
        npz_path = Path(out_path).with_suffix(".npz")
        np.savez(npz_path, W1=W1, b1=b1, W2=W2, b2=np.array([b2]))
        print(f"NumPy checkpoint → {npz_path}")

    evaluate(X_all[val_idx], y_all[val_idx], W1, b1, W2, b2, label="validation")
    return W1, b1, W2, b2


def _train_torch(X_all, y_all, tr_idx, val_idx, *,
                 epochs, lr, batch_size, seed, device, pos_weight):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(seed)
    dev = torch.device(device)

    X_tr = torch.tensor(X_all[tr_idx]).to(dev)
    y_tr = torch.tensor(y_all[tr_idx]).to(dev)
    X_va = torch.tensor(X_all[val_idx]).to(dev)
    y_va = torch.tensor(y_all[val_idx]).to(dev)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )

    model = nn.Sequential(
        nn.Linear(N_INPUTS, N_HIDDEN),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, 1),
        nn.Sigmoid(),
    ).to(dev)

    # Seed with the default cunn.h weights (approximates existing heuristic)
    with torch.no_grad():
        W1i, b1i, W2i, b2i = _default_weights()
        model[0].weight.copy_(torch.tensor(W1i))
        model[0].bias.copy_(torch.tensor(b1i))
        model[2].weight.copy_(torch.tensor(W2i).unsqueeze(0))
        model[2].bias.fill_(float(b2i))

    pw = None
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=dev)
    criterion = nn.BCELoss() if pw is None else nn.BCEWithLogitsLoss(pos_weight=pw)
    # If using BCEWithLogitsLoss we need logits, so swap the sigmoid layer
    if pw is not None:
        model = nn.Sequential(
            nn.Linear(N_INPUTS, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, 1),
        ).to(dev)
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor(W1i))
            model[0].bias.copy_(torch.tensor(b1i))
            model[2].weight.copy_(torch.tensor(W2i).unsqueeze(0))
            model[2].bias.fill_(float(b2i))

    optimiser = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    print(f"\nPyTorch training  {N_INPUTS}→{N_HIDDEN}→1  |  "
          f"{len(tr_idx):,} train  {len(val_idx):,} val  |  "
          f"{epochs} epochs  lr={lr}  device={device}")
    print(f"{'Ep':>4}  {'Train BCE':>10}  {'Val BCE':>9}  {'Val acc':>8}")
    print("─" * 38)

    best_val   = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in loader:
            optimiser.zero_grad()
            out  = model(Xb).squeeze(1)
            loss = criterion(out, yb)
            loss.backward()
            optimiser.step()
            tr_loss += loss.item() * len(Xb)
        tr_loss /= len(tr_idx)

        model.eval()
        with torch.no_grad():
            val_out  = model(X_va).squeeze(1)
            val_loss = criterion(val_out, y_va).item()
            if pw is not None:
                val_prob = torch.sigmoid(val_out)
            else:
                val_prob = val_out
            val_acc = ((val_prob > 0.5) == y_va.bool()).float().mean().item()

        scheduler.step()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % max(1, epochs // 10) == 0 or ep == epochs:
            print(f"{ep:>4}  {tr_loss:>10.5f}  {val_loss:>9.5f}  {val_acc:>7.2%}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        W1 = model[0].weight.cpu().numpy()          # (16, 8)
        b1 = model[0].bias.cpu().numpy()            # (16,)
        raw_W2 = model[2].weight.cpu().numpy()      # (1, 16) or (16,)
        W2 = raw_W2.flatten()                       # (16,)
        b2 = float(model[2].bias.cpu().numpy()[0])

    return W1, b1, W2, b2


def _train_numpy(X_all, y_all, tr_idx, val_idx, *,
                 epochs, lr, batch_size, seed):
    """Minimal SGD with momentum — no external dependencies beyond numpy."""
    rng = np.random.default_rng(seed)
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_va, y_va = X_all[val_idx], y_all[val_idx]

    W1, b1, W2, b2 = _default_weights()
    W1 = W1.copy().astype(np.float64)
    b1 = b1.copy().astype(np.float64)
    W2 = W2.copy().astype(np.float64)
    b2 = float(b2)

    # Momentum buffers
    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2); vb2 = 0.0
    mu  = 0.9

    n   = len(X_tr)
    print(f"\nNumPy SGD+momentum  {N_INPUTS}→{N_HIDDEN}→1  |  "
          f"{n:,} train  {len(val_idx):,} val  |  {epochs} epochs  lr={lr}")
    print(f"{'Ep':>4}  {'Val acc':>8}")
    print("─" * 15)

    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            Xb = X_tr[perm[start : start + batch_size]].astype(np.float64)
            yb = y_tr[perm[start : start + batch_size]].astype(np.float64)
            B  = len(Xb)

            # Forward
            H   = np.maximum(0.0, Xb @ W1.T + b1)            # (B, 16)
            p   = 1.0 / (1.0 + np.exp(-(H @ W2 + b2)))       # (B,)
            err = (p - yb) / B                                  # (B,)

            # Backward
            gW2 = H.T @ err
            gb2 = err.sum()
            gH  = np.outer(err, W2) * (H > 0)                  # (B, 16)
            gW1 = gH.T @ Xb
            gb1 = gH.sum(axis=0)

            # Momentum update
            vW1 = mu * vW1 + lr * gW1;  W1 -= vW1
            vb1 = mu * vb1 + lr * gb1;  b1 -= vb1
            vW2 = mu * vW2 + lr * gW2;  W2 -= vW2
            vb2 = mu * vb2 + lr * gb2;  b2 -= vb2

            # Cosine LR decay (per batch, roughly)
        if ep % max(1, epochs // 10) == 0 or ep == epochs:
            p_va = forward(X_va.astype(np.float64), W1, b1, W2, b2)
            acc  = ((p_va > 0.5) == (y_va > 0.5)).mean()
            print(f"{ep:>4}  {acc:>7.2%}")

    return (W1.astype(np.float32), b1.astype(np.float32),
            W2.astype(np.float32), np.float32(b2))


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand handlers
# ─────────────────────────────────────────────────────────────────────────────

def cmd_train(args):
    train(
        args.data, args.out,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch,
        val_frac=args.val, seed=args.seed, device=args.device,
        pos_weight=args.pos_weight,
        x265_depth=args.depth,
        save_npz=args.save_npz,
    )


def cmd_eval(args):
    df = _load_csv(args.data)
    X  = normalise(df, x265_depth=args.depth)
    y  = df["split"].to_numpy(np.float32)
    W1, b1, W2, b2 = load_weights(args.weights)
    evaluate(X, y, W1, b1, W2, b2, label=Path(args.weights).name)


def cmd_synth(args):
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas required: pip install pandas")
    print(f"Generating {args.n:,} synthetic samples → {args.out}")
    df = generate_synthetic(n=args.n, seed=args.seed, noise=args.noise)
    df.to_csv(args.out, index=False)
    print(f"Split rate: {df['split'].mean():.3f}  |  Saved to {args.out}")


def cmd_verify(args):
    """
    Print the forward pass for 8 canonical test inputs so you can
    cross-check against C++ (add a printf to cunnPredict for the same inputs).
    """
    W1, b1, W2, b2 = load_weights(args.weights)

    # Use the same canonical inputs so C++ comparison is deterministic
    rng = np.random.default_rng(0)
    X   = rng.random((8, N_INPUTS)).astype(np.float32)
    y   = forward(X.astype(np.float64), W1, b1, W2, b2)

    hdr = "  " + "  ".join(f"f{i:<5}" for i in range(N_INPUTS))
    print(f"\nForward-pass table  ({Path(args.weights).name})")
    print(hdr)
    print("─" * (len(hdr) + 4))
    for xi, pi in zip(X, y):
        vals = "  ".join(f"{v:.4f}" for v in xi)
        dec  = "skip" if pi > 0.5 else "recurse"
        print(f"  {vals}  →  {pi:.6f}  ({dec})")

    print()
    print("To verify in C++, add this just before 'return 1.f / ...' in cunnPredict():")
    print("  for (int i = 0; i < CUNN_INPUTS; i++) printf(\"%f \", features[i]);")
    print("  printf(\"-> %f\\n\", out);")


def cmd_export(args):
    """Convert a numpy .npz checkpoint (from --save-npz) to a binary file."""
    npz = np.load(args.npz)
    W1  = npz["W1"].astype(np.float32)
    b1  = npz["b1"].astype(np.float32)
    W2  = npz["W2"].astype(np.float32).flatten()
    b2  = float(npz["b2"].flatten()[0])
    save_weights(args.out, W1, b1, W2, b2)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        prog="cunn_trainer",
        description="Train the x265 CU split-decision MLP (cunn.h).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    t = sub.add_parser("train", help="Train from CSV, export binary weights")
    t.add_argument("--data",       required=True,
                   help="CSV file produced by X265_CUNN_LOG")
    t.add_argument("--out",        default="weights.bin",
                   help="Output binary weight file [weights.bin]")
    t.add_argument("--epochs",     type=int,   default=60,
                   help="Training epochs [60]")
    t.add_argument("--lr",         type=float, default=1e-2,
                   help="Initial learning rate [0.01]")
    t.add_argument("--batch",      type=int,   default=2048,
                   help="Mini-batch size [2048]")
    t.add_argument("--val",        type=float, default=0.15,
                   help="Fraction of data reserved for validation [0.15]")
    t.add_argument("--seed",       type=int,   default=0,
                   help="Random seed [0]")
    t.add_argument("--device",     default="cpu",
                   help="PyTorch device: cpu | cuda | mps [cpu]")
    t.add_argument("--pos-weight", type=float, default=None,
                   help="BCE pos_weight for imbalanced datasets "
                        "(e.g. 2.0 to penalise missed splits more)")
    t.add_argument("--depth",      type=int,   default=8,
                   help="x265 bit depth used when encoding [8]")
    t.add_argument("--save-npz",   action="store_true",
                   help="Also save a .npz checkpoint alongside the .bin")

    # ── eval ───────────────────────────────────────────────────────────────
    e = sub.add_parser("eval", help="Evaluate an existing weight file on CSV")
    e.add_argument("--data",    required=True)
    e.add_argument("--weights", required=True)
    e.add_argument("--depth",   type=int, default=8)

    # ── synth ──────────────────────────────────────────────────────────────
    s = sub.add_parser("synth",
                       help="Generate synthetic training data (pipeline test)")
    s.add_argument("--out",   default="synthetic.csv")
    s.add_argument("--n",     type=int,   default=50_000,
                   help="Number of CU samples [50000]")
    s.add_argument("--seed",  type=int,   default=42)
    s.add_argument("--noise", type=float, default=0.06,
                   help="Label noise fraction [0.06]")

    # ── verify ─────────────────────────────────────────────────────────────
    v = sub.add_parser("verify",
                       help="Print forward-pass table for C++ comparison")
    v.add_argument("--weights", required=True)

    # ── export ─────────────────────────────────────────────────────────────
    ex = sub.add_parser("export",
                        help="Convert .npz checkpoint to binary weight file")
    ex.add_argument("--npz", required=True,
                    help=".npz file saved with --save-npz")
    ex.add_argument("--out", default="weights.bin")

    args = p.parse_args()
    {
        "train":  cmd_train,
        "eval":   cmd_eval,
        "synth":  cmd_synth,
        "verify": cmd_verify,
        "export": cmd_export,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
