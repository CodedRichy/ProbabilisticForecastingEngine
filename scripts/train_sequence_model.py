"""
Train the Apollo Transformer sequence model.

Loads the historical club-match parquet, builds a :class:`SequenceDataset`
(running club-Elo computed internally — no external Elo file required for the
club teams), performs a strict chronological 80/20 train/val split, and trains
with Adam + ReduceLROnPlateau and early stopping on validation loss.

Usage
-----
    python -m scripts.train_sequence_model --epochs 100 --batch-size 256 \
        --seq-len 30 --device cpu

The best checkpoint (lowest val loss) is written to
``data/models/sequence_model.pt``. After every epoch we print:

    epoch | train_loss | val_loss | val_acc | val_brier

NOTE: run this as a module (``python -m scripts.train_sequence_model``) from the
project root. ``core/signal.py`` shadows the stdlib ``signal`` module if the
``core`` directory is placed first on ``sys.path`` (as happens when a script
inside it is run directly), which breaks the torch import.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable when invoked as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    missing = getattr(exc, "name", "torch")
    print(
        f"[train_sequence_model] Required dependency '{missing}' is not installed.\n"
        "Install with:\n\n    pip install torch numpy pandas pyarrow\n",
        file=sys.stderr,
    )
    sys.exit(1)

import pandas as pd

from core.sequence_model import SequenceDataset, SequenceModel

DEFAULT_PARQUET = ROOT / "data" / "processed" / "matches_xg.parquet"
DEFAULT_OUT = ROOT / "data" / "models" / "sequence_model.pt"


# ── Metrics ──────────────────────────────────────────────────────────────────
def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Multiclass Brier score: mean squared error vs one-hot targets.

    ``probs``: (N, 3) softmax probabilities. ``labels``: (N,) int in {0,1,2}.
    Lower is better; the trivial uniform predictor scores 2/3 ≈ 0.667.
    """
    onehot = torch.zeros_like(probs)
    onehot[torch.arange(probs.shape[0]), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(dim=1).mean().item())


# ── Train / eval epochs ──────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, device, optimizer=None) -> dict:
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_n = 0
    correct = 0
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    torch.set_grad_enabled(train)
    for home_seq, away_seq, home_mask, away_mask, context, comp_id, label in loader:
        home_seq = home_seq.to(device)
        away_seq = away_seq.to(device)
        home_mask = home_mask.to(device)
        away_mask = away_mask.to(device)
        context = context.to(device)
        comp_id = comp_id.to(device)
        label = label.to(device)

        logits = model(home_seq, away_seq, context, home_mask, away_mask, comp_id)
        loss = criterion(logits, label)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = label.shape[0]
        total_loss += loss.item() * bs
        total_n += bs

        probs = torch.softmax(logits.detach(), dim=-1)
        correct += (probs.argmax(dim=-1) == label).sum().item()
        all_probs.append(probs.cpu())
        all_labels.append(label.detach().cpu())

    torch.set_grad_enabled(True)

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    return {
        "loss": total_loss / max(total_n, 1),
        "acc": correct / max(total_n, 1),
        "brier": brier_score(probs, labels),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train Apollo sequence model.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--min-history", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--parquet", type=str, default=str(DEFAULT_PARQUET))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────────
    print(f"Loading {args.parquet} ...")
    df = pd.read_parquet(args.parquet)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    print(f"  {len(df):,} matches | {df['date'].min().date()} -> {df['date'].max().date()}")

    print("Building SequenceDataset (running club-Elo, no look-ahead) ...")
    dataset = SequenceDataset(
        df,
        elo_index=None,
        seq_len=args.seq_len,
        min_history=args.min_history,
    )
    n = len(dataset)
    if n == 0:
        print("ERROR: no training samples produced. Lower --min-history?", file=sys.stderr)
        sys.exit(1)
    print(f"  {n:,} labelled samples | {dataset.n_competitions} competitions")

    # ── Chronological split (NOT random → no leakage) ────────────────────
    # dataset.samples are already in chronological order (built in one pass).
    split = int(n * 0.80)
    train_idx = list(range(split))
    val_idx = list(range(split, n))
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    print(f"  train {len(train_ds):,} | val {len(val_ds):,} (chronological 80/20)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )

    # ── Class weights (the label distribution is skewed toward home wins) ──
    label_counts = torch.zeros(3)
    for s in dataset.samples[:split]:
        label_counts[s["label"]] += 1
    class_weights = (label_counts.sum() / (3 * label_counts.clamp(min=1))).to(device)
    print(f"  train label counts H/D/A: {label_counts.tolist()} "
          f"| weights: {[round(w, 3) for w in class_weights.tolist()]}")

    # ── Model / optim ────────────────────────────────────────────────────
    model = SequenceModel(
        n_competitions=dataset.n_competitions, seq_len=args.seq_len
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── Training loop with early stopping ────────────────────────────────
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'epoch':>5} | {'train_loss':>10} | {'val_loss':>8} | "
          f"{'val_acc':>7} | {'val_brier':>9} | {'lr':>8}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, device, optimizer)
        val_m = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_m["loss"])
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>5} | {train_m['loss']:>10.4f} | {val_m['loss']:>8.4f} | "
              f"{val_m['acc']:>7.4f} | {val_m['brier']:>9.4f} | {lr_now:>8.2e}")

        if val_m["loss"] < best_val - 1e-5:
            best_val = val_m["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            # Save best checkpoint immediately.
            torch.save({"hparams": model.hparams, "state_dict": best_state}, args.out)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no val improvement for {args.patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"hparams": model.hparams, "state_dict": best_state}, args.out)
        print(f"\nBest val loss: {best_val:.4f}")
        print(f"Saved best checkpoint -> {args.out}")
    else:
        print("WARNING: no checkpoint saved (training did not run).", file=sys.stderr)


if __name__ == "__main__":
    main()
