"""
Apollo — Transformer sequence model for probabilistic football forecasting.

Where XGBoost sees only a 5-match rolling window, this model attends over the
last N=30 matches for each team. A Transformer encoder learns *which* past
matches are most informative for predicting the next result, capturing
long-range patterns (post-European fixture fatigue, deceptive form where the
scoreline flatters a negative xG differential, etc.).

Per-match feature vector (10 dims, see ``encode_match_row``):
    [0:3]  result_encoded   one-hot win/draw/loss (from the team's perspective)
    [3]    goals_for_norm   goals_for / 3.0          (clipped 0..1)
    [4]    goals_against_norm
    [5]    xg_for_norm      xg_for / 3.0 (falls back to goals if xg missing)
    [6]    xg_against_norm
    [7]    is_home          1.0 home, 0.0 away
    [8]    days_since_prev  clip(days, 0, 14) / 14.0
    [9]    elo_delta_norm   (own_elo - opp_elo) / 800.0   (pre-match, own POV)

Public API
----------
    SequenceDataset(df, elo_index=None, seq_len=30, min_history=10)
    MatchEncoder(input_dim=10, d_model=64, nhead=4, num_layers=3)
    SequenceModel(d_model=64, n_competitions=10, dropout=0.1)
    sinusoidal_encoding(seq_len, d_model) -> Tensor

Run ``python core/sequence_model.py`` for a self-test that feeds random tensors
through the model and prints the output shape / a valid probability vector.
"""

from __future__ import annotations

import sys

# ── Hard dependency guard ────────────────────────────────────────────────────
try:
    import numpy as np
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - exercised only when torch absent
    missing = getattr(exc, "name", "torch")
    print(
        f"[core.sequence_model] Required dependency '{missing}' is not installed.\n"
        "This module needs PyTorch (and NumPy). Install with:\n\n"
        "    pip install torch numpy\n\n"
        "For a CPU-only build (smaller download):\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n",
        file=sys.stderr,
    )
    sys.exit(1)

import math
from pathlib import Path

import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
INPUT_DIM = 10          # per-match feature width
GOAL_NORM = 3.0         # goals / xg normalisation divisor
ELO_NORM = 800.0        # elo delta normalisation divisor
MAX_REST_DAYS = 14.0    # days_since_prev clip ceiling
DEFAULT_ELO = 1500.0
ELO_K = 20.0            # K-factor for the internal running club-Elo index
ELO_HOME_ADV = 60.0     # home advantage (rating points) for the internal Elo

# Label convention: matches XGBPredictor / EloModel ("H"=home win).
RESULT_TO_LABEL = {"H": 0, "D": 1, "A": 2}
LABEL_TO_KEY = {0: "p_home", 1: "p_draw", 2: "p_away"}


# ── Positional encoding ──────────────────────────────────────────────────────
def sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding.

    Returns a ``(seq_len, d_model)`` float32 tensor. Position 0 is the *oldest*
    match in the window and position ``seq_len-1`` is the most recent, so the
    encoder always sees a consistent temporal ordering regardless of padding.
    """
    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


# ── Feature encoding helpers ─────────────────────────────────────────────────
def encode_match_row(
    is_home: bool,
    goals_for: float,
    goals_against: float,
    xg_for: float | None,
    xg_against: float | None,
    days_since_prev: float,
    own_elo: float,
    opp_elo: float,
) -> list[float]:
    """Build the 10-dim per-match feature vector from one team's perspective.

    ``xg_*`` may be ``None``/NaN — goals are used as the fallback so the model
    degrades gracefully on the ~57% of historical rows with no Understat xG.
    """
    # Result one-hot from this team's POV.
    if goals_for > goals_against:
        result = [1.0, 0.0, 0.0]
    elif goals_for == goals_against:
        result = [0.0, 1.0, 0.0]
    else:
        result = [0.0, 0.0, 1.0]

    def _norm_goals(v: float) -> float:
        return float(min(max(v / GOAL_NORM, 0.0), 1.0))

    def _xg_or_goals(xg: float | None, goals: float) -> float:
        if xg is None or (isinstance(xg, float) and math.isnan(xg)):
            return _norm_goals(goals)
        return float(min(max(xg / GOAL_NORM, 0.0), 1.0))

    days = float(min(max(days_since_prev, 0.0), MAX_REST_DAYS)) / MAX_REST_DAYS
    elo_delta = float((own_elo - opp_elo) / ELO_NORM)

    return [
        result[0], result[1], result[2],
        _norm_goals(goals_for),
        _norm_goals(goals_against),
        _xg_or_goals(xg_for, goals_for),
        _xg_or_goals(xg_against, goals_against),
        1.0 if is_home else 0.0,
        days,
        elo_delta,
    ]


# ── Dataset ──────────────────────────────────────────────────────────────────
class SequenceDataset(torch.utils.data.Dataset):
    """Builds ``(home_seq, away_seq, context, label)`` samples from history.

    For every match (processed in chronological order) we gather the previous
    ``seq_len`` matches for the home team and the away team *strictly before*
    that match's date — guaranteeing no look-ahead leakage. Sequences shorter
    than ``seq_len`` are zero-padded at the front (oldest positions) and a
    boolean padding mask marks the pad slots so the Transformer ignores them.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: date, home_team, away_team, home_goals, away_goals,
        result, league. Optional: home_xg, away_xg, season.
    elo_index : dict[team, list[(timestamp, elo)]] | None
        Optional pre-computed Elo history. If ``None`` (the usual case for the
        club dataset, whose teams are absent from elo_national.json), a running
        club-Elo is computed internally during the same chronological pass, so
        ``elo_delta_norm`` is always populated without leakage.
    seq_len : int
        Window length N (default 30).
    min_history : int
        A match is only emitted as a training sample once *both* teams have at
        least this many prior matches (default 10). This keeps early-season /
        newly promoted teams with near-empty sequences out of the labels while
        still letting them accrue history for later matches.

    Each item is a tuple of float32 tensors:
        home_seq : (seq_len, INPUT_DIM)
        away_seq : (seq_len, INPUT_DIM)
        home_mask: (seq_len,)   True where padded (key_padding_mask convention)
        away_mask: (seq_len,)
        context  : (2,)         [elo_delta_norm, is_neutral]
        comp_id  : ()           int64 competition index for the learned embedding
        label    : ()           int64 in {0,1,2}
    """

    def __init__(
        self,
        df: pd.DataFrame,
        elo_index: dict | None = None,
        seq_len: int = 30,
        min_history: int = 10,
        competition_map: dict[str, int] | None = None,
    ) -> None:
        self.seq_len = int(seq_len)
        self.min_history = int(min_history)

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", kind="stable").reset_index(drop=True)

        # Competition (league) -> stable integer id for the learned embedding.
        if competition_map is None:
            leagues = sorted(df["league"].dropna().astype(str).unique().tolist())
            competition_map = {lg: i for i, lg in enumerate(leagues)}
        self.competition_map = competition_map
        self.n_competitions = max(len(competition_map), 1)

        has_xg = "home_xg" in df.columns and "away_xg" in df.columns
        has_neutral = "neutral" in df.columns

        # Running per-team match history: list of encoded 10-dim vectors.
        history: dict[str, list[list[float]]] = {}
        last_date: dict[str, pd.Timestamp] = {}
        # Internal running Elo (used when elo_index not supplied).
        elo: dict[str, float] = {}

        # Optional external Elo lookup (sorted timestamps per team).
        self._elo_index = elo_index

        self.samples: list[dict] = []

        for row in df.itertuples(index=False):
            home = str(row.home_team)
            away = str(row.away_team)
            date = row.date

            hg = float(row.home_goals) if not _isnan(row.home_goals) else None
            ag = float(row.away_goals) if not _isnan(row.away_goals) else None
            hxg = float(row.home_xg) if has_xg and not _isnan(row.home_xg) else None
            axg = float(row.away_xg) if has_xg and not _isnan(row.away_xg) else None
            result = getattr(row, "result", None)
            neutral = bool(getattr(row, "neutral", False)) if has_neutral else False
            league = str(row.league) if not _isnan(row.league) else ""

            # Pre-match Elo (does NOT include this match's outcome → no leakage).
            if elo_index is not None:
                home_elo = self._lookup_elo(home, date)
                away_elo = self._lookup_elo(away, date)
            else:
                home_elo = elo.get(home, DEFAULT_ELO)
                away_elo = elo.get(away, DEFAULT_ELO)

            home_hist = history.get(home, [])
            away_hist = history.get(away, [])

            # Emit a labelled sample only if both teams have enough history and
            # the match has a valid result + goals.
            if (
                result in RESULT_TO_LABEL
                and hg is not None
                and ag is not None
                and len(home_hist) >= self.min_history
                and len(away_hist) >= self.min_history
            ):
                elo_delta_norm = (home_elo + (0.0 if neutral else ELO_HOME_ADV) - away_elo) / ELO_NORM
                comp_id = self.competition_map.get(league, 0)
                self.samples.append(
                    {
                        "home_seq": _take_last(home_hist, self.seq_len),
                        "away_seq": _take_last(away_hist, self.seq_len),
                        "context": [float(elo_delta_norm), 1.0 if neutral else 0.0],
                        "comp_id": int(comp_id),
                        "label": RESULT_TO_LABEL[result],
                    }
                )

            # ── Update running state AFTER emitting (so the sample never sees
            # its own match). Skip rows with missing goals.
            if hg is None or ag is None:
                continue

            h_days = _days_between(last_date.get(home), date)
            a_days = _days_between(last_date.get(away), date)

            history.setdefault(home, []).append(
                encode_match_row(True, hg, ag, hxg, axg, h_days, home_elo, away_elo)
            )
            history.setdefault(away, []).append(
                encode_match_row(False, ag, hg, axg, hxg, a_days, away_elo, home_elo)
            )
            last_date[home] = date
            last_date[away] = date

            # Advance internal Elo with the realised result.
            if elo_index is None:
                elo[home], elo[away] = _elo_update(
                    home_elo, away_elo, hg, ag, neutral
                )

    # -- Elo lookup against an external index --------------------------------
    def _lookup_elo(self, team: str, date) -> float:
        idx = self._elo_index.get(team) if self._elo_index else None
        if not idx:
            return DEFAULT_ELO
        # idx is a list of (timestamp, elo); find the most recent entry < date.
        best = DEFAULT_ELO
        for ts, val in idx:
            if ts < date:
                best = val
            else:
                break
        return float(best)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        home_seq, home_mask = _pad_sequence(s["home_seq"], self.seq_len)
        away_seq, away_mask = _pad_sequence(s["away_seq"], self.seq_len)
        return (
            home_seq,
            away_seq,
            home_mask,
            away_mask,
            torch.tensor(s["context"], dtype=torch.float32),
            torch.tensor(s["comp_id"], dtype=torch.long),
            torch.tensor(s["label"], dtype=torch.long),
        )


# ── Dataset helper functions ─────────────────────────────────────────────────
def _isnan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v)
    except (TypeError, ValueError):
        return False


def _days_between(prev, cur) -> float:
    if prev is None:
        return MAX_REST_DAYS  # treat first-ever match as "well rested"
    delta = (cur - prev).days
    return float(delta)


def _take_last(hist: list[list[float]], n: int) -> list[list[float]]:
    return [list(v) for v in hist[-n:]]


def _pad_sequence(seq: list[list[float]], seq_len: int):
    """Left-pad ``seq`` (oldest first) with zeros to ``seq_len``.

    Returns ``(tensor(seq_len, INPUT_DIM), mask(seq_len,))`` where ``mask`` is
    True on padded positions (the key_padding_mask convention used by
    ``nn.TransformerEncoder``).
    """
    n = len(seq)
    arr = torch.zeros(seq_len, INPUT_DIM, dtype=torch.float32)
    mask = torch.ones(seq_len, dtype=torch.bool)  # True = padded/ignore
    if n > 0:
        data = torch.tensor(seq[-seq_len:], dtype=torch.float32)
        k = data.shape[0]
        arr[seq_len - k:] = data        # real matches occupy the tail
        mask[seq_len - k:] = False      # tail is real → not masked
    return arr, mask


def _elo_update(home_elo, away_elo, hg, ag, neutral):
    home_adv = 0.0 if neutral else ELO_HOME_ADV
    exp_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo - home_adv) / 400.0))
    if hg > ag:
        actual = 1.0
    elif hg < ag:
        actual = 0.0
    else:
        actual = 0.5
    gd = abs(hg - ag)
    mult = 1.0 + 0.5 * math.log1p(gd)
    delta = ELO_K * mult * (actual - exp_home)
    return home_elo + delta, away_elo - delta


# ── Per-team encoder ─────────────────────────────────────────────────────────
class MatchEncoder(nn.Module):
    """Encodes a team's match sequence into a fixed-size embedding.

    Linear(10 -> d_model) -> LayerNorm -> ReLU, plus sinusoidal positional
    encoding, then a stack of TransformerEncoder layers, then masked global
    average pooling over the (non-padded) sequence positions.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.ReLU()

        # Positional encoding buffer (registered → moves with .to(device), saved
        # in state_dict but not trained).
        self.register_buffer(
            "pos_encoding", sinusoidal_encoding(max_len, d_model), persistent=False
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,  # pre-LN: more stable training on short sequences
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> (batch, d_model).

        ``mask``: (batch, seq_len) bool, True on padded positions
        (key_padding_mask convention). May be None (no padding).
        """
        b, seq_len, _ = x.shape
        h = self.act(self.norm(self.input_proj(x)))
        h = h + self.pos_encoding[:seq_len].unsqueeze(0)

        out = self.transformer(h, src_key_padding_mask=mask)  # (b, seq_len, d)

        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()          # (b, seq_len, 1)
            summed = (out * valid).sum(dim=1)              # (b, d)
            counts = valid.sum(dim=1).clamp(min=1.0)       # (b, 1)
            pooled = summed / counts
        else:
            pooled = out.mean(dim=1)
        return pooled


# ── Full predictor ───────────────────────────────────────────────────────────
class SequenceModel(nn.Module):
    """Full match predictor combining home & away encoders.

    home_emb (d) | away_emb (d) | context_proj -> MLP -> 3 logits (H/D/A).

    match_context = [elo_delta_norm, is_neutral] concatenated with a learned
    ``competition`` embedding (8-dim), projected so the combined context block
    is 16-dim, matching the spec's 64 + 64 + 16 = 144 fusion width.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_competitions: int = 10,
        dropout: float = 0.1,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        seq_len: int = 30,
    ) -> None:
        super().__init__()
        # Persist hyper-params for clean save/load round-trips.
        self.hparams = {
            "d_model": d_model,
            "n_competitions": n_competitions,
            "dropout": dropout,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "seq_len": seq_len,
        }

        self.home_encoder = MatchEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.away_encoder = MatchEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )

        comp_emb_dim = 8
        self.comp_embedding = nn.Embedding(n_competitions, comp_emb_dim)
        # context = [elo_delta_norm, is_neutral] (2) + comp_emb (8) = 10 -> 16
        self.context_proj = nn.Sequential(
            nn.Linear(2 + comp_emb_dim, 16),
            nn.ReLU(),
        )

        fusion_dim = d_model + d_model + 16  # 144 with defaults
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

    def forward(
        self,
        home_seq: torch.Tensor,
        away_seq: torch.Tensor,
        context: torch.Tensor,
        home_mask: torch.Tensor | None = None,
        away_mask: torch.Tensor | None = None,
        comp_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns (batch, 3) logits for [home_win, draw, away_win].

        ``context``: (batch, 2) = [elo_delta_norm, is_neutral].
        ``comp_id``: (batch,) long. If None, competition 0 is assumed.
        """
        home_emb = self.home_encoder(home_seq, home_mask)
        away_emb = self.away_encoder(away_seq, away_mask)

        if comp_id is None:
            comp_id = torch.zeros(
                home_seq.shape[0], dtype=torch.long, device=home_seq.device
            )
        comp_emb = self.comp_embedding(comp_id)            # (b, 8)
        ctx = self.context_proj(torch.cat([context, comp_emb], dim=-1))  # (b, 16)

        fused = torch.cat([home_emb, away_emb, ctx], dim=-1)
        return self.mlp(fused)

    # ── Persistence ──────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hparams": self.hparams, "state_dict": self.state_dict()}, out)

    @classmethod
    def load(cls, path: str, map_location: str | None = None) -> "SequenceModel":
        ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        model = cls(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    # ── Single-sample inference ──────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        home_seq: "np.ndarray",
        away_seq: "np.ndarray",
        context: "np.ndarray",
        comp_id: int = 0,
    ) -> dict:
        """Predict outcome probabilities for one fixture.

        ``home_seq`` / ``away_seq``: (seq_len, INPUT_DIM) arrays (zero-padded
        rows are detected automatically and masked). ``context``: length-2
        array [elo_delta_norm, is_neutral].
        """
        self.eval()
        device = next(self.parameters()).device

        h = torch.as_tensor(home_seq, dtype=torch.float32, device=device)
        a = torch.as_tensor(away_seq, dtype=torch.float32, device=device)
        if h.dim() == 2:
            h = h.unsqueeze(0)
        if a.dim() == 2:
            a = a.unsqueeze(0)

        # Rows that are entirely zero are treated as padding.
        h_mask = (h.abs().sum(dim=-1) == 0)
        a_mask = (a.abs().sum(dim=-1) == 0)
        # Guard against an all-padding sequence (would produce empty pooling).
        h_mask[h_mask.all(dim=1)] = False
        a_mask[a_mask.all(dim=1)] = False

        ctx = torch.as_tensor(context, dtype=torch.float32, device=device)
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)
        cid = torch.tensor([int(comp_id)], dtype=torch.long, device=device)

        logits = self.forward(h, a, ctx, h_mask, a_mask, cid)
        probs = torch.softmax(logits, dim=-1)[0].tolist()
        return {
            "p_home": float(probs[0]),
            "p_draw": float(probs[1]),
            "p_away": float(probs[2]),
            "method": "transformer",
        }


# ── Self-test / demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    SEQ_LEN = 30
    BATCH = 4
    N_COMP = 8

    model = SequenceModel(n_competitions=N_COMP, seq_len=SEQ_LEN)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SequenceModel instantiated: {n_params:,} trainable parameters")

    # Random batch with mixed padding lengths.
    home = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
    away = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
    home_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    away_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    home_mask[0, :20] = True  # first sample only has 10 real matches
    context = torch.randn(BATCH, 2)
    comp_id = torch.randint(0, N_COMP, (BATCH,))

    model.eval()
    with torch.no_grad():
        logits = model(home, away, context, home_mask, away_mask, comp_id)
        probs = torch.softmax(logits, dim=-1)

    print(f"Batch forward  -> logits {tuple(logits.shape)} (expected ({BATCH}, 3))")
    assert logits.shape == (BATCH, 3), "unexpected logit shape"
    row_sums = probs.sum(dim=-1)
    print(f"Softmax row sums: {row_sums.tolist()} (expected ~1.0)")
    assert torch.allclose(row_sums, torch.ones(BATCH), atol=1e-5)

    # Single-sample predict() API on numpy input.
    h_np = np.random.randn(SEQ_LEN, INPUT_DIM).astype("float32")
    a_np = np.random.randn(SEQ_LEN, INPUT_DIM).astype("float32")
    h_np[:5] = 0.0  # simulate 25 real matches + 5 pad rows
    out = model.predict(h_np, a_np, np.array([0.25, 0.0], dtype="float32"), comp_id=2)
    print(f"predict() -> {out}")
    assert abs(out["p_home"] + out["p_draw"] + out["p_away"] - 1.0) < 1e-5
    assert out["method"] == "transformer"

    # sinusoidal_encoding sanity.
    pe = sinusoidal_encoding(SEQ_LEN, 64)
    print(f"sinusoidal_encoding -> {tuple(pe.shape)} (expected ({SEQ_LEN}, 64))")
    assert pe.shape == (SEQ_LEN, 64)

    print("\nAll self-tests passed.")
