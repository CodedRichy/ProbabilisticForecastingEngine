"""
BettingEnv: Gym environment for football match betting policy learning.

State  : 25-dim vector (Elo + form + goals + Poisson + market + edge + bankroll)
Actions: 0=no_bet  1=bet_home  2=bet_draw  3=bet_away
Reward : P&L in units (win: odds-1, loss: -1, no_bet: 0)
Episode: one chronological pass through all matches
"""

import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

OBS_DIM = 26


def _fair(home_odds, draw_odds, away_odds):
    raw = np.array([1/home_odds, 1/draw_odds, 1/away_odds], dtype=np.float32)
    total = raw.sum()
    return raw / total


class BettingEnv(gym.Env):
    metadata = {"render_modes": []}

    NO_BET   = 0
    BET_HOME = 1
    BET_DRAW = 2
    BET_AWAY = 3

    def __init__(self, matches_df: pd.DataFrame, stake_fraction: float = 0.02):
        super().__init__()
        self.df = matches_df.reset_index(drop=True)
        self.stake_fraction = stake_fraction
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self._idx = 0
        self._bankroll = 1.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        self._bankroll = 1.0
        return self._obs(), {}

    def step(self, action: int):
        row = self.df.iloc[self._idx]
        reward = self._reward(action, row)
        self._bankroll = max(0.01, self._bankroll + reward * self.stake_fraction)
        self._idx += 1
        done = self._idx >= len(self.df)
        obs = self._obs() if not done else np.zeros(OBS_DIM, dtype=np.float32)
        return obs, float(reward), done, False, {}

    def _obs(self) -> np.ndarray:
        r = self.df.iloc[self._idx]

        def g(col, default=0.0):
            v = r.get(col, default)
            return float(v) if pd.notna(v) else float(default)

        ho, do_, ao = g("home_odds", 3.0), g("draw_odds", 3.5), g("away_odds", 3.0)
        fi_h, fi_d, fi_a = _fair(ho, do_, ao)

        # Model probs: use Poisson/Elo-based probs if available, else fair implied
        p_h = g("p_home", fi_h)
        p_d = g("p_draw", fi_d)
        p_a = g("p_away", fi_a)

        edge_h = p_h - fi_h
        edge_d = p_d - fi_d
        edge_a = p_a - fi_a

        obs = np.array([
            # Elo (4)
            g("home_elo_k32", 1500) / 400.0,
            g("away_elo_k32", 1500) / 400.0,
            g("elo_delta_k32", 0)   / 400.0,
            g("elo_expected_home_k32", 0.5),
            # Form win rates (4)
            g("home_form_winrate_5", 0.33),
            g("away_form_winrate_5", 0.33),
            g("home_form_ppg_5",     1.0) / 3.0,
            g("away_form_ppg_5",     1.0) / 3.0,
            # Goals (4)
            g("home_form_gf_mean_5", 1.3) / 3.0,
            g("away_form_gf_mean_5", 1.3) / 3.0,
            g("home_form_ga_mean_5", 1.3) / 3.0,
            g("away_form_ga_mean_5", 1.3) / 3.0,
            # xG: real Understat when available, else Poisson model (2)
            g("home_xg", g("poisson_home_xg", 1.3)) / 3.0,
            g("away_xg", g("poisson_away_xg", 1.3)) / 3.0,
            # Market (3)
            fi_h, fi_d, fi_a,
            # Model probs (3)
            p_h, p_d, p_a,
            # Edge (3)
            edge_h, edge_d, edge_a,
            # Max edge + overround proxy (2)
            max(edge_h, edge_d, edge_a),
            (1/ho + 1/do_ + 1/ao) - 1.0,
            # Bankroll (1)
            min(self._bankroll, 3.0),
        ], dtype=np.float32)
        return obs

    def _reward(self, action: int, row) -> float:
        if action == self.NO_BET:
            return 0.0
        result = str(row.get("result", "")).upper().strip()
        if action == self.BET_HOME:
            return (float(row.get("home_odds", 3.0)) - 1.0) if result == "H" else -1.0
        if action == self.BET_DRAW:
            return (float(row.get("draw_odds", 3.5)) - 1.0) if result == "D" else -1.0
        return (float(row.get("away_odds", 3.0)) - 1.0) if result == "A" else -1.0

    @classmethod
    def from_processed_data(cls, parquet_path: str, **kwargs) -> "BettingEnv":
        """
        Build env from data/processed/matches.parquet (output of data_loader + feature_factory).
        Drops rows missing odds or result. Sorts chronologically.
        """
        from discovery.feature_factory import FeatureFactory
        from pathlib import Path

        # Prefer xG-enriched dataset when available
        xg_path = Path(parquet_path).parent / "matches_xg.parquet"
        load_path = str(xg_path) if xg_path.exists() else parquet_path
        if xg_path.exists():
            logger.info("Using xG-enriched dataset: %s", xg_path)

        df = pd.read_parquet(load_path)
        df = df.dropna(subset=["home_odds", "draw_odds", "away_odds", "result"])
        df = df.sort_values("date").reset_index(drop=True)

        logger.info("Computing features on %d club matches...", len(df))
        ff = FeatureFactory()
        df = ff.compute_all(df)
        logger.info("Features computed. State dim includes %d feature cols.", len(df.columns))

        return cls(df, **kwargs)

    @classmethod
    def from_international_csv(cls, csv_path_or_url: str, model, **kwargs) -> "BettingEnv":
        """Build env from martj42 international results CSV + EloModel (synthesised odds)."""
        import requests
        from io import StringIO
        from pathlib import Path

        p = Path(csv_path_or_url)
        if p.exists():
            df = pd.read_csv(p)
        else:
            r = requests.get(csv_path_or_url, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))

        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= "2010-01-01"].sort_values("date").reset_index(drop=True)

        MARGIN = 1.05
        records = []
        for _, row in df.iterrows():
            hs = row.get("home_score")
            as_ = row.get("away_score")
            if pd.isna(hs) or pd.isna(as_):
                continue
            home, away = row["home_team"], row["away_team"]
            pred = model.predict(home, away, neutral=bool(row.get("neutral", True)))
            ho  = round(MARGIN / max(pred["p_home"], 0.01), 3)
            do_ = round(MARGIN / max(pred["p_draw"], 0.01), 3)
            ao  = round(MARGIN / max(pred["p_away"], 0.01), 3)
            result = "H" if int(hs) > int(as_) else ("D" if int(hs) == int(as_) else "A")
            records.append({
                "home": home, "away": away, "date": row["date"],
                "home_elo_k32": pred["home_elo"], "away_elo_k32": pred["away_elo"],
                "elo_delta_k32": pred["home_elo"] - pred["away_elo"],
                "elo_expected_home_k32": 1/(1+10**((pred["away_elo"]-pred["home_elo"])/400)),
                "p_home": pred["p_home"], "p_draw": pred["p_draw"], "p_away": pred["p_away"],
                "home_odds": ho, "draw_odds": do_, "away_odds": ao,
                "result": result,
            })

        return cls(pd.DataFrame(records), **kwargs)
