import json
import logging
import math
import pickle
import warnings
from pathlib import Path

import pandas as pd
import requests

from core.team_names import TEAM_ALIASES, normalize_team

logger = logging.getLogger(__name__)

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

K_WEIGHTS = {
    "FIFA World Cup": 60,
    "Confederations Cup": 45,
    "Copa America": 45,
    "UEFA Euro": 45,
    "AFC Asian Cup": 45,
    "Africa Cup of Nations": 45,
    "CONCACAF Gold Cup": 45,
    "FIFA World Cup qualification": 25,
    "UEFA Euro qualification": 25,
    "Copa America qualification": 25,
    "AFC Asian Cup qualification": 25,
    "Africa Cup of Nations qualification": 25,
    "CONCACAF Gold Cup qualification": 25,
    "Friendly": 15,
}

HOME_ADVANTAGE = 75
DEFAULT_ELO = 1500
MIN_DATE = "2010-01-01"

def _resolve(team: str, ratings: dict) -> str:
    """Resolve *team* to a key present in *ratings*, or the best alias if absent.

    Resolution order:
      1. Exact match in ratings (fast path).
      2. normalize_team() alias lookup — resolves e.g. "USA" -> "United States".
      3. If still missing, logs a WARNING and returns the alias result so the
         caller receives DEFAULT_ELO via ratings.get(key, DEFAULT_ELO).
    """
    if team in ratings:
        return team
    resolved = normalize_team(team)
    if resolved not in ratings:
        logger.warning(
            "Team '%s' (resolved: '%s') not found in Elo ratings — "
            "defaulting to %d. Re-run scripts/build_elo.py to refresh ratings.",
            team, resolved, DEFAULT_ELO,
        )
    return resolved


def _k_factor(tournament: str) -> float:
    for key, k in K_WEIGHTS.items():
        if key.lower() in tournament.lower():
            return k
    return 20


def _expected(home_elo: float, away_elo: float, home_adv: float) -> float:
    return 1.0 / (1.0 + 10 ** ((away_elo - home_elo - home_adv) / 400))


def _margin_multiplier(goal_diff: int, elo_diff: float) -> float:
    """Return the FiveThirtyEight margin-of-victory multiplier.

    ``elo_diff`` is the winning team's Elo minus the losing team's Elo
    immediately before the match.  The autocorrelation correction
    (dividing by ``2.2 / (elo_diff * 0.001 + 2.2)``) dampens the
    multiplier when a strong favourite wins by a large margin, preventing
    runaway ratings.
    """
    multiplier = 1.0 + 0.5 * math.log1p(abs(goal_diff))
    multiplier /= 2.2 / (elo_diff * 0.001 + 2.2)
    return multiplier


class EloModel:
    def __init__(self):
        self._ratings: dict[str, float] = {}

    @property
    def ratings(self) -> dict[str, float]:
        return dict(self._ratings)

    def fit(self, df: pd.DataFrame) -> "EloModel":
        self._ratings = {}

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= MIN_DATE].sort_values("date").reset_index(drop=True)

        required = {"date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        df = df.dropna(subset=["home_score", "away_score"])
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        latest_date = df["date"].dropna().max()

        for row in df.itertuples(index=False):
            home = row.home_team
            away = row.away_team

            if home not in self._ratings:
                self._ratings[home] = DEFAULT_ELO
            if away not in self._ratings:
                self._ratings[away] = DEFAULT_ELO

            neutral = bool(row.neutral)
            home_adv = 0.0 if neutral else HOME_ADVANTAGE

            row_date = row.date
            if hasattr(row_date, "to_pydatetime"):
                row_date = row_date.to_pydatetime()
            days_ago = (latest_date.to_pydatetime() - row_date).days
            if days_ago < 90:
                time_mult = 1.5
            elif days_ago < 365:
                time_mult = 1.2
            elif days_ago < 730:
                time_mult = 1.0
            else:
                time_mult = 0.7
            k = _k_factor(row.tournament) * time_mult

            r_home = self._ratings[home]
            r_away = self._ratings[away]
            exp_home = _expected(r_home, r_away, home_adv)

            goal_diff = row.home_score - row.away_score
            if goal_diff > 0:
                actual_home = 1.0
            elif goal_diff < 0:
                actual_home = 0.0
            else:
                actual_home = 0.5

            # elo_diff = winner's Elo − loser's Elo (pre-match)
            if goal_diff > 0:
                elo_diff = r_home - r_away
            elif goal_diff < 0:
                elo_diff = r_away - r_home
            else:
                elo_diff = 0.0
            multiplier = _margin_multiplier(goal_diff, elo_diff)

            delta = k * multiplier * (actual_home - exp_home)
            self._ratings[home] = r_home + delta
            self._ratings[away] = r_away - delta

        return self

    @classmethod
    def from_csv(cls, url_or_path: str = RESULTS_URL) -> "EloModel":
        path = Path(url_or_path)
        if path.exists():
            df = pd.read_csv(path)
        else:
            response = requests.get(url_or_path, timeout=30)
            response.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

        model = cls()
        model.fit(df)
        return model

    def predict(self, home: str, away: str, neutral: bool = True) -> dict:
        home_key = _resolve(home, self._ratings)
        away_key = _resolve(away, self._ratings)

        home_elo = self._ratings.get(home_key, DEFAULT_ELO)
        away_elo = self._ratings.get(away_key, DEFAULT_ELO)

        home_adv = 0.0 if neutral else HOME_ADVANTAGE
        exp_home = _expected(home_elo, away_elo, home_adv)

        # Draw-rate lookup table keyed on absolute Elo delta between teams
        # (including home advantage already folded into home_elo via home_adv).
        # Buckets derived from historical international-match draw frequencies.
        elo_delta = abs((home_elo + home_adv) - away_elo)
        if elo_delta <= 50:
            draw_prob = 0.28
        elif elo_delta <= 100:
            draw_prob = 0.26
        elif elo_delta <= 150:
            draw_prob = 0.24
        elif elo_delta <= 200:
            draw_prob = 0.21
        elif elo_delta <= 300:
            draw_prob = 0.18
        else:
            draw_prob = 0.14

        # Split the remaining probability in proportion to the raw win expectations
        # then normalize all three so they sum exactly to 1.0.
        p_home = exp_home * (1.0 - draw_prob)
        p_away = (1.0 - exp_home) * (1.0 - draw_prob)

        total = p_home + draw_prob + p_away
        p_home /= total
        draw_prob /= total
        p_away /= total

        # Flag when either team fell back to DEFAULT_ELO so callers can
        # decide whether to trust or skip this prediction.
        elo_fallback = (
            self._ratings.get(home_key) is None
            or self._ratings.get(away_key) is None
        )

        return {
            "p_home": round(p_home, 6),
            "p_draw": round(draw_prob, 6),
            "p_away": round(p_away, 6),
            "home_elo": round(home_elo, 2),
            "away_elo": round(away_elo, 2),
            "elo_fallback": elo_fallback,
        }

    def save(self, path: str) -> None:
        p = Path(path)
        if p.suffix == ".json":
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self._ratings, f, indent=2, ensure_ascii=False)
        else:
            with open(p, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "EloModel":
        p = Path(path)
        if p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                ratings = json.load(f)
            model = cls()
            model._ratings = ratings
            return model
        else:
            with open(p, "rb") as f:
                return pickle.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(f"Downloading data from {RESULTS_URL} ...")
    model = EloModel.from_csv()
    top30 = sorted(model.ratings.items(), key=lambda x: x[1], reverse=True)[:30]
    print(f"\n{'Rank':<6}{'Team':<35}{'Elo':>8}")
    print("-" * 50)
    for rank, (team, elo) in enumerate(top30, start=1):
        print(f"{rank:<6}{team:<35}{elo:>8.1f}")
