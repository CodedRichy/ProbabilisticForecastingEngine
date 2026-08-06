"""
referee_model.py - Bayesian-shrunk referee home-bias model for Apollo.

Some referees produce a measurably higher home win rate than the league
baseline. That delta is partially priced by bookmakers but not fully, so it
represents exploitable edge. This module estimates a per-referee home-win
probability delta, shrinks it toward zero with a Bayesian prior (to tame
small-sample noise), and exposes the result as both a probability adjustment
and an approximate Elo equivalent.

Fully importable with no side effects at import time.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Bayesian shrinkage prior strength (pseudo-count of "league-average" matches).
# A referee needs ~prior_strength matches before half their raw delta survives.
PRIOR_STRENGTH = 50

# Minimum matches before a referee is modelled at all.
MIN_MATCHES = 10

# Clamp range for the returned probability delta.
DELTA_CLAMP = 0.12

# Probability-to-Elo conversion. The derivative of the logistic Elo expectation
# at 50% is 1 / 861 ≈ 0.00116 prob-points per Elo point, so 1% probability is
# roughly 8.6 Elo points. We round the divisor to 0.00125 (i.e. 8.0 Elo per 1%)
# for conservatism, matching the task brief.
PROB_PER_ELO = 0.00125


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class RefereeModel:
    def __init__(self):
        self._stats: dict[str, dict] = {}
        self._league_home_win_rate: float = 0.0

    @property
    def league_home_win_rate(self) -> float:
        return self._league_home_win_rate

    @property
    def stats(self) -> dict[str, dict]:
        return {ref: dict(s) for ref, s in self._stats.items()}

    def fit(self, df: pd.DataFrame, min_matches: int = MIN_MATCHES) -> "RefereeModel":
        """Fit per-referee statistics with Bayesian shrinkage.

        Expects columns: result ('H'/'D'/'A'), Referee, home_goals, away_goals.
        ``league`` is accepted but not required. Rows with a NaN/blank referee
        are skipped. The global home-win baseline is computed over all rows that
        carry a valid result, independent of whether the referee is known.
        """
        required = {"result", "Referee", "home_goals", "away_goals"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        self._stats = {}

        df = df.copy()
        # Only rows with a valid categorical result contribute to any rate.
        df = df[df["result"].isin(["H", "D", "A"])]

        if df.empty:
            logger.warning("No rows with valid 'result' values; model is empty.")
            self._league_home_win_rate = 0.0
            return self

        # 1. Global baseline home-win rate.
        self._league_home_win_rate = float((df["result"] == "H").mean())

        # Drop NaN / blank referees for the per-referee pass.
        ref_df = df[df["Referee"].notna()].copy()
        ref_df["Referee"] = ref_df["Referee"].astype(str).str.strip()
        ref_df = ref_df[ref_df["Referee"] != ""]

        if ref_df.empty:
            logger.warning("No rows with a valid Referee value; per-referee stats empty.")
            return self

        ref_df["_avg_goals"] = (
            ref_df["home_goals"].astype(float) + ref_df["away_goals"].astype(float)
        ) / 2.0

        for referee, group in ref_df.groupby("Referee"):
            n = int(len(group))
            if n < min_matches:
                continue

            home_win_rate_ref = float((group["result"] == "H").mean())
            draw_rate_ref = float((group["result"] == "D").mean())
            away_win_rate_ref = float((group["result"] == "A").mean())
            avg_goals = float(group["_avg_goals"].mean())

            raw_delta = home_win_rate_ref - self._league_home_win_rate
            # Bayesian shrinkage toward the league baseline.
            shrunk_delta = raw_delta * n / (n + PRIOR_STRENGTH)

            self._stats[referee] = {
                "delta": float(shrunk_delta),
                "n_matches": n,
                "avg_goals": avg_goals,
                "draw_rate": draw_rate_ref,
                # Kept for richer reporting; not part of the serialized schema.
                "home_win_rate": home_win_rate_ref,
                "away_win_rate": away_win_rate_ref,
                "raw_delta": float(raw_delta),
            }

        return self

    def get_prob_adjustment(self, referee: str) -> float:
        """Shrunk home-win probability delta, clamped to [-0.12, +0.12].

        Returns 0.0 for unknown referees or those below the match threshold.
        """
        if referee is None:
            return 0.0
        entry = self._stats.get(str(referee).strip())
        if entry is None:
            return 0.0
        return _clamp(entry["delta"], -DELTA_CLAMP, DELTA_CLAMP)

    def get_elo_adjustment(self, referee: str) -> float:
        """Approximate Elo equivalent of the referee's home-win prob delta.

        Returns 0.0 for unknown referees.
        """
        prob_adj = self.get_prob_adjustment(referee)
        if prob_adj == 0.0:
            return 0.0
        return prob_adj / PROB_PER_ELO

    def top_referees(self, n: int = 20) -> list[dict]:
        """Top ``n`` referees by absolute delta, each as a stats dict."""
        rows = []
        for referee, s in self._stats.items():
            row = {"referee": referee}
            row.update(s)
            rows.append(row)
        rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
        return rows[:n]

    def save(self, path: str) -> None:
        """Serialize to JSON with only native Python types."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "league_avg": float(self._league_home_win_rate),
            "referees": {
                referee: {
                    "delta": float(s["delta"]),
                    "n_matches": int(s["n_matches"]),
                    "avg_goals": float(s["avg_goals"]),
                    "draw_rate": float(s["draw_rate"]),
                }
                for referee, s in self._stats.items()
            },
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "RefereeModel":
        """Deserialize a model previously written by :meth:`save`."""
        with open(Path(path), "r", encoding="utf-8") as f:
            payload = json.load(f)

        model = cls()
        model._league_home_win_rate = float(payload.get("league_avg", 0.0))
        model._stats = {}
        for referee, s in payload.get("referees", {}).items():
            delta = float(s["delta"])
            league_avg = model._league_home_win_rate
            model._stats[referee] = {
                "delta": delta,
                "n_matches": int(s["n_matches"]),
                "avg_goals": float(s["avg_goals"]),
                "draw_rate": float(s["draw_rate"]),
                # Reconstruct best-effort derived fields for reporting.
                "home_win_rate": league_avg + delta,
                "away_win_rate": float("nan"),
                "raw_delta": delta,
            }
        return model

    def summary(self) -> str:
        """Human-readable report: top 10 home-biased + top 10 away-biased refs."""
        if not self._stats:
            return "RefereeModel: no referee statistics (empty model)."

        ordered = sorted(self._stats.items(), key=lambda kv: kv[1]["delta"], reverse=True)
        home_biased = ordered[:10]
        away_biased = list(reversed(ordered[-10:]))

        header = (
            f"{'Referee':<28}{'N':>6}{'Delta':>9}{'EloAdj':>9}"
            f"{'HomeWin%':>10}{'Draw%':>8}{'AvgGls':>8}"
        )
        sep = "-" * len(header)

        def fmt(referee: str, s: dict) -> str:
            elo = self.get_elo_adjustment(referee)
            hw = s.get("home_win_rate", self._league_home_win_rate + s["delta"])
            return (
                f"{referee[:27]:<28}{s['n_matches']:>6}"
                f"{s['delta'] * 100:>+8.2f}%{elo:>+9.1f}"
                f"{hw * 100:>9.1f}%{s['draw_rate'] * 100:>7.1f}%"
                f"{s['avg_goals']:>8.2f}"
            )

        lines = [
            "RefereeModel summary",
            f"League baseline home-win rate: {self._league_home_win_rate * 100:.2f}%",
            f"Referees modelled: {len(self._stats)}",
            "",
            "Top 10 HOME-biased referees (positive delta):",
            header,
            sep,
        ]
        lines += [fmt(r, s) for r, s in home_biased]
        lines += [
            "",
            "Top 10 AWAY-biased referees (negative delta):",
            header,
            sep,
        ]
        lines += [fmt(r, s) for r, s in away_biased]
        return "\n".join(lines)


if __name__ == "__main__":
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Quick self-test on synthetic data: one strongly home-biased referee,
    # one away-biased referee, and noise referees around a 0.45 baseline.
    rng = np.random.default_rng(42)
    rows = []
    specs = {
        "Home Hawk": (0.62, 200),
        "Away Owl": (0.30, 200),
        "Neutral Ned": (0.45, 150),
        "Sparse Sam": (0.80, 5),  # below threshold, should be ignored
    }
    for ref, (p_home, count) in specs.items():
        for _ in range(count):
            draw = rng.random()
            if draw < p_home:
                result = "H"
            elif draw < p_home + 0.25:
                result = "D"
            else:
                result = "A"
            rows.append(
                {
                    "result": result,
                    "Referee": ref,
                    "home_goals": int(rng.integers(0, 4)),
                    "away_goals": int(rng.integers(0, 4)),
                    "league": "TestLiga",
                }
            )
    # A handful of NaN-referee rows to confirm graceful skipping.
    for _ in range(30):
        rows.append(
            {
                "result": rng.choice(["H", "D", "A"]),
                "Referee": np.nan,
                "home_goals": 1,
                "away_goals": 1,
                "league": "TestLiga",
            }
        )

    df = pd.DataFrame(rows)
    model = RefereeModel().fit(df)
    print(model.summary())
    print("\nHome Hawk prob adj:", model.get_prob_adjustment("Home Hawk"))
    print("Home Hawk elo adj :", round(model.get_elo_adjustment("Home Hawk"), 1))
    print("Unknown ref adj   :", model.get_prob_adjustment("Nobody"))
    print("Sparse Sam adj    :", model.get_prob_adjustment("Sparse Sam"))
