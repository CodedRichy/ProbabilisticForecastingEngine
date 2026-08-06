"""
value_finder.py

Compares model probabilities against bookmaker odds to identify value bets.
No external dependencies beyond standard library and dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from core.team_names import normalize_team

if TYPE_CHECKING:
    from core.conformal import ConformalFilter


def _shin_z(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    iterations: int = 10,
) -> tuple[float, float, float]:
    """
    Estimate fair probabilities using Shin's (1993) iterative method.

    The Shin method accounts for insider trading by estimating a fraction z
    of stakes placed by informed bettors, then backing out fair probabilities.

    Args:
        home_odds: Decimal odds for home win.
        draw_odds: Decimal odds for draw.
        away_odds: Decimal odds for away win.
        iterations: Number of fixed-point iterations (10 is sufficient).

    Returns:
        Tuple of (home_prob, draw_prob, away_prob) fair probabilities summing to 1.0.
    """
    raw = [1 / home_odds, 1 / draw_odds, 1 / away_odds]
    overround = sum(raw)

    z = 0.0
    for _ in range(iterations):
        sqrt_sum = sum((p * overround) ** 0.5 for p in raw)
        z = (overround - 1) / (overround - sqrt_sum / (overround ** 0.5))
        raw = [
            (((p * overround) ** 0.5) * (1 - z) + z * p) / overround
            for p in raw
        ]

    # Normalise to guard against any residual floating-point drift
    total = sum(raw)
    return tuple(r / total for r in raw)


def remove_overround(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    method: str = "multiplicative",
) -> tuple[float, float, float]:
    """
    Convert decimal odds to fair implied probabilities by removing the bookmaker margin.

    Args:
        home_odds: Decimal odds for home win.
        draw_odds: Decimal odds for draw.
        away_odds: Decimal odds for away win.
        method: Normalisation method — "multiplicative" (default) or "shin".

    Returns:
        Tuple of (home_prob, draw_prob, away_prob) normalised to sum to 1.0.
    """
    if method == "shin":
        return _shin_z(home_odds, draw_odds, away_odds)

    # Default: multiplicative (proportional) normalisation
    raw = [1 / home_odds, 1 / draw_odds, 1 / away_odds]
    total = sum(raw)
    return tuple(r / total for r in raw)


def compute_edge(model_prob: float, fair_implied_prob: float) -> float:
    """
    Compute the edge of a bet as the difference between model probability and fair implied probability.

    Args:
        model_prob: Probability assigned by the forecasting model.
        fair_implied_prob: Fair (overround-removed) implied probability from bookmaker odds.

    Returns:
        Edge value (positive means the model sees value).
    """
    return model_prob - fair_implied_prob


def kelly_fraction(odds: float, model_prob: float, fraction: float = 0.25) -> float:
    """
    Compute the fractional Kelly criterion stake size.

    Args:
        odds: Decimal odds for the outcome.
        model_prob: Model probability for the outcome.
        fraction: Kelly fraction to apply (default 0.25 for quarter-Kelly).

    Returns:
        Recommended stake as a fraction of bankroll (0.0 if negative expectation).
    """
    b = odds - 1
    q = 1 - model_prob
    kelly_full = (b * model_prob - q) / b
    return max(0.0, kelly_full * fraction)


@dataclass
class ValueBet:
    """Represents a single value betting opportunity."""

    match: str
    outcome: str        # "home" | "draw" | "away"
    team: str
    odds: float
    model_prob: float
    fair_implied: float
    edge: float
    kelly: float

    def __str__(self) -> str:
        return (
            f"{self.match} | {self.outcome.upper():4s} ({self.team}) | "
            f"Odds: {self.odds:.2f} | "
            f"Model: {self.model_prob:.3f} | "
            f"Fair: {self.fair_implied:.3f} | "
            f"Edge: {self.edge:+.3f} | "
            f"Kelly: {self.kelly:.4f}"
        )


def find_value_bets(
    predictions: list[dict],
    bookmaker_odds: list[dict],
    min_edge: float = 0.03,
    min_kelly: float = 0.01,
    max_kelly_per_match: float = 0.25,
    conformal: Optional["ConformalFilter"] = None,
) -> list["ValueBet"]:
    """
    Identify value bets by comparing model probabilities to bookmaker odds.

    At most one outcome per match is selected — the one with the highest positive
    edge.  Betting full Kelly on two outcomes of the same event is undefined and
    leads to overbetting, so only the best opportunity per match is kept.

    The Kelly fraction for the selected bet is also capped at ``max_kelly_per_match``
    (default 0.25) so that no single match exceeds 25 % bankroll exposure.

    When a ``conformal`` filter is provided, any match where
    ``conformal.is_confident(probs)`` returns False is silently skipped — the
    model's prediction set contains more than one outcome, indicating it does not
    have genuine conviction and the apparent edge may be spurious overconfidence.

    Args:
        predictions: List of dicts with keys: home, away, p_home, p_draw, p_away.
        bookmaker_odds: List of dicts with keys: home, away, home_odds, draw_odds, away_odds.
        min_edge: Minimum edge threshold to include a bet (default 0.03).
        min_kelly: Minimum Kelly fraction to include a bet (default 0.01).
        max_kelly_per_match: Hard cap on Kelly fraction per match (default 0.25).
        conformal: Optional ConformalFilter instance.  When supplied, matches where
            the model is not confident (prediction set size > 1) are excluded.

    Returns:
        List of ValueBet objects sorted by edge descending, filtered by thresholds.
    """
    # Build a lookup from (canonical_home, canonical_away) -> odds dict.
    # normalize_team() is applied to both sides so that Odds API names
    # (e.g. "USA", "IR Iran") and ESPN names (e.g. "United States", "IR Iran")
    # resolve to the same canonical key regardless of source.
    odds_lookup: dict[tuple[str, str], dict] = {
        (normalize_team(o["home"]), normalize_team(o["away"])): o
        for o in bookmaker_odds
    }

    value_bets: list[ValueBet] = []

    for pred in predictions:
        key = (normalize_team(pred["home"]), normalize_team(pred["away"]))
        if key not in odds_lookup:
            continue

        # Conformal filter: skip this match if the model lacks genuine conviction.
        # is_confident() returns True only when the prediction set contains exactly
        # one outcome at the calibrated significance level.
        if conformal is not None:
            match_probs = {
                "p_home": pred.get("p_home", 0.0),
                "p_draw": pred.get("p_draw", 0.0),
                "p_away": pred.get("p_away", 0.0),
            }
            if not conformal.is_confident(match_probs):
                continue

        odds_entry = odds_lookup[key]
        match_label = f"{pred['home']} vs {pred['away']}"

        fair_home, fair_draw, fair_away = remove_overround(
            odds_entry["home_odds"],
            odds_entry["draw_odds"],
            odds_entry["away_odds"],
        )

        outcomes = [
            ("home",  pred["home"],  pred["p_home"],  odds_entry["home_odds"],  fair_home),
            ("draw",  "Draw",        pred["p_draw"],  odds_entry["draw_odds"],  fair_draw),
            ("away",  pred["away"],  pred["p_away"],  odds_entry["away_odds"],  fair_away),
        ]

        # Collect all candidates for this match, then keep only the best edge
        candidates: list[ValueBet] = []
        for outcome, team, model_prob, odds, fair_implied in outcomes:
            edge = compute_edge(model_prob, fair_implied)
            kelly = kelly_fraction(odds, model_prob)

            if edge >= min_edge and kelly >= min_kelly:
                candidates.append(
                    ValueBet(
                        match=match_label,
                        outcome=outcome,
                        team=team,
                        odds=odds,
                        model_prob=model_prob,
                        fair_implied=fair_implied,
                        edge=edge,
                        kelly=kelly,
                    )
                )

        if not candidates:
            continue

        # One-bet-per-match: select the single outcome with the highest edge
        best = max(candidates, key=lambda vb: vb.edge)

        # Apply per-match Kelly cap
        best.kelly = min(best.kelly, max_kelly_per_match)

        value_bets.append(best)

    value_bets.sort(key=lambda vb: vb.edge, reverse=True)
    return value_bets


def format_report(value_bets: list["ValueBet"]) -> str:
    """
    Format a list of ValueBet objects as a human-readable text table.

    Args:
        value_bets: List of ValueBet objects (typically already sorted by edge).

    Returns:
        Formatted string table.
    """
    if not value_bets:
        return "No value bets found."

    header = (
        f"{'Match':<35} {'Out':4s} {'Team':<20} "
        f"{'Odds':>6} {'Model':>7} {'Fair':>7} {'Edge':>7} {'Kelly':>7}"
    )
    separator = "-" * len(header)

    lines = [separator, header, separator]
    for vb in value_bets:
        lines.append(
            f"{vb.match:<35} {vb.outcome.upper():4s} {vb.team:<20} "
            f"{vb.odds:>6.2f} {vb.model_prob:>7.3f} {vb.fair_implied:>7.3f} "
            f"{vb.edge:>+7.3f} {vb.kelly:>7.4f}"
        )
    lines.append(separator)
    lines.append(f"Total value bets: {len(value_bets)}")

    return "\n".join(lines)
