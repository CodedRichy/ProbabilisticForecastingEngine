"""
core/explainer.py

LLM explanation layer for Apollo.  Uses a local Ollama instance (gemma4:e2b)
to generate plain-English explanations for value bet recommendations and
skipped matches.

Designed for graceful degradation: every public method returns an empty string
(or empty dict) when Ollama is unavailable, rather than raising.

Usage
-----
    from core.explainer import BetExplainer

    explainer = BetExplainer()
    if explainer.available():
        text = explainer.explain_bet(vb, elo_pred, steam=steam, lag=lag)

    # From Telegram bot /explain command:
    text = explain_match("Japan", "Morocco", "wc2026")
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class BetExplainer:
    """
    Generates plain-English bet explanations via a local Ollama model.

    All methods are safe to call when Ollama is offline — they return an
    empty string on any failure rather than raising.
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "gemma4:e2b"          # fast model for real-time explanations
    _PING_URL = "http://localhost:11434/api/tags"
    _TIMEOUT = 15                  # seconds for generation requests
    _PING_TIMEOUT = 3              # seconds for availability ping

    # ── Availability ──────────────────────────────────────────────────────────

    def available(self) -> bool:
        """Ping Ollama; return True if the service is responsive."""
        try:
            resp = requests.get(self._PING_URL, timeout=self._PING_TIMEOUT)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    # ── Core generation helper ────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        """
        Send *prompt* to Ollama and return the response text.
        Returns "" on any failure (network error, bad JSON, timeout, etc.).
        """
        try:
            resp = requests.post(
                self.OLLAMA_URL,
                json={
                    "model": self.MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 150,
                    },
                },
                timeout=self._TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except requests.Timeout:
            logger.warning("[BetExplainer] Ollama request timed out after %ss.", self._TIMEOUT)
            return ""
        except requests.RequestException as exc:
            logger.warning("[BetExplainer] Ollama request failed: %s", exc)
            return ""
        except (ValueError, KeyError) as exc:
            logger.warning("[BetExplainer] Unexpected Ollama response format: %s", exc)
            return ""

    # ── Context builders ──────────────────────────────────────────────────────

    @staticmethod
    def _build_bet_context(vb, elo_pred: dict, steam=None, lag=None) -> str:
        """
        Build a structured context string to feed into the LLM for a value bet.

        Parameters
        ----------
        vb : ValueBet-like
            Must expose: .match (str), .outcome (str), .team (str),
            .model_prob (float), .fair_implied (float), .edge (float),
            .kelly (float), .odds (float).
        elo_pred : dict
            Output of EloModel.predict() — must contain keys:
            home_elo, away_elo, p_home, p_draw, p_away.
        steam : dict | None
            Output of LineMonitor.detect_steam() — keys: is_steam, direction,
            magnitude, minutes_elapsed.
        lag : dict | None
            Output of LineMonitor.pinnacle_lag() — keys: lag_detected, outcome,
            edge_from_lag.
        """
        parts = vb.match.split(" vs ", 1)
        home = parts[0] if len(parts) == 2 else vb.match
        away = parts[1] if len(parts) == 2 else ""

        home_elo = elo_pred.get("home_elo", 1500.0)
        away_elo = elo_pred.get("away_elo", 1500.0)
        delta = home_elo - away_elo

        # EWMA PPG is an optional field on vb — default to "N/A" when missing
        home_ppg = getattr(vb, "home_form_ewma_ppg", None)
        away_ppg = getattr(vb, "away_form_ewma_ppg", None)
        home_ppg_str = f"{home_ppg:.2f}" if home_ppg is not None else "N/A"
        away_ppg_str = f"{away_ppg:.2f}" if away_ppg is not None else "N/A"

        # conformal flag — if vb carries it, use it; otherwise omit
        conformal = getattr(vb, "conformal_passed", None)
        conformal_str = "CONFIDENT" if conformal else ("UNCERTAIN" if conformal is False else "N/A")

        lines = [
            f"Match: {home} vs {away}",
            f"Bet: {vb.outcome.upper()} | Odds: {vb.odds:.2f} | Edge: {vb.edge:.1%} | Kelly: {vb.kelly:.1%}",
            f"Model prob: {vb.model_prob:.1%} | Market implied: {vb.fair_implied:.1%}",
            f"Home Elo: {home_elo:.0f} | Away Elo: {away_elo:.0f} | Elo delta: {delta:+.0f}",
            f"Home form EWMA PPG: {home_ppg_str} | Away form EWMA PPG: {away_ppg_str}",
        ]

        # Optional steam block
        if steam and steam.get("is_steam"):
            direction = steam.get("direction", "")
            magnitude = steam.get("magnitude", 0.0)
            minutes = steam.get("minutes_elapsed", "?")
            lines.append(
                f"Steam: {direction.upper()} {magnitude:.1%} in {minutes}min"
            )

        # Optional lag block
        if lag and lag.get("lag_detected"):
            lag_edge = lag.get("edge_from_lag", 0.0)
            lines.append(f"Lag edge: {lag_edge:.1%} vs Pinnacle")

        if conformal_str != "N/A":
            lines.append(f"Conformal: {conformal_str}")

        lines += [
            "",
            "In 2-3 sentences explain why Apollo recommends this bet. "
            "Be specific about the signals. Mention the biggest driver first. "
            "End with one key risk factor.",
        ]

        return "\n".join(lines)

    # ── Public explanation methods ────────────────────────────────────────────

    def explain_bet(self, vb, elo_pred: dict, steam=None, lag=None) -> str:
        """
        Generate a 2-3 sentence plain-English explanation for a value bet.

        Parameters are identical to _build_bet_context().

        Returns "" on any failure (Ollama down, timeout, bad response).
        """
        try:
            prompt = self._build_bet_context(vb, elo_pred, steam=steam, lag=lag)
            return self._generate(prompt)
        except Exception as exc:
            logger.warning("[BetExplainer] explain_bet failed unexpectedly: %s", exc)
            return ""

    def explain_skip(self, home: str, away: str, reason: str) -> str:
        """
        Generate a short explanation for why Apollo skipped a match.

        Parameters
        ----------
        home, away : str
            Team names.
        reason : str
            Machine-readable reason code, e.g. "low_confidence",
            "no_edge", "below_kelly_threshold".

        Returns "" on any failure.
        """
        # Map internal reason codes to human-readable descriptions
        _REASON_MAP = {
            "low_confidence":        "the conformal filter rated the model uncertain (prediction set size > 1)",
            "no_edge":               "no outcome had a model probability exceeding the fair implied probability by the minimum threshold",
            "below_kelly_threshold": "the Kelly criterion stake was below the minimum bet size",
            "odds_unavailable":      "live odds could not be retrieved from any source",
        }
        human_reason = _REASON_MAP.get(reason, reason)

        prompt = (
            f"Apollo's betting engine skipped the match {home} vs {away}. "
            f"Reason: {human_reason}. "
            "In 1-2 sentences, explain to a bettor why this match was skipped "
            "and what would need to change for Apollo to recommend a bet."
        )
        try:
            return self._generate(prompt)
        except Exception as exc:
            logger.warning("[BetExplainer] explain_skip failed unexpectedly: %s", exc)
            return ""


# ── Standalone helper for Telegram /explain command ──────────────────────────


def explain_match(
    home: str,
    away: str,
    competition: str = "wc2026",
    *,
    date_str: Optional[str] = None,
) -> str:
    """
    Fetch current odds, run Elo prediction, and return an LLM explanation.

    Designed to be called from the Telegram bot's /explain command handler:

        # In scripts/telegram_bot.py:
        # from core.explainer import explain_match
        # text = explain_match(home, away, competition)

    Parameters
    ----------
    home : str
        Home team name.
    away : str
        Away team name.
    competition : str
        Short competition key (default "wc2026").
    date_str : str | None
        ISO date for odds lookup. Defaults to today (UTC).

    Returns
    -------
    str
        Plain-English explanation, or an informative fallback message
        when data is unavailable.
    """
    # Lazy imports — keep module importable even when optional deps are missing
    try:
        from core.elo_model import EloModel
        from core.odds_fetcher import OddsFetcher
        from core.value_finder import ValueBet, compute_edge, kelly_fraction, remove_overround
    except ImportError as exc:
        logger.error("[explain_match] Missing dependency: %s", exc)
        return ""

    if date_str is None:
        date_str = date.today().isoformat()

    # ── Step 1: fetch live odds ───────────────────────────────────────────────
    odds_fetcher = OddsFetcher()
    odds = odds_fetcher.get_odds(home, away, date_str, competition)
    if not odds:
        return (
            f"Could not fetch live odds for {home} vs {away} "
            f"({competition}, {date_str}). No explanation available."
        )

    # ── Step 2: load Elo model ────────────────────────────────────────────────
    # Prefer a pre-built JSON ratings file; fall back to a warning message
    _ELO_PATHS = [
        "data/models/elo_ratings.json",
        "data/elo_ratings.json",
        "models/elo_ratings.json",
    ]
    elo_model = None
    for _path in _ELO_PATHS:
        import pathlib
        if pathlib.Path(_path).exists():
            try:
                elo_model = EloModel.load(_path)
                logger.debug("[explain_match] Loaded Elo from %s", _path)
                break
            except Exception as exc:
                logger.warning("[explain_match] Failed to load Elo from %s: %s", _path, exc)

    if elo_model is None:
        return (
            f"Elo ratings file not found. Run `scripts/build_elo.py` first. "
            f"Cannot explain {home} vs {away}."
        )

    elo_pred = elo_model.predict(home, away)

    # ── Step 3: build a synthetic ValueBet from Elo probs + fetched odds ─────
    home_odds = odds["home_odds"]
    draw_odds = odds["draw_odds"]
    away_odds = odds["away_odds"]

    fair_home, fair_draw, fair_away = remove_overround(home_odds, draw_odds, away_odds)

    # Find the highest-edge outcome from Elo probabilities
    candidates = [
        ("home", home, elo_pred["p_home"], home_odds, fair_home),
        ("draw", "Draw",  elo_pred["p_draw"], draw_odds, fair_draw),
        ("away", away, elo_pred["p_away"], away_odds, fair_away),
    ]
    best_outcome, best_team, best_prob, best_odds, best_fair = max(
        candidates, key=lambda c: c[2] - c[4]
    )
    edge = compute_edge(best_prob, best_fair)
    kelly = kelly_fraction(best_odds, best_prob)

    vb = ValueBet(
        match=f"{home} vs {away}",
        outcome=best_outcome,
        team=best_team,
        odds=best_odds,
        model_prob=best_prob,
        fair_implied=best_fair,
        edge=edge,
        kelly=kelly,
    )

    # ── Step 4: generate explanation ──────────────────────────────────────────
    explainer = BetExplainer()
    if not explainer.available():
        # Return structured summary without LLM explanation
        return (
            f"{home} vs {away} ({competition})\n"
            f"Best outcome: {best_outcome.upper()} @ {best_odds:.2f}\n"
            f"Model prob: {best_prob:.1%} | Market implied: {best_fair:.1%}\n"
            f"Edge: {edge:+.1%} | Kelly: {kelly:.1%}\n"
            f"Home Elo: {elo_pred['home_elo']:.0f} | Away Elo: {elo_pred['away_elo']:.0f}\n"
            "(Ollama unavailable — LLM explanation skipped)"
        )

    explanation = explainer.explain_bet(vb, elo_pred)
    if not explanation:
        explanation = "(LLM explanation unavailable)"

    return (
        f"{home} vs {away} ({competition})\n"
        f"Best outcome: {best_outcome.upper()} @ {best_odds:.2f} | "
        f"Edge: {edge:+.1%} | Kelly: {kelly:.1%}\n\n"
        f"{explanation}"
    )
