"""
line_monitor.py

Pinnacle line-movement monitor for detecting:
  1. Steam moves — sharp money causing rapid implied probability shifts at Pinnacle.
  2. Lag edge — soft books showing stale Pinnacle prices, creating a free-edge window.

The Odds API includes Pinnacle as a bookmaker key "pinnacle" in its bookmakers array.
This module extracts Pinnacle specifically from that response rather than building
a separate scraper.

Usage:
    monitor = LineMonitor()
    snapshots = monitor.snapshot("2026-06-25", "wc2026")
    steam = monitor.detect_steam("Argentina", "France", "2026-06-25")
    lag = monitor.pinnacle_lag("Argentina", "France", "2026-06-25")
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from core.odds_fetcher import (
    OddsFetcher,
    _OddsAPISource,
    _make_session,
    _normalise_team,
    _teams_match,
    _build_record,
    _COMPETITION_MAP,
    REQUEST_TIMEOUT,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Bookmaker key for Pinnacle inside The Odds API response
_PINNACLE_KEY = "pinnacle"

# Implied probability shift threshold for steam detection (3 % in 30 min)
_STEAM_THRESHOLD = 0.03

# Lag detection threshold — soft book implied differs from Pinnacle by this much
_LAG_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class LineSnapshot:
    """A single odds snapshot for one match from one source at one moment."""

    home: str
    away: str
    date: str
    source: str
    home_odds: float
    draw_odds: float
    away_odds: float
    fetched_at: str  # ISO 8601 timestamp in UTC


# ---------------------------------------------------------------------------
# Pinnacle extraction helper
# ---------------------------------------------------------------------------


def _extract_pinnacle_from_event(event: dict, home: str, away: str, date: str) -> Optional[dict]:
    """
    Pull Pinnacle odds from a raw Odds API event dict.

    The Odds API event structure:
        {
          "home_team": "...",
          "away_team": "...",
          "bookmakers": [
            {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [...]}]},
            ...
          ]
        }

    Returns a standard odds record or None if Pinnacle data not present.
    """
    for bookmaker in event.get("bookmakers", []):
        if bookmaker.get("key", "").lower() != _PINNACLE_KEY:
            continue
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            ho = outcomes.get(event.get("home_team", ""))
            ao = outcomes.get(event.get("away_team", ""))
            do_ = outcomes.get("Draw")
            if ho and do_ and ao:
                return _build_record(home, away, date, "pinnacle", ho, do_, ao)
    return None


def _extract_soft_from_event(event: dict, home: str, away: str, date: str) -> list[dict]:
    """
    Pull odds from all non-Pinnacle bookmakers in an Odds API event dict.
    Returns a list of standard odds records, one per soft book that has h2h odds.
    """
    records = []
    for bookmaker in event.get("bookmakers", []):
        key = bookmaker.get("key", "").lower()
        if key == _PINNACLE_KEY:
            continue
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            ho = outcomes.get(event.get("home_team", ""))
            ao = outcomes.get(event.get("away_team", ""))
            do_ = outcomes.get("Draw")
            if ho and do_ and ao:
                records.append(
                    _build_record(home, away, date, key, ho, do_, ao)
                )
            break
    return records


# ---------------------------------------------------------------------------
# LineMonitor
# ---------------------------------------------------------------------------


class LineMonitor:
    """
    Captures timestamped odds snapshots and detects steam moves and Pinnacle lag.

    Snapshots are stored as parquet files at:
        data/odds_history/{date}.parquet

    Each row is one LineSnapshot (one source, one match, one fetch time).
    Multiple calls on the same day append rows — this accumulates a time series.
    """

    SNAPSHOT_DIR = Path(__file__).parent.parent / "data" / "odds_history"

    def __init__(self) -> None:
        self.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        self._odds_api = _OddsAPISource()
        self._fetcher = OddsFetcher()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _parquet_path(self, date: str) -> Path:
        return self.SNAPSHOT_DIR / f"{date}.parquet"

    def _load_parquet(self, date: str) -> pd.DataFrame:
        p = self._parquet_path(date)
        if not p.exists():
            return pd.DataFrame(
                columns=["home", "away", "date", "source",
                         "home_odds", "draw_odds", "away_odds", "fetched_at"]
            )
        return pd.read_parquet(p)

    def _append_snapshots(self, date: str, snapshots: list[LineSnapshot]) -> None:
        if not snapshots:
            return
        new_rows = pd.DataFrame([vars(s) for s in snapshots])
        existing = self._load_parquet(date)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined.to_parquet(self._parquet_path(date), index=False)
        logger.info("[LineMonitor] Saved %d snapshot rows to %s.parquet", len(snapshots), date)

    def _to_implied(self, odds: float) -> float:
        """Convert decimal odds to implied probability (raw, not fair)."""
        if odds <= 0:
            return 0.0
        return 1.0 / odds

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def snapshot(self, date: str, competition: str) -> list[LineSnapshot]:
        """
        Fetch current odds from ALL available sources simultaneously.

        For The Odds API: extracts BOTH Pinnacle and soft-book odds from
        the single API response (no extra quota used).
        For other sources (SofaScore, Betfair, OddsChecker): one record
        per match tagged with the source name.

        Saves each snapshot as a parquet row to data/odds_history/{date}.parquet.
        Appends to existing file to accumulate a time series.

        Returns the list of LineSnapshot objects captured.
        """
        fetched_at = datetime.now(timezone.utc).isoformat()
        snapshots: list[LineSnapshot] = []

        # ── The Odds API: extract Pinnacle + soft books from single call ────
        if self._odds_api.available():
            try:
                # Force a fresh fetch (bypass cache) by using a unique cache key.
                # We want time-series data, not cached 5-min-old prices.
                self._odds_api._cache_key = ""  # invalidate cache
                events = self._odds_api.get_all(competition, date)

                for event in events:
                    h = event.get("home_team", "")
                    a = event.get("away_team", "")
                    if not h or not a:
                        continue

                    # Pinnacle record
                    pin_rec = _extract_pinnacle_from_event(event, h, a, date)
                    if pin_rec:
                        snapshots.append(LineSnapshot(
                            home=h, away=a, date=date,
                            source="pinnacle",
                            home_odds=pin_rec["home_odds"],
                            draw_odds=pin_rec["draw_odds"],
                            away_odds=pin_rec["away_odds"],
                            fetched_at=fetched_at,
                        ))

                    # All soft book records from same event
                    for rec in _extract_soft_from_event(event, h, a, date):
                        snapshots.append(LineSnapshot(
                            home=h, away=a, date=date,
                            source=rec["source"],
                            home_odds=rec["home_odds"],
                            draw_odds=rec["draw_odds"],
                            away_odds=rec["away_odds"],
                            fetched_at=fetched_at,
                        ))

            except Exception as exc:
                logger.warning("[LineMonitor] OddsAPI snapshot failed: %s", exc)

        # ── OddsChecker scraper (always available, tags as "oddschecker") ──
        try:
            from core.odds_fetcher import _OddsCheckerSource
            checker = _OddsCheckerSource()
            raw = checker._get_all_raw(competition)
            for m in raw:
                snapshots.append(LineSnapshot(
                    home=m["home"], away=m["away"], date=date,
                    source="oddschecker",
                    home_odds=m["home_odds"],
                    draw_odds=m["draw_odds"],
                    away_odds=m["away_odds"],
                    fetched_at=fetched_at,
                ))
        except Exception as exc:
            logger.warning("[LineMonitor] OddsChecker snapshot failed: %s", exc)

        self._append_snapshots(date, snapshots)
        logger.info(
            "[LineMonitor] Snapshot complete: %d records across %d sources.",
            len(snapshots),
            len({s.source for s in snapshots}),
        )
        return snapshots

    def get_history(self, home: str, away: str, date: str) -> pd.DataFrame:
        """
        Return all snapshots for a specific match, sorted by fetched_at.

        Columns: source, home_odds, draw_odds, away_odds, fetched_at
        """
        df = self._load_parquet(date)
        if df.empty:
            return df

        mask = df.apply(
            lambda row: _teams_match(str(row["home"]), home) and _teams_match(str(row["away"]), away),
            axis=1,
        )
        match_df = df[mask].copy()
        match_df = match_df.sort_values("fetched_at").reset_index(drop=True)
        return match_df[["source", "home_odds", "draw_odds", "away_odds", "fetched_at"]]

    def detect_steam(
        self, home: str, away: str, date: str, window_minutes: int = 30
    ) -> dict:
        """
        Look at the last `window_minutes` of Pinnacle snapshots for a match.

        Steam move = implied probability at Pinnacle shifted >= 3% in the window.

        Returns:
            {
                "direction": "home" | "draw" | "away" | None,
                "magnitude": float,   # % implied prob change in window
                "is_steam":  bool,    # True if magnitude > STEAM_THRESHOLD
            }
        """
        null_result = {"direction": None, "magnitude": 0.0, "is_steam": False}

        history = self.get_history(home, away, date)
        if history.empty:
            return null_result

        # Filter to Pinnacle only for steam detection
        pin_hist = history[history["source"] == "pinnacle"].copy()
        if pin_hist.empty:
            # Fall back to any available source if Pinnacle not present
            pin_hist = history.copy()

        pin_hist["fetched_at"] = pd.to_datetime(pin_hist["fetched_at"], utc=True)
        pin_hist = pin_hist.sort_values("fetched_at")

        cutoff = pin_hist["fetched_at"].max() - pd.Timedelta(minutes=window_minutes)
        window_df = pin_hist[pin_hist["fetched_at"] >= cutoff]

        if len(window_df) < 2:
            return null_result

        oldest = window_df.iloc[0]
        newest = window_df.iloc[-1]

        # Implied probability changes (raw, not fair — we care about direction/magnitude)
        changes = {
            "home": self._to_implied(newest["home_odds"]) - self._to_implied(oldest["home_odds"]),
            "draw": self._to_implied(newest["draw_odds"]) - self._to_implied(oldest["draw_odds"]),
            "away": self._to_implied(newest["away_odds"]) - self._to_implied(oldest["away_odds"]),
        }

        # Steam direction = outcome whose implied prob dropped the most (price shortened = sharpened)
        # A shortening implied prob means LESS probability (lower odds = shorter price),
        # which is counterintuitive. In betting: "price shortening" = implied prob INCREASES.
        # So we look for the LARGEST POSITIVE change in implied probability.
        direction = max(changes, key=lambda k: changes[k])
        magnitude = changes[direction]

        if magnitude <= 0:
            # No outcome saw a positive shift — no steam
            return null_result

        return {
            "direction": direction,
            "magnitude": round(magnitude, 4),
            "is_steam": magnitude >= _STEAM_THRESHOLD,
        }

    def pinnacle_lag(self, home: str, away: str, date: str) -> dict:
        """
        Compare the most recent Pinnacle implied probability against soft books.

        Lag detected = soft book implied probability differs from Pinnacle by > 2%.

        Returns:
            {
                "lag_detected": bool,
                "outcome":          str,   # "home" | "draw" | "away"
                "pinnacle_implied": float,
                "soft_implied":     float,
                "edge_from_lag":    float, # positive = soft book overprices this outcome
            }

        When lag_detected=True and edge_from_lag > 0, the soft book is offering
        a higher implied probability than Pinnacle (i.e. shorter odds than Pinnacle)
        which creates a free-edge window to bet against the soft book before it updates.

        Note: "edge_from_lag" is computed as (soft_implied - pinnacle_implied).
        A positive value means soft book thinks this outcome is MORE likely than
        Pinnacle does — i.e. soft book is offering LONGER odds, creating value.

        Wait — let's be precise:
            soft_implied > pinnacle_implied → soft book has SHORTER odds (thinks outcome more likely)
                                             → Pinnacle (sharper) disagrees → bet AGAINST this outcome
            soft_implied < pinnacle_implied → soft book has LONGER odds (thinks outcome less likely)
                                             → Pinnacle prices it higher → bet ON this outcome = edge

        edge_from_lag = pinnacle_implied - soft_implied
        Positive edge_from_lag = soft book underprices the outcome vs Pinnacle = free edge to bet ON it.
        """
        null_result = {
            "lag_detected": False,
            "outcome": "",
            "pinnacle_implied": 0.0,
            "soft_implied": 0.0,
            "edge_from_lag": 0.0,
        }

        history = self.get_history(home, away, date)
        if history.empty:
            return null_result

        # Get most recent Pinnacle snapshot
        pin_rows = history[history["source"] == "pinnacle"]
        if pin_rows.empty:
            return null_result

        pin_latest = pin_rows.sort_values("fetched_at").iloc[-1]

        # Get most recent soft book snapshot (any non-Pinnacle source)
        soft_rows = history[history["source"] != "pinnacle"]
        if soft_rows.empty:
            return null_result

        soft_latest = soft_rows.sort_values("fetched_at").iloc[-1]

        # Compute implied probabilities
        outcomes = {
            "home": (
                self._to_implied(pin_latest["home_odds"]),
                self._to_implied(soft_latest["home_odds"]),
            ),
            "draw": (
                self._to_implied(pin_latest["draw_odds"]),
                self._to_implied(soft_latest["draw_odds"]),
            ),
            "away": (
                self._to_implied(pin_latest["away_odds"]),
                self._to_implied(soft_latest["away_odds"]),
            ),
        }

        # Find the outcome with the largest divergence
        best_outcome = None
        best_edge = 0.0
        best_pin_implied = 0.0
        best_soft_implied = 0.0

        for outcome, (pin_impl, soft_impl) in outcomes.items():
            # edge = pinnacle_implied - soft_implied
            # positive → soft book underprices this outcome → bet on it
            edge = pin_impl - soft_impl
            if abs(edge) > abs(best_edge):
                best_edge = edge
                best_outcome = outcome
                best_pin_implied = pin_impl
                best_soft_implied = soft_impl

        if best_outcome is None or abs(best_edge) < _LAG_THRESHOLD:
            return null_result

        return {
            "lag_detected": True,
            "outcome": best_outcome,
            "pinnacle_implied": round(best_pin_implied, 4),
            "soft_implied": round(best_soft_implied, 4),
            "edge_from_lag": round(best_edge, 4),
        }
