"""
core/user_profile.py

User profile management for Apollo Forecasting Engine.
Handles bankroll, risk mode, staking calculations, and alert filtering.

Usage:
    from core.user_profile import UserProfile
    profile = UserProfile.load()
    stake = UserProfile.get_stake_amount(vb.kelly, profile["bankroll"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # ValueBet is duck-typed below

logger = logging.getLogger(__name__)

_DEFAULTS: dict = {
    "bankroll": 100.0,
    "risk_mode": "moderate",
    "kelly_multipliers": {"conservative": 0.25, "moderate": 0.5, "aggressive": 1.0},
    "min_edge": 0.03,
    "max_daily_exposure": 0.20,
    "competitions": ["wc2026"],
    "never_bet_draws": False,
    "min_odds": 1.30,
    "max_odds": 15.0,
    "telegram_alerts": True,
    "alert_on_steam": True,
    "alert_on_lag": True,
}


class UserProfile:
    """Static helper class for loading, saving, and using the user profile."""

    DEFAULT_PATH = Path(__file__).parent.parent / "config" / "user_profile.json"

    # ── I/O ──────────────────────────────────────────────────────────────────

    @staticmethod
    def load(path: Path | str | None = None) -> dict:
        """
        Load user profile from JSON.  Returns a copy of defaults merged with
        whatever is stored on disk — missing keys fall back to defaults.

        Parameters
        ----------
        path : Path | str | None
            Override the default profile path (useful in tests).
        """
        target = Path(path) if path else UserProfile.DEFAULT_PATH
        profile = dict(_DEFAULTS)

        if target.exists():
            try:
                with target.open("r", encoding="utf-8") as fh:
                    stored = json.load(fh)
                # Shallow merge: stored values override defaults; missing keys kept
                profile.update(stored)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read user profile (%s) — using defaults.", exc)
        else:
            logger.debug("User profile not found at %s — using defaults.", target)

        return profile

    @staticmethod
    def save(profile: dict, path: Path | str | None = None) -> None:
        """
        Persist the profile dict to JSON.

        Parameters
        ----------
        profile : dict
            The profile to write.
        path : Path | str | None
            Override the default profile path.
        """
        target = Path(path) if path else UserProfile.DEFAULT_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("w", encoding="utf-8") as fh:
                json.dump(profile, fh, indent=2)
            logger.debug("User profile saved to %s.", target)
        except OSError as exc:
            logger.error("Failed to save user profile: %s", exc)

    # ── Staking helpers ───────────────────────────────────────────────────────

    @staticmethod
    def get_kelly_fraction(vb_kelly: float, profile: dict | None = None) -> float:
        """
        Apply the profile's risk-mode Kelly multiplier to a raw Kelly fraction.

        The raw Kelly from value_finder.py already applies a conservative 0.25
        fraction.  This method multiplies by the risk multiplier then caps the
        result so that a single bet never exceeds 25% of bankroll.

        Parameters
        ----------
        vb_kelly : float
            Raw Kelly fraction from ValueBet.kelly (already quarter-Kelly).
        profile : dict | None
            Loaded profile dict.  Loaded from disk if not provided.

        Returns
        -------
        float
            Adjusted Kelly fraction (0.0 – 0.25).
        """
        if profile is None:
            profile = UserProfile.load()

        risk_mode   = profile.get("risk_mode", "moderate")
        multipliers = profile.get("kelly_multipliers",
                                  _DEFAULTS["kelly_multipliers"])
        multiplier  = multipliers.get(risk_mode, 0.5)

        # The stored vb_kelly is already quarter-Kelly; rescale relative to 0.25 baseline
        # so that moderate (0.5) doubles it and aggressive (1.0) quadruples it.
        baseline   = multipliers.get("conservative", 0.25)
        scale      = multiplier / baseline if baseline > 0 else 1.0
        adjusted   = vb_kelly * scale

        # Hard cap: no single bet > 25% of bankroll
        return min(adjusted, 0.25)

    @staticmethod
    def get_stake_amount(vb_kelly: float, bankroll: float,
                         profile: dict | None = None) -> float:
        """
        Convert a Kelly fraction to an actual stake in currency units.

        Parameters
        ----------
        vb_kelly : float
            Raw Kelly fraction from ValueBet.kelly.
        bankroll : float
            Current bankroll in currency units (e.g. ₹100).
        profile : dict | None
            Loaded profile dict.  Loaded from disk if not provided.

        Returns
        -------
        float
            Stake amount in currency units, rounded to 2 decimal places.
        """
        fraction = UserProfile.get_kelly_fraction(vb_kelly, profile)
        return round(fraction * bankroll, 2)

    @staticmethod
    def format_stake(vb_kelly: float, profile: dict | None = None) -> str:
        """
        Format the stake as a human-readable string, e.g. "₹4.20 (4.2%)".

        Parameters
        ----------
        vb_kelly : float
            Raw Kelly fraction from ValueBet.kelly.
        profile : dict | None
            Loaded profile dict.  Loaded from disk if not provided.

        Returns
        -------
        str
            Formatted stake string.
        """
        if profile is None:
            profile = UserProfile.load()

        bankroll = float(profile.get("bankroll", 100))
        fraction = UserProfile.get_kelly_fraction(vb_kelly, profile)
        amount   = round(fraction * bankroll, 2)
        pct      = fraction * 100
        return f"₹{amount:.2f} ({pct:.1f}%)"

    # ── Alert filtering ───────────────────────────────────────────────────────

    @staticmethod
    def should_alert(vb, profile: dict | None = None) -> bool:
        """
        Return True if this ValueBet should trigger an alert based on profile filters.

        Checks applied (in order):
        1. edge >= profile.min_edge
        2. odds >= profile.min_odds
        3. odds <= profile.max_odds
        4. if never_bet_draws is True, skip draw outcomes

        Parameters
        ----------
        vb : ValueBet-like
            Must expose .edge (float), .odds (float), .outcome (str).
        profile : dict | None
            Loaded profile dict.  Loaded from disk if not provided.

        Returns
        -------
        bool
        """
        if profile is None:
            profile = UserProfile.load()

        if vb.edge < profile.get("min_edge", 0.03):
            return False
        if vb.odds < profile.get("min_odds", 1.30):
            return False
        if vb.odds > profile.get("max_odds", 15.0):
            return False
        if profile.get("never_bet_draws", False) and vb.outcome == "draw":
            return False

        return True

    # ── Mute helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def is_muted(profile: dict | None = None) -> bool:
        """
        Return True if alerts are currently muted (mute_until is in the future).

        Parameters
        ----------
        profile : dict | None
        """
        import time
        if profile is None:
            profile = UserProfile.load()

        mute_until = profile.get("mute_until")
        if mute_until is None:
            return False
        return float(mute_until) > time.time()
