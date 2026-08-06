"""
core/notifier.py

Telegram notification system for Apollo Forecasting Engine.
Uses raw requests to the Telegram Bot API — no python-telegram-bot dependency.
Formatting: HTML parse_mode (safer than MarkdownV2 for dynamic content like odds/prices).

Usage:
    from core.notifier import TelegramNotifier
    n = TelegramNotifier()
    if n.available():
        n.alert_value_bet(vb, conformal_passed=True)
"""

from __future__ import annotations

import html
import os
import time
import logging
from typing import TYPE_CHECKING

import requests
from dotenv import load_dotenv

try:
    from core.explainer import BetExplainer
    _HAS_EXPLAINER = True
except ImportError:
    _HAS_EXPLAINER = False

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.telegram.org/bot{token}/{method}"
_TIMEOUT  = 10


def h(val) -> str:
    """Escape a value for safe embedding in HTML parse_mode messages."""
    return html.escape(str(val))


def _api(token: str, method: str, **payload) -> dict:
    """POST to Telegram API with exponential backoff on 429."""
    url = _BASE_URL.format(token=token, method=method)
    for attempt in range(4):
        try:
            resp = requests.post(url, json=payload, timeout=_TIMEOUT)
            data = resp.json()
            if resp.status_code == 429:
                wait = data.get("parameters", {}).get("retry_after", 2 ** attempt)
                logger.warning("Telegram 429 — retrying in %ss", wait)
                time.sleep(wait)
                continue
            return data
        except requests.RequestException as exc:
            logger.warning("Telegram %s failed (attempt %d): %s", method, attempt + 1, exc)
            time.sleep(2 ** attempt)
    return {"ok": False}


class TelegramNotifier:
    """Sends HTML-formatted messages to a Telegram chat via Bot API."""

    def __init__(self) -> None:
        load_dotenv()
        self._token   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self._chat_id = os.getenv("TELEGRAM_CHAT_ID",   "").strip()
        self._explainer = BetExplainer() if _HAS_EXPLAINER else None

    def available(self) -> bool:
        return bool(self._token and self._chat_id)

    def send(self, message: str, disable_notification: bool = False) -> bool:
        """Send an HTML-formatted message. Returns True on success."""
        if not self.available():
            logger.debug("Telegram not configured; skipping send.")
            return False
        data = _api(
            self._token, "sendMessage",
            chat_id=self._chat_id,
            text=message,
            parse_mode="HTML",
            disable_web_page_preview=True,
            disable_notification=disable_notification,
        )
        ok = data.get("ok", False)
        if not ok:
            logger.warning("Telegram send failed: %s", data.get("description", data))
        return ok

    # ── High-level alert methods ──────────────────────────────────────────────

    def alert_value_bet(
        self,
        vb,
        conformal_passed: bool,
        steam=None,
        lag=None,
        elo_pred: dict | None = None,
        profile: dict | None = None,
    ) -> bool:
        try:
            from core.user_profile import UserProfile
            _profile = profile if profile is not None else UserProfile.load()
        except Exception:
            _profile = {}

        try:
            from core.user_profile import UserProfile
            if UserProfile.is_muted(_profile):
                logger.debug("Alert suppressed — mute active.")
                return False
            if not UserProfile.should_alert(vb, _profile):
                logger.debug("Alert skipped — profile filter rejected.")
                return False
        except Exception:
            pass

        if steam and steam.get("is_steam") and not _profile.get("alert_on_steam", True):
            steam = None
        if lag and lag.get("lag_detected") and not _profile.get("alert_on_lag", True):
            lag = None

        stake_str = ""
        try:
            from core.user_profile import UserProfile
            stake_str = UserProfile.format_stake(vb.kelly, _profile)
        except Exception:
            stake_str = f"Kelly: {vb.kelly:.1%}"

        parts = vb.match.split(" vs ", 1)
        home  = parts[0] if len(parts) == 2 else vb.match
        away  = parts[1] if len(parts) == 2 else ""

        star     = "★" if conformal_passed else "☆"
        edge_pct = int(round(vb.edge * 100))

        if vb.outcome == "home":
            team_label = home
        elif vb.outcome == "away":
            team_label = away
        else:
            team_label = "Draw"

        time_str   = getattr(vb, "time_utc", "")
        source     = getattr(vb, "source", "")
        model_pct  = f"{vb.model_prob * 100:.1f}%" if hasattr(vb, "model_prob") else "—"
        market_pct = f"{vb.fair_implied * 100:.1f}%" if hasattr(vb, "fair_implied") else "—"
        conf_label = "High confidence" if conformal_passed else "Uncertain"
        conf_emoji = "✅" if conformal_passed else "⚠️"

        lines: list[str] = [
            f"🔮 <b>Apollo Value Bet</b>  {star}",
            "",
            f"⚽ <b>{h(home)} vs {h(away)}</b>",
            f"🏷 Bet: <b>{h(team_label.upper())}</b> ({h(team_label)})",
            f"📊 Model: <code>{h(model_pct)}</code> | Market: <code>{h(market_pct)}</code>",
            f"💰 Edge: <code>+{h(edge_pct)}%</code> | Kelly: <code>{h(vb.kelly * 100:.1f)}%</code>",
        ]

        odds_line = f"🎯 Odds: <code>{h(f'{vb.odds:.2f}')}</code>"
        if source:
            odds_line += f" ({h(source)})"
        lines.append(odds_line)

        if time_str:
            lines.append(f"⏰ Kick-off: <b>{h(time_str)} UTC</b>")

        lines.append(f"💵 Stake: <code>{h(stake_str)}</code>")
        lines.append(f"{conf_emoji} {h(conf_label)}")

        if steam and steam.get("is_steam"):
            direction = steam.get("direction", "").upper()
            magnitude = steam.get("magnitude", 0.0)
            minutes   = steam.get("minutes_elapsed", "?")
            lines.append(
                f"🔥 Steam: <b>{h(direction)}</b> {h(f'{magnitude:+.1%}')} "
                f"in {h(minutes)}min"
            )

        if lag and lag.get("lag_detected"):
            lag_edge = lag.get("edge_from_lag", 0.0)
            lines.append(f"⚡ Lag edge: <code>{h(f'{lag_edge:+.1%}')}</code> vs Pinnacle")

        if conformal_passed and elo_pred is not None and self._explainer is not None:
            try:
                if self._explainer.available():
                    explanation = self._explainer.explain_bet(vb, elo_pred, steam=steam, lag=lag)
                    if explanation:
                        lines += ["", f"💬 <i>{h(explanation)}</i>"]
            except Exception as exc:
                logger.warning("LLM explanation failed: %s", exc)

        lines += ["", "<i>Apollo Forecasting Engine</i>"]
        return self.send("\n".join(lines))

    def alert_steam(self, home: str, away: str, steam_result: dict) -> bool:
        try:
            from core.user_profile import UserProfile
            _profile = UserProfile.load()
            if not _profile.get("alert_on_steam", True) or UserProfile.is_muted(_profile):
                return False
        except Exception:
            pass

        direction  = steam_result.get("direction", "").upper()
        team_label = away if direction == "AWAY" else home
        magnitude  = steam_result.get("magnitude", 0.0)
        minutes    = steam_result.get("minutes_elapsed", "?")
        to_ko      = steam_result.get("minutes_to_kickoff")

        lines: list[str] = [
            "🔥 <b>Steam Move Detected</b>",
            "",
            f"⚽ <b>{h(home)} vs {h(away)}</b>",
            f"📈 Direction: <b>{h(direction)}</b> ({h(team_label)})",
            f"📉 Pinnacle shift: <code>{h(f'{magnitude:+.1%}')}</code> in {h(minutes)}min",
        ]
        if to_ko is not None:
            hours, mins = divmod(int(to_ko), 60)
            lines.append(f"⏰ {h(hours)}h {mins:02d}min to kick-off")
        lines += ["", "<i>Monitor picks for full analysis</i>"]
        return self.send("\n".join(lines))

    def alert_lag_edge(self, home: str, away: str, lag_result: dict) -> bool:
        try:
            from core.user_profile import UserProfile
            _profile = UserProfile.load()
            if not _profile.get("alert_on_lag", True) or UserProfile.is_muted(_profile):
                return False
        except Exception:
            pass

        outcome      = lag_result.get("outcome", "").upper()
        team_label   = home if outcome == "HOME" else (away if outcome == "AWAY" else "Draw")
        pin_implied  = lag_result.get("pinnacle_implied", 0.0)
        soft_implied = lag_result.get("soft_implied", 0.0)
        lag_edge     = lag_result.get("edge_from_lag", 0.0)

        lines: list[str] = [
            "⚡ <b>Pinnacle Lag Edge</b>",
            "",
            f"⚽ <b>{h(home)} vs {h(away)}</b>",
            f"🎯 Outcome: <b>{h(outcome)}</b> ({h(team_label)})",
            f"📌 Pinnacle implied: <code>{h(f'{pin_implied:.1%}')}</code>",
            f"📖 Soft book implied: <code>{h(f'{soft_implied:.1%}')}</code>",
            f"💰 Lag edge: <code>{h(f'{lag_edge:+.1%}')}</code> (bet before they update)",
            "",
            "<i>Apollo Forecasting Engine</i>",
        ]
        return self.send("\n".join(lines))

    def alert_pipeline_done(self, event: str, details: str) -> bool:
        lines: list[str] = [
            f"<b>{h(event)}</b>",
            "",
            h(details),
            "",
            "<i>Apollo Forecasting Engine</i>",
        ]
        return self.send("\n".join(lines))
