"""
monitor_lines.py

CLI tool for continuous line movement monitoring.

Usage:
    python scripts/monitor_lines.py --date 2026-06-25 --competition wc2026 --interval 300

Takes a snapshot every `interval` seconds, prints steam alerts and Pinnacle lag
signals to console. Runs until Ctrl+C.

Alerts:
  STEAM  — Pinnacle implied probability shifted >= 3% in the last 30 minutes.
           Indicates sharp money. Reduce conviction if moving against your bet;
           increase conviction if moving your way.

  LAG    — Soft book implied probability diverges from Pinnacle by > 2%.
           Free edge window before soft book updates to match Pinnacle.
"""

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.line_monitor import LineMonitor
from core.notifier import TelegramNotifier

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _fmt_direction(direction: str) -> str:
    return {"home": "HOME", "draw": "DRAW", "away": "AWAY"}.get(direction, direction.upper())


def _print_header(date_str: str, competition: str, interval: int) -> None:
    print()
    print("=" * 72)
    print("  Apollo Line Movement Monitor")
    print(f"  Date: {date_str}   Competition: {competition}   Interval: {interval}s")
    print("=" * 72)
    print("  STEAM = Pinnacle implied prob shifted >=3% in 30 min (sharp money)")
    print("  LAG   = Soft book diverges from Pinnacle >2% (free edge window)")
    print("=" * 72)
    print()


def _run_snapshot(monitor: LineMonitor, date_str: str, competition: str,
                  notifier: "TelegramNotifier | None" = None) -> None:
    """Take one snapshot and report alerts."""
    print(f"\n[{_now()}] Taking snapshot...")

    snapshots = monitor.snapshot(date_str, competition)

    if not snapshots:
        print("  No odds data returned. Check API keys or try again later.")
        return

    # Deduplicate matches from snapshots
    seen_matches: set[tuple[str, str]] = set()
    matches: list[tuple[str, str]] = []
    for s in snapshots:
        key = (s.home, s.away)
        if key not in seen_matches:
            seen_matches.add(key)
            matches.append(key)

    sources = {s.source for s in snapshots}
    print(f"  {len(snapshots)} snapshots saved   {len(matches)} matches   sources: {', '.join(sorted(sources))}")

    steam_alerts = []
    lag_alerts = []

    for home, away in matches:
        steam = monitor.detect_steam(home, away, date_str, window_minutes=30)
        lag = monitor.pinnacle_lag(home, away, date_str)

        if steam["is_steam"]:
            steam_alerts.append((home, away, steam))
        if lag["lag_detected"]:
            lag_alerts.append((home, away, lag))

    if not steam_alerts and not lag_alerts:
        print("  No alerts.")
        return

    if steam_alerts:
        print()
        print("  STEAM ALERTS")
        print("  " + "-" * 60)
        for home, away, steam in steam_alerts:
            direction = _fmt_direction(steam["direction"])
            magnitude = steam["magnitude"]
            print(
                f"  🔥 STEAM  {home} vs {away}"
                f"  → {direction}  magnitude={magnitude:+.2%}"
            )
            if notifier and notifier.available():
                try:
                    notifier.alert_steam(home, away, steam)
                except Exception as _exc:
                    logger.warning("Telegram steam alert failed: %s", _exc)

    if lag_alerts:
        print()
        print("  LAG EDGE ALERTS")
        print("  " + "-" * 60)
        for home, away, lag in lag_alerts:
            outcome = _fmt_direction(lag["outcome"])
            edge = lag["edge_from_lag"]
            pin = lag["pinnacle_implied"]
            soft = lag["soft_implied"]
            direction_label = "BET ON" if edge > 0 else "FADE"
            print(
                f"  ⚡ LAG    {home} vs {away}"
                f"  → {direction_label} {outcome}"
                f"  pin={pin:.2%} soft={soft:.2%} edge={edge:+.2%}"
            )
            if notifier and notifier.available():
                try:
                    notifier.alert_lag_edge(home, away, lag)
                except Exception as _exc:
                    logger.warning("Telegram lag alert failed: %s", _exc)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apollo line movement monitor — snapshots + steam/lag alerts"
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Match date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--competition",
        default="wc2026",
        choices=["wc2026", "epl", "laliga", "seriea", "bundesliga", "ligue1"],
        help="Competition key. Default: wc2026.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between snapshots. Default: 300 (5 min).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Take a single snapshot then exit (no loop).",
    )
    args = parser.parse_args()

    monitor = LineMonitor()
    notifier = TelegramNotifier()
    _print_header(args.date, args.competition, args.interval)

    if notifier.available():
        print("  📱 Telegram alerts: ON")
    else:
        print("  📱 Telegram alerts: OFF (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env)")

    if args.once:
        _run_snapshot(monitor, args.date, args.competition, notifier=notifier)
        return

    print(f"Monitoring started. Press Ctrl+C to stop.\n")

    try:
        while True:
            _run_snapshot(monitor, args.date, args.competition, notifier=notifier)
            print(f"\n  Next snapshot in {args.interval}s...")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
