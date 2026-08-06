"""
Resolve completed predictions: fetch closing odds + actual results,
compute CLV and P&L, update log, print performance report.

Usage:
    python scripts/track_clv.py                  # resolve all unresolved
    python scripts/track_clv.py --report         # just print report, no updates
    python scripts/track_clv.py --from 2026-06-01
"""

import sys, argparse, logging
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from core.prediction_logger import load_predictions, _load, _save, LOG_PATH
from core.clv_tracker import (
    fetch_closing_odds, fetch_result, compute_clv, compute_pnl
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def resolve_unresolved(df: pd.DataFrame) -> pd.DataFrame:
    today = date.today().isoformat()
    unresolved = df[~df["resolved"].fillna(False)].copy()

    # Only try to resolve past matches (not today — they might still be playing)
    unresolved = unresolved[unresolved["match_date"].fillna("") < today]

    if unresolved.empty:
        logger.info("No unresolved past matches to update.")
        return df

    updated = 0
    for idx in unresolved.index:
        row = df.loc[idx]
        home, away = row["home"], row["away"]
        match_date = row["match_date"]
        competition = row.get("competition", "wc2026") or "wc2026"

        # Fetch result
        result = fetch_result(home, away, match_date, competition)
        if result is None:
            logger.debug("Result not available yet: %s vs %s %s", home, away, match_date)
            continue

        df.at[idx, "actual_result"] = result

        # Closing odds
        closing = fetch_closing_odds(home, away, match_date, competition)
        if closing:
            df.at[idx, "close_home_odds"] = closing["home_odds"]
            df.at[idx, "close_draw_odds"] = closing["draw_odds"]
            df.at[idx, "close_away_odds"] = closing["away_odds"]
            df.at[idx, "close_source"]    = closing["source"]

            # CLV (only meaningful if a bet was placed)
            bet_outcome = row.get("bet_outcome")
            bet_odds    = row.get("bet_odds")
            if pd.notna(bet_outcome) and pd.notna(bet_odds) and closing:
                close_odds_map = {
                    "home": closing["home_odds"],
                    "draw": closing["draw_odds"],
                    "away": closing["away_odds"],
                }
                close_odds = close_odds_map.get(str(bet_outcome))
                if close_odds:
                    clv_abs, clv_pct = compute_clv(float(bet_odds), float(close_odds))
                    df.at[idx, "clv"]     = clv_abs
                    df.at[idx, "clv_pct"] = clv_pct

        # P&L
        bet_outcome = row.get("bet_outcome")
        bet_odds    = row.get("bet_odds")
        if pd.notna(bet_outcome) and pd.notna(bet_odds):
            df.at[idx, "profit_loss"] = compute_pnl(str(bet_outcome), float(bet_odds), result)

        df.at[idx, "resolved"] = True
        updated += 1
        logger.info("Resolved: %s vs %s  result=%s  pnl=%s  clv_pct=%s",
                    home, away, result,
                    df.at[idx, "profit_loss"], df.at[idx, "clv_pct"])

    logger.info("Resolved %d/%d unresolved matches.", updated, len(unresolved))
    return df


def print_report(df: pd.DataFrame):
    resolved = df[df["resolved"].fillna(False)].copy()
    bets = resolved[resolved["bet_outcome"].notna()].copy()

    print("\n" + "=" * 65)
    print("  APOLLO PERFORMANCE REPORT")
    print("=" * 65)

    # ── All predictions (no bet required) ────────────────────────
    print(f"\n  Predictions logged : {len(df)}")
    print(f"  Resolved           : {len(resolved)}")
    print(f"  Pending            : {len(df) - len(resolved)}")

    if resolved.empty:
        print("\n  No resolved predictions yet.")
        return

    # ── Bet performance ───────────────────────────────────────────
    if bets.empty:
        print("\n  No bets placed yet.")
    else:
        wins     = (bets["profit_loss"] > 0).sum()
        losses   = (bets["profit_loss"] < 0).sum()
        win_rate = wins / len(bets) * 100
        total_pl = bets["profit_loss"].sum()
        roi      = total_pl / len(bets) * 100
        avg_odds = bets["bet_odds"].mean()

        print(f"\n  ── Bet Record ──────────────────────────────────")
        print(f"  Bets placed  : {len(bets)}")
        print(f"  Wins / Losses: {wins} / {losses}")
        print(f"  Win rate     : {win_rate:.1f}%")
        print(f"  Total P&L    : {total_pl:+.2f} units")
        print(f"  ROI          : {roi:+.1f}%")
        print(f"  Avg bet odds : {avg_odds:.2f}")

        # CLV section
        clv_bets = bets[bets["clv_pct"].notna()]
        if not clv_bets.empty:
            avg_clv     = clv_bets["clv_pct"].mean()
            pos_clv_pct = (clv_bets["clv_pct"] > 0).mean() * 100
            print(f"\n  ── Closing Line Value (CLV) ────────────────────")
            print(f"  Bets with CLV data : {len(clv_bets)}")
            print(f"  Avg CLV            : {avg_clv:+.2f}%")
            print(f"  Beat closing line  : {pos_clv_pct:.0f}% of bets")
            if avg_clv > 0:
                print(f"  ✓ Positive CLV — model has genuine edge")
            else:
                print(f"  ✗ Negative CLV — need more data or model improvement")

        # By outcome type
        print(f"\n  ── By Bet Type ─────────────────────────────────")
        for outcome in ["home", "draw", "away"]:
            sub = bets[bets["bet_outcome"] == outcome]
            if sub.empty:
                continue
            sub_wins = (sub["profit_loss"] > 0).sum()
            sub_roi  = sub["profit_loss"].sum() / len(sub) * 100
            print(f"  {outcome.upper():<5}: {len(sub):>3} bets  "
                  f"{sub_wins}/{len(sub)} wins  ROI {sub_roi:+.1f}%")

    # ── Recent predictions table ──────────────────────────────────
    print(f"\n  ── Recent Results ──────────────────────────────")
    recent = resolved.tail(10)[["match_date","home","away","bet_outcome",
                                 "bet_odds","actual_result","profit_loss","clv_pct"]]
    for _, r in recent.iterrows():
        bet = f"{r['bet_outcome']} @ {r['bet_odds']:.2f}" if pd.notna(r['bet_outcome']) else "no bet"
        pnl = f"{r['profit_loss']:+.2f}" if pd.notna(r['profit_loss']) else "—"
        clv = f"CLV {r['clv_pct']:+.1f}%" if pd.notna(r['clv_pct']) else ""
        print(f"  {r['match_date']}  {r['home'][:12]} v {r['away'][:12]:<12}"
              f"  {bet:<18}  res={r['actual_result']}  P&L {pnl}  {clv}")

    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="Print report only, no updates")
    parser.add_argument("--from",   dest="date_from", default=None, help="YYYY-MM-DD")
    args = parser.parse_args()

    df = load_predictions(date_from=args.date_from)

    if df.empty:
        print("No predictions logged yet. Run: python scripts/predict_today.py --odds")
        return

    if not args.report:
        df_all = _load()
        df_all = resolve_unresolved(df_all)
        _save(df_all)
        df = load_predictions(date_from=args.date_from)

    print_report(df)


if __name__ == "__main__":
    main()
