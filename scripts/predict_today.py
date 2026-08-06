import argparse, sys, logging
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.elo_model import EloModel
from core.fixtures_fetcher import get_today_fixtures
from core.value_finder import find_value_bets, format_report

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Elo is primary for international; XGB adds calibration when market features present
_ELO_W = 0.70
_XGB_W = 0.30

# ── Ensemble weight table ─────────────────────────────────────────────────────
# The blend depends on which optional components are active. Weights are keyed by
# a frozenset of the active component names; the helper _ensemble_weights() picks
# the right row and renormalises if a configured component turned out unavailable.
#
#   elo + xgb                          -> 0.70 / 0.30
#   elo + xgb + player_model           -> 0.50 / 0.30 / 0.20
#   elo + xgb + transformer            -> 0.45 / 0.30 / 0.25
#   elo + xgb + player_model + transf  -> 0.40 / 0.25 / 0.20 / 0.15
_ENSEMBLE_WEIGHTS: dict[frozenset, dict[str, float]] = {
    frozenset({"elo", "xgb"}):                          {"elo": 0.70, "xgb": 0.30},
    frozenset({"elo", "xgb", "player"}):                {"elo": 0.50, "xgb": 0.30, "player": 0.20},
    frozenset({"elo", "xgb", "transformer"}):           {"elo": 0.45, "xgb": 0.30, "transformer": 0.25},
    frozenset({"elo", "xgb", "player", "transformer"}): {"elo": 0.40, "xgb": 0.25, "player": 0.20, "transformer": 0.15},
}


def _ensemble_weights(active: list[str]) -> dict[str, float]:
    """Return normalised ensemble weights for the set of *active* components.

    ``active`` always contains "elo"; "xgb", "player" and "transformer" are
    present only when their model produced a usable prediction for the match.
    Falls back to an Elo-only / proportional blend for any combination not in
    the explicit table (e.g. elo+player with no xgb).
    """
    key = frozenset(active)
    if key in _ENSEMBLE_WEIGHTS:
        return dict(_ENSEMBLE_WEIGHTS[key])
    if active == ["elo"]:
        return {"elo": 1.0}
    # Unlisted combination: take each component's weight from the richest table
    # row that contains it, then renormalise over the active components only.
    base = {"elo": 0.70, "xgb": 0.30, "player": 0.20, "transformer": 0.25}
    w = {c: base.get(c, 0.0) for c in active}
    total = sum(w.values())
    if total <= 0:
        return {"elo": 1.0}
    return {c: v / total for c, v in w.items()}

# Path to historical club-match data (used for rolling form at inference time).
# This dataset covers EPL, Championship, Bundesliga, LaLiga, SerieA, Ligue1.
# International teams (e.g. WC2026 fixtures) will NOT be found here — form
# features will remain NaN for those matches, falling back to training medians.
_HIST_PARQUET = Path("data/processed/matches_xg.parquet")

# League average goals per team per match — used for Poisson xG approximation
_LEAGUE_AVG_GOALS = 1.3


def _load_form_index(parquet_path: Path) -> dict | None:
    """
    Load the historical match parquet and build a per-team form index.

    Returns a dict mapping team name -> sorted list of dicts
    {date, result ('W'/'D'/'L'), gf, ga} so we can efficiently
    compute rolling 5-game form at query time.

    Returns None if the parquet file is missing or unreadable.
    """
    if not parquet_path.exists():
        logger.debug("Historical parquet not found at %s; form features will be NaN.", parquet_path)
        return None
    try:
        df = pd.read_parquet(parquet_path, columns=["date", "home_team", "away_team",
                                                     "home_goals", "away_goals", "result"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["home_goals", "away_goals", "result"])
        df = df.sort_values("date").reset_index(drop=True)

        index: dict[str, list[dict]] = {}

        for row in df.itertuples(index=False):
            # Determine result for each side
            if row.result == "H":
                h_res, a_res = "W", "L"
            elif row.result == "A":
                h_res, a_res = "L", "W"
            else:
                h_res, a_res = "D", "D"

            for team, res, gf, ga in [
                (row.home_team, h_res, row.home_goals, row.away_goals),
                (row.away_team, a_res, row.away_goals, row.home_goals),
            ]:
                if team not in index:
                    index[team] = []
                index[team].append({
                    "date":   row.date,
                    "result": res,
                    "gf":     float(gf),
                    "ga":     float(ga),
                })

        logger.debug("Form index built: %d teams from %s.", len(index), parquet_path)
        return index
    except Exception as exc:
        logger.warning("Could not build form index from %s: %s", parquet_path, exc)
        return None


def _compute_form(form_index: dict | None, team: str, as_of: str, window: int = 5) -> dict:
    """
    Compute rolling form features for *team* using up to *window* matches
    played strictly before *as_of* date.

    Returns a dict with keys:
        winrate_5, ppg_5, gf_mean_5, ga_mean_5

    All values are NaN when the team has < window matches of history,
    or when the team is not present in the form index (e.g. international teams).
    """
    nan = {
        "winrate_5": np.nan,
        "ppg_5":     np.nan,
        "gf_mean_5": np.nan,
        "ga_mean_5": np.nan,
    }

    if form_index is None or team not in form_index:
        return nan

    cutoff = pd.Timestamp(as_of)
    matches = [m for m in form_index[team] if m["date"] < cutoff]

    if len(matches) < window:
        return nan

    recent = matches[-window:]
    wins   = sum(1 for m in recent if m["result"] == "W")
    draws  = sum(1 for m in recent if m["result"] == "D")
    points = wins * 3 + draws

    return {
        "winrate_5": wins  / window,
        "ppg_5":     points / window,
        "gf_mean_5": float(np.mean([m["gf"] for m in recent])),
        "ga_mean_5": float(np.mean([m["ga"] for m in recent])),
    }


def _poisson_xg_from_elo(elo_expected_home: float,
                          league_avg: float = _LEAGUE_AVG_GOALS) -> tuple[float, float]:
    """
    Derive Poisson expected-goal rates from the Elo-based home win probability.

    Uses a Dixon-Coles-style parameterisation:
      - Total expected goals fixed at 2 * league_avg.
      - Home share proportional to elo_expected_home (win prob before draw allocation).
      - Away share takes the complement.

    This is a first-order approximation that replaces the full attack/defence
    rating system when historical goals data are unavailable (e.g. international
    fixtures). It is strictly better than defaulting to training-set medians
    because it encodes the Elo strength differential.

    Parameters
    ----------
    elo_expected_home : float
        Elo expected score (probability home wins a binary contest), range (0, 1).
    league_avg : float
        Assumed average goals per team per match (default 1.3 for international).

    Returns
    -------
    (poisson_home_xg, poisson_away_xg) : tuple[float, float]
    """
    total_xg = 2.0 * league_avg
    home_share = elo_expected_home
    away_share = 1.0 - elo_expected_home
    return float(home_share * total_xg), float(away_share * total_xg)


def _ensemble(elo: dict, xgb: dict | None,
              player: dict | None = None,
              transformer: dict | None = None) -> dict:
    """Blend 1X2 probabilities across the active model components.

    ``elo`` is always present. ``xgb`` / ``player`` / ``transformer`` are each
    blended in only when their dict is not None. Weights come from the
    _ENSEMBLE_WEIGHTS table (see _ensemble_weights), renormalised over whichever
    components actually contributed.
    """
    comps: dict[str, dict] = {"elo": elo}
    if xgb is not None:
        comps["xgb"] = xgb
    if player is not None:
        comps["player"] = player
    if transformer is not None:
        comps["transformer"] = transformer

    weights = _ensemble_weights(list(comps.keys()))

    out = {"p_home": 0.0, "p_draw": 0.0, "p_away": 0.0}
    for name, pred in comps.items():
        w = weights.get(name, 0.0)
        for k in out:
            out[k] += w * pred[k]

    total = out["p_home"] + out["p_draw"] + out["p_away"]
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def _elo_probs_from_ratings(home_elo: float, away_elo: float,
                            neutral: bool = True) -> dict:
    """Recompute 1X2 probabilities from (possibly adjusted) raw Elo ratings.

    Mirrors ``EloModel.predict`` exactly — same draw-rate bucket table and the
    same proportional split — but takes effective Elos directly so that referee
    and fatigue Elo deltas can be folded in before the probability calculation.
    """
    home_adv = 0.0 if neutral else 75.0  # HOME_ADVANTAGE in core.elo_model
    exp_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo - home_adv) / 400))

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

    p_home = exp_home * (1.0 - draw_prob)
    p_away = (1.0 - exp_home) * (1.0 - draw_prob)
    total = p_home + draw_prob + p_away
    return {
        "p_home": p_home / total,
        "p_draw": draw_prob / total,
        "p_away": p_away / total,
        "home_elo": home_elo,
        "away_elo": away_elo,
    }


def _load_seq_index(parquet_path: Path) -> dict | None:
    """Build a per-team chronological list of encoded match rows for the
    Transformer sequence model.

    Returns a dict mapping team -> list of (date, encoded_10dim_vector) tuples,
    built with a running internal club-Elo (mirrors SequenceDataset's no-leakage
    pass). Returns None if the parquet is missing/unreadable or torch/the encoder
    are unavailable. International (WC2026) teams will be absent → transformer is
    skipped for those fixtures and the ensemble renormalises without it.
    """
    if not parquet_path.exists():
        return None
    try:
        from core.sequence_model import (
            encode_match_row, _days_between, _elo_update, DEFAULT_ELO,
        )
    except Exception as exc:
        logger.warning("Cannot import sequence encoder: %s", exc)
        return None
    try:
        cols = ["date", "home_team", "away_team", "home_goals", "away_goals"]
        df = pd.read_parquet(parquet_path, columns=cols)
        has_xg = False
        try:
            df_xg = pd.read_parquet(parquet_path, columns=["home_xg", "away_xg"])
            df = pd.concat([df, df_xg], axis=1)
            has_xg = True
        except Exception:
            has_xg = False

        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["home_goals", "away_goals"]).sort_values("date")

        index: dict[str, list] = {}
        last_date: dict[str, object] = {}
        elo: dict[str, float] = {}

        for row in df.itertuples(index=False):
            home, away = str(row.home_team), str(row.away_team)
            hg, ag = float(row.home_goals), float(row.away_goals)
            hxg = float(row.home_xg) if has_xg and pd.notna(getattr(row, "home_xg", None)) else None
            axg = float(row.away_xg) if has_xg and pd.notna(getattr(row, "away_xg", None)) else None

            home_elo = elo.get(home, DEFAULT_ELO)
            away_elo = elo.get(away, DEFAULT_ELO)

            h_days = _days_between(last_date.get(home), row.date)
            a_days = _days_between(last_date.get(away), row.date)

            index.setdefault(home, []).append(
                (row.date, encode_match_row(True, hg, ag, hxg, axg, h_days, home_elo, away_elo))
            )
            index.setdefault(away, []).append(
                (row.date, encode_match_row(False, ag, hg, axg, hxg, a_days, away_elo, home_elo))
            )
            last_date[home] = row.date
            last_date[away] = row.date
            elo[home], elo[away] = _elo_update(home_elo, away_elo, hg, ag, neutral=False)

        logger.debug("Sequence index built for %d teams.", len(index))
        return index
    except Exception as exc:
        logger.warning("Could not build sequence index from %s: %s", parquet_path, exc)
        return None


def _seq_predict(sequence_model, seq_index: dict | None,
                 home: str, away: str, as_of: str,
                 home_elo: float, away_elo: float,
                 neutral: bool = True, seq_len: int = 30) -> dict | None:
    """Run the Transformer on one fixture, or return None if it can't be built.

    Requires both teams to have at least ``seq_len // 6`` prior encoded matches
    in ``seq_index`` (a soft minimum — the model pads shorter sequences). Returns
    None for teams absent from the index (e.g. international sides), which causes
    the ensemble to drop the transformer component for that match.
    """
    if sequence_model is None or not seq_index:
        return None
    try:
        import numpy as _np
        from core.sequence_model import INPUT_DIM, ELO_NORM, ELO_HOME_ADV

        cutoff = pd.Timestamp(as_of)
        h_hist = [v for (d, v) in seq_index.get(home, []) if d < cutoff]
        a_hist = [v for (d, v) in seq_index.get(away, []) if d < cutoff]
        if not h_hist or not a_hist:
            return None

        def _pad(hist: list) -> "_np.ndarray":
            arr = _np.zeros((seq_len, INPUT_DIM), dtype="float32")
            tail = hist[-seq_len:]
            if tail:
                k = len(tail)
                arr[seq_len - k:] = _np.asarray(tail, dtype="float32")
            return arr

        home_seq = _pad(h_hist)
        away_seq = _pad(a_hist)
        elo_delta_norm = (home_elo + (0.0 if neutral else ELO_HOME_ADV) - away_elo) / ELO_NORM
        context = _np.asarray([elo_delta_norm, 1.0 if neutral else 0.0], dtype="float32")
        return sequence_model.predict(home_seq, away_seq, context, comp_id=0)
    except Exception as exc:
        logger.debug("Transformer predict failed for %s vs %s: %s", home, away, exc)
        return None


def _apply_lineup(probs: dict, adj: dict) -> dict:
    h = adj.get("home_adj", 0.0)
    a = adj.get("away_adj", 0.0)
    ph = max(0.01, probs["p_home"] + h - a)
    pd_ = max(0.01, probs["p_draw"])
    pa = max(0.01, probs["p_away"] + a - h)
    t = ph + pd_ + pa
    return {"p_home": ph / t, "p_draw": pd_ / t, "p_away": pa / t}


def main():
    parser = argparse.ArgumentParser(description="Apollo: match predictions + value bets")
    parser.add_argument("--date",      default=date.today().isoformat())
    parser.add_argument("--odds",      action="store_true", help="Fetch live odds, show value bets")
    parser.add_argument("--lineup",    action="store_true", help="Fetch injury/lineup news")
    parser.add_argument("--referee",   action="store_true",
                        help="Apply referee home-bias Elo adjustment per match")
    parser.add_argument("--fatigue",   action="store_true",
                        help="Apply rest/travel fatigue Elo adjustments")
    parser.add_argument("--player-model", action="store_true",
                        help="Blend the player-parametric Dixon-Coles model into the ensemble")
    parser.add_argument("--transformer",  action="store_true",
                        help="Blend the Transformer sequence model into the ensemble (if available)")
    parser.add_argument("--model",     default="data/models/elo_national.json")
    parser.add_argument("--min-edge",  type=float, default=0.03)
    parser.add_argument("--competition", default="wc2026")
    args = parser.parse_args()

    # ── Load models ──────────────────────────────────────────────────────────
    if not Path(args.model).exists():
        print(f"Elo model not found. Run: python scripts/build_elo.py")
        sys.exit(1)
    elo_model = EloModel.load(args.model)

    xgb_model = None
    xgb_path = Path("data/models/xgb_predictor.pkl")
    if xgb_path.exists():
        try:
            from core.xgb_predictor import XGBPredictor
            xgb_model = XGBPredictor.load(str(xgb_path))
        except Exception as e:
            logger.warning("XGBoost load failed: %s", e)

    # ── Optional new model components (all silent-fail) ──────────────────────
    referee_model = None
    if args.referee:
        ref_path = Path("data/models/referee_model.json")
        if ref_path.exists():
            try:
                from core.referee_model import RefereeModel
                referee_model = RefereeModel.load(str(ref_path))
            except Exception as e:
                logger.warning("Referee model load failed: %s", e)
        else:
            logger.warning("--referee set but %s not found; skipping.", ref_path)

    fatigue_model = None
    if args.fatigue:
        sched_path = Path("data/processed/matches.parquet")
        if sched_path.exists():
            try:
                from core.fatigue_model import FatigueModel
                fatigue_model = FatigueModel()
                fatigue_model.build_schedule_index(pd.read_parquet(sched_path))
            except Exception as e:
                logger.warning("Fatigue model init failed: %s", e)
                fatigue_model = None
        else:
            logger.warning("--fatigue set but %s not found; skipping.", sched_path)

    player_model = None
    if args.player_model:
        pm_path = Path("data/models/player_model.json")
        if pm_path.exists():
            try:
                from core.player_model import PlayerModel
                player_model = PlayerModel.load(str(pm_path))
            except Exception as e:
                logger.warning("Player model load failed: %s", e)
        else:
            logger.warning("--player-model set but %s not found; skipping.", pm_path)

    sequence_model = None
    seq_form_index = None
    if args.transformer:
        seq_path = Path("data/models/sequence_model.pt")
        if seq_path.exists():
            try:
                from core.sequence_model import SequenceModel
                sequence_model = SequenceModel.load(str(seq_path))
                # The transformer needs per-team match sequences at inference.
                # Build a sequence index from the club-match parquet (best-effort).
                seq_form_index = _load_seq_index(_HIST_PARQUET)
            except Exception as e:
                logger.warning("Sequence model load failed: %s", e)
                sequence_model = None
        else:
            logger.warning("--transformer set but %s not found; skipping.", seq_path)

    # ── Load historical form index (for rolling form features) ───────────────
    # Populated from club-league data; international teams return NaN form.
    form_index = _load_form_index(_HIST_PARQUET) if xgb_model else None

    # ── Fetch fixtures ───────────────────────────────────────────────────────
    fixtures = get_today_fixtures(date=args.date)
    if not fixtures:
        print(f"No fixtures found for {args.date}")
        sys.exit(0)

    # ── Optionally fetch odds early (needed for XGB market features) ─────────
    bookmaker_odds = []
    odds_source = None
    if args.odds:
        print("Fetching odds...")
        try:
            from core.odds_fetcher import OddsFetcher
            fetcher = OddsFetcher()
            bookmaker_odds = fetcher.get_all_today(args.date, competition=args.competition)
            if bookmaker_odds:
                odds_source = bookmaker_odds[0].get("source", "?")
        except Exception as e:
            logger.warning("Odds fetch failed: %s", e)

    odds_by_match = {(o["home"], o["away"]): o for o in bookmaker_odds}

    # ── Build predictions ────────────────────────────────────────────────────
    predictions = []
    lineup_adjustments = {}

    referee_adjustments = {}
    fatigue_adjustments = {}

    for f in fixtures:
        home, away = f["home"], f["away"]

        elo_pred = elo_model.predict(home, away, neutral=True)

        # ── Referee + fatigue Elo deltas (applied BEFORE probability calc) ───
        # home_elo_effective = home_elo + home_fatigue_adj + referee_elo_adj/2
        # away_elo_effective = away_elo + away_fatigue_adj - referee_elo_adj/2
        referee_elo_adj = 0.0
        home_fatigue_adj = 0.0
        away_fatigue_adj = 0.0

        if referee_model is not None:
            referee_name = f.get("referee") or f.get("Referee") or ""
            if referee_name:
                try:
                    referee_elo_adj = referee_model.get_elo_adjustment(referee_name)
                except Exception:
                    referee_elo_adj = 0.0
                if referee_elo_adj:
                    referee_adjustments[f"{home} vs {away}"] = {
                        "referee": referee_name, "elo_adj": referee_elo_adj,
                    }

        if fatigue_model is not None:
            try:
                fa = fatigue_model.get_match_adjustments(home, away, args.date)
                home_fatigue_adj = fa.get("home_elo_adj", 0.0)
                away_fatigue_adj = fa.get("away_elo_adj", 0.0)
                if home_fatigue_adj or away_fatigue_adj:
                    fatigue_adjustments[f"{home} vs {away}"] = fa
            except Exception:
                pass

        home_elo_eff = elo_pred["home_elo"] + home_fatigue_adj + (referee_elo_adj / 2.0)
        away_elo_eff = elo_pred["away_elo"] + away_fatigue_adj - (referee_elo_adj / 2.0)

        # Recompute Elo probabilities with the adjusted effective Elos when any
        # adjustment was applied; otherwise keep the original elo_pred.
        if (referee_elo_adj or home_fatigue_adj or away_fatigue_adj):
            elo_eff = _elo_probs_from_ratings(home_elo_eff, away_elo_eff, neutral=True)
        else:
            elo_eff = elo_pred

        # XGB: populate as many of the 17 trained features as possible.
        # Features the model was trained on (from core/xgb_predictor.py FEATURE_COLS):
        #   Elo (4):     home_elo_k32, away_elo_k32, elo_delta_k32, elo_expected_home_k32
        #   Form (8):    home/away_form_winrate_5, ppg_5, gf_mean_5, ga_mean_5
        #   Poisson (2): poisson_home_xg, poisson_away_xg
        #   Market (3):  mkt_home_implied, mkt_draw_implied, mkt_away_implied
        #
        # Train/test mismatch note: the model was trained on CLOSING odds from
        # the historical parquet (home_implied/draw_implied/away_implied columns
        # from bet365/Pinnacle). At inference we supply OPENING odds from the
        # live odds fetcher. This mismatch will compress the market edge estimate;
        # closing odds are typically sharper than opening odds.
        #
        # home_xg / away_xg (Understat real xG) were excluded from training because
        # they had >30% null rate. They do not appear in the stored model feature_cols.
        xgb_pred = None
        if xgb_model:
            odds_entry = odds_by_match.get((home, away), {})

            # ── Elo features (all computable) ────────────────────────────
            elo_expected_home_k32 = 1.0 / (
                1.0 + 10 ** ((elo_pred["away_elo"] - elo_pred["home_elo"]) / 400)
            )

            # ── Poisson xG — derived from Elo via Dixon-Coles approximation ──
            # When historical goals data for these specific teams are unavailable
            # (e.g. WC2026 international fixtures not in club-league parquet),
            # this gives a better signal than training-set medians because it
            # encodes the Elo strength differential.
            poisson_home_xg, poisson_away_xg = _poisson_xg_from_elo(
                elo_expected_home_k32, league_avg=_LEAGUE_AVG_GOALS
            )

            # ── Rolling form features — requires historical match data ────
            # For club matches: computed from data/processed/matches_xg.parquet
            # For international fixtures (WC2026): teams not in parquet → NaN
            #   → model falls back to training medians (winrate=0.40, ppg=1.40,
            #     gf_mean=1.20, ga_mean=1.20/1.40).
            # To fix this gap: load a separate international results CSV/parquet
            # (e.g. from martj42/international_results) and build a parallel
            # form index for national teams.
            home_form = _compute_form(form_index, home, args.date)
            away_form = _compute_form(form_index, away, args.date)

            xgb_feats = {
                # Elo
                "home_elo_k32":          elo_pred["home_elo"],
                "away_elo_k32":          elo_pred["away_elo"],
                "elo_delta_k32":         elo_pred["home_elo"] - elo_pred["away_elo"],
                "elo_expected_home_k32": elo_expected_home_k32,
                # Form — NaN for international teams (see comment above)
                "home_form_winrate_5":   home_form["winrate_5"],
                "away_form_winrate_5":   away_form["winrate_5"],
                "home_form_ppg_5":       home_form["ppg_5"],
                "away_form_ppg_5":       away_form["ppg_5"],
                "home_form_gf_mean_5":   home_form["gf_mean_5"],
                "away_form_gf_mean_5":   away_form["gf_mean_5"],
                "home_form_ga_mean_5":   home_form["ga_mean_5"],
                "away_form_ga_mean_5":   away_form["ga_mean_5"],
                # Poisson xG — derived from Elo (always computable)
                "poisson_home_xg":       poisson_home_xg,
                "poisson_away_xg":       poisson_away_xg,
                # Market — from live odds fetcher (opening odds; see train/test note above)
                # NaN when --odds not passed or odds fetch failed
                "mkt_home_implied":      odds_entry.get("fair_home"),
                "mkt_draw_implied":      odds_entry.get("fair_draw"),
                "mkt_away_implied":      odds_entry.get("fair_away"),
            }
            try:
                xgb_pred = xgb_model.predict(xgb_feats)
            except Exception:
                pass

        # ── Player model (Dixon-Coles goal-based 1X2) ───────────────────────
        player_pred = None
        if player_model is not None:
            try:
                player_pred = player_model.predict_outcome(
                    home, away, home_absent=None, away_absent=None, neutral=True
                )
            except Exception:
                player_pred = None

        # ── Transformer sequence model (3rd ensemble component if buildable) ─
        transformer_pred = None
        if sequence_model is not None:
            transformer_pred = _seq_predict(
                sequence_model, seq_form_index, home, away, args.date,
                elo_pred["home_elo"], elo_pred["away_elo"], neutral=True,
            )

        # Ensemble uses the referee/fatigue-adjusted Elo probabilities.
        probs = _ensemble(elo_eff, xgb_pred, player_pred, transformer_pred)

        # Lineup adjustments
        adj = {"home_adj": 0.0, "away_adj": 0.0}
        if args.lineup:
            try:
                from core.news_parser import parse_lineup_adjustments
                adj = parse_lineup_adjustments(home, away, args.date)
                if adj["home_adj"] != 0.0 or adj["away_adj"] != 0.0:
                    lineup_adjustments[f"{home} vs {away}"] = adj
            except Exception:
                pass

        probs = _apply_lineup(probs, adj)

        predictions.append({
            **probs,
            "home":       home,
            "away":       away,
            "time_utc":   f.get("time_utc", "?"),
            "group":      f.get("group", ""),
            "home_elo":   elo_pred["home_elo"],
            "away_elo":   elo_pred["away_elo"],
            "home_elo_eff": round(home_elo_eff, 2),
            "away_elo_eff": round(away_elo_eff, 2),
            "xgb_used":   xgb_pred is not None,
            "player_used": player_pred is not None,
            "transformer_used": transformer_pred is not None,
        })

    # ── Print predictions table ───────────────────────────────────────────────
    # Report the ensemble weights that actually applied to the slate (use the
    # richest combination that appeared on any fixture).
    active = ["elo"]
    if xgb_model and any(p["xgb_used"] for p in predictions):
        active.append("xgb")
    if player_model and any(p["player_used"] for p in predictions):
        active.append("player")
    if sequence_model and any(p["transformer_used"] for p in predictions):
        active.append("transformer")
    applied_weights = _ensemble_weights(active)

    mode_parts = ["Elo"]
    if "xgb" in active:
        mode_parts.append("XGB")
    if "player" in active:
        mode_parts.append("Player")
    if "transformer" in active:
        mode_parts.append("Transformer")
    blend_str = "/".join(f"{applied_weights[c]:.0%}" for c in active)
    mode = " + ".join(mode_parts) + f" ({blend_str} blend)"
    if referee_model is not None:
        mode += " +referee"
    if fatigue_model is not None:
        mode += " +fatigue"
    if args.lineup:
        mode += " +lineup"

    print(f"\nApollo — {args.date}  [{mode}]")
    print("=" * 82)
    print(f"  {'Time':>5}  {'Match':<34} {'Elo-H':>5} {'Elo-A':>5}  {'P(H)':>6} {'P(D)':>6} {'P(A)':>6}  {'Pick'}")
    print("-" * 82)
    for p in predictions:
        match = f"{p['home']} vs {p['away']}"[:33]
        pick_idx = [p["p_home"], p["p_draw"], p["p_away"]].index(
            max(p["p_home"], p["p_draw"], p["p_away"])
        )
        pick = [p["home"], "Draw", p["away"]][pick_idx]
        marks = ""
        marks += "*" if p["xgb_used"] else ""
        marks += "P" if p.get("player_used") else ""
        marks += "T" if p.get("transformer_used") else ""
        mark = (marks or " ")[:1]
        print(f"{mark} {p['time_utc']:>5}  {match:<34} {p['home_elo']:>5.0f} {p['away_elo']:>5.0f}"
              f"  {p['p_home']:>6.1%} {p['p_draw']:>6.1%} {p['p_away']:>6.1%}  {pick}")
    legend = []
    if "xgb" in active:
        legend.append("* = XGB")
    if "player" in active:
        legend.append("P = PlayerModel")
    if "transformer" in active:
        legend.append("T = Transformer")
    if legend:
        print("  (" + " · ".join(legend) + " ensemble active)")

    # ── Referee adjustments ───────────────────────────────────────────────────
    if referee_adjustments:
        print("\nReferee Elo adjustments applied:")
        for match, ra in referee_adjustments.items():
            print(f"  {match}: {ra['referee']}  Elo {ra['elo_adj']:+.1f} (home favour)")

    # ── Fatigue adjustments ───────────────────────────────────────────────────
    if fatigue_adjustments:
        print("\nFatigue/travel Elo adjustments applied:")
        for match, fa in fatigue_adjustments.items():
            print(f"  {match}: home {fa['home_elo_adj']:+.1f}  away {fa['away_elo_adj']:+.1f}"
                  f"  (rest H{fa['home_rest_days']}/A{fa['away_rest_days']}d, "
                  f"travel {fa['travel_km']:.0f}km)")

    # ── Lineup adjustments ────────────────────────────────────────────────────
    if lineup_adjustments:
        print("\nLineup adjustments applied:")
        for match, adj in lineup_adjustments.items():
            print(f"  {match}: home {adj['home_adj']:+.3f}  away {adj['away_adj']:+.3f}")

    # ── Value bets ────────────────────────────────────────────────────────────
    value_bets = []
    if args.odds:
        if not bookmaker_odds:
            print("\nNo odds retrieved — add ODDS_API_KEY to .env or configure Betfair credentials.")
        else:
            print(f"\nOdds source: {odds_source}")
            value_bets = find_value_bets(predictions, bookmaker_odds, min_edge=args.min_edge)
            if value_bets:
                print(f"\nValue Bets  (edge > {args.min_edge:.0%}, quarter-Kelly sizing):")
                print(format_report(value_bets))
                total_k = sum(vb.kelly for vb in value_bets)
                print(f"\n  Total bankroll exposure: {total_k:.2%}")
            else:
                print(f"\nNo value bets above {args.min_edge:.0%} edge threshold.")

            # Auto-log every run with odds for CLV tracking
            try:
                from core.prediction_logger import log_predictions
                n = log_predictions(predictions, bookmaker_odds, value_bets,
                                    competition=args.competition,
                                    match_date=args.date)
                if n:
                    print(f"\n  [CLV] Logged {n} predictions → data/predictions/log.parquet")
            except Exception as e:
                logger.debug("Prediction logging failed: %s", e)

    # ── Telegram summary ──────────────────────────────────────────────────────
    if value_bets:
        try:
            from core.notifier import TelegramNotifier
            _notifier = TelegramNotifier()
            if _notifier.available():
                _lines = [f"• {vb.match}  {vb.outcome.upper()}  edge={vb.edge:+.1%}  odds={vb.odds:.2f}"
                          for vb in value_bets]
                _total_k = sum(vb.kelly for vb in value_bets)
                _details = (
                    f"Date: {args.date}  |  {len(value_bets)} bet(s) found\n"
                    + "\n".join(_lines)
                    + f"\n\nTotal Kelly exposure: {_total_k:.1%}"
                )
                _notifier.alert_pipeline_done(
                    event=f"predict\\_today — {len(value_bets)} value bet(s)",
                    details=_details,
                )
                print("\n  [Telegram] Summary sent.")
        except Exception as _e:
            logger.debug("Telegram notify failed: %s", _e)

    print()


if __name__ == "__main__":
    main()
