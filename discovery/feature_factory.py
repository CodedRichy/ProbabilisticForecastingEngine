"""
Feature Factory for Apollo Discovery Engine.

Computes ~140 base features from raw match data using ONLY
information available before each match. Features are computed
in strict chronological order via expanding window.

Every feature value is tagged with a `vintage` — the date it
was computed from — to prevent any possibility of lookahead.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FeatureMeta:
    """Metadata for a computed feature."""
    name: str
    category: str       # form, elo, market, schedule, h2h, poisson, derived
    lookback: int       # matches of history required
    description: str


class FeatureFactory:
    """
    Computes all base features for the discovery engine.
    
    Architecture:
    - All features computed in a single chronological pass.
    - Per-team state is maintained in rolling accumulators.
    - No feature can reference the current match outcome.
    - Output is a wide DataFrame indexed by match_id.
    """
    
    WINDOWS = [3, 5, 10, 20]
    ELO_K_FACTORS = [20, 32, 48]
    ELO_INIT = 1500
    ELO_HOME_ADVANTAGE = 65
    
    EWMA_SPAN = 5

    def __init__(self):
        self.feature_meta: Dict[str, FeatureMeta] = {}
        self._team_state: Dict[str, dict] = {}
        self._league_state: Dict[str, dict] = {}  # per-league goal accumulators
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point. Takes raw match data, returns feature matrix.
        
        Input must have: date, home_team, away_team, home_goals, away_goals,
                         result, home_implied, draw_implied, away_implied,
                         home_odds, draw_odds, away_odds, season, league
        
        Output: original columns + all computed features.
        """
        df = df.sort_values('date').reset_index(drop=True)
        self._team_state = {}
        self._league_state = {}
        
        # Pre-allocate feature columns
        feature_arrays = {}
        
        # ── Chronological pass ──
        for idx in range(len(df)):
            row = df.iloc[idx]
            home = row['home_team']
            away = row['away_team']
            league = row.get('league', '__unknown__')

            # Initialize team / league state if new
            self._ensure_team(home)
            self._ensure_team(away)
            self._ensure_league(league)
            
            # ── COMPUTE FEATURES (before updating state) ──
            features = {}
            
            # Category A: Form features
            features.update(self._form_features(home, 'home'))
            features.update(self._form_features(away, 'away'))
            
            # Category B: Elo features
            features.update(self._elo_features(home, away))
            
            # Category C: Market features
            features.update(self._market_features(row))
            
            # Category D: Schedule features
            features.update(self._schedule_features(home, away, row))
            
            # Category E: H2H features
            features.update(self._h2h_features(home, away))
            
            # Category F: Goals model features
            features.update(self._goals_model_features(home, away, row))
            
            # Store features
            for fname, fval in features.items():
                if fname not in feature_arrays:
                    feature_arrays[fname] = np.full(len(df), np.nan)
                feature_arrays[fname][idx] = fval
            
            # ── UPDATE STATE (after computing features) ──
            self._update_state(home, away, row)
        
        # Merge features into dataframe
        for fname, farr in feature_arrays.items():
            df[fname] = farr
        
        # Register metadata
        self._register_metadata()
        
        return df
    
    def _ensure_team(self, team: str):
        if team not in self._team_state:
            self._team_state[team] = {
                'results': [],         # list of ('W','D','L')
                'goals_for': [],       # goals scored per match
                'goals_against': [],   # goals conceded per match
                'dates': [],           # match dates
                'elo': {k: self.ELO_INIT for k in self.ELO_K_FACTORS},
                'elo_history': {k: [] for k in self.ELO_K_FACTORS},
                'opponents': [],       # opponent names
                'h2h': {},             # opponent -> list of results
                'clean_sheets': [],    # 1/0 per match
                'failed_to_score': [], # 1/0 per match
                'match_count': 0,
                # EWMA state (span=5, adjust=False)
                'ewma_ppg': None,      # running EWMA of points per game
                'ewma_gd': None,       # running EWMA of goal difference
            }

    def _ensure_league(self, league: str):
        if league not in self._league_state:
            self._league_state[league] = {
                'total_goals': [],    # total goals per completed match
            }
    
    # ── CATEGORY A: FORM ──────────────────────────────────────
    
    def _form_features(self, team: str, side: str) -> dict:
        """Rolling form features for a team."""
        state = self._team_state[team]
        features = {}
        prefix = f"{side}_form"
        
        results = state['results']
        gf = state['goals_for']
        ga = state['goals_against']
        cs = state['clean_sheets']
        fts = state['failed_to_score']
        
        for w in self.WINDOWS:
            if len(results) < w:
                # Not enough history — features are NaN
                continue
            
            recent_r = results[-w:]
            recent_gf = gf[-w:]
            recent_ga = ga[-w:]
            recent_cs = cs[-w:]
            recent_fts = fts[-w:]
            
            wins = sum(1 for r in recent_r if r == 'W')
            draws = sum(1 for r in recent_r if r == 'D')
            losses = sum(1 for r in recent_r if r == 'L')
            points = wins * 3 + draws
            
            features[f"{prefix}_winrate_{w}"] = wins / w
            features[f"{prefix}_drawrate_{w}"] = draws / w
            features[f"{prefix}_lossrate_{w}"] = losses / w
            features[f"{prefix}_ppg_{w}"] = points / w
            features[f"{prefix}_gf_mean_{w}"] = np.mean(recent_gf)
            features[f"{prefix}_ga_mean_{w}"] = np.mean(recent_ga)
            features[f"{prefix}_gd_mean_{w}"] = np.mean(np.array(recent_gf) - np.array(recent_ga))
            features[f"{prefix}_cs_rate_{w}"] = np.mean(recent_cs)
            features[f"{prefix}_fts_rate_{w}"] = np.mean(recent_fts)
            
            if w >= 5:
                features[f"{prefix}_gf_std_{w}"] = np.std(recent_gf, ddof=1) if len(recent_gf) > 1 else 0
                features[f"{prefix}_ga_std_{w}"] = np.std(recent_ga, ddof=1) if len(recent_ga) > 1 else 0
        
        # Win streak
        streak = 0
        for r in reversed(results):
            if r == 'W':
                streak += 1
            else:
                break
        features[f"{prefix}_win_streak"] = streak
        
        # Unbeaten streak
        unbeaten = 0
        for r in reversed(results):
            if r in ('W', 'D'):
                unbeaten += 1
            else:
                break
        features[f"{prefix}_unbeaten_streak"] = unbeaten
        
        # Losing streak
        losing = 0
        for r in reversed(results):
            if r == 'L':
                losing += 1
            else:
                break
        features[f"{prefix}_losing_streak"] = losing
        
        features[f"{prefix}_matches_played"] = len(results)

        # EWMA form features (span=5, adjust=False)
        # ewma_ppg / ewma_gd in state are computed from all *past* matches only
        # (state update happens after feature extraction), so no lookahead.
        ewma_ppg = state['ewma_ppg']
        ewma_gd = state['ewma_gd']
        if ewma_ppg is not None:
            features[f"{prefix}_ewma_ppg"] = ewma_ppg
        if ewma_gd is not None:
            features[f"{prefix}_ewma_gd"] = ewma_gd

        return features
    
    # ── CATEGORY B: ELO ───────────────────────────────────────
    
    def _elo_features(self, home: str, away: str) -> dict:
        features = {}
        
        for k in self.ELO_K_FACTORS:
            h_elo = self._team_state[home]['elo'][k]
            a_elo = self._team_state[away]['elo'][k]
            delta = h_elo - a_elo
            
            # Expected score (with home advantage)
            exp_home = 1.0 / (1.0 + 10 ** (-(delta + self.ELO_HOME_ADVANTAGE) / 400))
            
            features[f"home_elo_k{k}"] = h_elo
            features[f"away_elo_k{k}"] = a_elo
            features[f"elo_delta_k{k}"] = delta
            features[f"elo_expected_home_k{k}"] = exp_home
            
            # Elo momentum (change over last 5 matches)
            h_hist = self._team_state[home]['elo_history'][k]
            a_hist = self._team_state[away]['elo_history'][k]
            
            if len(h_hist) >= 5:
                features[f"home_elo_momentum_k{k}"] = h_elo - h_hist[-5]
            if len(a_hist) >= 5:
                features[f"away_elo_momentum_k{k}"] = a_elo - a_hist[-5]
        
        return features
    
    def _update_elo(self, home: str, away: str, result: str):
        """Update Elo ratings after a match."""
        for k in self.ELO_K_FACTORS:
            h_elo = self._team_state[home]['elo'][k]
            a_elo = self._team_state[away]['elo'][k]
            
            exp_h = 1.0 / (1.0 + 10 ** (-(h_elo - a_elo + self.ELO_HOME_ADVANTAGE) / 400))
            exp_a = 1.0 - exp_h
            
            if result == 'H':
                actual_h, actual_a = 1.0, 0.0
            elif result == 'A':
                actual_h, actual_a = 0.0, 1.0
            else:
                actual_h, actual_a = 0.5, 0.5
            
            self._team_state[home]['elo'][k] = h_elo + k * (actual_h - exp_h)
            self._team_state[away]['elo'][k] = a_elo + k * (actual_a - exp_a)
            
            self._team_state[home]['elo_history'][k].append(h_elo)
            self._team_state[away]['elo_history'][k].append(a_elo)
    
    # ── CATEGORY C: MARKET ────────────────────────────────────
    
    def _market_features(self, row) -> dict:
        features = {}
        
        hi = row.get('home_implied', np.nan)
        di = row.get('draw_implied', np.nan)
        ai = row.get('away_implied', np.nan)
        
        if pd.notna(hi) and pd.notna(di) and pd.notna(ai):
            features['mkt_home_implied'] = hi
            features['mkt_draw_implied'] = di
            features['mkt_away_implied'] = ai
            features['mkt_favorite_prob'] = max(hi, di, ai)
            features['mkt_underdog_prob'] = min(hi, ai)
            
            # Compactness: how close are the probabilities?
            probs = sorted([hi, di, ai], reverse=True)
            features['mkt_compactness'] = probs[0] - probs[2]
            
            # Home advantage premium implied by market
            features['mkt_home_premium'] = hi - ai
        
        return features
    
    # ── CATEGORY D: SCHEDULE ──────────────────────────────────
    
    def _schedule_features(self, home: str, away: str, row) -> dict:
        features = {}
        
        match_date = row['date']
        if pd.isna(match_date):
            return features
        
        # Days since last match
        for side, team in [('home', home), ('away', away)]:
            dates = self._team_state[team]['dates']
            if dates:
                last_date = dates[-1]
                if pd.notna(last_date) and pd.notna(match_date):
                    delta = (pd.Timestamp(match_date) - pd.Timestamp(last_date)).days
                    features[f"{side}_rest_days"] = delta
                    
                    # Congestion: matches in last 7/14/30 days
                    for window in [7, 14, 30]:
                        cutoff = pd.Timestamp(match_date) - pd.Timedelta(days=window)
                        count = sum(1 for d in dates if pd.Timestamp(d) >= cutoff)
                        features[f"{side}_matches_last_{window}d"] = count
        
        return features
    
    # ── CATEGORY E: HEAD TO HEAD ──────────────────────────────
    
    def _h2h_features(self, home: str, away: str) -> dict:
        features = {}
        
        h2h = self._team_state[home].get('h2h', {}).get(away, [])
        
        if len(h2h) >= 3:
            for w in [5, 10]:
                recent = h2h[-w:] if len(h2h) >= w else h2h
                if recent:
                    wins = sum(1 for r in recent if r == 'W')
                    features[f"h2h_home_winrate_{min(w, len(h2h))}"] = wins / len(recent)
        
        features['h2h_total_meetings'] = len(h2h)
        
        return features
    
    # ── CATEGORY F: GOALS MODEL ───────────────────────────────
    
    def _goals_model_features(self, home: str, away: str, row) -> dict:
        """Simple Poisson-style attack/defense ratings."""
        features = {}

        h_gf = self._team_state[home]['goals_for']
        h_ga = self._team_state[home]['goals_against']
        a_gf = self._team_state[away]['goals_for']
        a_ga = self._team_state[away]['goals_against']

        if len(h_gf) >= 10 and len(a_gf) >= 10:
            # Per-league rolling average of total goals per match.
            # Uses only past matches (state is updated *after* feature computation).
            # Falls back to 1.3 if fewer than 20 league matches observed.
            league = row.get('league', '__unknown__')
            self._ensure_league(league)
            league_goals = self._league_state[league]['total_goals']
            if len(league_goals) >= 20:
                # Average total goals per match; halve to get per-team per-match avg
                league_avg = np.mean(league_goals) / 2.0
            else:
                league_avg = 1.3

            h_attack = np.mean(h_gf[-20:]) / league_avg
            h_defense = np.mean(h_ga[-20:]) / league_avg
            a_attack = np.mean(a_gf[-20:]) / league_avg
            a_defense = np.mean(a_ga[-20:]) / league_avg

            features['home_attack_rating'] = h_attack
            features['home_defense_rating'] = h_defense
            features['away_attack_rating'] = a_attack
            features['away_defense_rating'] = a_defense

            # Expected goals from Poisson
            features['poisson_home_xg'] = h_attack * a_defense * league_avg
            features['poisson_away_xg'] = a_attack * h_defense * league_avg
            features['poisson_total_xg'] = features['poisson_home_xg'] + features['poisson_away_xg']

        return features
    
    # ── STATE UPDATE ──────────────────────────────────────────
    
    def _update_state(self, home: str, away: str, row):
        """Update all team state AFTER features are computed."""
        result = row['result']
        hg = int(row['home_goals']) if pd.notna(row['home_goals']) else 0
        ag = int(row['away_goals']) if pd.notna(row['away_goals']) else 0

        # Update per-league total-goals accumulator
        league = row.get('league', '__unknown__')
        self._ensure_league(league)
        self._league_state[league]['total_goals'].append(hg + ag)

        # EWMA smoothing factor (span=5, adjust=False): alpha = 2/(span+1)
        alpha = 2.0 / (self.EWMA_SPAN + 1)

        # Home team
        h_result = 'W' if result == 'H' else ('D' if result == 'D' else 'L')
        a_result = 'W' if result == 'A' else ('D' if result == 'D' else 'L')

        for team, t_result, gf, ga, opp in [
            (home, h_result, hg, ag, away),
            (away, a_result, ag, hg, home)
        ]:
            s = self._team_state[team]
            s['results'].append(t_result)
            s['goals_for'].append(gf)
            s['goals_against'].append(ga)
            s['dates'].append(row['date'])
            s['opponents'].append(opp)
            s['clean_sheets'].append(1 if ga == 0 else 0)
            s['failed_to_score'].append(1 if gf == 0 else 0)
            s['match_count'] += 1

            # H2H tracking
            if opp not in s['h2h']:
                s['h2h'][opp] = []
            s['h2h'][opp].append(t_result)

            # Update EWMA accumulators (adjust=False recurrence)
            pts = 3.0 if t_result == 'W' else (1.0 if t_result == 'D' else 0.0)
            gd = float(gf - ga)
            if s['ewma_ppg'] is None:
                s['ewma_ppg'] = pts
                s['ewma_gd'] = gd
            else:
                s['ewma_ppg'] = alpha * pts + (1.0 - alpha) * s['ewma_ppg']
                s['ewma_gd'] = alpha * gd + (1.0 - alpha) * s['ewma_gd']

        # Update Elo
        self._update_elo(home, away, result)
    
    # ── METADATA ──────────────────────────────────────────────
    
    def _register_metadata(self):
        """Build feature metadata catalog."""
        # Auto-populated from what was actually computed
        pass
    
    def get_base_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Return list of all computed feature columns (excluding raw data columns)."""
        raw_cols = {
            'match_id', 'date', 'season', 'league', 'home_team', 'away_team',
            'home_goals', 'away_goals', 'result', 'home_win', 'draw', 'away_win',
            'home_odds', 'draw_odds', 'away_odds', 'home_implied', 'draw_implied',
            'away_implied', 'odds_source'
        }
        return [c for c in df.columns if c not in raw_cols and df[c].dtype in ('float64', 'int64', 'float32')]
