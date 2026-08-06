"""
Validation Cascade for Apollo Discovery Engine.

A hypothesis that passes Gate 1 (discovery + multiple testing correction)
is still more likely than not to be a false positive. The cascade applies
three additional filters, each independent:

Gate 2: Walk-forward out-of-sample backtest (same league)
Gate 3: Cross-domain replication (different leagues)
Gate 4: Holdout confirmation (reserved seasons, never previously seen)

Only signals that survive all four gates enter the signal database.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Callable
from scipy import stats

from core.metrics import brier_score, brier_skill_score, roi_simulation
from core.stats import paired_brier_test


class ValidationCascade:
    """
    Multi-gate validation for candidate signals.
    """
    
    def __init__(self,
                 holdout_seasons: List[str] = None,
                 min_oos_observations: int = 200,
                 min_leagues_confirmed: int = 2,
                 cross_league_alpha: float = 0.10,
                 holdout_alpha: float = 0.05):
        self.holdout_seasons = holdout_seasons or ['2023-24', '2024-25']
        self.min_oos = min_oos_observations
        self.min_leagues = min_leagues_confirmed
        self.cross_alpha = cross_league_alpha
        self.holdout_alpha = holdout_alpha
    
    def gate2_walkforward(self, 
                          df: pd.DataFrame,
                          signal_func: Callable,
                          hypothesis_id: str,
                          target: str = 'home_win',
                          baseline_col: str = 'home_implied',
                          odds_col: str = 'home_odds',
                          min_train: int = 300) -> dict:
        """
        Gate 2: Walk-forward out-of-sample test.
        
        The signal function is re-estimated at each step using only
        past data. Predictions are collected out-of-sample and compared
        against the bookmaker baseline.
        
        Parameters
        ----------
        df : DataFrame sorted by date (EXCLUDING holdout seasons)
        signal_func : function(train_df, test_row) -> adjusted_probability
            Takes historical data and a single test row, returns
            a probability for the target outcome.
        hypothesis_id : for tracking
        target : binary outcome column
        baseline_col : market implied probability column
        odds_col : decimal odds column
        min_train : minimum matches before first prediction
        """
        df = df.sort_values('date').reset_index(drop=True)
        df = df[~df['season'].isin(self.holdout_seasons)]
        
        # Drop rows missing essential columns
        df = df.dropna(subset=[target, baseline_col, odds_col])
        
        predictions = []
        baselines = []
        outcomes = []
        odds = []
        
        for i in range(min_train, len(df)):
            train = df.iloc[:i]
            test_row = df.iloc[i]
            
            try:
                pred = signal_func(train, test_row)
                if pred is not None and 0 < pred < 1:
                    predictions.append(pred)
                    baselines.append(test_row[baseline_col])
                    outcomes.append(test_row[target])
                    odds.append(test_row[odds_col])
            except Exception:
                continue
        
        predictions = np.array(predictions)
        baselines = np.array(baselines)
        outcomes = np.array(outcomes)
        odds_arr = np.array(odds)
        
        n = len(predictions)
        
        if n < self.min_oos:
            return {
                'hypothesis_id': hypothesis_id,
                'gate': 2,
                'passed': False,
                'reason': f'insufficient OOS observations ({n} < {self.min_oos})',
                'n_observations': n,
            }
        
        # Evaluate
        model_brier = brier_score(predictions, outcomes)
        base_brier = brier_score(baselines, outcomes)
        bss = brier_skill_score(model_brier, base_brier)
        
        sig = paired_brier_test(predictions, baselines, outcomes)
        
        roi = roi_simulation(predictions, odds_arr, outcomes, edge_threshold=0.03)
        
        passed = sig['p_value'] < 0.05 and sig['model_better']
        
        return {
            'hypothesis_id': hypothesis_id,
            'gate': 2,
            'passed': passed,
            'n_observations': n,
            'model_brier': float(model_brier),
            'baseline_brier': float(base_brier),
            'brier_skill_score': float(bss),
            'p_value': sig['p_value'],
            't_statistic': sig['t_stat'],
            'roi_3pct': roi['roi'],
            'n_bets_3pct': roi['n_bets'],
            'per_bet_returns': (predictions * odds_arr - 1) if roi['n_bets'] > 0 else None,
        }
    
    def gate3_cross_league(self,
                           all_data: pd.DataFrame,
                           signal_func: Callable,
                           hypothesis_id: str,
                           primary_league: str,
                           target: str = 'home_win',
                           baseline_col: str = 'home_implied') -> dict:
        """
        Gate 3: Cross-domain replication.
        
        Tests the signal independently in each non-primary league.
        Requires consistent sign and p < 0.10 in at least
        min_leagues_confirmed other leagues.
        """
        other_leagues = [
            l for l in all_data['league'].unique() 
            if l != primary_league
        ]
        
        # Exclude holdout seasons
        all_data = all_data[~all_data['season'].isin(self.holdout_seasons)]
        
        league_results = {}
        leagues_confirmed = 0
        primary_sign = None
        
        for league in other_leagues:
            league_df = all_data[all_data['league'] == league].copy()
            league_df = league_df.dropna(subset=[target, baseline_col])
            
            if len(league_df) < 200:
                league_results[league] = {'skipped': True, 'reason': 'insufficient_data'}
                continue
            
            try:
                result = signal_func(league_df)
                
                if result is None:
                    league_results[league] = {'skipped': True, 'reason': 'signal_func_returned_none'}
                    continue
                
                # Check consistency
                if primary_sign is None:
                    primary_sign = np.sign(result.get('effect_direction', 0))
                
                league_sign = np.sign(result.get('effect_direction', 0))
                p = result.get('p_value', 1.0)
                
                consistent = (league_sign == primary_sign) and (p < self.cross_alpha)
                
                if consistent:
                    leagues_confirmed += 1
                
                league_results[league] = {
                    'p_value': p,
                    'effect_direction': result.get('effect_direction', 0),
                    'effect_size': result.get('effect_size', 0),
                    'n_observations': result.get('n_observations', 0),
                    'consistent': consistent,
                }
            except Exception as e:
                league_results[league] = {'skipped': True, 'reason': str(e)}
        
        passed = leagues_confirmed >= self.min_leagues
        
        return {
            'hypothesis_id': hypothesis_id,
            'gate': 3,
            'passed': passed,
            'leagues_confirmed': leagues_confirmed,
            'leagues_required': self.min_leagues,
            'league_results': league_results,
        }
    
    def gate4_holdout(self,
                      holdout_data: pd.DataFrame,
                      signal_func: Callable,
                      hypothesis_id: str,
                      target: str = 'home_win',
                      baseline_col: str = 'home_implied',
                      odds_col: str = 'home_odds') -> dict:
        """
        Gate 4: Final holdout confirmation.
        
        This function loads ONLY holdout season data. It is the last
        check. No further analysis is permitted after failure here.
        
        The holdout data must never have been used in any prior stage.
        """
        holdout = holdout_data[
            holdout_data['season'].isin(self.holdout_seasons)
        ].copy()
        
        holdout = holdout.dropna(subset=[target, baseline_col, odds_col])
        
        if len(holdout) < 100:
            return {
                'hypothesis_id': hypothesis_id,
                'gate': 4,
                'passed': False,
                'reason': f'insufficient holdout data ({len(holdout)} matches)',
            }
        
        try:
            predictions = signal_func(holdout)
            
            if predictions is None or len(predictions) != len(holdout):
                return {
                    'hypothesis_id': hypothesis_id,
                    'gate': 4,
                    'passed': False,
                    'reason': 'signal_func returned invalid predictions',
                }
            
            baselines = holdout[baseline_col].values
            outcomes = holdout[target].values
            odds_arr = holdout[odds_col].values
            
            model_brier = brier_score(predictions, outcomes)
            base_brier = brier_score(baselines, outcomes)
            bss = brier_skill_score(model_brier, base_brier)
            
            sig = paired_brier_test(predictions, baselines, outcomes)
            roi = roi_simulation(predictions, odds_arr, outcomes, edge_threshold=0.03)
            
            passed = sig['p_value'] < self.holdout_alpha and sig['model_better']
            
            return {
                'hypothesis_id': hypothesis_id,
                'gate': 4,
                'passed': passed,
                'n_observations': len(holdout),
                'model_brier': float(model_brier),
                'baseline_brier': float(base_brier),
                'brier_skill_score': float(bss),
                'p_value': sig['p_value'],
                'roi_3pct': roi['roi'],
                'n_bets_3pct': roi['n_bets'],
                'seasons': self.holdout_seasons,
            }
        
        except Exception as e:
            return {
                'hypothesis_id': hypothesis_id,
                'gate': 4,
                'passed': False,
                'reason': str(e),
            }
