"""
Walk-forward backtesting engine for Apollo.

CRITICAL DESIGN RULE:
Never use future information to predict the past.

All backtests use expanding or rolling windows:
- Train on data up to time T
- Predict at time T+1
- Move forward one step
- Repeat

This prevents lookahead bias, the single most common
cause of false positives in forecasting research.
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional, List
from core.metrics import brier_score, log_loss, roi_simulation


def walk_forward_backtest(
    df: pd.DataFrame,
    signal_func: Callable,
    prediction_func: Callable,
    min_train_size: int = 200,
    step_size: int = 1,
    window: Optional[int] = None,
    date_col: str = 'date',
    outcome_col: str = 'home_win',
    baseline_col: str = 'home_implied',
    odds_col: str = 'home_odds',
) -> pd.DataFrame:
    """
    Walk-forward backtest with expanding or rolling window.
    
    Parameters
    ----------
    df : DataFrame sorted by date
    signal_func : function(train_df) -> model/state object
    prediction_func : function(model, test_row) -> probability
    min_train_size : minimum observations before first prediction
    step_size : how many matches to advance per step
    window : if set, use rolling window of this size; else expanding
    date_col : column with match dates
    outcome_col : binary outcome column
    baseline_col : bookmaker implied probability column
    odds_col : decimal odds column
    
    Returns
    -------
    DataFrame with columns: date, prediction, baseline, outcome, odds
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    
    results = []
    
    i = min_train_size
    while i < len(df):
        # Define training window
        if window is not None:
            train_start = max(0, i - window)
        else:
            train_start = 0
        
        train_df = df.iloc[train_start:i].copy()
        
        # Train/fit on historical data only
        model = signal_func(train_df)
        
        # Predict next batch
        end = min(i + step_size, len(df))
        for j in range(i, end):
            test_row = df.iloc[j]
            
            try:
                pred = prediction_func(model, test_row)
            except Exception:
                pred = np.nan
            
            results.append({
                'index': j,
                'date': test_row[date_col],
                'prediction': pred,
                'baseline': test_row[baseline_col],
                'outcome': test_row[outcome_col],
                'odds': test_row[odds_col],
            })
        
        i = end
    
    result_df = pd.DataFrame(results)
    result_df = result_df.dropna(subset=['prediction', 'baseline', 'outcome'])
    
    return result_df


def evaluate_backtest(result_df: pd.DataFrame) -> dict:
    """
    Evaluate walk-forward backtest results.
    """
    preds = result_df['prediction'].values
    baseline = result_df['baseline'].values
    outcomes = result_df['outcome'].values
    odds = result_df['odds'].values
    
    model_brier = brier_score(preds, outcomes)
    base_brier = brier_score(baseline, outcomes)
    
    return {
        'n_predictions': len(result_df),
        'model_brier': model_brier,
        'baseline_brier': base_brier,
        'brier_improvement': base_brier - model_brier,
        'model_logloss': log_loss(preds, outcomes),
        'baseline_logloss': log_loss(baseline, outcomes),
        'roi_0pct': roi_simulation(preds, odds, outcomes, edge_threshold=0.0),
        'roi_3pct': roi_simulation(preds, odds, outcomes, edge_threshold=0.03),
        'roi_5pct': roi_simulation(preds, odds, outcomes, edge_threshold=0.05),
    }


def cross_league_backtest(
    df: pd.DataFrame,
    signal_func: Callable,
    prediction_func: Callable,
    leagues: Optional[List[str]] = None,
    **kwargs,
) -> dict:
    """
    Run walk-forward backtest independently per league.
    
    A signal that works in only one league is suspicious.
    A signal that works across multiple leagues is more credible.
    """
    if leagues is None:
        leagues = df['league'].unique().tolist()
    
    league_results = {}
    
    for league in leagues:
        league_df = df[df['league'] == league].copy()
        
        if len(league_df) < kwargs.get('min_train_size', 200) * 2:
            print(f"  {league}: insufficient data ({len(league_df)} matches)")
            continue
        
        print(f"  {league}: {len(league_df)} matches")
        bt = walk_forward_backtest(league_df, signal_func, prediction_func, **kwargs)
        
        if len(bt) > 0:
            league_results[league] = evaluate_backtest(bt)
            league_results[league]['n_matches'] = len(league_df)
    
    return league_results


def cross_season_backtest(
    df: pd.DataFrame,
    signal_func: Callable,
    prediction_func: Callable,
    **kwargs,
) -> dict:
    """
    Run evaluation per season to test temporal stability.
    """
    season_results = {}
    
    for season in sorted(df['season'].unique()):
        season_df = df[df['season'] == season].copy()
        
        if len(season_df) < 100:
            continue
        
        # For per-season eval, we use all prior seasons as training
        prior_df = df[df['season'] < season].copy()
        
        if len(prior_df) < kwargs.get('min_train_size', 200):
            continue
        
        model = signal_func(prior_df)
        
        preds = []
        for _, row in season_df.iterrows():
            try:
                preds.append(prediction_func(model, row))
            except Exception:
                preds.append(np.nan)
        
        season_df = season_df.copy()
        season_df['prediction'] = preds
        season_df = season_df.dropna(subset=['prediction'])
        
        if len(season_df) > 50:
            season_results[season] = {
                'model_brier': brier_score(
                    season_df['prediction'].values,
                    season_df['home_win'].values
                ),
                'baseline_brier': brier_score(
                    season_df['home_implied'].values,
                    season_df['home_win'].values
                ),
                'n_matches': len(season_df),
            }
    
    return season_results
