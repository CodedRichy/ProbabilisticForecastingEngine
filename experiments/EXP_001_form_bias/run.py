"""
EXP_001: Recent Form Bias

Hypothesis: Teams on 5+ match winning streaks are overpriced by 
bookmakers because public money overweights recent form.

Test: For each match, compute the home team's recent form (wins in 
last N matches). When a team is on a winning streak above the threshold,
compare their actual win rate against the bookmaker implied probability.

If the market is efficient, streaking teams should win at approximately
the rate the odds imply. If they're overpriced, they win LESS often
than implied.

This experiment does NOT try to build a better model. It tests whether
a specific market bias exists.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from pathlib import Path

from core.experiment import Experiment


class FormBiasExperiment(Experiment):
    
    @property
    def hypothesis(self) -> str:
        return "Teams on 5+ match winning streaks are overpriced by bookmakers"
    
    @property
    def experiment_id(self) -> str:
        return "EXP_001"
    
    def compute_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling win streak for each team.
        
        CRITICAL: This uses only PAST information. For each match,
        the streak is computed from matches BEFORE the current date.
        """
        streak_threshold = self.config.get('streak_threshold', 5)
        
        df = df.sort_values('date').copy()
        
        # Compute per-team rolling form
        # We need to iterate chronologically to avoid lookahead
        team_history = {}  # team -> list of recent results
        
        home_streaks = []
        away_streaks = []
        
        for idx, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            # Get current streak BEFORE this match
            home_streak = self._current_streak(team_history.get(home, []))
            away_streak = self._current_streak(team_history.get(away, []))
            
            home_streaks.append(home_streak)
            away_streaks.append(away_streak)
            
            # Update history AFTER recording the streak
            if home not in team_history:
                team_history[home] = []
            if away not in team_history:
                team_history[away] = []
            
            result = row['result']
            team_history[home].append('W' if result == 'H' else ('D' if result == 'D' else 'L'))
            team_history[away].append('W' if result == 'A' else ('D' if result == 'D' else 'L'))
            
            # Keep only last 20 results per team
            team_history[home] = team_history[home][-20:]
            team_history[away] = team_history[away][-20:]
        
        df['home_win_streak'] = home_streaks
        df['away_win_streak'] = away_streaks
        
        # Flag: is the home team on a hot streak?
        df['home_on_streak'] = (df['home_win_streak'] >= streak_threshold).astype(int)
        df['away_on_streak'] = (df['away_win_streak'] >= streak_threshold).astype(int)
        
        print(f"  Home teams on streak (>={streak_threshold}): {df['home_on_streak'].sum()}")
        print(f"  Away teams on streak (>={streak_threshold}): {df['away_on_streak'].sum()}")
        
        return df
    
    def _current_streak(self, results: list) -> int:
        """Count consecutive wins from most recent backward."""
        streak = 0
        for r in reversed(results):
            if r == 'W':
                streak += 1
            else:
                break
        return streak
    
    def get_predictions(self, df: pd.DataFrame) -> np.ndarray:
        """
        This experiment doesn't build a predictive model.
        It tests whether the BASELINE (market) is miscalibrated 
        for streak teams.
        
        For the evaluation framework to work, we return the market
        implied probability as our "prediction" — the test is whether
        streak vs non-streak subgroups show different calibration.
        
        The real analysis is in the analysis.py companion script.
        """
        # Return market probabilities — we're testing calibration, not building a model
        return df['home_implied'].values
    
    def run_calibration_analysis(self):
        """
        The core analysis: compare market calibration for streak vs non-streak matches.
        """
        print(f"\n{'='*60}")
        print("  EXP_001: Calibration Analysis — Form Bias")
        print(f"{'='*60}\n")
        
        df = self.load_data(exclude_holdout=True)
        df = self.compute_signal(df)
        
        # Drop matches without odds
        df = df.dropna(subset=['home_implied', 'home_win'])
        
        streak_threshold = self.config.get('streak_threshold', 5)
        
        # Split into streak vs non-streak
        streak_mask = df['home_on_streak'] == 1
        
        streak_df = df[streak_mask]
        normal_df = df[~streak_mask]
        
        print(f"\nStreak matches: {len(streak_df)}")
        print(f"Normal matches: {len(normal_df)}")
        
        if len(streak_df) < 50:
            print("  WARNING: Very small streak sample. Results unreliable.")
        
        # Compare: implied probability vs actual win rate
        streak_implied = streak_df['home_implied'].mean()
        streak_actual = streak_df['home_win'].mean()
        
        normal_implied = normal_df['home_implied'].mean()
        normal_actual = normal_df['home_win'].mean()
        
        print(f"\n{'─'*40}")
        print(f"  STREAK teams:")
        print(f"    Market expects: {streak_implied:.4f}")
        print(f"    Actually win:   {streak_actual:.4f}")
        print(f"    Difference:     {streak_actual - streak_implied:+.4f}")
        print(f"  NON-STREAK teams:")
        print(f"    Market expects: {normal_implied:.4f}")
        print(f"    Actually win:   {normal_actual:.4f}")
        print(f"    Difference:     {normal_actual - normal_implied:+.4f}")
        print(f"{'─'*40}")
        
        # If difference is negative for streak teams: they win LESS than
        # the market expects → overpriced → hypothesis supported
        
        # Statistical test: are streak teams' residuals different from normal?
        from scipy import stats
        
        streak_residuals = streak_df['home_win'].values - streak_df['home_implied'].values
        normal_residuals = normal_df['home_win'].values - normal_df['home_implied'].values
        
        t_stat, p_value = stats.ttest_ind(streak_residuals, normal_residuals)
        
        print(f"\n  t-test (streak vs normal residuals):")
        print(f"    t = {t_stat:.4f}, p = {p_value:.6f}")
        
        if p_value < 0.05 and streak_actual < streak_implied:
            print(f"\n  >>> POTENTIAL SIGNAL: Streak teams appear overpriced (p={p_value:.4f})")
            print(f"  >>> Needs FDR correction and robustness checks before conclusion.")
        else:
            print(f"\n  >>> NO SIGNAL detected at α=0.05")
        
        return {
            'streak_n': len(streak_df),
            'normal_n': len(normal_df),
            'streak_implied': streak_implied,
            'streak_actual': streak_actual,
            'streak_gap': streak_actual - streak_implied,
            'normal_implied': normal_implied,
            'normal_actual': normal_actual,
            'normal_gap': normal_actual - normal_implied,
            't_stat': t_stat,
            'p_value': p_value,
        }


if __name__ == "__main__":
    exp_dir = Path(__file__).parent
    exp = FormBiasExperiment(experiment_dir=str(exp_dir))
    
    # Run the calibration analysis (the real test for this hypothesis)
    results = exp.run_calibration_analysis()
