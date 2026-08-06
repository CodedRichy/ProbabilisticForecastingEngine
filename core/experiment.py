"""
Experiment framework for Apollo.

Each experiment:
1. Has a pre-registered hypothesis
2. Loads data without seeing hold-out seasons
3. Computes a signal
4. Evaluates against bookmaker baseline
5. Tests significance
6. Generates a report

Usage:
    python -m experiments.EXP_001_form_bias.run
"""

import os
import yaml
import json
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from core.metrics import brier_score, log_loss, calibration_bins, roi_simulation, brier_skill_score
from core.stats import paired_brier_test, bootstrap_confidence_interval


# ── Load global settings ──────────────────────────────────────────────

def load_settings() -> dict:
    settings_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(settings_path) as f:
        return yaml.safe_load(f)


SETTINGS = load_settings()
HOLDOUT_SEASONS = SETTINGS['defaults']['holdout_seasons']
RANDOM_SEED = SETTINGS['defaults']['random_seed']
MIN_SAMPLE = SETTINGS['defaults']['min_sample_size']


# ── Experiment Base Class ─────────────────────────────────────────────

class Experiment(ABC):
    """
    Base class for all Apollo experiments.
    
    Subclass this and implement:
        - hypothesis (property): plain-English hypothesis
        - compute_signal(df): returns signal column(s)
        - get_predictions(df): returns predicted probabilities
    """
    
    def __init__(self, experiment_dir: str, config_path: Optional[str] = None):
        self.experiment_dir = Path(experiment_dir)
        self.results_dir = self.experiment_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load experiment-specific config
        if config_path is None:
            config_path = self.experiment_dir / "config.yaml"
        
        if Path(config_path).exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        self.seed = self.config.get('random_seed', RANDOM_SEED)
        np.random.seed(self.seed)
        
        self.run_timestamp = datetime.now().isoformat()
        self._results = {}
    
    @property
    @abstractmethod
    def hypothesis(self) -> str:
        """Plain-English hypothesis statement."""
        pass
    
    @property
    @abstractmethod
    def experiment_id(self) -> str:
        """e.g., 'EXP_001'"""
        pass
    
    @abstractmethod
    def compute_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add signal columns to the dataframe.
        Must NOT use outcome columns (home_goals, away_goals, result).
        Must NOT use hold-out season data.
        
        Returns modified DataFrame with new signal columns.
        """
        pass
    
    @abstractmethod
    def get_predictions(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predicted probabilities from the signal.
        Returns array of probabilities for the target outcome.
        """
        pass
    
    def load_data(self, leagues: Optional[list] = None, 
                  exclude_holdout: bool = True) -> pd.DataFrame:
        """
        Load processed match data.
        
        By default, EXCLUDES hold-out seasons to prevent leakage.
        """
        data_dir = Path("data/processed")
        
        if not data_dir.exists():
            raise FileNotFoundError(
                f"No processed data found at {data_dir}. "
                "Run the data pipeline first."
            )
        
        frames = []
        for f in sorted(data_dir.glob("*.parquet")):
            df = pd.read_parquet(f)
            frames.append(df)
        
        if not frames:
            raise FileNotFoundError("No parquet files found in data/processed/")
        
        df = pd.concat(frames, ignore_index=True)
        
        # Filter leagues
        if leagues:
            df = df[df['league'].isin(leagues)]
        
        # Exclude hold-out seasons
        if exclude_holdout:
            df = df[~df['season'].isin(HOLDOUT_SEASONS)]
            print(f"  Hold-out seasons excluded: {HOLDOUT_SEASONS}")
        
        print(f"  Loaded {len(df)} matches across {df['league'].nunique()} leagues, "
              f"{df['season'].nunique()} seasons")
        
        return df
    
    def evaluate(self, predictions: np.ndarray, baseline_probs: np.ndarray,
                 outcomes: np.ndarray, market_odds: np.ndarray) -> dict:
        """
        Full evaluation suite against bookmaker baseline.
        """
        results = {}
        
        # Core metrics
        results['model_brier'] = brier_score(predictions, outcomes)
        results['baseline_brier'] = brier_score(baseline_probs, outcomes)
        results['brier_skill_score'] = brier_skill_score(
            results['model_brier'], results['baseline_brier']
        )
        
        results['model_logloss'] = log_loss(predictions, outcomes)
        results['baseline_logloss'] = log_loss(baseline_probs, outcomes)
        
        # Significance test
        sig_test = paired_brier_test(predictions, baseline_probs, outcomes)
        results['significance'] = sig_test
        
        # ROI simulation
        results['roi_0pct'] = roi_simulation(predictions, market_odds, outcomes, 
                                              edge_threshold=0.0)
        results['roi_3pct'] = roi_simulation(predictions, market_odds, outcomes,
                                              edge_threshold=0.03)
        results['roi_5pct'] = roi_simulation(predictions, market_odds, outcomes,
                                              edge_threshold=0.05)
        
        # Calibration
        results['calibration'] = calibration_bins(predictions, outcomes).to_dict('records')
        
        # Bootstrap CI on Brier improvement
        brier_diffs = (predictions - outcomes)**2 - (baseline_probs - outcomes)**2
        point, ci_lo, ci_hi = bootstrap_confidence_interval(brier_diffs, seed=self.seed)
        results['brier_improvement_ci'] = {
            'point': point, 'ci_lower': ci_lo, 'ci_upper': ci_hi
        }
        
        # Sample size check
        results['n_observations'] = int(len(outcomes))
        results['sufficient_sample'] = len(outcomes) >= MIN_SAMPLE
        
        return results
    
    def run(self):
        """Execute the full experiment pipeline."""
        print(f"\n{'='*60}")
        print(f"  {self.experiment_id}: {self.hypothesis}")
        print(f"{'='*60}\n")
        
        # Step 1: Load data
        print("[1/5] Loading data...")
        df = self.load_data(
            leagues=self.config.get('leagues'),
            exclude_holdout=True
        )
        
        # Step 2: Compute signal
        print("[2/5] Computing signal...")
        df = self.compute_signal(df)
        
        # Step 3: Generate predictions
        print("[3/5] Generating predictions...")
        predictions = self.get_predictions(df)
        
        # Step 4: Extract baseline and outcomes
        # Subclass should define which outcome to target
        target_col = self.config.get('target', 'home_implied')
        outcome_col = self.config.get('outcome', 'home_win')
        odds_col = self.config.get('odds_col', 'home_odds')
        
        baseline_probs = df[target_col].values
        outcomes = df[outcome_col].values
        market_odds = df[odds_col].values
        
        # Remove NaN rows
        valid = ~(np.isnan(predictions) | np.isnan(baseline_probs) | 
                  np.isnan(outcomes) | np.isnan(market_odds))
        predictions = predictions[valid]
        baseline_probs = baseline_probs[valid]
        outcomes = outcomes[valid]
        market_odds = market_odds[valid]
        
        print(f"  Valid observations: {len(predictions)}")
        
        # Step 5: Evaluate
        print("[4/5] Evaluating...")
        self._results = self.evaluate(predictions, baseline_probs, outcomes, market_odds)
        self._results['hypothesis'] = self.hypothesis
        self._results['experiment_id'] = self.experiment_id
        self._results['run_timestamp'] = self.run_timestamp
        self._results['config'] = self.config
        self._results['seed'] = self.seed
        
        # Step 6: Generate report
        print("[5/5] Generating report...")
        self._save_results()
        self._generate_report()
        
        # Print summary
        self._print_summary()
        
        return self._results
    
    def _save_results(self):
        """Save raw results as JSON."""
        out_path = self.results_dir / "results.json"
        with open(out_path, 'w') as f:
            json.dump(self._results, f, indent=2, default=str)
        print(f"  Results saved: {out_path}")
    
    def _generate_report(self):
        """Generate markdown report."""
        r = self._results
        sig = r['significance']
        
        verdict = "SIGNIFICANT" if sig['p_value'] < 0.05 and sig['model_better'] else "NOT SIGNIFICANT"
        if not r['sufficient_sample']:
            verdict = "INSUFFICIENT DATA"
        
        report = f"""# {self.experiment_id} — Experiment Report

**Hypothesis:** {self.hypothesis}

**Run:** {self.run_timestamp}

**Verdict:** {verdict}

---

## Results Summary

| Metric | Model | Baseline (Market) | Difference |
|---|---|---|---|
| Brier Score | {r['model_brier']:.6f} | {r['baseline_brier']:.6f} | {r['model_brier'] - r['baseline_brier']:+.6f} |
| Log Loss | {r['model_logloss']:.6f} | {r['baseline_logloss']:.6f} | {r['model_logloss'] - r['baseline_logloss']:+.6f} |
| Brier Skill Score | {r['brier_skill_score']:.4f} | — | — |

## Significance

| Metric | Value |
|---|---|
| t-statistic | {sig['t_stat']:.4f} |
| p-value | {sig['p_value']:.6f} |
| Mean Brier difference | {sig['mean_diff']:.6f} |
| 95% CI | [{sig['ci_lower']:.6f}, {sig['ci_upper']:.6f}] |
| n | {sig['n']} |
| Model better? | {sig['model_better']} |

## ROI Simulation (flat stake)

| Edge Threshold | ROI | N Bets | Win Rate |
|---|---|---|---|
| 0% | {r['roi_0pct']['roi']:.4f} | {r['roi_0pct']['n_bets']} | {r['roi_0pct']['win_rate']:.4f} |
| 3% | {r['roi_3pct']['roi']:.4f} | {r['roi_3pct']['n_bets']} | {r['roi_3pct']['win_rate']:.4f} |
| 5% | {r['roi_5pct']['roi']:.4f} | {r['roi_5pct']['n_bets']} | {r['roi_5pct']['win_rate']:.4f} |

## Bootstrap CI on Brier Improvement

Point estimate: {r['brier_improvement_ci']['point']:.6f}
95% CI: [{r['brier_improvement_ci']['ci_lower']:.6f}, {r['brier_improvement_ci']['ci_upper']:.6f}]

(Negative = model better than baseline)

## Sample

Observations: {r['n_observations']}
Sufficient: {r['sufficient_sample']}

---

*Report auto-generated by Apollo Experiment Framework.*
"""
        report_path = self.results_dir / "report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"  Report saved: {report_path}")
    
    def _print_summary(self):
        r = self._results
        sig = r['significance']
        
        print(f"\n{'─'*40}")
        print(f"  Brier:  model={r['model_brier']:.6f}  market={r['baseline_brier']:.6f}")
        print(f"  BSS:    {r['brier_skill_score']:.4f}")
        print(f"  p-value: {sig['p_value']:.6f}")
        print(f"  ROI @3%: {r['roi_3pct']['roi']:.4f} ({r['roi_3pct']['n_bets']} bets)")
        
        if sig['p_value'] < 0.05 and sig['model_better']:
            print(f"  >>> POTENTIALLY SIGNIFICANT — needs FDR correction + robustness check")
        else:
            print(f"  >>> NOT SIGNIFICANT against market baseline")
        print(f"{'─'*40}\n")
