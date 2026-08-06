"""
Testing Pipeline for Apollo Discovery Engine.

Takes a list of hypotheses and runs each through rigorous
statistical evaluation against the bookmaker baseline.

Every test measures the same thing: does this hypothesis identify
situations where the market is miscalibrated?

The test statistic is always: actual_outcome - market_implied.
If this residual correlates with a feature or differs systematically
in a subgroup, the market isn't fully pricing that information.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy import stats
from multiprocessing import Pool, cpu_count
import time

from discovery.generators.hypothesis_generator import Hypothesis


def test_single_hypothesis(args: tuple) -> dict:
    """
    Test one hypothesis. Designed for multiprocessing.
    
    Parameters (packed as tuple for Pool.map):
        hypothesis: Hypothesis object
        df: DataFrame with features and outcomes
    
    Returns dict with test results.
    """
    hypothesis, df_dict_data, feature_cols_info = args
    
    # Reconstruct DataFrame from serialized form
    df = pd.DataFrame(df_dict_data)
    
    h = hypothesis
    
    try:
        result = _run_test(h, df)
    except Exception as e:
        result = {
            'hypothesis_id': h.hypothesis_id,
            'error': str(e),
            'p_value': 1.0,
            'effect_size': 0.0,
            'n_observations': 0,
            'passed_gate1': False,
        }
    
    return result


def _run_test(h: Hypothesis, df: pd.DataFrame) -> dict:
    """Execute the appropriate test for a hypothesis type."""
    
    # Determine baseline column
    target_to_baseline = {
        'home_win': 'home_implied',
        'draw': 'draw_implied',
        'away_win': 'away_implied',
    }
    baseline_col = target_to_baseline.get(h.target, 'home_implied')
    
    # Compute residuals: actual - market_implied
    valid = df[h.target].notna() & df[baseline_col].notna()
    for feat in h.features:
        if feat in df.columns:
            valid = valid & df[feat].notna()
    
    df_valid = df[valid].copy()
    
    if len(df_valid) < 100:
        return {
            'hypothesis_id': h.hypothesis_id,
            'p_value': 1.0,
            'effect_size': 0.0,
            'n_observations': len(df_valid),
            'passed_gate1': False,
            'reason': 'insufficient_data',
        }
    
    residuals = df_valid[h.target].values - df_valid[baseline_col].values
    
    if h.operator == 'residual_correlation':
        return _test_residual_correlation(h, df_valid, residuals)
    elif h.operator == 'conditional_calibration':
        return _test_conditional_calibration(h, df_valid, residuals)
    elif h.operator == 'pairwise_interaction':
        return _test_pairwise_interaction(h, df_valid, residuals)
    elif h.operator == 'tree_partition':
        return _test_tree_partition(h, df_valid, residuals)
    else:
        return {
            'hypothesis_id': h.hypothesis_id,
            'p_value': 1.0,
            'error': f'unknown operator: {h.operator}',
            'passed_gate1': False,
        }


def _test_residual_correlation(h: Hypothesis, df: pd.DataFrame, 
                                residuals: np.ndarray) -> dict:
    """
    Level 1: Does feature X correlate with calibration residuals?
    
    Uses Spearman rank correlation (robust to outliers and nonlinearity).
    """
    feat = h.features[0]
    feature_values = df[feat].values
    
    corr, p_value = stats.spearmanr(feature_values, residuals, nan_policy='omit')
    
    # Effect size: the correlation coefficient itself
    effect = abs(corr)
    
    return {
        'hypothesis_id': h.hypothesis_id,
        'p_value': float(p_value),
        'effect_size': float(effect),
        'correlation': float(corr),
        'n_observations': int(len(df)),
        'test_type': 'spearman',
    }


def _test_conditional_calibration(h: Hypothesis, df: pd.DataFrame,
                                   residuals: np.ndarray) -> dict:
    """
    Level 2: Is market calibration broken in a specific quintile of feature X?
    
    Splits data by feature quintile, tests whether the residual mean
    in the target quintile differs significantly from zero.
    """
    feat = h.features[0]
    
    try:
        quintiles = pd.qcut(df[feat], 5, labels=False, duplicates='drop')
    except (ValueError, TypeError):
        return {
            'hypothesis_id': h.hypothesis_id,
            'p_value': 1.0,
            'reason': 'quintile_computation_failed',
            'passed_gate1': False,
        }
    
    # Parse target quintile from condition
    target_q = int(h.condition.split('_')[1]) if h.condition else 0
    
    mask = quintiles == target_q
    subgroup_residuals = residuals[mask]
    
    if len(subgroup_residuals) < 30:
        return {
            'hypothesis_id': h.hypothesis_id,
            'p_value': 1.0,
            'n_observations': len(subgroup_residuals),
            'reason': 'insufficient_subgroup_data',
            'passed_gate1': False,
        }
    
    # One-sample t-test: is the mean residual significantly different from 0?
    t_stat, p_value = stats.ttest_1samp(subgroup_residuals, 0)
    
    effect = abs(subgroup_residuals.mean()) / subgroup_residuals.std() if subgroup_residuals.std() > 0 else 0
    
    return {
        'hypothesis_id': h.hypothesis_id,
        'p_value': float(p_value),
        'effect_size': float(effect),
        't_statistic': float(t_stat),
        'mean_residual': float(subgroup_residuals.mean()),
        'n_observations': int(len(subgroup_residuals)),
        'n_total': int(len(df)),
        'quintile': target_q,
        'test_type': 'one_sample_t',
    }


def _test_pairwise_interaction(h: Hypothesis, df: pd.DataFrame,
                                residuals: np.ndarray) -> dict:
    """
    Level 3: Does the interaction of features X and Y predict residuals?
    
    Strategy: Split each feature at median. Create 4 quadrants.
    Test whether the residual distribution differs across quadrants
    using Kruskal-Wallis (non-parametric ANOVA).
    
    Then test the most extreme quadrant (both high or both low)
    for significant miscalibration.
    """
    f1, f2 = h.features[0], h.features[1]
    
    med1 = df[f1].median()
    med2 = df[f2].median()
    
    # Four quadrants
    q_hh = (df[f1] >= med1) & (df[f2] >= med2)  # both high
    q_hl = (df[f1] >= med1) & (df[f2] < med2)   # f1 high, f2 low
    q_lh = (df[f1] < med1) & (df[f2] >= med2)   # f1 low, f2 high
    q_ll = (df[f1] < med1) & (df[f2] < med2)    # both low
    
    groups = [residuals[q_hh], residuals[q_hl], residuals[q_lh], residuals[q_ll]]
    groups = [g for g in groups if len(g) >= 20]
    
    if len(groups) < 3:
        return {
            'hypothesis_id': h.hypothesis_id,
            'p_value': 1.0,
            'reason': 'insufficient_quadrant_data',
            'passed_gate1': False,
        }
    
    # Kruskal-Wallis: do quadrants differ?
    kw_stat, kw_p = stats.kruskal(*groups)
    
    # Also test the most extreme quadrant (both high)
    extreme_residuals = residuals[q_hh]
    other_residuals = residuals[~q_hh]
    
    if len(extreme_residuals) >= 20 and len(other_residuals) >= 20:
        mw_stat, mw_p = stats.mannwhitneyu(
            extreme_residuals, other_residuals, alternative='two-sided'
        )
    else:
        mw_p = 1.0
    
    # Use the more conservative (larger) p-value
    p_value = max(kw_p, mw_p)
    
    # Effect size: difference between extreme quadrant and rest
    effect = abs(extreme_residuals.mean() - other_residuals.mean()) if len(extreme_residuals) > 0 else 0
    
    return {
        'hypothesis_id': h.hypothesis_id,
        'p_value': float(p_value),
        'p_kruskal': float(kw_p),
        'p_mannwhitney': float(mw_p),
        'effect_size': float(effect),
        'mean_residual_hh': float(extreme_residuals.mean()),
        'mean_residual_other': float(other_residuals.mean()),
        'n_quadrant_hh': int(q_hh.sum()),
        'n_total': int(len(df)),
        'test_type': 'kruskal_mannwhitney',
    }


def _test_tree_partition(h: Hypothesis, df: pd.DataFrame,
                          residuals: np.ndarray) -> dict:
    """Level 4: Test a tree-discovered partition."""
    # Parse condition and apply as filter
    # Conditions come as strings like "feat1 > 0.5 AND feat2 < 0.3"
    try:
        mask = df.eval(h.condition)
        subgroup = residuals[mask]
    except Exception:
        return {
            'hypothesis_id': h.hypothesis_id,
            'p_value': 1.0,
            'reason': 'condition_parse_failed',
            'passed_gate1': False,
        }
    
    if len(subgroup) < 30:
        return {
            'hypothesis_id': h.hypothesis_id,
            'p_value': 1.0,
            'reason': 'insufficient_subgroup_data',
            'passed_gate1': False,
        }
    
    t_stat, p_value = stats.ttest_1samp(subgroup, 0)
    effect = abs(subgroup.mean()) / subgroup.std() if subgroup.std() > 0 else 0
    
    return {
        'hypothesis_id': h.hypothesis_id,
        'p_value': float(p_value),
        'effect_size': float(effect),
        'mean_residual': float(subgroup.mean()),
        'n_subgroup': int(len(subgroup)),
        'n_total': int(len(df)),
        'test_type': 'one_sample_t_tree',
    }


class TestingPipeline:
    """
    Orchestrates batch testing of hypotheses.
    
    Supports multiprocessing for Level 3 (thousands of pairwise tests).
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        self.n_workers = n_workers or max(1, cpu_count() - 1)
    
    def run_batch(self, hypotheses: List[Hypothesis], 
                  df: pd.DataFrame,
                  parallel: bool = True) -> List[dict]:
        """
        Test all hypotheses in a batch.
        
        Returns list of result dicts, one per hypothesis.
        """
        print(f"\n  Testing {len(hypotheses)} hypotheses...")
        start = time.time()
        
        if parallel and len(hypotheses) > 100 and self.n_workers > 1:
            results = self._run_parallel(hypotheses, df)
        else:
            results = self._run_sequential(hypotheses, df)
        
        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.1f}s ({len(hypotheses)/max(elapsed,0.1):.0f} tests/sec)")
        
        return results
    
    def _run_sequential(self, hypotheses: List[Hypothesis],
                        df: pd.DataFrame) -> List[dict]:
        results = []
        df_dict = df.to_dict('list')
        
        for i, h in enumerate(hypotheses):
            if (i + 1) % 500 == 0:
                print(f"    Progress: {i+1}/{len(hypotheses)}")
            result = test_single_hypothesis((h, df_dict, None))
            results.append(result)
        
        return results
    
    def _run_parallel(self, hypotheses: List[Hypothesis],
                      df: pd.DataFrame) -> List[dict]:
        """Parallel execution for large hypothesis sets."""
        df_dict = df.to_dict('list')
        
        args_list = [(h, df_dict, None) for h in hypotheses]
        
        print(f"    Using {self.n_workers} workers...")
        with Pool(self.n_workers) as pool:
            results = pool.map(test_single_hypothesis, args_list, chunksize=50)
        
        return results
