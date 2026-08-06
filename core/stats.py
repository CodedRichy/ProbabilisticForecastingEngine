"""
Statistical significance testing for Apollo experiments.

Key principle: We are running many experiments simultaneously.
Single-experiment p-values are insufficient. 
All results must survive multiple testing correction.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Optional


def paired_brier_test(model_probs: np.ndarray, baseline_probs: np.ndarray,
                      outcomes: np.ndarray) -> dict:
    """
    Test whether model Brier scores are significantly different from baseline.
    Uses paired t-test on per-observation Brier scores (Diebold-Mariano style).
    
    Parameters
    ----------
    model_probs : model's predicted probabilities
    baseline_probs : bookmaker implied probabilities (baseline)
    outcomes : binary outcomes
    
    Returns
    -------
    dict with t_stat, p_value, mean_diff, ci_lower, ci_upper, n
    """
    model_errors = (model_probs - outcomes) ** 2
    baseline_errors = (baseline_probs - outcomes) ** 2
    
    # Difference: negative = model is better
    diff = model_errors - baseline_errors
    
    n = len(diff)
    mean_diff = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    
    t_stat = mean_diff / se if se > 0 else 0
    p_value = stats.t.sf(abs(t_stat), df=n - 1) * 2  # two-sided
    
    ci_margin = stats.t.ppf(0.975, df=n - 1) * se
    
    return {
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'mean_diff': float(mean_diff),
        'ci_lower': float(mean_diff - ci_margin),
        'ci_upper': float(mean_diff + ci_margin),
        'n': int(n),
        'model_better': mean_diff < 0
    }


def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Benjamini-Hochberg FDR correction for multiple testing.
    
    Given a list of p-values from multiple experiments,
    returns which hypotheses are significant after correction.
    
    Parameters
    ----------
    p_values : list of raw p-values from experiments
    alpha : target false discovery rate
    
    Returns
    -------
    List of booleans — True if significant after correction
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    
    significant = [False] * n
    max_significant_rank = -1
    
    for rank, (original_idx, p) in enumerate(indexed, 1):
        threshold = (rank / n) * alpha
        if p <= threshold:
            max_significant_rank = rank
    
    # All hypotheses with rank <= max_significant_rank are significant
    if max_significant_rank > 0:
        for rank, (original_idx, p) in enumerate(indexed, 1):
            if rank <= max_significant_rank:
                significant[original_idx] = True
    
    return significant


def effect_size_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Cohen's d for two independent groups.
    
    |d| < 0.2: negligible
    0.2 <= |d| < 0.5: small
    0.5 <= |d| < 0.8: medium
    |d| >= 0.8: large
    """
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = group_a.var(ddof=1), group_b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return 0.0
    return (group_a.mean() - group_b.mean()) / pooled_std


def bootstrap_confidence_interval(data: np.ndarray, stat_func=np.mean,
                                   n_bootstrap: int = 10000,
                                   ci: float = 0.95,
                                   seed: int = 42) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval.
    
    Returns (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    point_estimate = stat_func(data)
    
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_func(sample))
    
    boot_stats = np.array(boot_stats)
    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100
    
    return (
        float(point_estimate),
        float(np.percentile(boot_stats, lower_pct)),
        float(np.percentile(boot_stats, upper_pct))
    )


def robustness_check(experiment_func, param_name: str,
                     base_value: float, variations: List[float],
                     **kwargs) -> pd.DataFrame:
    """
    Run an experiment with parameter variations to test robustness.
    
    If the result is sensitive to small parameter changes, it's fragile.
    
    Parameters
    ----------
    experiment_func : function that takes **kwargs and returns a dict with 'p_value' and 'effect'
    param_name : which parameter to vary
    base_value : original parameter value
    variations : list of alternative values to test
    
    Returns
    -------
    DataFrame with columns [param_value, p_value, effect, significant]
    """
    all_values = [base_value] + variations
    rows = []
    
    for val in all_values:
        kwargs[param_name] = val
        result = experiment_func(**kwargs)
        rows.append({
            'param_value': val,
            'p_value': result.get('p_value', 1.0),
            'effect': result.get('effect', 0.0),
            'is_base': val == base_value
        })
    
    df = pd.DataFrame(rows)
    df['significant'] = df['p_value'] < 0.05
    return df
