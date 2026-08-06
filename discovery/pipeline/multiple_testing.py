"""
Multiple Testing Controller for Apollo Discovery Engine.

When testing thousands of hypotheses, conventional significance
thresholds are meaningless. This module implements three layers
of correction:

Layer 1: Benjamini-Yekutieli (BY) FDR correction
    Valid under arbitrary dependence between tests.
    More conservative than Benjamini-Hochberg, which assumes
    independence we don't have.

Layer 2: Minimum Bayes Factor
    Converts p-values to the minimum Bayes Factor consistent
    with the observed p-value. Provides a second, independent
    filter from a Bayesian perspective.

Layer 3: Deflated Sharpe Ratio (López de Prado)
    For signals with economic significance (positive ROI),
    adjusts for the total number of strategies tried,
    plus skewness and kurtosis of returns.

The Trial Counter is a monotonically increasing integer that
tracks ALL tests ever conducted. It never resets.
"""

import numpy as np
import sqlite3
from typing import List, Tuple, Optional
from datetime import datetime
from scipy import stats


class TrialCounter:
    """
    Persistent, append-only counter of total hypotheses tested.
    
    Stored in SQLite. Never decremented. Used by BY correction
    and Deflated Sharpe Ratio to account for the full history
    of researcher exploration.
    """
    
    def __init__(self, db_path: str = "discovery/tracking/apollo.db"):
        self.db_path = db_path
        self._ensure_table()
    
    def _ensure_table(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trial_counter (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                tests_in_batch INTEGER NOT NULL,
                cumulative_total INTEGER NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def get_total(self) -> int:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT cumulative_total FROM trial_counter ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return row[0] if row else 0
    
    def record_batch(self, batch_id: str, n_tests: int) -> int:
        """Record a batch of tests. Returns new cumulative total."""
        current = self.get_total()
        new_total = current + n_tests
        
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO trial_counter (timestamp, batch_id, tests_in_batch, cumulative_total) "
            "VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), batch_id, n_tests, new_total)
        )
        conn.commit()
        conn.close()
        
        return new_total


class MultipleTestingController:
    """
    Orchestrates all three layers of multiple testing correction.
    
    Usage:
        controller = MultipleTestingController(db_path)
        results = controller.correct(raw_p_values, roi_series_list, batch_id)
        # results contains: by_adjusted_p, bayes_factor, deflated_sharpe, passed
    """
    
    def __init__(self, db_path: str = "discovery/tracking/apollo.db",
                 fdr_target: float = 0.05,
                 bayes_threshold: float = 0.1,
                 dsr_threshold: float = 0.0):
        self.trial_counter = TrialCounter(db_path)
        self.fdr_target = fdr_target
        self.bayes_threshold = bayes_threshold     # BF_min must be below this
        self.dsr_threshold = dsr_threshold          # DSR must be above this
    
    def correct(self, 
                raw_p_values: List[float],
                hypothesis_ids: List[str],
                batch_id: str,
                roi_series: Optional[List[np.ndarray]] = None
               ) -> List[dict]:
        """
        Apply all correction layers.
        
        Parameters
        ----------
        raw_p_values : p-values from individual hypothesis tests
        hypothesis_ids : corresponding hypothesis IDs
        batch_id : unique identifier for this batch
        roi_series : optional list of per-bet return arrays (for DSR)
        
        Returns
        -------
        List of dicts, one per hypothesis, with correction results.
        """
        n = len(raw_p_values)
        
        # Record this batch in the trial counter
        total_ever = self.trial_counter.record_batch(batch_id, n)
        print(f"  Trial counter: {total_ever} total tests (this batch: {n})")
        
        # Layer 1: Benjamini-Yekutieli
        by_adjusted = self._benjamini_yekutieli(raw_p_values)
        
        # Layer 2: Minimum Bayes Factor
        bayes_factors = [self._minimum_bayes_factor(p) for p in raw_p_values]
        
        # Layer 3: Deflated Sharpe Ratio (only for hypotheses with ROI data)
        dsrs = [None] * n
        if roi_series is not None:
            for i, roi in enumerate(roi_series):
                if roi is not None and len(roi) > 30:
                    dsrs[i] = self._deflated_sharpe_ratio(roi, total_ever)
        
        # Combine results
        results = []
        for i in range(n):
            passed_by = by_adjusted[i] < self.fdr_target
            passed_bf = bayes_factors[i] < self.bayes_threshold
            
            # DSR check only applies if ROI data exists
            passed_dsr = True
            if dsrs[i] is not None:
                passed_dsr = dsrs[i] > self.dsr_threshold
            
            results.append({
                'hypothesis_id': hypothesis_ids[i],
                'p_value_raw': raw_p_values[i],
                'p_value_by': by_adjusted[i],
                'bayes_factor_min': bayes_factors[i],
                'deflated_sharpe': dsrs[i],
                'passed_by': passed_by,
                'passed_bf': passed_bf,
                'passed_dsr': passed_dsr,
                'passed_all': passed_by and passed_bf and passed_dsr,
                'total_tests_at_time': total_ever,
            })
        
        n_passed = sum(1 for r in results if r['passed_all'])
        print(f"  Passed BY: {sum(1 for r in results if r['passed_by'])}/{n}")
        print(f"  Passed BF: {sum(1 for r in results if r['passed_bf'])}/{n}")
        print(f"  Passed ALL: {n_passed}/{n}")
        
        return results
    
    # ── LAYER 1: BENJAMINI-YEKUTIELI ─────────────────────────
    
    def _benjamini_yekutieli(self, p_values: List[float]) -> List[float]:
        """
        BY procedure: valid under arbitrary dependence.
        
        More conservative than BH. The harmonic number correction
        accounts for the worst-case correlation structure.
        
        Adjusted p_i = p_i * n * c(n) / rank_i
        where c(n) = sum(1/k for k in 1..n) ≈ ln(n) + 0.5772
        """
        n = len(p_values)
        if n == 0:
            return []
        
        # Harmonic number
        c_n = sum(1.0 / k for k in range(1, n + 1))
        
        # Sort p-values, track original indices
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        
        adjusted = [0.0] * n
        
        # Step-up procedure
        prev_adj = 1.0
        for rank_minus_1 in range(n - 1, -1, -1):
            original_idx, p = indexed[rank_minus_1]
            rank = rank_minus_1 + 1
            
            adj_p = p * n * c_n / rank
            adj_p = min(adj_p, prev_adj)  # enforce monotonicity
            adj_p = min(adj_p, 1.0)
            
            adjusted[original_idx] = adj_p
            prev_adj = adj_p
        
        return adjusted
    
    # ── LAYER 2: MINIMUM BAYES FACTOR ────────────────────────
    
    def _minimum_bayes_factor(self, p_value: float) -> float:
        """
        Minimum Bayes Factor: the smallest BF consistent with
        the observed p-value, assuming any alternative hypothesis.
        
        BF_min = -e * p * ln(p) for p < 1/e
        BF_min = 1              for p >= 1/e
        
        Interpretation:
            BF < 1/100: decisive evidence
            BF < 1/10:  strong evidence
            BF < 1/3:   moderate evidence
            BF > 1:     no evidence against null
        
        Reference: Sellke, Bayarri, Berger (2001)
        """
        if p_value <= 0:
            return 0.0
        if p_value >= 1.0 / np.e:
            return 1.0
        return float(-np.e * p_value * np.log(p_value))
    
    # ── LAYER 3: DEFLATED SHARPE RATIO ───────────────────────
    
    def _deflated_sharpe_ratio(self, returns: np.ndarray, 
                                n_trials: int) -> float:
        """
        Deflated Sharpe Ratio (López de Prado, 2014).
        
        Tests whether the observed Sharpe ratio would be expected
        under the null hypothesis that the best strategy was selected
        from n_trials independent trials.
        
        Accounts for:
        - Number of strategies tried (n_trials)
        - Non-normality of returns (skewness, kurtosis)
        - Sample size
        
        Returns a DSR value. DSR > 0 means the signal survives
        the multiple-testing haircut.
        
        Reference: "The Deflated Sharpe Ratio: Correcting for
        Selection Bias, Backtest Overfitting, and Non-Normality"
        """
        T = len(returns)
        if T < 30:
            return -1.0  # insufficient data
        
        sr = returns.mean() / returns.std() if returns.std() > 0 else 0
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))  # excess kurtosis
        
        # Expected maximum Sharpe ratio under null
        # E[max(SR)] ≈ sqrt(2 * ln(N)) * (1 - gamma/sqrt(2*ln(N))) + gamma/sqrt(2*ln(N))
        # where gamma ≈ 0.5772 (Euler-Mascheroni constant)
        # Simplified: E[max(SR)] ≈ sqrt(2 * ln(n_trials))
        
        if n_trials <= 1:
            sr_expected_max = 0
        else:
            euler_gamma = 0.5772156649
            z = np.sqrt(2 * np.log(n_trials))
            sr_expected_max = z - (np.log(np.pi) + euler_gamma) / (2 * z)
        
        # Standard error of SR estimate (accounting for non-normality)
        se_sr = np.sqrt(
            (1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / (T - 1)
        )
        
        if se_sr <= 0:
            return -1.0
        
        # DSR = probability that observed SR > expected max SR under null
        # Expressed as a z-score (higher = more likely to be genuine)
        dsr = (sr - sr_expected_max) / se_sr
        
        return float(dsr)
