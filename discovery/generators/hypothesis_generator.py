"""
Hypothesis Generator for Apollo Discovery Engine.

Systematically enumerates all testable hypotheses from the feature
space. No human curation. No narrative. Pure combinatorial enumeration.

Hypothesis types:
    Level 1: Univariate — does feature X predict calibration residuals?
    Level 2: Conditional — does market calibration break down in specific
             quintiles of feature X?
    Level 3: Pairwise — does the interaction of X and Y predict residuals?
    Level 4: Tree-based — recursive partitioning discovers subgroups
             with systematic miscalibration.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
import hashlib
import json


@dataclass
class Hypothesis:
    """A testable hypothesis in structured form."""
    hypothesis_id: str
    level: int                          # 1, 2, 3, or 4
    features: List[str]                 # features involved
    operator: str                       # test type
    target: str                         # outcome variable
    condition: Optional[str] = None     # filter expression
    description: str = ""               # auto-generated English description
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def config_hash(self) -> str:
        """Deterministic hash for deduplication."""
        content = json.dumps({
            'level': self.level,
            'features': sorted(self.features),
            'operator': self.operator,
            'target': self.target,
            'condition': self.condition,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


class HypothesisGenerator:
    """
    Enumerates the hypothesis space mechanically.
    
    No human judgment about which hypotheses are "interesting."
    Every valid combination is generated. Statistical testing
    decides what survives, not intuition.
    """
    
    TARGETS = ['home_win', 'draw', 'away_win']
    
    def __init__(self, 
                 max_pairwise_features: int = 80,
                 quintile_bins: int = 5,
                 tree_max_depth: int = 3):
        self.max_pairwise = max_pairwise_features
        self.quintile_bins = quintile_bins
        self.tree_max_depth = tree_max_depth
        self._counter = 0
        self.hypotheses: List[Hypothesis] = []
    
    def generate_all(self, feature_names: List[str], 
                     df: pd.DataFrame) -> List[Hypothesis]:
        """
        Generate the full hypothesis catalog.
        
        Parameters
        ----------
        feature_names : list of computed feature column names
        df : DataFrame with features (used only for quintile computation
             and feature selection — NOT for outcome data)
        """
        self.hypotheses = []
        self._counter = 0
        
        # Feature quality filter: drop features with >50% NaN
        valid_features = [
            f for f in feature_names 
            if df[f].notna().mean() > 0.5
        ]
        print(f"  Valid features (>50% non-null): {len(valid_features)}/{len(feature_names)}")
        
        # Level 1: Univariate
        print("  Generating Level 1 (univariate)...")
        self._generate_level1(valid_features)
        
        # Level 2: Conditional calibration
        print("  Generating Level 2 (conditional quintiles)...")
        self._generate_level2(valid_features, df)
        
        # Level 3: Pairwise interactions
        # Select top features by variance for pairwise to keep tractable
        pairwise_features = self._select_pairwise_features(valid_features, df)
        print(f"  Generating Level 3 (pairwise, {len(pairwise_features)} features)...")
        self._generate_level3(pairwise_features)
        
        # Level 4: Tree-based discovery (generated during testing, not here)
        # Placeholder — these are extracted from fitted trees
        
        print(f"\n  Total hypotheses generated: {len(self.hypotheses)}")
        print(f"    Level 1: {sum(1 for h in self.hypotheses if h.level == 1)}")
        print(f"    Level 2: {sum(1 for h in self.hypotheses if h.level == 2)}")
        print(f"    Level 3: {sum(1 for h in self.hypotheses if h.level == 3)}")
        
        return self.hypotheses
    
    def _next_id(self) -> str:
        self._counter += 1
        return f"H-{self._counter:05d}"
    
    # ── LEVEL 1: UNIVARIATE ───────────────────────────────────
    
    def _generate_level1(self, features: List[str]):
        """
        For each feature: does it correlate with calibration residuals?
        
        Test: Spearman correlation between feature value and 
        (actual_outcome - market_implied) for each target.
        """
        for feat in features:
            for target in self.TARGETS:
                baseline_col = target.replace('_win', '_implied').replace('draw', 'draw_implied')
                self.hypotheses.append(Hypothesis(
                    hypothesis_id=self._next_id(),
                    level=1,
                    features=[feat],
                    operator='residual_correlation',
                    target=target,
                    description=f"Does {feat} predict {target} calibration residuals?"
                ))
    
    # ── LEVEL 2: CONDITIONAL CALIBRATION ──────────────────────
    
    def _generate_level2(self, features: List[str], df: pd.DataFrame):
        """
        For each feature, bin into quintiles. Test whether market
        calibration is systematically off in each quintile.
        
        This finds: "When feature X is very high/low, the market
        over/underestimates probability Y."
        """
        for feat in features:
            col = df[feat].dropna()
            if len(col) < 100:
                continue
            
            try:
                quintiles = pd.qcut(col, self.quintile_bins, duplicates='drop')
                n_bins = quintiles.nunique()
            except (ValueError, TypeError):
                continue
            
            if n_bins < 3:
                continue
            
            # Test extreme quintiles only (top and bottom)
            # Middle quintiles rarely show signal and inflate test count
            for quintile_idx in [0, n_bins - 1]:
                for target in ['home_win']:  # Primary target only for Level 2
                    self.hypotheses.append(Hypothesis(
                        hypothesis_id=self._next_id(),
                        level=2,
                        features=[feat],
                        operator='conditional_calibration',
                        target=target,
                        condition=f"quintile_{quintile_idx}",
                        description=(
                            f"Is market miscalibrated for {target} when "
                            f"{feat} is in quintile {quintile_idx}?"
                        )
                    ))
    
    # ── LEVEL 3: PAIRWISE ────────────────────────────────────
    
    def _generate_level3(self, features: List[str]):
        """
        For each pair of features: does their joint condition
        predict calibration residuals?
        
        Test: stratify by both features (above/below median),
        test residual in each quadrant.
        """
        for f1, f2 in combinations(features, 2):
            # Only test the most interesting quadrant: both extreme
            # (both high or both low vs mixed)
            for target in ['home_win']:  # Primary target only
                self.hypotheses.append(Hypothesis(
                    hypothesis_id=self._next_id(),
                    level=3,
                    features=[f1, f2],
                    operator='pairwise_interaction',
                    target=target,
                    description=(
                        f"Does the interaction of {f1} and {f2} "
                        f"predict {target} calibration residuals?"
                    )
                ))
    
    def _select_pairwise_features(self, features: List[str], 
                                   df: pd.DataFrame) -> List[str]:
        """
        Select features for pairwise testing.
        
        Strategy: Pick features with highest variance (most informative)
        and lowest mutual correlation (most independent).
        This keeps the pairwise count manageable while maximizing
        the chance of finding novel interactions.
        """
        if len(features) <= self.max_pairwise:
            return features
        
        # Score by: variance rank (higher = better) - correlation penalty
        # Simple approach: take top N by variance, then prune highly correlated
        variances = {}
        for f in features:
            col = df[f].dropna()
            if len(col) > 50:
                # Normalize variance by range to make comparable
                rng = col.quantile(0.95) - col.quantile(0.05)
                variances[f] = col.std() / rng if rng > 0 else 0
        
        # Sort by normalized variance, take top 2x target
        candidates = sorted(variances.keys(), key=lambda x: variances[x], reverse=True)
        candidates = candidates[:self.max_pairwise * 2]
        
        # Prune: remove features with |correlation| > 0.9 with a higher-ranked feature
        selected = []
        for feat in candidates:
            if len(selected) >= self.max_pairwise:
                break
            
            is_redundant = False
            for existing in selected:
                try:
                    corr = df[[feat, existing]].dropna().corr().iloc[0, 1]
                    if abs(corr) > 0.9:
                        is_redundant = True
                        break
                except Exception:
                    continue
            
            if not is_redundant:
                selected.append(feat)
        
        return selected
    
    def add_tree_hypotheses(self, tree_splits: List[dict]):
        """
        Add Level 4 hypotheses extracted from a fitted decision tree
        on calibration residuals.
        
        Called by the testing pipeline after tree fitting.
        
        tree_splits: list of dicts with keys:
            features, condition, target, n_samples, mean_residual
        """
        for split in tree_splits:
            if split['n_samples'] < 50:  # minimum leaf size
                continue
            
            self.hypotheses.append(Hypothesis(
                hypothesis_id=self._next_id(),
                level=4,
                features=split['features'],
                operator='tree_partition',
                target=split['target'],
                condition=split['condition'],
                description=f"Tree-discovered subgroup: {split['condition']}"
            ))
    
    def summary(self) -> Dict[str, int]:
        counts = {}
        for h in self.hypotheses:
            key = f"level_{h.level}_{h.operator}"
            counts[key] = counts.get(key, 0) + 1
        return counts
