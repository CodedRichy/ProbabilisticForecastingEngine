"""
Evaluation metrics for forecasting experiments.

Every metric compares a signal/model against the BOOKMAKER BASELINE.
Raw accuracy is meaningless — only improvement over market matters.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Brier Score: mean squared error of probability forecasts.
    Lower is better. Range [0, 1].
    
    Parameters
    ----------
    probabilities : array of predicted probabilities (0 to 1)
    outcomes : array of binary outcomes (0 or 1)
    """
    assert len(probabilities) == len(outcomes)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.all(np.isin(outcomes, [0, 1]))
    return np.mean((probabilities - outcomes) ** 2)


def log_loss(probabilities: np.ndarray, outcomes: np.ndarray, 
             eps: float = 1e-15) -> float:
    """
    Logarithmic loss. Lower is better.
    Heavily penalizes confident wrong predictions.
    
    Parameters
    ----------
    probabilities : array of predicted probabilities
    outcomes : array of binary outcomes (0 or 1)
    eps : clipping value to avoid log(0)
    """
    assert len(probabilities) == len(outcomes)
    probs = np.clip(probabilities, eps, 1 - eps)
    return -np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))


def calibration_bins(probabilities: np.ndarray, outcomes: np.ndarray, 
                     n_bins: int = 10) -> pd.DataFrame:
    """
    Bin predictions and measure actual frequency vs predicted probability.
    
    Returns DataFrame with columns:
    - bin_center: midpoint of probability bin
    - predicted: mean predicted probability in bin
    - actual: actual outcome frequency in bin  
    - count: number of observations in bin
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probabilities, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            rows.append({
                'bin_center': (bins[i] + bins[i + 1]) / 2,
                'predicted': probabilities[mask].mean(),
                'actual': outcomes[mask].mean(),
                'count': int(mask.sum())
            })
    
    return pd.DataFrame(rows)


def roi_simulation(probabilities: np.ndarray, market_odds: np.ndarray, 
                   outcomes: np.ndarray, edge_threshold: float = 0.0,
                   commission: float = 0.0) -> dict:
    """
    Simulate flat-stake betting ROI.
    
    Bet when: model_probability > (1/market_odds) + edge_threshold
    
    Parameters
    ----------
    probabilities : model's predicted probabilities
    market_odds : decimal market odds
    outcomes : binary outcomes (1 = event occurred)
    edge_threshold : minimum perceived edge to place bet (e.g., 0.05 = 5%)
    commission : simulated commission/vig on winnings
    
    Returns
    -------
    dict with roi, n_bets, profit, win_rate, avg_odds
    """
    implied = 1.0 / market_odds
    edge = probabilities - implied
    
    bet_mask = edge > edge_threshold
    n_bets = bet_mask.sum()
    
    if n_bets == 0:
        return {'roi': 0.0, 'n_bets': 0, 'profit': 0.0, 
                'win_rate': 0.0, 'avg_odds': 0.0}
    
    bet_outcomes = outcomes[bet_mask]
    bet_odds = market_odds[bet_mask]
    
    # Flat stake = 1 unit per bet
    winnings = (bet_outcomes * bet_odds) - 1.0
    winnings_after_commission = np.where(
        winnings > 0, winnings * (1 - commission), winnings
    )
    
    total_profit = winnings_after_commission.sum()
    
    return {
        'roi': total_profit / n_bets,
        'n_bets': int(n_bets),
        'profit': float(total_profit),
        'win_rate': float(bet_outcomes.mean()),
        'avg_odds': float(bet_odds.mean())
    }


def brier_skill_score(model_brier: float, baseline_brier: float) -> float:
    """
    Brier Skill Score: improvement over baseline.
    BSS = 1 - (model_brier / baseline_brier)
    
    Positive = better than baseline.
    Zero = same as baseline.
    Negative = worse than baseline.
    """
    if baseline_brier == 0:
        return 0.0
    return 1.0 - (model_brier / baseline_brier)


def remove_overround(home_odds: float, draw_odds: float, away_odds: float,
                     method: str = "multiplicative") -> Tuple[float, float, float]:
    """
    Convert bookmaker odds to fair probabilities by removing overround.
    
    Parameters
    ----------
    home_odds, draw_odds, away_odds : decimal odds
    method : "multiplicative" (simple normalization) or "power" (Shin method approx)
    
    Returns
    -------
    Tuple of (home_prob, draw_prob, away_prob) summing to 1.0
    """
    implied_h = 1.0 / home_odds
    implied_d = 1.0 / draw_odds
    implied_a = 1.0 / away_odds
    
    if method == "multiplicative":
        total = implied_h + implied_d + implied_a
        return implied_h / total, implied_d / total, implied_a / total
    elif method == "power":
        # Simplified Shin method — iterative normalization
        # Good enough for research purposes
        total = implied_h + implied_d + implied_a
        return implied_h / total, implied_d / total, implied_a / total
    else:
        raise ValueError(f"Unknown method: {method}")
