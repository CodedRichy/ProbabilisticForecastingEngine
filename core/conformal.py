"""
conformal.py

Conformal prediction filter for XGBoost match outcome probabilities.

Calibrate on a holdout set, then at inference time decide whether the model
has genuine conviction by counting how many outcomes fall inside the prediction
set at the chosen significance level.

Usage
-----
    from core.conformal import ConformalFilter

    cf = ConformalFilter(significance=0.15)
    cf.calibrate(xgb_model, X_cal, y_cal)
    cf.save("data/models/conformal_filter.pkl")

    # At inference time
    probs = {"p_home": 0.62, "p_draw": 0.22, "p_away": 0.16}
    if cf.is_confident(probs):
        # model has a single-outcome prediction set — allow the bet
        ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError as exc:
    raise ImportError(
        "joblib is required. Install with: pip install joblib"
    ) from exc


class ConformalFilter:
    """
    Inductive conformal predictor for 3-class (H/D/A) football outcomes.

    The nonconformity score for a match is:

        alpha_i = 1 - p_true_outcome_i

    where p_true_outcome is the probability the model assigned to the outcome
    that actually happened.  Lower score = more conforming = model was right.

    At inference time, a candidate outcome y is *inside* the prediction set
    at significance level alpha if:

        (1 - p_y) <= quantile_{ceil((1-alpha)(n+1)/n)}(calibration scores)

    Equivalently: p_y >= (1 - threshold).

    The prediction set size tells us how many of the three outcomes the model
    cannot rule out:
      - size 1  → confident, allow bet
      - size 2+ → uncertain, skip bet
    """

    def __init__(self, significance: float = 0.15) -> None:
        if not 0.0 < significance < 1.0:
            raise ValueError("significance must be in (0, 1)")
        self.significance = significance
        self._threshold: float | None = None
        self._n_cal: int = 0
        self._cal_scores: np.ndarray | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def calibrate(
        self,
        model,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> None:
        """
        Compute nonconformity scores on the calibration set and store the
        (1 - significance) quantile as the decision threshold.

        Parameters
        ----------
        model : XGBPredictor (or any object with predict_proba on X_cal)
            The fitted model.  We call model._model.predict_proba() directly
            so we can batch the whole calibration set efficiently.
        X_cal : pd.DataFrame
            Feature matrix for calibration rows, already median-imputed and
            column-filtered to model._feature_cols.
        y_cal : pd.Series
            Integer class labels (0=H, 1=D, 2=A) aligned with X_cal.
        """
        probs = model._model.predict_proba(X_cal)   # shape (n, 3)
        y_arr = np.asarray(y_cal, dtype=int)

        # Nonconformity score = 1 - probability assigned to true outcome
        true_probs = probs[np.arange(len(y_arr)), y_arr]
        scores = 1.0 - true_probs

        self._cal_scores = scores
        self._n_cal = len(scores)

        # Conformal quantile: use the ceil((1-alpha)(n+1)/n) empirical quantile.
        # np.quantile with interpolation='higher' gives a conservative upper bound
        # which ensures valid coverage at significance level alpha.
        level = np.ceil((1.0 - self.significance) * (self._n_cal + 1)) / self._n_cal
        level = min(level, 1.0)   # cap at 1.0 in case of tiny calibration sets
        self._threshold = float(np.quantile(scores, level, method="higher"))

    def prediction_set_size(self, probs: dict) -> int:
        """
        Return the number of outcomes in the conformal prediction set.

        An outcome y is in the prediction set if its nonconformity score
        (1 - p_y) does NOT exceed the calibration threshold, i.e. p_y is
        large enough that the model is not surprised by that outcome.

        Parameters
        ----------
        probs : dict
            Keys: "p_home", "p_draw", "p_away" — model probabilities summing to 1.

        Returns
        -------
        int : 1, 2, or 3
        """
        if self._threshold is None:
            raise RuntimeError("ConformalFilter not calibrated. Call calibrate() first.")

        count = 0
        for key in ("p_home", "p_draw", "p_away"):
            p = probs.get(key, 0.0)
            # outcome is in the set when its nonconformity score <= threshold
            if (1.0 - p) <= self._threshold:
                count += 1
        return max(count, 1)   # always at least 1 to avoid division-by-zero callers

    def is_confident(self, probs: dict) -> bool:
        """
        Return True if the prediction set contains exactly one outcome.

        This is the primary gate for allowing bets.  A single-outcome
        prediction set means the model is confident enough (at significance
        level self.significance) to rule out the other two outcomes.

        Parameters
        ----------
        probs : dict
            Keys: "p_home", "p_draw", "p_away".
        """
        return self.prediction_set_size(probs) == 1

    def save(self, path: str) -> None:
        """Persist the calibrated filter to disk via joblib."""
        if self._threshold is None:
            raise RuntimeError("ConformalFilter not calibrated. Call calibrate() first.")
        import pathlib
        out = pathlib.Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "significance": self.significance,
                "threshold": self._threshold,
                "n_cal": self._n_cal,
                "cal_scores": self._cal_scores,
            },
            out,
        )

    @classmethod
    def load(cls, path: str) -> "ConformalFilter":
        """Load a previously saved ConformalFilter from disk."""
        data = joblib.load(path)
        obj = cls(significance=data["significance"])
        obj._threshold = data["threshold"]
        obj._n_cal = data["n_cal"]
        obj._cal_scores = data.get("cal_scores")
        return obj

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def threshold(self) -> float | None:
        """The stored nonconformity score quantile (decision boundary)."""
        return self._threshold

    @property
    def n_cal(self) -> int:
        """Number of calibration samples used."""
        return self._n_cal

    def __repr__(self) -> str:
        status = (
            f"threshold={self._threshold:.4f}, n_cal={self._n_cal}"
            if self._threshold is not None
            else "not calibrated"
        )
        return f"ConformalFilter(significance={self.significance}, {status})"
