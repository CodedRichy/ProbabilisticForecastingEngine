"""
Calibrated XGBoost predictor for match outcome probabilities.
Produces P(H), P(D), P(A) suitable for betting edge calculations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.model_selection import PredefinedSplit
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        f"Required dependency missing: {e}. "
        "Install with: pip install xgboost scikit-learn joblib"
    ) from e

sys.path.insert(0, str(Path(__file__).parent.parent))
from discovery.feature_factory import FeatureFactory


FEATURE_COLS = [
    # ── Elo (always available at inference time) ──────────────────────────────
    # Computed from EloModel trained on international results (k=32).
    "home_elo_k32", "away_elo_k32", "elo_delta_k32", "elo_expected_home_k32",

    # ── Form — rolling 5-match window ────────────────────────────────────────
    # Available at inference time ONLY for teams in the club-league historical
    # parquet (EPL, Championship, Bundesliga, LaLiga, SerieA, Ligue1).
    # International fixtures (WC2026) will have NaN → training-median imputation.
    # Fix: build a parallel form index from an international results dataset
    # (e.g. martj42/international_results on GitHub).
    "home_form_winrate_5", "away_form_winrate_5",
    "home_form_ppg_5", "away_form_ppg_5",

    # ── Form — EWMA (span=5, adjust=False) ───────────────────────────────────
    # Exponentially weighted form features; more responsive to recent matches
    # than the simple rolling window. Computed from all past matches (no minimum
    # window required — available after the team's very first match).
    "home_form_ewma_ppg", "away_form_ewma_ppg",
    "home_form_ewma_gd", "away_form_ewma_gd",

    # ── Goals — rolling 5-match window ───────────────────────────────────────
    # Same availability caveat as form features above.
    "home_form_gf_mean_5", "away_form_gf_mean_5",
    "home_form_ga_mean_5", "away_form_ga_mean_5",

    # ── Poisson expected goals ────────────────────────────────────────────────
    # Training: computed by FeatureFactory._goals_model_features() from rolling
    # attack/defence ratings (requires >=10 matches of history per team).
    # Inference: approximated via Dixon-Coles from elo_expected_home_k32
    # (see _poisson_xg_from_elo() in scripts/predict_today.py). Always computable
    # and better than training-median when Elo differential is large.
    "poisson_home_xg", "poisson_away_xg",

    # ── Real xG from Understat ────────────────────────────────────────────────
    # TRAIN: ~51% null (Understat covers selected leagues from ~2014 onward).
    # Excluded from the STORED MODEL at training time (null rate > NAN_THRESHOLD=30%).
    # Does not appear in model.feature_cols, so never passed at inference.
    # To unlock: source a real-time xG feed (StatsBomb open data, Opta, etc.).
    "home_xg", "away_xg",

    # ── Market implied probabilities ──────────────────────────────────────────
    # TRAIN/TEST MISMATCH: Model trained on CLOSING odds stored in the historical
    # parquet (home_implied/draw_implied/away_implied, sourced from bet365/Pinnacle
    # via football-data.co.uk). At inference, predict_today.py supplies OPENING
    # odds from the live OddsFetcher. Closing odds are sharper (reflect late
    # information), so using opening odds at inference will underestimate the
    # true market edge signal. Mitigation: retrain with opening-odds columns,
    # or apply a shrinkage correction (e.g. pull mkt_* implied probs toward 1/3).
    "mkt_home_implied", "mkt_draw_implied", "mkt_away_implied",
]

TARGET_MAP = {"H": 0, "D": 1, "A": 2}
TARGET_INVERSE = {0: "p_home", 1: "p_draw", 2: "p_away"}
NAN_THRESHOLD = 0.30


class XGBPredictor:
    """
    Calibrated multiclass XGBoost predictor for H/D/A outcomes.

    Training uses a strict chronological 80/20 split to prevent leakage.
    Probabilities are isotonic-calibrated for betting use.
    """

    def __init__(self) -> None:
        self._model: CalibratedClassifierCV | None = None
        self._feature_cols: list[str] = []
        self._medians: pd.Series | None = None
        self._label_encoder = LabelEncoder()

    # ── Public API ────────────────────────────────────────────────────────

    def fit(
        self,
        parquet_path: str = str(Path(__file__).parent.parent / "data" / "processed" / "matches_xg.parquet"),
    ) -> "XGBPredictor":
        df = pd.read_parquet(parquet_path)

        ff = FeatureFactory()
        df = ff.compute_all(df)

        candidates = [c for c in FEATURE_COLS if c in df.columns]
        null_rates = df[candidates].isna().mean()
        feature_cols = null_rates[null_rates <= NAN_THRESHOLD].index.tolist()

        self._medians = df[feature_cols].median()
        X = df[feature_cols].fillna(self._medians)
        y_raw = df["result"].map(TARGET_MAP)

        valid_mask = y_raw.notna()
        X = X[valid_mask].reset_index(drop=True)
        y = y_raw[valid_mask].astype(int).reset_index(drop=True)

        split = int(len(X) * 0.80)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        # Step 1: fit base XGB with early stopping to find optimal n_estimators
        _base_probe = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
            random_state=42,
        )
        _base_probe.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_n = _base_probe.best_iteration + 1

        # Step 2: calibrate with isotonic regression on val fold via PredefinedSplit.
        # CalibratedClassifierCV trains the base on the -1 fold and calibrates on the 0 fold.
        X_all = pd.concat([X_train, X_val], ignore_index=True)
        y_all = pd.concat([y_train, y_val], ignore_index=True)
        test_fold = np.concatenate([
            -np.ones(len(X_train), dtype=int),
            np.zeros(len(X_val), dtype=int),
        ])
        ps = PredefinedSplit(test_fold)

        base = XGBClassifier(
            n_estimators=best_n,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
        )

        self._model = CalibratedClassifierCV(
            estimator=base,
            method="isotonic",
            cv=ps,
        )
        self._model.fit(X_all, y_all)
        self._feature_cols = feature_cols

        self._eval_metrics = self._evaluate(X_val, y_val)
        return self

    def predict(self, features: dict) -> dict:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        # Coerce None → NaN so XGBoost receives float columns.
        # features.get() may return None when a caller passes None explicitly
        # (e.g. market features when odds are unavailable); NaN is then imputed
        # with the training-set median inside the fill step below.
        def _val(col: str) -> float:
            v = features.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return float(self._medians[col])
            return float(v)

        row = {col: _val(col) for col in self._feature_cols}
        X = pd.DataFrame([row])
        probs = self._model.predict_proba(X)[0]
        return {
            "p_home": float(probs[0]),
            "p_draw": float(probs[1]),
            "p_away": float(probs[2]),
        }

    def save(self, path: str = str(Path(__file__).parent.parent / "data" / "models" / "xgb_predictor.pkl")) -> None:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "feature_cols": self._feature_cols,
                "medians": self._medians,
                "eval_metrics": getattr(self, "_eval_metrics", {}),
            },
            out,
        )

    @classmethod
    def load(cls, path: str = str(Path(__file__).parent.parent / "data" / "models" / "xgb_predictor.pkl")) -> "XGBPredictor":
        data = joblib.load(path)
        obj = cls()
        obj._model = data["model"]
        obj._feature_cols = data["feature_cols"]
        obj._medians = data["medians"]
        obj._eval_metrics = data.get("eval_metrics", {})
        return obj

    @property
    def feature_importance(self) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        base = self._model.calibrated_classifiers_[0].estimator
        scores = base.feature_importances_
        return (
            pd.Series(scores, index=self._feature_cols)
            .sort_values(ascending=False)
        )

    # ── Internal ──────────────────────────────────────────────────────────

    def _evaluate(self, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        probs = self._model.predict_proba(X_val)
        ll = log_loss(y_val, probs)

        y_arr = np.array(y_val)
        brier = {}
        for cls_idx, name in TARGET_INVERSE.items():
            y_bin = (y_arr == cls_idx).astype(int)
            brier[name] = brier_score_loss(y_bin, probs[:, cls_idx])

        return {"log_loss": ll, "brier": brier}
