"""
calibrate_conformal.py

Build and persist a ConformalFilter calibrated on the chronologically last
20% of matches_xg.parquet.

The split is intentionally time-ordered (not random) to avoid lookahead bias:
the calibration set represents future data relative to the training window.

Usage
-----
    python scripts/calibrate_conformal.py

Output
------
    data/models/conformal_filter.pkl

Prints
------
    - Calibration set size
    - Nonconformity score quantile threshold
    - % of calibration matches where model is confident (prediction set size == 1)
      at the configured significance level (0.15)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from core.xgb_predictor import XGBPredictor, TARGET_MAP
from core.conformal import ConformalFilter

PARQUET = str(Path(__file__).parent.parent / "data" / "processed" / "matches_xg.parquet")
XGB_MODEL = str(Path(__file__).parent.parent / "data" / "models" / "xgb_predictor.pkl")
CONFORMAL_OUT = str(Path(__file__).parent.parent / "data" / "models" / "conformal_filter.pkl")

SIGNIFICANCE = 0.15
CAL_FRACTION = 0.20   # chronologically last 20% used for calibration


def main() -> None:
    # ── 1. Load raw data ──────────────────────────────────────────────────────
    print(f"Loading data from {PARQUET} ...")
    df_raw = pd.read_parquet(PARQUET)
    print(f"  Total rows: {len(df_raw):,}")

    # ── 2. Load trained XGBoost model ─────────────────────────────────────────
    print(f"\nLoading XGBoost model from {XGB_MODEL} ...")
    xgb = XGBPredictor.load(XGB_MODEL)
    print(f"  Feature cols ({len(xgb._feature_cols)}): {xgb._feature_cols}")

    # ── 3. Build features exactly as in training ──────────────────────────────
    print("\nBuilding features via FeatureFactory ...")
    from discovery.feature_factory import FeatureFactory
    ff = FeatureFactory()
    df = ff.compute_all(df_raw)

    # ── 4. Chronological calibration split (last 20%) ─────────────────────────
    # The data is assumed to be sorted chronologically (it comes from train_xgb.py
    # which also sorts by row order, matching the parquet on-disk order which is
    # date-sorted).  We mirror the exact same 80/20 boundary used in XGBPredictor.fit()
    # so the calibration set is strictly after the training set.
    split_idx = int(len(df) * (1.0 - CAL_FRACTION))
    df_cal = df.iloc[split_idx:].copy()

    print(f"\nChronological split:")
    print(f"  Training set rows  : 0 – {split_idx - 1}  ({split_idx:,} rows)")
    print(f"  Calibration set rows: {split_idx} – {len(df) - 1}  ({len(df_cal):,} rows)")

    # ── 5. Prepare X_cal / y_cal ──────────────────────────────────────────────
    y_raw = df_cal["result"].map(TARGET_MAP)
    valid = y_raw.notna()
    df_cal = df_cal[valid]
    y_cal = y_raw[valid].astype(int)

    X_cal = df_cal[xgb._feature_cols].fillna(xgb._medians)

    print(f"\nCalibration set after dropping rows with missing result:")
    print(f"  Rows : {len(X_cal):,}")
    print(f"  Class distribution: H={( y_cal == 0).sum()}, D={(y_cal == 1).sum()}, A={(y_cal == 2).sum()}")

    # ── 6. Calibrate ──────────────────────────────────────────────────────────
    print(f"\nCalibrating ConformalFilter at significance={SIGNIFICANCE} ...")
    cf = ConformalFilter(significance=SIGNIFICANCE)
    cf.calibrate(xgb, X_cal, y_cal)

    print(f"\n  Calibration set size : {cf.n_cal:,}")
    print(f"  Nonconformity quantile threshold : {cf.threshold:.6f}")
    print(f"  Equivalent min model-prob to enter set : {1 - cf.threshold:.6f}")

    # ── 7. Analyse pass rate ──────────────────────────────────────────────────
    probs_matrix = xgb._model.predict_proba(X_cal)   # (n, 3)
    conf_flags = np.array([
        cf.is_confident({
            "p_home": float(row[0]),
            "p_draw": float(row[1]),
            "p_away": float(row[2]),
        })
        for row in probs_matrix
    ])
    set_sizes = np.array([
        cf.prediction_set_size({
            "p_home": float(row[0]),
            "p_draw": float(row[1]),
            "p_away": float(row[2]),
        })
        for row in probs_matrix
    ])

    n_total = len(conf_flags)
    n_confident = conf_flags.sum()
    pct_confident = n_confident / n_total * 100

    print(f"\nPass-rate analysis at significance={SIGNIFICANCE}:")
    print(f"  Prediction set size distribution:")
    for sz in [1, 2, 3]:
        cnt = (set_sizes == sz).sum()
        print(f"    size={sz} : {cnt:>5,}  ({cnt/n_total*100:.1f}%)")
    print(f"\n  Matches that PASS filter (size=1) : {n_confident:,} / {n_total:,}  ({pct_confident:.1f}%)")
    print(f"  Matches that FAIL filter (size>1) : {n_total - n_confident:,} / {n_total:,}  ({100 - pct_confident:.1f}%)")

    # ── 8. Save ───────────────────────────────────────────────────────────────
    cf.save(CONFORMAL_OUT)
    print(f"\nConformal filter saved to: {CONFORMAL_OUT}")
    print(f"\nDone. Run apollo.py (streamlit) — the Value Bets page will now load")
    print(f"the filter and skip bets where the model is uncertain.")


if __name__ == "__main__":
    main()
