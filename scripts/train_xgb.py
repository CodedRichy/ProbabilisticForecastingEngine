import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.calibration import calibration_curve

from core.xgb_predictor import XGBPredictor

PARQUET = "data/processed/matches_xg.parquet"
MODEL_OUT = "data/models/xgb_predictor.pkl"


def main() -> None:
    print("Training XGBPredictor …")
    predictor = XGBPredictor()
    predictor.fit(parquet_path=PARQUET)

    metrics = predictor._eval_metrics
    print(f"\nVal log-loss : {metrics['log_loss']:.4f}")
    print("Val Brier scores:")
    for outcome, score in metrics["brier"].items():
        print(f"  {outcome:<8}: {score:.4f}")

    print("\nTop 10 features by importance:")
    top10 = predictor.feature_importance.head(10)
    for feat, imp in top10.items():
        print(f"  {feat:<35} {imp:.4f}")

    import numpy as np
    import pandas as pd
    from discovery.feature_factory import FeatureFactory
    from core.xgb_predictor import TARGET_MAP

    df = pd.read_parquet(PARQUET)
    ff = FeatureFactory()
    df = ff.compute_all(df)
    df_val = df.iloc[int(len(df) * 0.80):].copy()
    y_val = df_val["result"].map(TARGET_MAP).dropna().astype(int)
    X_val = df_val.loc[y_val.index, predictor._feature_cols].fillna(predictor._medians)
    probs = predictor._model.predict_proba(X_val)

    print("\nCalibration check (fraction of positives vs mean predicted prob):")
    outcome_labels = {0: "home", 1: "draw", 2: "away"}
    for cls_idx, label in outcome_labels.items():
        y_bin = (np.array(y_val) == cls_idx).astype(int)
        frac_pos, mean_pred = calibration_curve(y_bin, probs[:, cls_idx], n_bins=5)
        print(f"  {label}:")
        for fp, mp in zip(frac_pos, mean_pred):
            print(f"    pred={mp:.2f}  actual={fp:.2f}")

    predictor.save(MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
