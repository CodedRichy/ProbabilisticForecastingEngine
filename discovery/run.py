"""
Apollo Discovery Runner — Main Orchestrator

Runs the full automated discovery pipeline:
    1. Compute features from raw data
    2. Generate hypotheses from feature space
    3. Test all hypotheses (Gate 1)
    4. Apply multiple testing correction
    5. Promote survivors to Gate 2, 3, 4
    6. Record everything

Usage:
    cd apollo/
    python -m discovery.run --league EPL --batch my_run_001
    python -m discovery.run --league EPL --gate1-only
    python -m discovery.run --summary
"""

import argparse
import sys
import os
import uuid
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from discovery.feature_factory import FeatureFactory
from discovery.generators.hypothesis_generator import HypothesisGenerator
from discovery.pipeline.testing import TestingPipeline
from discovery.pipeline.multiple_testing import MultipleTestingController
from discovery.tracking.tracker import ExperimentTracker


def run_discovery(league: str = 'EPL',
                  batch_id: str = None,
                  gate1_only: bool = False,
                  max_pairwise: int = 60,
                  parallel: bool = True):
    """
    Full discovery pipeline execution.
    """
    batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db_path = "discovery/tracking/apollo.db"
    
    print(f"\n{'='*60}")
    print(f"  APOLLO DISCOVERY ENGINE")
    print(f"  Batch: {batch_id}")
    print(f"  League: {league}")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")
    
    tracker = ExperimentTracker(db_path)
    
    # ── STEP 1: Load Data ─────────────────────────────────────
    
    print("[1/6] Loading processed data...")
    data_path = Path("data/processed/matches.parquet")
    
    if not data_path.exists():
        print("  ERROR: No processed data found.")
        print("  Run: python -m core.data_loader download")
        print("  Then: python -m core.data_loader process")
        return
    
    df_all = pd.read_parquet(data_path)
    print(f"  Total matches: {len(df_all)}")
    print(f"  Leagues: {df_all['league'].unique().tolist()}")
    
    # Primary league for discovery
    df_primary = df_all[df_all['league'] == league].copy()
    
    # Exclude holdout seasons
    holdout = ['2023-24', '2024-25']
    df_train = df_primary[~df_primary['season'].isin(holdout)].copy()
    df_holdout = df_primary[df_primary['season'].isin(holdout)].copy()
    
    print(f"  {league} training matches: {len(df_train)}")
    print(f"  {league} holdout matches: {len(df_holdout)}")
    
    # ── STEP 2: Compute Features ──────────────────────────────
    
    print("\n[2/6] Computing features...")
    factory = FeatureFactory()
    df_featured = factory.compute_all(df_train)
    
    feature_names = factory.get_base_feature_names(df_featured)
    print(f"  Features computed: {len(feature_names)}")
    
    # ── STEP 3: Generate Hypotheses ───────────────────────────
    
    print("\n[3/6] Generating hypotheses...")
    generator = HypothesisGenerator(max_pairwise_features=max_pairwise)
    hypotheses = generator.generate_all(feature_names, df_featured)
    
    # Record all hypotheses
    tracker.record_hypotheses(hypotheses, batch_id)
    
    # Deduplicate against previously tested
    new_hypotheses = []
    for h in hypotheses:
        if not tracker.hypothesis_already_tested(h.config_hash()):
            new_hypotheses.append(h)
    
    print(f"  New (untested) hypotheses: {len(new_hypotheses)}/{len(hypotheses)}")
    
    if not new_hypotheses:
        print("  All hypotheses already tested. Nothing to do.")
        print(tracker.summary())
        return
    
    # ── STEP 4: Gate 1 — Statistical Testing ──────────────────
    
    print("\n[4/6] Gate 1 — Testing hypotheses against market baseline...")
    pipeline = TestingPipeline(n_workers=None)
    
    gate1_results = pipeline.run_batch(new_hypotheses, df_featured, parallel=parallel)
    
    # Record all test runs
    for h, result in zip(new_hypotheses, gate1_results):
        run_id = f"{batch_id}_{h.hypothesis_id}_g1"
        result['passed'] = result.get('p_value', 1.0) < 0.05  # raw, pre-correction
        tracker.record_test_run(
            run_id=run_id,
            hypothesis_id=h.hypothesis_id,
            gate=1,
            result=result,
            league=league,
        )
    
    # ── STEP 5: Multiple Testing Correction ───────────────────
    
    print("\n[5/6] Applying multiple testing correction...")
    controller = MultipleTestingController(db_path)
    
    raw_p_values = [r.get('p_value', 1.0) for r in gate1_results]
    hypothesis_ids = [h.hypothesis_id for h in new_hypotheses]
    
    corrected = controller.correct(
        raw_p_values=raw_p_values,
        hypothesis_ids=hypothesis_ids,
        batch_id=batch_id,
    )
    
    # Update test records with corrected values
    survivors = []
    for h, raw_result, corr_result in zip(new_hypotheses, gate1_results, corrected):
        if corr_result['passed_all']:
            survivors.append((h, raw_result, corr_result))
    
    print(f"\n  Gate 1 survivors (after correction): {len(survivors)}/{len(new_hypotheses)}")
    
    if survivors:
        print("\n  Surviving hypotheses:")
        for h, raw, corr in survivors:
            print(f"    {h.hypothesis_id}: {h.description[:70]}...")
            print(f"      p_raw={raw.get('p_value', '?'):.6f}  "
                  f"p_BY={corr['p_value_by']:.6f}  "
                  f"BF={corr['bayes_factor_min']:.4f}")
    
    if gate1_only:
        print("\n  (Gate 1 only mode — stopping here)")
        print(f"\n{tracker.summary()}")
        return
    
    # ── STEP 6: Gates 2-4 ────────────────────────────────────
    
    print(f"\n[6/6] Running validation cascade on {len(survivors)} survivors...")
    
    # Gate 2, 3, 4 require signal-specific prediction functions.
    # For the automated pipeline, we convert each surviving hypothesis
    # into a simple conditional adjustment model.
    
    validated_count = 0
    
    for h, raw_result, corr_result in survivors:
        print(f"\n  Validating {h.hypothesis_id}...")
        
        # Build a signal function from the hypothesis
        # (This is the bridge between abstract hypothesis and concrete prediction)
        signal_func = _build_signal_function(h, df_featured)
        
        if signal_func is None:
            print(f"    Could not build signal function. Skipping.")
            continue
        
        # Gate 2: Walk-forward
        # (Simplified — full implementation would use the cascade module)
        print(f"    Gate 2: Walk-forward OOS...")
        # Record as attempted
        tracker.record_test_run(
            run_id=f"{batch_id}_{h.hypothesis_id}_g2",
            hypothesis_id=h.hypothesis_id,
            gate=2,
            result={'passed': False, 'reason': 'requires_signal_specific_implementation'},
            league=league,
        )
        
        # NOTE: Gates 2-4 require per-hypothesis signal functions that
        # produce adjusted probabilities. This is where the automated
        # system meets the need for experiment-specific logic.
        #
        # The architecture supports two approaches:
        # 1. Generic: Use the residual direction to adjust market probabilities
        # 2. Specific: Manually implement for promising Gate 1 survivors
        #
        # For now, Gate 1 survivors are flagged for manual Gate 2+ review.
        
        print(f"    → Flagged for manual Gate 2-4 validation")
    
    # ── SUMMARY ───────────────────────────────────────────────
    
    print(f"\n{'='*60}")
    print(f"  DISCOVERY RUN COMPLETE")
    print(f"{'='*60}")
    print(f"\n{tracker.summary()}")


def _build_signal_function(hypothesis, df):
    """
    Attempt to build a generic signal function from a hypothesis.
    
    For Level 1 (correlation) and Level 2 (conditional calibration),
    we can build automatic adjustment functions. For Level 3+,
    this requires more complex logic.
    
    Returns None if automatic construction isn't feasible.
    """
    # Placeholder — this is where the system transitions from
    # fully automated discovery to semi-automated validation.
    # The discovery identifies WHAT is worth testing;
    # the researcher implements HOW to turn it into a prediction.
    return None


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Apollo Discovery Engine')
    parser.add_argument('--league', default='EPL', help='Primary league for discovery')
    parser.add_argument('--batch', default=None, help='Batch identifier')
    parser.add_argument('--gate1-only', action='store_true', help='Run Gate 1 only')
    parser.add_argument('--max-pairwise', type=int, default=60, help='Max features for pairwise')
    parser.add_argument('--no-parallel', action='store_true', help='Disable multiprocessing')
    parser.add_argument('--summary', action='store_true', help='Print summary and exit')
    
    args = parser.parse_args()
    
    if args.summary:
        tracker = ExperimentTracker("discovery/tracking/apollo.db")
        print(tracker.summary())
        return
    
    run_discovery(
        league=args.league,
        batch_id=args.batch,
        gate1_only=args.gate1_only,
        max_pairwise=args.max_pairwise,
        parallel=not args.no_parallel,
    )


if __name__ == "__main__":
    main()
