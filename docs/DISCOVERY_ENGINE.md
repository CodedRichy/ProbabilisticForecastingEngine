# Apollo Discovery Engine — Automated Hypothesis Generation & Validation

## The Problem With Manual Hypotheses

The 25-hypothesis catalog in HYPOTHESES.md has a fatal flaw: a human chose them. That means researcher degrees of freedom contaminate the process before a single test runs. The researcher selects hypotheses they find plausible, ignoring thousands of interactions they can't imagine. This introduces:

- **Selection bias**: We test what we expect to work.
- **Narrative bias**: We frame tests around stories, not data geometry.
- **Dimensionality blindness**: Humans can't reason about 3-way or 4-way feature interactions.
- **Anchoring**: Literature-driven hypotheses cluster around known effects, missing novel ones.

The solution: enumerate the hypothesis space programmatically, test exhaustively, and let statistical rigor — not human intuition — determine what survives.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     APOLLO DISCOVERY ENGINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   FEATURE    │───▶│  HYPOTHESIS  │───▶│   TESTING    │      │
│  │  FACTORY     │    │  GENERATOR   │    │   PIPELINE   │      │
│  │              │    │              │    │              │      │
│  │ Raw data  ──▶│    │ Enumerate    │    │ Walk-forward │      │
│  │ Compute 500+ │    │ all testable │    │ backtest per │      │
│  │ features     │    │ combinations │    │ hypothesis   │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │              │
│                                                  ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   SIGNAL     │◀───│  VALIDATION  │◀───│  MULTIPLE    │      │
│  │  DATABASE    │    │  CASCADE     │    │  TESTING     │      │
│  │              │    │              │    │  CONTROLLER  │      │
│  │ Permanent    │    │ OOS ──▶      │    │              │      │
│  │ registry of  │    │ Cross-league │    │ FDR correct  │      │
│  │ all results  │    │ ──▶ Holdout  │    │ across ALL   │      │
│  └──────────────┘    └──────────────┘    │ tests ever   │      │
│                                          └──────────────┘      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  EXPERIMENT TRACKER                      │   │
│  │  SQLite database recording every test, every parameter, │   │
│  │  every result, every version. Nothing is ever deleted.   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Feature Factory

The Feature Factory transforms raw match data into a high-dimensional feature matrix. Every feature is computed using ONLY information available BEFORE the match kicks off.

### Feature Categories

**Category A — Form (rolling windows)**
For each team, compute over the last N matches (N ∈ {3, 5, 10, 20}):
- Win rate, draw rate, loss rate
- Goals scored mean/std
- Goals conceded mean/std
- Goal difference mean
- Points per game
- Clean sheet rate
- Scoring failure rate (% matches with 0 goals)

4 windows × ~12 metrics × 2 teams = ~96 features

**Category B — Elo / Strength**
- Elo rating (multiple K-factors: 20, 32, 48)
- Elo delta (home - away)
- Elo momentum (Elo change over last 5/10 matches)
- League-adjusted Elo (cross-league normalization)

~12 features

**Category C — Market**
- Home/draw/away implied probability (overround-removed)
- Implied probability rank within the matchday
- Favorite strength (max implied prob)
- Odds compactness (how close the three probabilities are)

~8 features

**Category D — Schedule / Context**
- Days since last match (home, away)
- Matches played in last 7/14/30 days
- Home/away sequence (nth consecutive home/away game)
- Season phase (match week / total weeks)
- Is derby (computed from historical H2H frequency)

~12 features

**Category E — Head-to-Head**
- H2H win rate (last 5/10 meetings)
- H2H goal difference
- H2H home advantage delta

~6 features

**Category F — Goals Model**
- Poisson attack/defense ratings
- Expected goals differential
- Over/under 2.5 implied probability

~8 features

**Category G — Derived / Interaction (auto-generated)**
- Ratios: feature_A / feature_B for selected pairs
- Differences: feature_A - feature_B
- Products: feature_A × feature_B (captures interactions)
- Threshold indicators: feature_A > percentile_75

This is where the combinatorial explosion lives. Managed by the Hypothesis Generator.

**Total base features: ~140**
**With interactions and transformations: 2,000–10,000 depending on depth setting**

### Data Leakage Prevention

Every feature computation follows one inviolable rule:

> **The feature for match M must be computable using ONLY data from matches completed BEFORE match M.**

This is enforced architecturally:
- Features are computed in chronological order via an expanding-window loop.
- The feature matrix is indexed by (match_id, computation_date).
- A `vintage` column records when each feature value was computed.
- The testing pipeline joins features to matches using `vintage < match_date`.

---

## 2. Hypothesis Generator

A "hypothesis" in this system is not a paragraph of English. It is a structured object:

```
Hypothesis:
    feature_set: [feature_1, feature_2, ...]    # 1-4 features
    operator: conditional_calibration | regression | threshold_signal
    target: home_win | draw | away_win | over_2.5 | ...
    baseline: market_implied_probability
    condition: optional filter (e.g., season_phase > 0.8)
    direction: optional expected sign
```

### Generation Strategy: Structured Enumeration

**Level 1 — Univariate (each feature alone)**
For each of the ~140 base features: does this feature predict calibration residuals?

The "calibration residual" is: actual_outcome - market_implied_probability.

If feature X correlates with the residual, then X contains information the market hasn't fully priced.

~140 hypotheses.

**Level 2 — Conditional Calibration**
For each feature, bin into quintiles. Does market calibration break down in specific quintiles?

Example: When `home_form_5` is in the top quintile AND `home_implied > 0.6`, do home teams underperform expectations?

~140 features × 5 quintiles = ~700 hypotheses.

**Level 3 — Pairwise Interactions**
For each pair of base features: does their interaction predict residuals?

This finds "when A is high AND B is low, the market mispredicts."

C(140, 2) = 9,730 pairs. Too many? No — this is where novel signals live. But it demands extreme multiple testing correction.

**Level 4 — Conditional Subgroups**
Recursive partitioning (decision tree on residuals) to find pockets of mispricing. The tree splits are hypotheses.

Limited to depth 3 to prevent overfitting: generates ~50-200 hypotheses depending on data.

**Total hypothesis space: ~11,000 at Level 1-3, plus ~200 from Level 4.**

### What We Do NOT Do

- We do NOT generate hypotheses by looking at results first. The enumeration is mechanical and complete — it does not depend on what "looks promising."
- We do NOT cherry-pick feature pairs. All pairs are tested.
- We do NOT use stepwise selection, which is statistically invalid.

---

## 3. Multiple Testing Controller

This is the most critical component. Testing 11,000 hypotheses at α = 0.05 guarantees ~550 false positives by chance alone. The controller must prevent this.

### Three-Layer Correction

**Layer 1: Benjamini-Yekutieli (BY) FDR Control**

Standard Benjamini-Hochberg assumes independence between tests. Our tests are correlated (many features share information). BY is valid under arbitrary dependence. Target FDR: 5%.

The cost: BY is conservative. Many true signals will be missed. This is acceptable — false negatives are cheap, false positives are catastrophic.

**Layer 2: Minimum Bayes Factor**

For any hypothesis that survives BY, compute the minimum Bayes Factor:

    BF_min = -e × p × ln(p)

If BF_min < 1/10 (strong evidence), proceed. This provides a second independent check that doesn't depend on frequentist assumptions.

**Layer 3: Deflated Sharpe Ratio (López de Prado)**

For any hypothesis that produces a tradeable signal (positive ROI), compute the Deflated Sharpe Ratio, which adjusts for:
- Number of trials (all hypotheses ever tested)
- Skewness and kurtosis of returns
- Variance of Sharpe ratio estimate

A positive DSR means the signal would survive even if it were the best out of N random strategies. This is the most severe test.

### The Trial Counter

A global counter records the TOTAL number of hypotheses ever tested across all runs. This counter only increases. It is used in BY correction and DSR computation. This prevents the "restart trick" — running the system repeatedly until something passes.

---

## 4. Validation Cascade

A hypothesis must pass FOUR sequential gates before entering the signal database as "validated."

```
Gate 1: Discovery (in-sample, single league)
    ↓ passes BY-corrected FDR
Gate 2: Out-of-Sample (walk-forward backtest, same league)
    ↓ p < 0.05 in walk-forward
Gate 3: Cross-Domain (at least 2 additional leagues)
    ↓ consistent sign and p < 0.10 in ≥2 other leagues
Gate 4: Holdout Confirmation (reserved seasons, never seen)
    ↓ p < 0.05 on holdout data
    
    ──▶ VALIDATED SIGNAL
```

### Gate 1: Discovery

Run on the training partition of the primary league (EPL recommended due to depth). Apply all multiple testing corrections. Anything that survives gets a provisional signal ID.

Survival rate expectation: ~0.5-2% of hypotheses (50-200 out of 11,000).

### Gate 2: Out-of-Sample Walk-Forward

For each survivor, run a proper walk-forward backtest:
- Expanding window, minimum 200 matches training.
- Predict one matchday at a time.
- Measure Brier Skill Score vs market baseline.
- Require p < 0.05 on Diebold-Mariano test.

Survival rate expectation: ~30-50% of Gate 1 survivors (15-100 signals).

### Gate 3: Cross-Domain

Test in La Liga, Serie A, Bundesliga, Ligue 1. A real signal should generalize. We relax significance to p < 0.10 per league but require consistency in at least 2 of 4 leagues:
- Same sign of effect
- p < 0.10 in each

Survival rate expectation: ~20-40% of Gate 2 survivors (3-40 signals).

### Gate 4: Holdout

The 2023-24 and 2024-25 seasons are NEVER touched until this stage. This is the final confirmation. Standard significance (p < 0.05) required.

A signal that fails holdout is REJECTED regardless of how strong it looked in earlier gates. No appeals. No "well, it was close." Dead.

Survival rate expectation: ~30-60% of Gate 3 survivors (1-25 signals).

### Expected Final Yield

From ~11,000 hypotheses: **1-25 validated signals.**

If zero survive, that is a legitimate and valuable finding: "the market is efficient at the resolution of our feature set."

---

## 5. Experiment Tracker (SQLite)

Every hypothesis test is recorded permanently.

### Schema

```sql
-- Every hypothesis ever generated
CREATE TABLE hypotheses (
    hypothesis_id    TEXT PRIMARY KEY,   -- H-00001, H-00002, ...
    created_at       TIMESTAMP,
    feature_set      TEXT,               -- JSON array of feature names
    operator         TEXT,               -- "conditional_calibration", "regression", etc.
    target           TEXT,               -- "home_win", "draw", etc.
    condition        TEXT,               -- JSON filter condition or NULL
    level            INTEGER,            -- 1=univariate, 2=conditional, 3=pairwise, 4=tree
    generation_batch TEXT                -- which run generated it
);

-- Every test execution
CREATE TABLE test_runs (
    run_id           TEXT PRIMARY KEY,
    hypothesis_id    TEXT REFERENCES hypotheses,
    gate             INTEGER,            -- 1, 2, 3, or 4
    league           TEXT,
    seasons          TEXT,               -- JSON array
    started_at       TIMESTAMP,
    completed_at     TIMESTAMP,
    
    -- Core results
    n_observations   INTEGER,
    brier_model      REAL,
    brier_baseline   REAL,
    brier_skill      REAL,
    log_loss_model   REAL,
    log_loss_baseline REAL,
    
    -- Significance
    p_value_raw      REAL,
    t_statistic      REAL,
    effect_size      REAL,
    
    -- ROI
    roi_0pct         REAL,
    roi_3pct         REAL,
    roi_5pct         REAL,
    n_bets_0pct      INTEGER,
    n_bets_3pct      INTEGER,
    
    -- Multiple testing context
    total_tests_at_time INTEGER,         -- global counter at time of test
    p_value_by       REAL,               -- BY-corrected p-value
    bayes_factor_min REAL,
    deflated_sharpe  REAL,
    
    -- Verdict
    passed           BOOLEAN,
    rejection_reason TEXT,               -- NULL if passed
    
    -- Reproducibility
    random_seed      INTEGER,
    code_version     TEXT,
    data_version     TEXT,
    config_hash      TEXT                -- SHA256 of full config
);

-- Cumulative trial counter (append-only)
CREATE TABLE trial_counter (
    counter_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TIMESTAMP,
    batch_id         TEXT,
    hypotheses_tested INTEGER,
    cumulative_total INTEGER
);

-- Validated signals (final output)
CREATE TABLE signals (
    signal_id        TEXT PRIMARY KEY,
    hypothesis_id    TEXT REFERENCES hypotheses,
    
    -- Gate results
    gate1_p          REAL,
    gate2_p          REAL,
    gate3_leagues    TEXT,               -- JSON: which leagues confirmed
    gate4_p          REAL,
    
    -- Strength
    brier_skill_avg  REAL,
    roi_avg          REAL,
    effect_size_avg  REAL,
    
    -- Status
    status           TEXT,               -- "validated", "monitoring", "decayed", "rejected"
    validated_at     TIMESTAMP,
    last_checked     TIMESTAMP,
    
    -- Decay monitoring
    decay_checks     INTEGER DEFAULT 0,
    decay_failures   INTEGER DEFAULT 0
);
```

### Append-Only Principle

No row in `test_runs` is ever updated or deleted. If a test is re-run, a new row is created. This creates a complete audit trail. The trial counter is also append-only — you can always verify the total number of tests ever conducted.

---

## 6. Signal Database Design

The signal database is the permanent output of Apollo. It answers: "What do we know?"

### Signal Lifecycle

```
CANDIDATE  →  DISCOVERED  →  OOS_CONFIRMED  →  CROSS_VALIDATED  →  VALIDATED
                                                                        │
                                                                        ▼
                                                                   MONITORING
                                                                        │
                                                              ┌─────────┴─────────┐
                                                              ▼                   ▼
                                                           ACTIVE              DECAYED
```

### Decay Monitoring

Validated signals are re-tested every season on new data. If a signal fails significance in 2 consecutive seasons, its status changes to `DECAYED`. This handles regime change and market adaptation.

### Signal Card (what gets stored)

```yaml
signal_id: SIG-0042
hypothesis_id: H-07831
name: "Fixture Congestion × Elo Delta Interaction"
description: >
  When the home team has played 3+ matches in 7 days AND 
  the Elo delta favors them by 100+ points, the market 
  overestimates home win probability by ~4.2 percentage points.
features:
  - home_matches_last_7d
  - elo_delta
operator: conditional_calibration
target: home_win
condition: "home_matches_7d >= 3 AND elo_delta > 100"

# Evidence
discovery:
  league: EPL
  seasons: ["2010-11", "2011-12", ..., "2022-23"]
  p_value_raw: 0.0003
  p_value_by: 0.018
  bayes_factor: 0.004
  brier_skill: 0.023
  n: 847
  
oos_walkforward:
  brier_skill: 0.019
  p_value: 0.007
  roi_3pct: 0.041
  n_bets: 312

cross_validation:
  LaLiga: {p: 0.034, bss: 0.015, n: 623}
  SerieA: {p: 0.087, bss: 0.011, n: 591}
  Bundesliga: {p: 0.142, bss: 0.008, n: 418}  # did not confirm
  Ligue1: {p: 0.063, bss: 0.013, n: 502}
  leagues_confirmed: 3

holdout:
  seasons: ["2023-24", "2024-25"]
  p_value: 0.029
  brier_skill: 0.017
  roi_3pct: 0.033

status: validated
validated_at: "2026-07-15"
```

---

## 7. Computational Requirements

### Workload Estimation

| Stage | Operations | Time (RTX 4060 laptop) |
|---|---|---|
| Feature computation | 140 features × ~50K matches | ~2 minutes |
| Level 1 hypotheses | 140 tests | ~30 seconds |
| Level 2 hypotheses | 700 tests | ~3 minutes |
| Level 3 hypotheses | 9,730 pairwise tests | ~45 minutes |
| Level 4 hypotheses | 1 tree fit + extraction | ~2 minutes |
| Gate 2 walk-forward | ~100 survivors × 5-league backtest | ~2 hours |
| Gate 3 cross-league | ~40 survivors × 4 leagues | ~1 hour |
| Gate 4 holdout | ~15 survivors | ~5 minutes |

**Total: ~4 hours per full discovery run.**

Feasible on a laptop. No GPU needed — these are statistical tests on tabular data, not deep learning.

### Memory

Feature matrix: ~50K matches × 500 features × 8 bytes ≈ 200MB. Fits in RAM trivially.

SQLite database: grows ~10MB per full discovery run. Negligible.

### Parallelization

Level 3 (pairwise tests) is embarrassingly parallel. Use `multiprocessing` with `n_cores - 1` workers. On an 8-core laptop, this cuts the 45-minute step to ~7 minutes.

---

## Architecture Principles

1. **The system has no opinion.** It tests everything mechanically. Human intuition enters only in feature definition, never in hypothesis selection.

2. **The trial counter never lies.** Every test ever run is counted. You cannot "reset" the multiple testing penalty.

3. **Holdout data is sacred.** It is loaded only in Gate 4. The Gate 4 function physically reads from a separate file path. No accident can contaminate it.

4. **Negative results are stored.** The 10,975 hypotheses that failed are as important as the 25 that passed. They represent the market's efficiency frontier.

5. **Decay is expected.** A signal that worked in 2015-2022 may stop working in 2025. The monitoring system detects this. Signals are not permanent truths — they are conditional, temporal, and fragile.
