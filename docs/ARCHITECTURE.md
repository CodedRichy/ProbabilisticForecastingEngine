# Project Apollo — System Architecture

## Design Principles

1. **Experiments are atomic.** Each experiment is a self-contained directory with its own config, code, data references, and output. You can delete any experiment without affecting others.
2. **Signals earn their place.** No signal enters the model layer until it has passed out-of-sample testing, cross-league validation, and false discovery correction.
3. **Negative results are first-class.** A signal that fails is as valuable as one that succeeds. Both get cataloged permanently.
4. **Reproducibility is mandatory.** Every experiment records its random seed, data version, code version, and exact parameters. Anyone re-running it gets identical results.
5. **Simplicity over cleverness.** Parquet files over databases. Scripts over frameworks. Flat config over dependency injection.

---

## Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Ecosystem depth, scipy/statsmodels/sklearn |
| Data storage | Parquet files | Columnar, fast, zero infrastructure |
| Data manipulation | pandas + polars | pandas for exploration, polars for large backtests |
| Statistics | scipy.stats, statsmodels | Hypothesis testing, regression, calibration |
| ML | scikit-learn, xgboost | Standard, well-documented |
| Visualization | matplotlib, seaborn | Publication-quality, no server needed |
| Notebooks | Jupyter | Exploration only — production code is scripts |
| Config | YAML | Human-readable experiment configs |
| Reports | Markdown + matplotlib PNGs | Portable, version-controllable |

No cloud. No Docker (yet). No databases. No APIs (initially).

---

## Directory Structure

```
apollo/
│
├── ARCHITECTURE.md          # This document
├── HYPOTHESES.md            # Master hypothesis catalog
├── README.md                # Project overview
├── requirements.txt         # Python dependencies
│
├── config/
│   └── settings.yaml        # Global settings (paths, defaults)
│
├── core/                    # Shared library code
│   ├── __init__.py
│   ├── data_loader.py       # Unified data loading interface
│   ├── experiment.py        # Experiment base class and runner
│   ├── signal.py            # Signal definition and evaluation
│   ├── metrics.py           # Brier score, log loss, calibration, ROI
│   ├── stats.py             # Significance testing, FDR correction
│   ├── backtest.py          # Walk-forward backtesting engine
│   └── report.py            # Auto-generate experiment reports
│
├── data/
│   ├── raw/                 # Untouched source files (CSV, JSON)
│   ├── processed/           # Cleaned, normalized parquet files
│   └── cache/               # Intermediate computation cache
│
├── experiments/             # One subdirectory per experiment
│   └── EXP_001_form_bias/
│       ├── config.yaml      # Experiment parameters
│       ├── run.py           # Execution script
│       ├── analysis.py      # Statistical analysis
│       ├── results/         # Output data, plots, tables
│       └── report.md        # Auto-generated findings
│
├── signals/                 # Validated signal registry
│   └── registry.yaml        # Master signal catalog with scores
│
├── backtests/               # Cross-experiment backtest results
│
├── reports/                 # Published research summaries
│
├── models/                  # Only after signals validated
│
├── evaluation/              # Model evaluation framework
│
├── notebooks/               # Jupyter exploration (not production)
│
└── dashboard/               # Simple local dashboard (later)
```

---

## Experiment Lifecycle

```
[1] Define hypothesis (HYPOTHESES.md)
        ↓
[2] Create experiment directory (experiments/EXP_NNN_name/)
        ↓
[3] Write config.yaml (parameters, data sources, date ranges)
        ↓
[4] Write run.py (load data → compute signal → measure against outcomes)
        ↓
[5] Execute: python -m experiments.EXP_001_form_bias.run
        ↓
[6] Statistical analysis (significance, effect size, robustness)
        ↓
[7] Auto-generate report.md
        ↓
[8] If significant → register signal in signals/registry.yaml
   If not significant → document failure in report, still valuable
        ↓
[9] Cross-validation across leagues/seasons
        ↓
[10] If survives → promote to model layer for integration testing
```

---

## Data Architecture

### Raw Data Schema (after normalization)

All match data normalizes to a single schema:

| Column | Type | Description |
|---|---|---|
| match_id | str | Unique identifier |
| date | date | Match date |
| season | str | e.g. "2023-24" |
| league | str | e.g. "EPL", "LaLiga" |
| home_team | str | Standardized team name |
| away_team | str | Standardized team name |
| home_goals | int | Full-time home goals |
| away_goals | int | Full-time away goals |
| result | str | "H", "D", "A" |
| home_odds | float | Best available closing odds (decimal) |
| draw_odds | float | Best available closing odds (decimal) |
| away_odds | float | Best available closing odds (decimal) |
| home_implied | float | Implied probability (overround removed) |
| draw_implied | float | Implied probability (overround removed) |
| away_implied | float | Implied probability (overround removed) |

### Extended Features (joined from secondary sources)

| Column | Type | Source |
|---|---|---|
| home_xg | float | FBref |
| away_xg | float | FBref |
| home_form_5 | float | Computed: points from last 5 |
| away_form_5 | float | Computed: points from last 5 |
| home_elo | float | Computed: Elo rating |
| away_elo | float | Computed: Elo rating |
| home_rest_days | int | Computed: days since last match |
| away_rest_days | int | Computed: days since last match |

---

## Evaluation Metrics

Every experiment measures:

| Metric | Purpose | What "good" looks like |
|---|---|---|
| Brier Score | Probability calibration | Lower than bookmaker baseline |
| Log Loss | Probability sharpness | Lower than bookmaker baseline |
| Calibration Plot | Visual calibration check | Points near diagonal |
| ROI | Economic significance | Positive after vig simulation |
| p-value | Statistical significance | < 0.05 after FDR correction |
| Effect Size | Practical significance | Cohen's d > 0.2 or meaningful ROI |
| Sample Size | Reliability | n > 200 minimum per condition |
| Cross-league stability | Robustness | Effect persists in 2+ leagues |
| Cross-season stability | Robustness | Effect persists in 3+ seasons |

### Critical Rule: The Bookmaker Baseline

Every signal is measured AGAINST bookmaker implied probabilities, not against a naive baseline. The question is never "does this predict outcomes?" — it's "does this predict outcomes BETTER than the market already does?"

If a signal can't beat the market baseline, it's noise regardless of its raw accuracy.

---

## False Discovery Prevention

With 25+ hypotheses, multiple testing is a real danger. Protections:

1. **Benjamini-Hochberg FDR correction** applied across all experiments.
2. **Pre-registration:** Hypothesis and analysis plan written BEFORE seeing results.
3. **Hold-out seasons:** Final 2 seasons reserved for confirmation. Never touched during exploration.
4. **Effect size requirements:** Statistical significance alone is insufficient. Require meaningful effect size.
5. **Robustness checks:** Vary parameters ±20%. If the result disappears, it's fragile.
6. **Cross-domain validation:** A signal found in EPL must show up in at least one other league.

---

## What We Are NOT Building

- A betting bot
- A real-time prediction API
- A user-facing product
- A dashboard for picking winners
- Anything that requires paid infrastructure

We are building a research instrument. It produces knowledge, not picks.
