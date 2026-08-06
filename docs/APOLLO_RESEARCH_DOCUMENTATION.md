# Apollo — Research Platform Documentation

Version 2.0 | June 2026

A Quantitative Discovery Engine for Football Market Inefficiency

---

## Part I: Architectural Autopsy

Before redesigning anything, the current system must be taken apart and its failure modes identified with precision. Every component that can break will break. The question is whether we know how.

### 1.1 Statistical Weaknesses

**The Residual-Correlation Paradigm Is Too Narrow**

The entire testing pipeline asks one question: does feature X correlate with (actual_outcome − market_implied)? This is Spearman rank correlation for Level 1, t-tests for Level 2, Kruskal-Wallis for Level 3. Every test is a linear or rank-based association test on calibration residuals.

This misses three categories of real signal. First, nonlinear effects. A feature might have zero Spearman correlation with the residual while having a strong U-shaped relationship — the market could misprice both extremes of a variable in opposite directions, and rank correlation would read zero. Second, conditional effects that don't manifest in marginal tests. Feature A might only predict residuals when Feature B is in a specific range, but the Level 3 pairwise test splits on median, which may not be the relevant threshold. Third, distributional effects. A feature might not shift the mean residual but compress or expand its variance — meaning it identifies matches where the market is uncertain, not matches where the market is wrong in a particular direction. Variance-predictive signals are invisible to mean-based tests.

**Benjamini-Yekutieli Is Simultaneously Too Conservative and Insufficient**

BY correction is valid under arbitrary dependence, which sounds safe. The cost is that it is extremely conservative — the harmonic number penalty (roughly ln(N)) applied to every p-value means that with 11,000 tests, the effective significance threshold drops to approximately α / (11,000 × 9.7) ≈ 4.7 × 10⁻⁷. This kills every weak-but-real signal. The football market is not grossly inefficient; real signals likely have small effect sizes with p-values in the 10⁻³ to 10⁻⁴ range. BY will annihilate them all.

But the more fundamental problem is that FDR correction, regardless of variant, assumes the tests are pre-specified. The discovery engine generates hypotheses mechanically, but the feature space itself was designed by a human. The researcher chose which features to compute, which windows to use, which transformations to apply. These are uncounted researcher degrees of freedom that no multiple testing correction can fix, because they happened before the testing pipeline was invoked.

**The Holdout Is a Single Shot**

Gate 4 uses two reserved seasons as a final confirmation. This sounds rigorous. The problem: you get one attempt. If a signal fails holdout, it is rejected permanently. But holdout failure can mean the signal is false (correct rejection), the signal is real but weak (type II error due to small holdout sample), or the signal is real but regime-dependent (it worked in 2015-2022 but the market adapted). A single holdout test cannot distinguish these cases.

Worse: once you have tested 25 candidates against the holdout, the holdout itself is contaminated. You now know that 25 specific hypotheses failed or passed against those two seasons. Any future analysis conditioned on that knowledge is no longer truly out-of-sample. The holdout is a wasting asset — each use reduces its statistical independence.

**The Trial Counter Creates Perverse Incentives**

The monotonically increasing trial counter is designed to prevent the restart trick. But it creates a different problem: it penalizes exploration. A researcher who has tested 50,000 hypotheses faces a much harsher BY correction than one who has tested 5,000, even if the 50,000-hypothesis researcher's additional tests were in completely unrelated feature spaces. The counter treats every test as drawing from the same well of false discovery risk, which is correct in the worst case but profoundly wasteful when the hypothesis space has independent clusters.

### 1.2 Machine Learning Weaknesses

**No Representation Learning**

The feature factory computes hand-designed features: rolling means, streaks, Elo ratings. These are level-one representations — direct aggregations of raw data. The system has no capacity to learn higher-order representations. A rolling 5-match win rate is a human-legible statistic, but it may not be the optimal encoding of recent form. Learned embeddings of team state (via recurrent networks or transformers on match sequences) could capture richer dynamics — momentum shifts, tactical transitions, squad rotation patterns — that no rolling window can express.

The system also cannot represent team identity as a latent variable. Two teams with identical rolling statistics are treated identically, but a team's identity encodes structural information (playing style, squad depth, managerial philosophy) that rolling form cannot capture. Without embeddings, this information is lost.

**No Temporal Modeling**

Features are computed per-match with rolling windows, but there is no explicit temporal model. The system cannot detect regime changes (a team's true ability shifted after a managerial change), seasonality (December congestion affects results differently from August), or trends (a team is improving/declining over a season). These require sequential models — hidden Markov models, changepoint detection, state-space models — which are entirely absent.

**No Multi-Task Learning**

The system tests home_win, draw, and away_win as separate targets. But these are components of a single multinomial outcome. Testing them independently wastes statistical power and misses joint structure. A multinomial model that simultaneously predicts all three outcomes, subject to the constraint that probabilities sum to one, would be strictly more efficient than three independent tests.

### 1.3 Data Weaknesses

**Single Source Dependency**

Football-Data.co.uk is the sole data source. This creates three risks. Survival risk: if the site goes offline or changes format, the entire pipeline breaks. Coverage risk: the site covers major European leagues but misses second divisions, cups, and international competitions where market inefficiency is likely larger (less liquid, less sophisticated). Quality risk: odds data reflects whichever bookmakers the site scrapes, not necessarily the sharpest or most efficient market. Pinnacle closing lines (the gold standard) are available only for recent seasons.

**No Event-Level Data**

Match-level data (goals, results, odds) is the coarsest possible resolution. It misses shot-level information (which FBref provides), passing networks (which StatsBomb provides), player-level data (injuries, suspensions, transfers), and in-play events (red cards, penalty decisions, tactical substitutions). These contain signal that aggregate match data cannot capture. A team that won 1-0 from an own goal while being outshot 20-3 looks identical in the data to a team that won 1-0 with 70% possession and 4 xG. The feature factory cannot distinguish them.

**Odds Data Lacks Depth**

The system uses closing odds, which are point estimates. It doesn't capture the full price trajectory — opening odds, line movements, volume. Line movement data is where sharp money reveals itself. A match where the line moved from 2.10 to 1.85 between opening and close tells a completely different story from one that stayed at 1.95 throughout. Without tick-level odds data, the entire Market Physics layer (Task 6) is impossible.

### 1.4 Discovery Weaknesses

**The Feature Space Is Closed**

The hypothesis generator enumerates combinations of pre-defined features. It cannot invent new features. If the real signal is "the ratio of home team xG in the last 3 away matches to away team xG in the last 3 home matches," the system cannot discover this unless someone hand-codes it. The combinatorial search operates within a fixed vocabulary; it cannot expand the vocabulary itself.

**No Learning From Failure**

When 10,975 hypotheses fail and 25 pass, the system does nothing with the 10,975 failures except store them. But failures contain information. If every hypothesis involving "home_form_winrate_5" fails, that tells us something about the 5-match window. If all Level 3 hypotheses involving Elo fail but some involving market features pass, that reveals the structure of market efficiency. The system has no mechanism to analyze its own failure patterns and redirect its search.

**No Adversarial Testing**

The system tests whether signals predict residuals. It does not test whether signals are robust to adversarial perturbation. Could the signal be an artifact of data cleaning? Would it survive if 5% of outcomes were randomly corrupted? Is it driven by a handful of outlier matches? Robustness testing beyond parameter-variation checks is absent.

### 1.5 Scalability Weaknesses

**DataFrame Serialization Bottleneck**

The parallel testing pipeline serializes the entire DataFrame as a Python dict for each worker process. With 500 features and 50,000 matches, this is approximately 200MB serialized per worker. With 8 workers, that is 1.6GB of redundant copies. At 100,000 hypotheses, the serialization overhead dominates compute time.

**SQLite Concurrency**

SQLite supports single-writer semantics. The experiment tracker uses SQLite for all recording. Under parallel execution, write contention will serialize test recording, creating a bottleneck. At 10,000+ tests per batch with parallel workers, this becomes a real throughput limiter.

**No Incremental Computation**

The feature factory recomputes all features from scratch every run. If you add one new match, every rolling window, every Elo rating, every H2H statistic is recomputed from the beginning. With 50,000 matches and 140 features, this takes minutes. At 500,000 matches with 1,000 features, it becomes impractical.

---

## Part II: Apollo V2 — The Feature Discovery Engine

The defining weakness of V1 is that features are human-designed. V2 replaces the static Feature Factory with a generative Feature Discovery Engine that constructs, evaluates, and selects features automatically.

### 2.1 Design Philosophy

The insight: a feature is a function from raw match data to a real number. The space of all such functions is infinite, but the space of *useful* features has structure. Useful features tend to be compositions of a small set of operators (rolling mean, lag, difference, rank, ratio) applied to a small set of base variables (goals, results, dates, odds). This structure can be exploited by defining a grammar that generates features compositionally.

### 2.2 The Feature Grammar

Define a context-free grammar over feature-construction operators:

```
FEATURE   → AGGREGATE(FILTER(BASE, SCOPE), WINDOW)
          | TRANSFORM(FEATURE)
          | COMBINE(FEATURE, FEATURE)

BASE      → goals_for | goals_against | result | odds_home | odds_draw
          | odds_away | xg | shots | shots_target | possession | ...

SCOPE     → home_only | away_only | all | vs_opponent | vs_top6 | vs_bottom6

WINDOW    → last_3 | last_5 | last_10 | last_20 | season | last_60d | last_90d

AGGREGATE → mean | std | sum | max | min | trend | skewness | count_above_median

TRANSFORM → rank_percentile | z_score | log | diff_from_league_avg
          | momentum(Δ between two windows) | regime_indicator

COMBINE   → ratio | difference | product | max | min | correlation_over_window
```

This grammar generates features like:

- `mean(goals_for, home_only, last_5)` → average home goals in last 5 home matches
- `ratio(mean(goals_for, all, last_5), mean(goals_against, all, last_5))` → offensive/defensive ratio
- `momentum(mean(xg, all, last_5), mean(xg, all, last_20))` → xG form vs longer-run xG
- `rank_percentile(std(goals_for, all, last_10))` → how volatile a team's scoring is relative to the league
- `regime_indicator(mean(result, all, last_3), mean(result, all, last_20))` → is the team over/underperforming its baseline?

### 2.3 Search Strategy: Grammar-Guided Genetic Programming

Enumerate features using genetic programming (GP) constrained by the grammar:

**Initialization.** Generate a population of 5,000 random features by sampling production rules from the grammar. Depth-limited to 4 levels of composition to prevent absurd complexity.

**Fitness Evaluation.** Each candidate feature is evaluated by:
1. Computing its values across the training dataset (chronologically, no lookahead).
2. Measuring mutual information with the calibration residual (target − market_implied). Mutual information captures nonlinear relationships that Spearman correlation misses.
3. Penalizing correlation with existing features in the accepted set (to encourage diversity).
4. Penalizing complexity (deeper parse trees get a regularization penalty).

Fitness = mutual_information − λ_redundancy × max_correlation_with_existing − λ_complexity × tree_depth

**Selection and Reproduction.**
- Tournament selection (size 7) picks parents.
- Crossover: swap subtrees between two parent features (respecting type constraints from the grammar).
- Mutation: randomly replace one subtree with a new random subtree from the grammar.
- Elitism: top 10% survive unchanged.

**Termination.** Run for 50 generations. Extract the Pareto front of features that are high-information, low-redundancy, and low-complexity.

### 2.4 Regime Detection

A dedicated sub-module searches for regime indicators — features that identify when the market's error distribution shifts.

**Method: Changepoint-Conditional Feature Search**

1. Run changepoint detection (PELT algorithm) on the rolling market calibration residual series.
2. Identify structural breaks: periods where the market was systematically biased in one direction.
3. Search the feature space for features that predict which regime is active (i.e., features that were different before vs. after a changepoint).
4. Any feature that retroactively explains a changepoint is a candidate regime indicator.

This is dangerous (overfitting to historical regime boundaries), so regime indicators require a dedicated validation step: they must predict out-of-sample regime shifts, not just explain past ones.

### 2.5 Validation of Discovered Features

Discovered features are not automatically trusted. They enter the same hypothesis testing pipeline as V1, but with an additional penalty: the multiple testing correction must account for the GP search. If 5,000 candidate features were evaluated over 50 generations (250,000 fitness evaluations), the trial counter increases by 250,000. This is severe but honest — it reflects the true extent of researcher exploration, even though the "researcher" was an algorithm.

A mitigating strategy: split data into a discovery partition (for GP fitness evaluation) and a testing partition (for hypothesis testing). Features discovered on the first half of the data are tested on the second half. This halves the available data for each stage but ensures genuine out-of-sample evaluation.

---

## Part III: Apollo V3 — The Autonomous Scientific Agent

V3 transforms Apollo from a pipeline that tests pre-generated hypotheses into an agent that conducts research autonomously — generating, testing, analyzing, and iterating.

### 3.1 The Research Cycle

```
    ┌──────────────┐
    │  OBSERVE     │ ← Analyze failure patterns, residual structure,
    │              │   knowledge graph state
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  HYPOTHESIZE │ ← Generate new hypotheses informed by observations
    │              │   (not random enumeration)
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  TEST        │ ← Standard statistical testing with full correction
    │              │
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  ANALYZE     │ ← Why did it pass/fail? What does the result
    │              │   tell us about the broader question?
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  INTEGRATE   │ ← Update knowledge graph, signal genome,
    │              │   research priorities
    └──────┬───────┘
           │
           └────────▶ back to OBSERVE
```

### 3.2 Directed Hypothesis Generation

V1 generates hypotheses by exhaustive enumeration. V3 generates hypotheses by directed search informed by accumulated knowledge.

**Failure Analysis Module**

After each testing batch, the system analyzes failures:

- Cluster failed hypotheses by feature family. If 90% of hypotheses involving rolling form metrics failed, the system learns: "rolling form is well-priced by the market."
- Identify near-misses: hypotheses with p-values between 0.05 and 0.20 after correction. These are regions of the feature space where weak signal may exist. Rather than discarding them, the system generates refined hypotheses in the neighborhood — different windows, different transformations, different conditioning variables.
- Detect contradiction: if Feature A predicts positive residuals and Feature B (highly correlated with A) predicts negative residuals, something is wrong. The system flags contradictions for investigation.

**Refinement Operators**

Given a failed hypothesis H, the system generates refinements:

| Operator | Description | Example |
|---|---|---|
| Narrow | Add a conditioning variable | "Form bias" → "Form bias when favorite is heavy" |
| Widen | Remove a condition | "Form bias in derbies" → "Form bias" |
| Shift window | Change temporal parameters | "Last 5 matches" → "Last 10 matches" |
| Transform | Apply a different function | "Mean of X" → "Trend of X" |
| Substitute | Replace one feature with a related one | "Goals scored" → "xG" |
| Invert | Test the opposite direction | "Streaks overpriced" → "Streaks underpriced" |
| Combine | Merge two near-miss hypotheses | "A is weak + B is weak" → "A AND B together" |

The refinement operators are not random. They are prioritized by the failure analysis: if the near-miss was a window-length issue, window shifts are tried first.

### 3.3 Research Memory

The agent maintains a structured research memory:

**Established Facts.** Propositions the system considers settled. Example: "Draw markets are well-calibrated in the EPL for the period 2010-2023." Each fact has an evidence chain — the experiments that established it, the p-values, the sample sizes. Facts can be overturned if new evidence contradicts them.

**Open Questions.** Areas where evidence is ambiguous. Example: "Fixture congestion appears to matter in the Bundesliga but not in La Liga. Why?" Open questions drive future hypothesis generation — they are the system's research agenda.

**Dead Ends.** Regions of the hypothesis space that have been thoroughly explored and found empty. Example: "All simple rolling-form features at all windows have been tested against all three targets across five leagues. None survive correction." Dead ends prevent the system from re-exploring exhausted terrain.

**Anomalies.** Results that don't fit existing knowledge. Example: "A signal involving referee identity passed Gate 1 in Serie A but no other league." Anomalies trigger targeted investigation — are they real (league-specific effects) or artifacts?

### 3.4 The Research Director

The agent needs a meta-controller that allocates research effort. This is the Research Director — a priority function that decides what to investigate next.

**Priority Score Calculation:**

```
Priority(question) = 
    expected_information_gain × feasibility
    ÷ (computational_cost × redundancy_with_existing_knowledge)
```

Where:
- Expected information gain: estimated from how close neighboring hypotheses came to significance.
- Feasibility: does the required data exist? Is the sample size sufficient?
- Computational cost: how many tests are needed?
- Redundancy: how much does this overlap with what we already know?

The Research Director maintains a priority queue. Each cycle, it selects the highest-priority open question and generates hypotheses to address it. This transforms Apollo from an exhaustive searcher into a directed scientist.

---

## Part IV: The Signal Genome

### 4.1 Core Concept

In V1, a signal is a flat record: feature, operator, target, p-value, effect size. This loses the compositional structure of how the signal was discovered and how it relates to other signals.

The Signal Genome represents signals as structured, heritable objects with lineage, relationships, and evolutionary dynamics.

### 4.2 Signal DNA

A signal's DNA is its complete specification in a decomposable format:

```
Signal DNA:
    Base Signal:     market_calibration_bias
    Feature Gene 1:  home_form_winrate(window=5)
    Feature Gene 2:  elo_delta(k=32)
    Operator Gene:   conditional_calibration
    Condition Gene:  feature_1 > percentile_80 AND feature_2 > 100
    Target Gene:     home_win
    Regime Gene:     season_phase > 0.7 (late season)
```

Each "gene" is a modular component that can be independently varied. The full DNA specifies the complete signal — what features, what operator, what conditions, what target, under what regime.

### 4.3 Signal Inheritance

When a validated signal is found, the system generates child signals by mutating individual genes:

**Parent Signal:** "Home teams on winning streaks are overpriced when Elo delta is large."

**Child Signals (single-gene mutations):**
- Feature mutation: Replace win streak with unbeaten streak.
- Window mutation: Change streak threshold from 5 to 3.
- Condition mutation: Replace "Elo delta > 100" with "Elo delta > 150."
- Target mutation: Test against draw instead of home_win.
- Regime mutation: Test in early season instead of late season.
- Scope mutation: Test in La Liga instead of EPL.

Children inherit all genes from the parent except the mutated one. This maintains structural similarity while exploring the neighborhood.

### 4.4 Signal Family Trees

Every signal records its parent (if any) and its children. This creates a tree structure:

```
SIG-0001: Market calibration bias (root)
├── SIG-0023: Form bias (feature mutation)
│   ├── SIG-0089: Form bias + Elo interaction (combine)
│   │   ├── SIG-0134: Form + Elo + late season (regime mutation)
│   │   └── SIG-0156: Form + Elo + congestion (feature addition) [EXTINCT]
│   └── SIG-0091: Form bias + travel fatigue (combine) [REJECTED at Gate 3]
├── SIG-0027: Elo miscalibration (feature substitution)
│   └── SIG-0044: Elo miscalibration in cup matches (condition) [EXTINCT]
└── SIG-0031: Draw underpricing (target mutation) [VALIDATED]
```

Family trees reveal the structure of market inefficiency. If a signal family has many validated descendants, the underlying mechanism is likely robust. If all children go extinct, the parent signal may be an artifact.

### 4.5 Signal Extinction

Signals are not permanent. A validated signal is monitored each season. If it fails significance in two consecutive seasons, it is marked EXTINCT. Extinction events are recorded with hypothesized causes:

| Extinction Type | Description |
|---|---|
| Market adaptation | The market learned to price this information. |
| Regime change | The underlying mechanism stopped operating. |
| Data artifact | The signal was driven by a data quality issue that has been corrected. |
| Statistical fluke | Despite passing all gates, the signal was a false positive. |

Extinction is valuable data. A signal that went extinct due to market adaptation confirms that the market was, at one point, genuinely inefficient — and that the signal was real. This is evidence for the broader theory, even though the signal is no longer tradeable.

### 4.6 Signal Fitness Landscape

The genome metaphor enables fitness landscape analysis. Each signal occupies a point in a high-dimensional space defined by its genes. The "fitness" at each point is the Brier Skill Score (or ROI). The system can map which regions of the landscape are fertile (many viable signals) and which are barren (exhaustively tested, nothing survives).

Over time, the fitness landscape itself shifts (markets adapt, regimes change). Mapping this drift is one of Apollo's most valuable long-term outputs — it measures the evolution of market efficiency.

---

## Part V: The Research Knowledge Graph

### 5.1 Structure

The graph has five node types and four edge types:

**Nodes:**
- **Feature nodes**: Every computed feature.
- **Signal nodes**: Every tested hypothesis (passed or failed).
- **Model nodes**: Every model configuration tested.
- **Regime nodes**: Identified market regimes.
- **Fact nodes**: Established propositions.

**Edges:**
- **CORRELATES_WITH** (Feature ↔ Feature): Measured correlation, direction, strength.
- **SUPPORTS** (Signal → Fact): A signal that provides evidence for a proposition.
- **CONTRADICTS** (Signal → Fact): A signal that provides evidence against a proposition.
- **REQUIRES** (Signal → Feature): A signal depends on this feature for its computation.
- **ACTIVE_DURING** (Signal → Regime): A signal is operative only during this regime.
- **DERIVED_FROM** (Feature → Feature): One feature is a transformation of another.
- **PARENT_OF** (Signal → Signal): Signal genome lineage.

### 5.2 Evidence Accumulation

Facts have an evidence weight that evolves over time:

```
evidence_weight(fact) = Σ (support_i × quality_i) − Σ (contradiction_j × quality_j)
```

Where quality reflects sample size, gate level reached, and recency. A fact with strong positive evidence weight is considered established. A fact with negative weight is considered refuted. A fact near zero is considered open.

### 5.3 Cluster Analysis for Research Frontiers

Community detection algorithms (Louvain, label propagation) applied to the knowledge graph identify clusters of related features, signals, and facts. These clusters correspond to research themes — "form-based signals," "market microstructure signals," "schedule effects."

The boundaries between clusters are research frontiers — unexplored combinations of features from different clusters that haven't been tested together. The Research Director prioritizes hypotheses at cluster boundaries because these represent the most novel combinations.

### 5.4 Contradiction Detection

When a new test result contradicts an established fact, the system raises a flag:

```
CONTRADICTION DETECTED:
    New result: Signal SIG-0412 shows draws are OVERPRICED in La Liga (p=0.003)
    Existing fact: FACT-0019: Draw markets are well-calibrated (evidence weight: +4.7)
    
    Possible resolutions:
    1. La Liga is an exception (league-specific effect)
    2. FACT-0019 is outdated (regime change since it was established)
    3. SIG-0412 is a false positive (check multiple testing context)
    
    Recommended action: Test draw calibration in La Liga specifically, 
    using recent seasons only.
```

Contradictions are high-priority research items because they indicate either a gap in knowledge or an error in established beliefs.

---

## Part VI: The Market Physics Layer

### 6.1 Conceptual Framework

Standard forecasting asks: given features X, what is the probability of outcome Y? Market physics asks a different question: how does information get incorporated into prices, and where does that process break down?

The market is not a single agent. It is an ecology of participants:

- **Sharp bettors**: Quantitative operators who move the line with large, informed wagers.
- **Recreational bettors**: Public money driven by narratives, loyalty, and entertainment.
- **Bookmakers**: Market makers who set and adjust lines to manage risk.
- **Media**: Amplifies narratives that drive public sentiment.

Market inefficiency arises when these participants interact in predictable ways. Apollo's Market Physics layer models these interactions.

### 6.2 Observable Market Behaviors

**Public Overreaction**

Measurable as: a systematic relationship between the magnitude of public attention (news volume, social media mentions) and the direction of line movement, where the line moves beyond what fundamentals justify.

Discovery method: correlate the residual (outcome − implied) with media volume. If high-media-attention matches have systematically worse calibration, the market is overreacting to narratives.

**Momentum and Reversal**

Measurable as: autocorrelation in line movements. If an early line move in one direction predicts further movement (momentum) or a correction (reversal), there is a systematic pattern in price discovery.

Discovery method: compute opening-to-midweek and midweek-to-closing line changes. Test autocorrelation and cross-correlation with residuals.

**Sharp vs. Public Money**

Measurable as: differential calibration between opening lines (set by bookmakers) and closing lines (influenced by sharp money). If closing lines are systematically better calibrated than opening lines, the difference represents the information that sharp bettors add.

Discovery method: compare Brier scores of opening vs. closing implied probabilities. Quantify the information content of line movement itself.

**Market Efficiency Breakdown Conditions**

The most valuable discovery: identifying conditions under which the market becomes systematically less efficient. Candidates:

- Low-liquidity matches (small leagues, early rounds, friendlies).
- High-narrative matches (derbies, relegation battles, title deciders).
- After information shocks (managerial sacking, star player injury).
- Temporal windows (early season when priors are stale, post-international-break).

Discovery method: segment matches by condition. Compute the variance of calibration residuals per segment. Higher variance indicates lower market efficiency (more room for signals to exist).

### 6.3 Information Flow Model

Model the market as an information processing system:

```
True state (team strength, form, conditions)
    ↓
Fundamental models (Elo, Poisson, xG models)
    ↓
Opening line (bookmaker's initial estimate)
    ↓
Sharp money (informed bettors correct the line)
    ↓
Public money (recreational bettors add noise)
    ↓
News/narratives (media creates correlated public action)
    ↓
Closing line (final market consensus)
    ↓
Outcome (match result)
```

At each stage, information is added and noise is added. Apollo's opportunity is at stages where noise dominates signal — where the closing line is worse than it should be given the information available. The Market Physics layer identifies which stages introduce systematic error, not just random noise.

### 6.4 Measuring Market Efficiency Over Time

A long-term output: a time series of market efficiency itself. Define efficiency as the Brier score of market-implied probabilities per league, per season.

If efficiency is improving (Brier scores declining over time), the market is getting harder to beat. If efficiency fluctuates, there are windows of opportunity. If efficiency varies by league, the least efficient leagues are the most promising targets.

This meta-analysis — the study of how efficient the market is, rather than the study of specific signals — is arguably Apollo's most defensible long-term output. It requires years of accumulated data that no competitor can shortcut.

---

## Part VII: Scaling to Millions of Hypotheses

### 7.1 The Problem

Testing 11,000 hypotheses is computationally trivial but statistically expensive (BY correction with N = 11,000). Testing 10,000,000 hypotheses requires solving both the computational problem and the statistical problem — at N = 10⁷, BY correction makes any individual test essentially impossible to pass.

### 7.2 Hierarchical Testing Architecture

The solution: don't test all hypotheses at the same level of rigor. Use a funnel:

```
Level 0: Coarse Screen (10,000,000 hypotheses)
    Method: Mutual information estimate with random subsampling
    Cost per test: ~1ms
    Total time: ~3 hours
    Survival: ~100,000 (top 1%)
    
Level 1: Medium Screen (100,000 hypotheses)
    Method: Full Spearman correlation on complete data
    Cost per test: ~50ms
    Total time: ~1.5 hours
    Survival: ~5,000 (top 5%)
    
Level 2: Full Statistical Test (5,000 hypotheses)
    Method: Complete test suite (t-test, Kruskal-Wallis, bootstrap CI)
    Cost per test: ~500ms
    Total time: ~40 minutes
    Survival: ~200 (after FDR correction at N=5,000, not 10M)
    
Level 3: Walk-Forward Validation (200 hypotheses)
    Method: Full walk-forward backtest
    Cost per test: ~30 seconds
    Total time: ~100 minutes
    Survival: ~20-50
    
Level 4: Holdout (20-50 hypotheses)
    Standard holdout confirmation
```

The critical statistical insight: the multiple testing penalty at Level 2 applies only to the 5,000 hypotheses that reached Level 2, NOT to the 10,000,000 that were screened at Level 0. This is valid if the Level 0 screen is independent of the Level 2 test statistic — which it is, because Level 0 uses mutual information on a random subsample while Level 2 uses parametric tests on the full data. The two screens are measuring different things on different data, so the effective multiple testing burden is dramatically reduced.

This is formally justified by the conditional testing framework (Fithian, Sun, and Taylor, 2017): the p-value at a later stage is valid conditional on having passed the earlier screen, provided the earlier screen is a function of a different sufficient statistic.

### 7.3 Random Projection for Feature Interaction Search

Testing all C(N, 2) pairwise interactions when N = 1,000 features means ~500,000 pairs. All C(N, 3) triples means ~166 million. Exhaustive search is impossible.

Solution: random projection. Johnson-Lindenstrauss lemma guarantees that random projections approximately preserve distances. Project the feature matrix into random low-dimensional subspaces (dimension 5-10) and test whether the projected features predict residuals. If a random projection captures signal, it implies that some combination of the original features along that projection axis is predictive. The system then decomposes the projection to identify which original features contributed.

This is used by quant funds for exactly this purpose: it searches exponentially large interaction spaces in polynomial time, at the cost of missing some signals (no guarantee of completeness).

### 7.4 Budget Allocation

Given finite computational resources, how should testing budget be allocated across hypothesis families?

Use a multi-armed bandit framework. Each family (form signals, market signals, schedule signals, interaction signals) is an arm. The "reward" is the rate of passing Level 1 screening. Families with higher discovery rates get more budget. Families with consistently zero discoveries get their budget reduced (but never eliminated, to maintain exploration).

Thompson sampling with a Beta prior on each family's discovery rate provides an asymptotically optimal allocation.

### 7.5 Computational Infrastructure at Scale

| Hypothesis Count | Hardware | Time | Storage |
|---|---|---|---|
| 11,000 | Laptop (8 cores) | ~4 hours | 100MB SQLite |
| 100,000 | Laptop (8 cores) | ~12 hours | 500MB SQLite |
| 1,000,000 | Laptop (8 cores) + hierarchical screening | ~24 hours | 2GB SQLite → migrate to DuckDB |
| 10,000,000 | Small cluster (4 × 32-core machines) or cloud burst | ~8 hours | 10GB DuckDB or Parquet |

The transition from SQLite to DuckDB (or Parquet + analytics engine) happens at approximately 1M records. DuckDB is column-oriented, supports concurrent reads, and handles analytical queries on billions of rows efficiently. It is the right tool for a research database at this scale.

---

## Part VIII: The Defensible Moat

### 8.1 What Can Be Copied

Everything visible can be copied: the algorithms, the model architectures, the statistical tests, the feature grammar, the genome structure, the knowledge graph schema. If you publish a paper describing the methodology, anyone can replicate it. Open-source the code, and it takes a weekend.

### 8.2 What Cannot Be Copied

**The Negative Results Library**

After 10 years, Apollo has tested 10 million hypotheses. Of these, perhaps 200 are validated signals. The other 9,999,800 are documented failures. This negative results library is unreplicable — no one else has spent the compute, time, or data to prove those 9,999,800 hypotheses don't work.

A competitor starting from scratch must re-explore all of this dead space, or risk wasting resources on already-disproved ideas. Apollo knows where not to look. That knowledge is worth more than knowing where to look, because the barren regions are far larger.

**Temporal Depth**

Apollo's experiment database stretches back years. It has tested signals in 2016 that have since gone extinct, and can trace the extinction cause. It has watched market efficiency evolve season by season. It has a longitudinal view of which signal families gain and lose viability over time.

This temporal depth enables meta-analysis that is literally impossible for a new entrant. You cannot observe how signals decay over 10 years if you have been operating for 6 months. Time is the ultimate moat.

**The Knowledge Graph**

The graph of relationships between features, signals, facts, regimes, and contradictions is a compound asset. Each new experiment adds nodes and edges. The graph's value grows super-linearly — new connections between existing nodes emerge as more experiments fill in gaps. A competitor can replicate any individual node, but replicating the graph requires replicating the full history of research.

**Signal Lineage**

The genome system records not just what signals exist, but how they were discovered — which parents, which mutations, which gates they passed and failed. This lineage data reveals the productive regions of the search space. It is a map of where in the feature-interaction-regime landscape signals tend to emerge. This map cannot be derived from the signals alone; it requires the full evolutionary history.

### 8.3 The Compound Knowledge Thesis

Apollo's moat is cumulative and self-reinforcing:

```
Year 1:  Test 100K hypotheses. Discover 10 signals. Map 100K dead regions.
Year 2:  Directed search avoids dead regions. Test 200K new hypotheses.
         Discover 20 signals. Dead region map now covers 300K.
Year 3:  Research Director focuses effort on frontier regions.
         Discovery rate improves. Map covers 600K.
...
Year 10: 10M hypotheses tested. Knowledge graph has 500K edges.
         A new competitor starting today faces 10M hypotheses they
         must test (or risk duplicating) before reaching Apollo's frontier.
```

The moat is not any single signal or algorithm. The moat is the accumulated history of scientific investigation — the map of the territory.

---

## Part IX: The 10-Year Vision

### 9.1 Year 1-2: Foundation

Build the V1 pipeline. Test the first 50,000 hypotheses. Establish the experiment database, the signal registry, the knowledge graph skeleton. Discover or refute the obvious hypotheses (form bias, home advantage erosion, draw pricing, favorite calibration). The output is a map of the most-studied region of the hypothesis space, establishing which well-known effects are real and which are noise.

### 9.2 Year 3-4: Discovery Engine

Deploy V2 (Feature Discovery Engine). Grammar-guided GP generates novel features. The hypothesis space expands from hand-crafted to algorithmically generated. First discovered features that no human hypothesized. First signal genome lineages emerge. The knowledge graph reaches critical density — contradictions begin appearing, driving new research directions.

### 9.3 Year 5-6: Autonomous Researcher

Deploy V3 (Scientific Agent). Apollo directs its own research. The Research Director allocates budget. The failure analysis module generates refined hypotheses. The system begins producing genuine surprises — discoveries in corners of the feature space that no human would have explored. First cross-domain signals (features from one league predicting inefficiencies in another). First market physics results (identified conditions under which efficiency breaks down).

### 9.4 Year 7-8: Market Efficiency Observatory

Apollo's most valuable output is no longer individual signals (most of which have decayed by now). It is the longitudinal measurement of market efficiency itself. Apollo can answer:

- How efficient is the EPL betting market in 2033 vs. 2025?
- Which leagues have become more efficient? Which haven't?
- What types of information are fastest/slowest to be priced?
- How quickly do newly discovered signals get arbitraged away?

This is original scientific output. It could be published in academic journals. It is unreplicable by any entity that hasn't been measuring these quantities continuously for years.

### 9.5 Year 9-10: Cross-Domain Generalization

Apply the methodology to adjacent domains:

- Political prediction markets (same framework: features → hypotheses → test against market baseline).
- Financial markets (much harder — the efficient market hypothesis is more strongly defended, and the data is enormous. But the methodology transfers.)
- Macroeconomic forecasting (who should we trust among central bank forecasts, survey-based forecasts, and model-based forecasts?)
- Epidemiological forecasting (where are the prediction markets for pandemic trajectories systematically miscalibrated?)

The domain transfer requires new features and new data pipelines, but the core architecture — grammar-guided feature generation, automated hypothesis testing, multi-gate validation, knowledge graph, signal genome — is domain-agnostic.

### 9.6 The Final Form

After 10 years, Apollo is not a betting system. It is a forecasting research institute.

Its assets are:

| Asset | Description | Replicability |
|---|---|---|
| Signal database | ~200 validated signals with full provenance and lineage | Low (requires years of continuous testing) |
| Negative results library | ~10M disproved hypotheses with full documentation | Very low (requires the compute history) |
| Knowledge graph | ~500K edges connecting features, signals, facts, regimes | Very low (requires the experimental sequence) |
| Market efficiency time series | 10+ years of calibrated efficiency measurement per league | Impossible without the time investment |
| Signal extinction archive | Record of every signal that worked then stopped, with analysis of why | Impossible retrospectively |
| Discovery methodology | Proven, refined, battle-tested framework | High (can be described and replicated) |
| Feature grammar | Evolved grammar with pruned dead branches and productive patterns | Medium (grammar is copyable, pruning history is not) |

The methodology is replicable. The accumulated knowledge is not. This is the same moat that protects research institutions, intelligence agencies, and long-lived quant funds. The code is trivial. The database is the asset.

### 9.7 What Would Make Apollo Unique

There is, to the best of available knowledge, no public system that:

1. Systematically tests millions of forecasting hypotheses against market calibration residuals.
2. Applies hedge-fund-grade multiple testing correction.
3. Maintains a persistent, append-only experimental database spanning years.
4. Models signal evolution with a genome and lineage tracking system.
5. Autonomously directs its own research based on accumulated knowledge.
6. Measures market efficiency longitudinally across leagues and years.

Academic researchers study individual hypotheses. Quant funds keep their findings secret. Prediction markets aggregate opinions but don't decompose why the market is right or wrong. Sports analytics companies build models but don't validate them with the rigor of a clinical trial.

Apollo occupies a unique position: the systematic, rigorous, automated study of what predicts outcomes and what doesn't, accumulated over years, with every result documented and no result ever deleted.

The world has many prediction systems. It has very few knowledge discovery systems. That is the gap.

---

## Appendix A: Glossary

| Term | Definition |
|---|---|
| Brier Score | Mean squared error of probabilistic forecasts. Range [0, 1]. Lower is better. |
| Brier Skill Score | 1 − (model Brier / baseline Brier). Positive means better than baseline. |
| Calibration residual | Actual outcome − market implied probability. The target variable for all hypothesis tests. |
| BY correction | Benjamini-Yekutieli false discovery rate control. Valid under arbitrary test dependence. |
| Deflated Sharpe Ratio | Sharpe ratio adjusted for the number of strategies tried, skewness, and kurtosis. |
| Gate | A validation stage in the cascade. Gate 1 (discovery) through Gate 4 (holdout). |
| Regime | A period during which the statistical properties of market behavior are stable. |
| Signal DNA | The complete decomposable specification of a validated signal. |
| Trial counter | Monotonically increasing count of all hypotheses ever tested. |
| Feature grammar | A context-free grammar that generates feature-construction expressions. |

## Appendix B: Key References

| Topic | Reference |
|---|---|
| Forecast combination | Bates and Granger (1969) |
| False discovery rate | Benjamini and Hochberg (1995); Benjamini and Yekutieli (2001) |
| Deflated Sharpe Ratio | López de Prado (2014) |
| Multiple testing in finance | Harvey, Liu, Zhu (2016) "...and the Cross-Section of Expected Returns" |
| Conditional testing | Fithian, Sun, Taylor (2017) |
| Genetic programming | Koza (1992); Poli, Langdon, McPhee (2008) |
| Superforecasting | Tetlock and Gardner (2015) |
| Market microstructure | O'Hara (1995); Harris (2003) |
| Changepoint detection | Killick, Fearnhead, Eckley (2012) — PELT algorithm |
| Information theory for feature selection | Brown, Pocock, Zhao, Luján (2012) |
| Online learning with expert advice | Cesa-Bianchi and Lugosi (2006) |
| Efficient market hypothesis in sports | Sauer (1998); Woodland and Woodland (1994) |
