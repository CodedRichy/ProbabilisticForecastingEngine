# Project Apollo — Hypothesis Catalog

Every hypothesis must specify: what we expect, why (mechanism), how to test it,
and what data it requires. Hypotheses are grouped by tier based on testability
with free data and a single developer.

---

## Tier 1: Immediately Testable (Football-Data.co.uk only)

These require nothing beyond match results and closing odds.

---

### H-001: Recent Form Overpricing

**Hypothesis:** Teams on extended winning streaks (5+ matches) are overpriced by bookmakers because public money overweights recent results.

**Mechanism:** Recency bias in public betting shifts odds toward streaking teams, creating value on the other side.

**Test:** Compare implied probability of streak teams winning vs actual win rate. Measure ROI of fading streak teams.

**Variables:** Last 5/10 match results, closing odds, outcome.

**Data:** Match results + odds. Available.

**Priority:** HIGH — clean test, large sample, well-defined signal.

---

### H-002: Home Advantage Erosion

**Hypothesis:** Home advantage has declined over the past decade, but bookmaker odds have not fully adjusted.

**Mechanism:** VAR, empty stadiums (COVID era), tactical evolution. Markets may be slow to update a deeply embedded prior.

**Test:** Measure home win rate vs home implied probability per season. Test for statistically significant trend and lag in odds adjustment.

**Variables:** Home/away result, odds, season.

**Data:** Match results + odds. Available.

**Priority:** HIGH — testable longitudinal hypothesis with large n.

---

### H-003: Draw Underpricing

**Hypothesis:** Draws are systematically underpriced because bettors have a psychological bias toward decisive outcomes (win/loss).

**Mechanism:** Draws are unsatisfying. Recreational bettors underbet draws, leaving value.

**Test:** Compare draw implied probability vs actual draw frequency across leagues and seasons. Measure ROI of systematic draw betting.

**Variables:** Draw odds, draw outcome.

**Data:** Match results + odds. Available.

**Priority:** HIGH — well-studied in literature, good calibration test.

---

### H-004: Promoted Team Underpricing

**Hypothesis:** Newly promoted teams are underpriced early in the season because markets anchor to their lower-division status.

**Test:** Compare promoted team performance vs market expectations in first 10 matches. Measure ROI.

**Variables:** Promoted team flag (manual or derive from prior season data), odds, results.

**Data:** Match results + odds + league membership. Available with minor enrichment.

**Priority:** MEDIUM — smaller sample per season (2-3 teams promoted per league).

---

### H-005: Fixture Congestion Effect

**Hypothesis:** Teams playing 3+ matches in 7 days underperform market expectations due to fatigue.

**Mechanism:** Physical fatigue, rotation, reduced intensity.

**Test:** Compute days since last match for each team. Compare performance vs implied probability when rest < 4 days.

**Variables:** Match dates (to compute rest days), odds, results.

**Data:** Match results + dates + odds. Available.

**Priority:** HIGH — computable from dates alone.

---

### H-006: Season Phase Effects

**Hypothesis:** The relationship between odds and outcomes varies by season phase (early, mid, late).

**Mechanism:** Early season: less information, wider odds. Late season: motivation varies (relegation battle vs mid-table). Markets may not fully price these.

**Test:** Evaluate bookmaker calibration separately for matches 1-10, 11-28, 29-38 of each season.

**Variables:** Match week, odds, results.

**Data:** Match results + odds. Available.

**Priority:** MEDIUM — needs match-week computation.

---

### H-007: Derby Match Mispricing

**Hypothesis:** Derby/rivalry matches have systematically different outcomes vs market expectations.

**Mechanism:** Derbies produce more draws and more upsets due to heightened intensity and reduced skill gap under pressure.

**Test:** Identify major derbies per league. Compare market calibration in derbies vs non-derbies.

**Variables:** Derby flag (requires a lookup table), odds, results.

**Data:** Match results + odds + derby list. Available with manual enrichment.

**Priority:** MEDIUM — small sample per derby pair.

---

### H-008: Goal Totals Market Efficiency

**Hypothesis:** Over/under 2.5 goals markets are less efficient than match result markets.

**Mechanism:** Match result markets attract more sophisticated money. Totals markets have more recreational action.

**Test:** Compare calibration of over/under implied probabilities vs actual frequencies.

**Variables:** Total goals, over/under odds.

**Data:** Football-Data.co.uk includes totals odds for recent seasons. Partially available.

**Priority:** MEDIUM — odds availability varies by season.

---

### H-009: Heavy Favorite Underperformance

**Hypothesis:** When implied probability exceeds 75%, favorites win less often than the market suggests.

**Mechanism:** Diminishing returns on dominance — a 75% team can't be "twice as good" as a 37% team in a single match. Variance compresses at extremes.

**Test:** Bin matches by favorite implied probability. Compare actual vs implied in each bin. Calibration analysis.

**Variables:** Odds, results.

**Data:** Available.

**Priority:** HIGH — pure calibration study, large sample.

---

### H-010: Odds Movement as Signal

**Hypothesis:** Large pre-match odds movements (opening to closing) predict something beyond the closing price.

**Mechanism:** Late sharp money may overshoot. Alternatively, large movements signal information arrival that hasn't fully resolved.

**Test:** Compute opening-to-closing odds change. Test whether extreme movers have different calibration.

**Variables:** Opening odds, closing odds, results. Football-Data.co.uk has some opening odds.

**Data:** Partially available — not all seasons have opening odds.

**Priority:** MEDIUM — data availability limits sample.

---

## Tier 2: Testable with FBref or Computed Features

Require either scraping FBref or computing derived statistics.

---

### H-011: xG vs Market Efficiency

**Hypothesis:** Expected Goals (xG) models contain information not fully reflected in betting odds.

**Mechanism:** xG captures shot quality; market odds may rely more on results. Process-based metrics may lag result-based pricing.

**Test:** Build rolling xG-based predictions. Compare calibration against bookmaker implied probabilities.

**Variables:** xG data (FBref), odds, results.

**Data:** FBref — free but requires scraping. Available from ~2017.

**Priority:** HIGH — xG is the strongest candidate signal.

---

### H-012: Shot Quality vs Shot Volume

**Hypothesis:** Shot quality (xG per shot) is more predictive than shot volume.

**Test:** Compare predictive power of total shots vs xG/shot in forecasting future results.

**Variables:** Shots, xG (FBref).

**Data:** FBref scraping needed.

**Priority:** MEDIUM — derivative of H-011.

---

### H-013: Elo Rating Edge

**Hypothesis:** A well-calibrated Elo model produces better calibrated probabilities than bookmaker odds for certain match types.

**Mechanism:** Elo captures long-run team strength without overfitting to recent results. May outperform in early season or after long breaks.

**Test:** Build Elo system. Compare Brier score vs bookmaker baseline across conditions.

**Variables:** Match results (to compute Elo), odds.

**Data:** Available — Elo is computed from results only.

**Priority:** HIGH — fully self-contained, strong baseline model.

---

### H-014: Poisson Model Edge

**Hypothesis:** A Poisson regression model produces better score predictions than implied by odds.

**Test:** Fit attack/defense strength Poisson model. Compare predicted score distributions vs market-implied probabilities.

**Variables:** Goals scored/conceded per team.

**Data:** Available.

**Priority:** HIGH — classical approach, good benchmark.

---

### H-015: Defensive Stability Underpricing

**Hypothesis:** Teams with consistently low goals conceded are underpriced because markets overweight attacking metrics.

**Mechanism:** Goals scored are salient; defensive solidity is harder to observe and price.

**Test:** Compute rolling defensive metrics. Test whether low-concession teams outperform market expectations.

**Variables:** Goals conceded, odds, results.

**Data:** Available.

**Priority:** MEDIUM.

---

### H-016: Manager Bounce Effect

**Hypothesis:** New manager appointments produce a short-term performance improvement that markets underprice.

**Mechanism:** Motivation spike, tactical reset, "new broom" effect.

**Test:** Identify manager changes. Measure performance vs market expectations in first 5 matches post-appointment.

**Variables:** Manager change dates (manual or scraped), odds, results.

**Data:** Requires enrichment — Transfermarkt or manual compilation.

**Priority:** LOW — labor-intensive data collection, small sample per event.

---

## Tier 3: Testable with External Data

Require sentiment data, weather, or other external sources.

---

### H-017: Social Sentiment as Contrarian Signal

**Hypothesis:** Extreme public sentiment (measured via Reddit/Twitter) is a negative predictor — the crowd is wrong when most confident.

**Mechanism:** Herd behavior, narrative-driven betting.

**Test:** Collect historical Reddit/Twitter sentiment for matches. Measure whether extreme sentiment predicts underperformance vs odds.

**Variables:** Sentiment scores, odds, results.

**Data:** Would need historical social data — expensive/difficult retroactively. Best tested prospectively.

**Priority:** LOW for backtest, HIGH for future prospective study.

---

### H-018: Weather Impact on Goals

**Hypothesis:** Extreme weather (heavy rain, very cold, high wind) reduces total goals below market expectations.

**Mechanism:** Physical difficulty, conservative tactics, poor passing.

**Test:** Match historical weather data to match dates/locations. Compare total goals in extreme vs normal conditions.

**Variables:** Weather data (OpenMeteo free API), match results, totals odds.

**Data:** Available — OpenMeteo has free historical data.

**Priority:** MEDIUM — fun hypothesis, clean test, but may have small effect size.

---

### H-019: Travel Distance Effect

**Hypothesis:** Away teams traveling long distances underperform market expectations.

**Mechanism:** Fatigue, disrupted routines, unfamiliar conditions.

**Test:** Compute travel distances between stadiums. Test performance vs market expectations as function of distance.

**Variables:** Stadium coordinates (available), match results, odds.

**Data:** Available with geocoding enrichment.

**Priority:** MEDIUM — clean mechanism, but may be absorbed by home advantage.

---

### H-020: Referee Bias Signal

**Hypothesis:** Certain referees systematically produce more home wins, more cards, or more goals, and markets don't fully price this.

**Test:** Measure referee-level statistics. Test whether knowing the referee improves prediction beyond odds.

**Variables:** Referee assignments (available in Football-Data.co.uk for recent seasons), outcomes.

**Data:** Partially available.

**Priority:** MEDIUM — interesting but small marginal signal expected.

---

## Tier 4: Advanced / Longer-Term

---

### H-021: Market Overreaction to Blowouts

**Hypothesis:** After a team wins or loses by 3+ goals, the market overadjusts in the next match.

**Test:** Measure performance vs odds in the match following a blowout result.

**Variables:** Score margins, subsequent match odds and results.

**Data:** Available.

**Priority:** MEDIUM.

---

### H-022: Cup Competition Fatigue Spillover

**Hypothesis:** Teams playing midweek cup matches underperform in the following weekend league match.

**Test:** Identify midweek cup participation. Compare weekend performance vs market expectations.

**Variables:** Cup fixture dates, league fixture dates, odds, results.

**Data:** Requires cup fixture data — available from FBref.

**Priority:** MEDIUM — related to H-005 but with a specific cup angle.

---

### H-023: Closing Line Value (CLV) as Meta-Signal

**Hypothesis:** Identifying which matches have the largest closing line movement identifies where the sharpest information entered the market.

**Mechanism:** CLV is the gold standard in professional betting. If we can predict where CLV will be large, we identify where information asymmetry exists.

**Test:** Measure opening-to-closing line movement. Test whether features predict large CLV.

**Variables:** Opening odds, closing odds, match features.

**Data:** Partially available.

**Priority:** HIGH conceptually, MEDIUM practically due to data gaps.

---

### H-024: League Strength Mispricing in European Competition

**Hypothesis:** Markets misprice teams from weaker leagues in European competition (Champions League, Europa League).

**Mechanism:** Markets may underweight league-strength differentials or overweight name recognition.

**Test:** Build league-strength index. Compare European competition results vs market expectations by league-strength gap.

**Variables:** European competition results and odds, league rankings.

**Data:** Would need European competition data — available from Football-Data.co.uk.

**Priority:** LOW — smaller sample, complex setup.

---

### H-025: Second-Half Season Regression

**Hypothesis:** Teams that significantly overperform xG in the first half of the season regress in the second half, and markets are slow to adjust.

**Mechanism:** "Luck" (outperforming underlying process) is mean-reverting. Markets may anchor to first-half results.

**Test:** Identify first-half xG overperformers. Measure second-half performance vs market expectations.

**Variables:** xG data, half-season splits, odds, results.

**Data:** Requires FBref xG data.

**Priority:** MEDIUM — elegant hypothesis, clean mechanism, moderate data needs.

---

## Execution Order

Based on data availability, sample size, and expected insight:

| Phase | Hypotheses | Rationale |
|---|---|---|
| Phase 1 | H-001, H-002, H-003, H-009, H-005 | Odds + results only. Largest samples. Immediate execution. |
| Phase 2 | H-013, H-014, H-010, H-004, H-006 | Computed features from existing data. No new sources needed. |
| Phase 3 | H-011, H-012, H-015, H-025 | Requires FBref scraping. xG is highest-value signal candidate. |
| Phase 4 | H-007, H-008, H-020, H-021, H-022 | Enrichment needed. Moderate effort, targeted hypotheses. |
| Phase 5 | H-017, H-018, H-019, H-016, H-023, H-024 | External data. Higher effort, uncertain payoff. |

---

## Rules

1. Test one hypothesis at a time. Do not peek at multiple results simultaneously.
2. Pre-register the analysis plan before running the experiment.
3. Apply Benjamini-Hochberg correction across all tested hypotheses at each checkpoint.
4. A signal is not "validated" until it survives holdout season testing.
5. Document every failure. Negative results prevent future waste.
