"""
portfolio.py

Correlated-Kelly portfolio optimiser for Apollo.

Standard fractional Kelly (see ``core.value_finder.kelly_fraction``) sizes every
bet as if it were statistically independent of the others.  In reality a slate of
football matches placed on the same day in the same league shares hidden risk
factors — weather, the referee pool, congestion / rotation effects, league-wide
scoring regimes.  Treating correlated bets as independent systematically *over*-bets
the book: realised variance is higher than Kelly assumes, and the bankroll hits
ruin materially faster (roughly ~40 %% faster for typical same-league slates) than
the independent model predicts.

This module:

  1. Estimates a pairwise correlation matrix from cheap, transparent heuristics.
  2. Solves a mean-variance approximation of the *joint* Kelly problem, which
     shrinks stakes on mutually correlated bets and caps total book exposure.
  3. Applies a drawdown brake that halves all stakes once the bankroll falls a
     configurable distance below its running peak.

scipy is optional.  When it is available the exact constrained optimum is found
with SLSQP; otherwise a closed-form shrinkage heuristic is used as a fallback.
numpy is required.

No side effects occur at import time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.value_finder import ValueBet, kelly_fraction

# scipy is an optional dependency.  Probe for it once at import time without
# executing anything; the flag gates the optimiser branch in PortfolioOptimizer.
try:  # pragma: no cover - trivial import guard
    from scipy.optimize import minimize as _scipy_minimize

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - exercised only when scipy is absent
    _scipy_minimize = None  # type: ignore[assignment]
    _HAS_SCIPY = False


# --------------------------------------------------------------------------- #
# Match-string parsing helpers
# --------------------------------------------------------------------------- #
def _parse_teams(match: str) -> tuple[str, str]:
    """
    Split a ``"Home vs Away"`` match label into its two team names.

    Args:
        match: Match label, conventionally ``"Home vs Away"``.

    Returns:
        Tuple ``(home, away)``.  If the separator is missing the whole string is
        returned as the home team and the away team is empty.
    """
    for sep in (" vs ", " v ", " - ", " vs. "):
        if sep in match:
            home, _, away = match.partition(sep)
            return home.strip(), away.strip()
    return match.strip(), ""


def _league_hint(bet: ValueBet, context: Optional[dict] = None) -> str:
    """
    Best-effort extraction of a league/competition label for a bet.

    The :class:`~core.value_finder.ValueBet` dataclass does not itself carry a
    league field, so we look it up from an optional ``context`` dict keyed by the
    match label.  Absent any hint we return an empty string, which the correlation
    heuristic treats as "unknown".

    Args:
        bet: The bet to label.
        context: Optional mapping ``{match_label: {"league": str, "date": str,
            "competition": str}}``.

    Returns:
        A league string (possibly empty).
    """
    if context:
        info = context.get(bet.match)
        if isinstance(info, dict):
            return str(info.get("league", "") or info.get("competition", ""))
    return ""


def _date_hint(bet: ValueBet, context: Optional[dict] = None) -> str:
    """Extract a date string for a bet from the optional context dict."""
    if context:
        info = context.get(bet.match)
        if isinstance(info, dict):
            return str(info.get("date", ""))
    return ""


# --------------------------------------------------------------------------- #
# Correlation estimation
# --------------------------------------------------------------------------- #
def estimate_correlation(
    bet_a: ValueBet,
    bet_b: ValueBet,
    context: Optional[dict] = None,
    competition: str = "",
) -> float:
    """
    Estimate the correlation between two bets using transparent heuristics.

    The ladder, from strongest to weakest shared risk:

    ======================================  ===========
    Relationship                            Correlation
    ======================================  ===========
    Same match (one bet per match — n/a)    1.00
    Same league, same day                   0.25
    Same competition (e.g. both WC2026)     0.15
    Different competition, same day         0.05
    Different competition, different day    0.00
    ======================================  ===========

    These are deliberately conservative point estimates rather than fitted values
    — the goal is to *bias stakes downward* on plausibly-correlated bets, not to
    nail the true covariance.  Over-estimating correlation merely leaves edge on
    the table; under-estimating it courts ruin, so we err toward caution.

    Args:
        bet_a: First bet.
        bet_b: Second bet.
        context: Optional per-match metadata (see :func:`_league_hint`).
        competition: Optional global competition label applied to all bets in the
            slate (e.g. ``"WC2026"``).

    Returns:
        Estimated Pearson-style correlation in ``[0.0, 1.0]``.
    """
    # Same match is impossible under the one-bet-per-match rule, but we honour the
    # contract for completeness / defensive callers.
    if bet_a.match == bet_b.match:
        return 1.0

    league_a = _league_hint(bet_a, context)
    league_b = _league_hint(bet_b, context)
    date_a = _date_hint(bet_a, context)
    date_b = _date_hint(bet_b, context)

    same_day = bool(date_a) and bool(date_b) and date_a == date_b
    # When no dates are supplied we assume the slate is for a single matchday,
    # which is the common Apollo usage; this keeps the heuristic protective.
    if not date_a and not date_b:
        same_day = True

    same_league = bool(league_a) and bool(league_b) and league_a == league_b

    if same_league and same_day:
        return 0.25

    # A shared overarching competition (passed in globally) is a weaker but real
    # link — shared travel, climate, scheduling body, ball, VAR protocols.
    if competition:
        return 0.15

    if same_day:
        return 0.05

    return 0.0


def build_correlation_matrix(
    bets: list[ValueBet],
    competition: str = "",
    context: Optional[dict] = None,
) -> np.ndarray:
    """
    Build the symmetric ``N x N`` correlation matrix for a slate of bets.

    The diagonal is 1.0 (each bet is perfectly correlated with itself); the
    off-diagonals come from :func:`estimate_correlation`.  A tiny jitter is added
    to the diagonal if needed so the matrix is positive semi-definite (PSD), which
    the downstream mean-variance objective requires for a well-posed optimum.

    Args:
        bets: Slate of value bets.
        competition: Optional global competition label.
        context: Optional per-match metadata.

    Returns:
        An ``(N, N)`` numpy array, symmetric and PSD.
    """
    n = len(bets)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    C = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            rho = estimate_correlation(
                bets[i], bets[j], context=context, competition=competition
            )
            C[i, j] = rho
            C[j, i] = rho

    # Ensure positive semi-definiteness.  The smallest eigenvalue of a correlation
    # matrix built from arbitrary heuristics can dip slightly negative; nudge the
    # diagonal until it is non-negative within tolerance.
    if n > 1:
        eigmin = float(np.linalg.eigvalsh(C).min())
        if eigmin < 1e-8:
            np.fill_diagonal(C, np.diag(C) + (abs(eigmin) + 1e-6))

    return C


# --------------------------------------------------------------------------- #
# Portfolio optimiser
# --------------------------------------------------------------------------- #
class PortfolioOptimizer:
    """
    Correlated-Kelly portfolio optimiser using a mean-variance approximation.

    The exact growth-optimal (log-utility) stake vector under correlation has no
    closed form, but a second-order Taylor expansion of expected log-wealth around
    small stakes yields the classic mean-variance objective:

        maximise   g(f) = Σ_i f_i b_i p_i  -  ½ Σ_i Σ_j f_i f_j C_ij σ_i σ_j

        subject to f_i >= 0,  Σ_i f_i <= max_total_exposure,  f_i <= max_single_bet

    where for each bet ``i``:

        b_i  = odds_i - 1                  (net decimal payout)
        p_i  = model probability
        σ_i  = sqrt(p_i (1 - p_i)) * b_i   (per-unit-stake return std-dev)
        C_ij = estimated correlation       (build_correlation_matrix)

    The linear term rewards edge; the quadratic term penalises *covariance* of
    returns, so correlated bets are jointly throttled rather than each sized in
    isolation.  We seed the solver from the per-bet fractional-Kelly stakes and
    solve with scipy SLSQP.  If scipy is unavailable we fall back to a closed-form
    shrinkage heuristic that scales each Kelly stake down by its average
    correlation with the rest of the book.

    A drawdown brake is applied last: once the bankroll sits below
    ``(1 - drawdown_threshold)`` of its running peak, every stake is multiplied by
    ``drawdown_reduction``.
    """

    def __init__(
        self,
        max_total_exposure: float = 0.20,
        max_single_bet: float = 0.10,
        kelly_fraction: float = 0.25,
        drawdown_threshold: float = 0.15,
        drawdown_reduction: float = 0.50,
    ) -> None:
        self.max_total_exposure = float(max_total_exposure)
        self.max_single_bet = float(max_single_bet)
        self.kelly_fraction = float(kelly_fraction)
        self.drawdown_threshold = float(drawdown_threshold)
        self.drawdown_reduction = float(drawdown_reduction)

    # -- internal maths ----------------------------------------------------- #
    def _raw_kelly(self, bets: list[ValueBet]) -> np.ndarray:
        """
        Per-bet fractional-Kelly stakes, recomputed from odds and model prob so the
        optimiser is self-consistent with ``self.kelly_fraction``.
        """
        return np.array(
            [
                kelly_fraction(b.odds, b.model_prob, fraction=self.kelly_fraction)
                for b in bets
            ],
            dtype=float,
        )

    def _edges(self, bets: list[ValueBet]) -> tuple[np.ndarray, np.ndarray]:
        """
        Return per-bet expected net return ``μ_i`` and an effective risk scale
        ``σ_i`` used to build the quadratic penalty.

        We anchor ``σ_i`` so that the *uncorrelated, unconstrained* mean-variance
        optimum reproduces the exact fractional-Kelly stake.  For a single binary
        bet, true Kelly maximises log-growth at ``f* = (b p − q) / b = μ / b``.
        Our diagonal objective ``f μ − ½ σ² f²`` is optimised at ``f = μ / σ²``,
        so choosing ``σ_i² = b_i / kelly_fraction`` makes the diagonal optimum equal
        ``kelly_fraction · μ / b`` — i.e. fractional Kelly — exactly.  The
        off-diagonal correlation terms then *only ever reduce* stakes below that
        anchor, which is the conservative behaviour the portfolio model exists to
        provide.  (The naive ``σ² = Var[return]`` overshoots because the small-stake
        Taylor expansion understates log-utility curvature at long odds.)
        """
        b = np.array([bet.odds - 1.0 for bet in bets], dtype=float)
        p = np.array([bet.model_prob for bet in bets], dtype=float)
        mean_ret = b * p - (1.0 - p)            # μ_i: E[return per unit stake]
        kf = self.kelly_fraction if self.kelly_fraction > 0 else 1.0
        # σ_i² = b_i / kelly_fraction  →  σ_i = sqrt(b_i / kelly_fraction)
        sigma = np.sqrt(np.clip(b, 1e-9, None) / kf)
        return mean_ret, sigma

    def _solve(
        self,
        bets: list[ValueBet],
        C: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the constrained mean-variance problem, returning portfolio fractions.

        Falls back to the shrinkage heuristic when scipy is unavailable or the
        numerical solve fails to converge.
        """
        n = len(bets)
        raw = self._raw_kelly(bets)
        if n == 0:
            return raw

        mean_ret, sigma = self._edges(bets)
        # Quadratic penalty matrix Σ_ij = C_ij σ_i σ_j.  σ_i is already calibrated
        # in _edges so the uncorrelated diagonal optimum equals fractional Kelly;
        # the off-diagonal correlation terms shrink correlated stakes from there.
        cov = C * np.outer(sigma, sigma)

        if not _HAS_SCIPY:
            return self._shrinkage_fallback(raw, C)

        def neg_growth(f: np.ndarray) -> float:
            return -(f @ mean_ret - 0.5 * (f @ cov @ f))

        def neg_growth_grad(f: np.ndarray) -> np.ndarray:
            return -(mean_ret - cov @ f)

        # f_i in [0, max_single_bet]; Σ f_i <= max_total_exposure.
        bounds = [(0.0, self.max_single_bet)] * n
        constraints = [
            {
                "type": "ineq",
                "fun": lambda f: self.max_total_exposure - np.sum(f),
                "jac": lambda f: -np.ones_like(f),
            }
        ]
        # Seed from clipped raw-Kelly so the solver starts inside the feasible set.
        x0 = np.clip(raw, 0.0, self.max_single_bet)
        if x0.sum() > self.max_total_exposure and x0.sum() > 0:
            x0 = x0 * (self.max_total_exposure / x0.sum())

        try:
            res = _scipy_minimize(
                neg_growth,
                x0,
                jac=neg_growth_grad,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-9},
            )
            if res.success:
                return np.clip(res.x, 0.0, self.max_single_bet)
        except Exception:
            pass  # fall through to heuristic

        return self._shrinkage_fallback(raw, C)

    def _shrinkage_fallback(self, raw: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Closed-form fallback when scipy is unavailable.

        Each Kelly stake is shrunk by its mean off-diagonal correlation::

            f_i_corr = f_i_kelly * (1 - avg_corr_i * 0.5)

        then the per-bet and total-exposure caps are enforced (the latter by
        proportional scaling).
        """
        n = len(raw)
        if n == 0:
            return raw
        if n == 1:
            return np.clip(raw, 0.0, self.max_single_bet)

        off_diag_sum = C.sum(axis=1) - np.diag(C)
        avg_corr = off_diag_sum / (n - 1)
        shrunk = raw * (1.0 - avg_corr * 0.5)
        shrunk = np.clip(shrunk, 0.0, self.max_single_bet)

        total = shrunk.sum()
        if total > self.max_total_exposure and total > 0:
            shrunk = shrunk * (self.max_total_exposure / total)
        return shrunk

    def _apply_drawdown_control(
        self,
        fractions: np.ndarray,
        bankroll: float,
        peak: float,
    ) -> np.ndarray:
        """
        Halve (scale by ``drawdown_reduction``) all stakes while in drawdown.

        The brake engages when ``bankroll / peak < 1 - drawdown_threshold``.  This
        protects against the losing streaks that are statistically certain even
        with a genuine edge: cutting stakes during a drawdown reduces the chance of
        compounding losses into ruin, at the cost of slower recovery.

        Args:
            fractions: Pre-brake stake fractions.
            bankroll: Current bankroll.
            peak: Running peak bankroll.

        Returns:
            Possibly-reduced fractions.
        """
        if peak > 0 and (bankroll / peak) < (1.0 - self.drawdown_threshold):
            return fractions * self.drawdown_reduction
        return fractions

    # -- public API --------------------------------------------------------- #
    def optimize(
        self,
        bets: list[ValueBet],
        bankroll: float,
        peak_bankroll: float,
        competition: str = "",
        context: Optional[dict] = None,
    ) -> list[dict]:
        """
        Optimise stakes across a slate of correlated bets.

        Args:
            bets: Candidate value bets (one per match).
            bankroll: Current bankroll, used to convert fractions to amounts.
            peak_bankroll: Running peak bankroll, used for the drawdown brake.
            competition: Optional global competition label feeding the correlation
                heuristic.
            context: Optional per-match metadata ``{match: {"league", "date", ...}}``.

        Returns:
            List of dicts (one per bet) sorted by ``stake_amount`` descending; see
            the module docstring / task spec for the schema.
        """
        if not bets:
            return []

        raw_kelly = self._raw_kelly(bets)
        C = build_correlation_matrix(bets, competition=competition, context=context)
        portfolio_kelly = self._solve(bets, C)

        # Drawdown brake applies to the final, correlation-adjusted stakes.
        final = self._apply_drawdown_control(
            portfolio_kelly, bankroll, peak_bankroll
        )

        results: list[dict] = []
        for i, bet in enumerate(bets):
            frac = float(final[i])
            results.append(
                {
                    "match": bet.match,
                    "outcome": bet.outcome,
                    "team": bet.team,
                    "odds": float(bet.odds),
                    "edge": float(bet.edge),
                    "kelly_raw": float(raw_kelly[i]),
                    "kelly_portfolio": float(portfolio_kelly[i]),
                    "stake_fraction": frac,
                    "stake_amount": frac * float(bankroll),
                }
            )

        results.sort(key=lambda r: r["stake_amount"], reverse=True)
        return results

    def summary(self, result: list[dict], bankroll: float) -> str:
        """
        Human-readable one-paragraph portfolio summary.

        Reports total exposure (₹ and %% of bankroll) and a diversification score
        in ``[0, 1]``, where 1 means the correlation optimiser left stakes intact
        (well-diversified) and lower values mean stakes were heavily shrunk due to
        correlation / exposure caps / drawdown.
        """
        if not result:
            return "Portfolio is empty — no qualifying bets."

        total_stake = sum(r["stake_amount"] for r in result)
        total_frac = total_stake / bankroll if bankroll else 0.0
        raw_total = sum(r["kelly_raw"] for r in result)
        port_total = sum(r["kelly_portfolio"] for r in result)
        diversification = (port_total / raw_total) if raw_total > 0 else 1.0

        return (
            f"Portfolio: {len(result)} bets | "
            f"Total exposure: ₹{total_stake:,.2f} ({total_frac:.1%} of bankroll) | "
            f"Raw-Kelly would stake {raw_total:.1%}, "
            f"correlation-adjusted to {port_total:.1%} | "
            f"Diversification retention score: {diversification:.2f}"
        )


# --------------------------------------------------------------------------- #
# Pretty-printing
# --------------------------------------------------------------------------- #
def format_portfolio(result: list[dict], bankroll: float) -> str:
    """
    Render a portfolio result as a human-readable text table.

    Shows per-bet: match, outcome/team, odds, edge%, raw-Kelly%, portfolio-Kelly%
    and ₹ stake, plus footer lines for total exposure and drawdown status.

    Args:
        result: Output of :meth:`PortfolioOptimizer.optimize`.
        bankroll: Current bankroll (for the exposure %% footer).

    Returns:
        Formatted multi-line string.
    """
    if not result:
        return "No portfolio — no qualifying bets."

    header = (
        f"{'Match':<32} {'Bet':<18} "
        f"{'Odds':>6} {'Edge':>7} {'KellyRaw':>9} {'KellyPort':>10} {'Stake':>12}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    total_stake = 0.0
    for r in result:
        total_stake += r["stake_amount"]
        bet_label = f"{r['outcome'].upper()} {r['team']}".strip()
        lines.append(
            f"{r['match']:<32.32} {bet_label:<18.18} "
            f"{r['odds']:>6.2f} {r['edge']:>+7.2%} "
            f"{r['kelly_raw']:>9.2%} {r['kelly_portfolio']:>10.2%} "
            f"₹{r['stake_amount']:>11,.2f}"
        )

    lines.append(sep)
    exposure_frac = total_stake / bankroll if bankroll else 0.0
    lines.append(
        f"TOTAL EXPOSURE: ₹{total_stake:,.2f} / ₹{bankroll:,.2f} "
        f"({exposure_frac:.1%} of bankroll)"
    )

    # Infer drawdown status from whether portfolio stakes were uniformly halved.
    raw_total = sum(r["kelly_portfolio"] for r in result)
    stake_total_frac = sum(r["stake_fraction"] for r in result)
    if raw_total > 0 and stake_total_frac < raw_total - 1e-9:
        lines.append("DRAWDOWN BRAKE: ENGAGED — stakes reduced for capital protection.")
    else:
        lines.append("DRAWDOWN BRAKE: off — bankroll near peak.")
    lines.append(sep)

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Three sample bets: two in the same league/day (correlated) and one isolated.
    demo_bets = [
        ValueBet(
            match="Arsenal vs Chelsea",
            outcome="home",
            team="Arsenal",
            odds=2.10,
            model_prob=0.55,
            fair_implied=0.476,
            edge=0.074,
            kelly=0.0,
        ),
        ValueBet(
            match="Liverpool vs Everton",
            outcome="home",
            team="Liverpool",
            odds=1.80,
            model_prob=0.62,
            fair_implied=0.556,
            edge=0.064,
            kelly=0.0,
        ),
        ValueBet(
            match="Real Madrid vs Barcelona",
            outcome="away",
            team="Barcelona",
            odds=3.20,
            model_prob=0.38,
            fair_implied=0.313,
            edge=0.067,
            kelly=0.0,
        ),
    ]

    # Context links the two English fixtures to the same league + date.
    demo_context = {
        "Arsenal vs Chelsea": {"league": "EPL", "date": "2026-06-25"},
        "Liverpool vs Everton": {"league": "EPL", "date": "2026-06-25"},
        "Real Madrid vs Barcelona": {"league": "LaLiga", "date": "2026-06-25"},
    }

    optimizer = PortfolioOptimizer()
    bankroll = 10_000.0

    print("=== Normal bankroll (at peak) ===")
    res = optimizer.optimize(
        demo_bets, bankroll=bankroll, peak_bankroll=bankroll, context=demo_context
    )
    print(format_portfolio(res, bankroll))
    print(optimizer.summary(res, bankroll))

    print("\n=== In drawdown (bankroll 20% below peak) ===")
    res_dd = optimizer.optimize(
        demo_bets, bankroll=8_000.0, peak_bankroll=10_000.0, context=demo_context
    )
    print(format_portfolio(res_dd, 8_000.0))
    print(optimizer.summary(res_dd, 8_000.0))

    print(f"\nscipy available: {_HAS_SCIPY}")
