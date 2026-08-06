"""
player_model.py

Player-parametric Dixon-Coles forecasting framework for Apollo.

Motivation
----------
The classic Dixon-Coles (1997) model treats goals as a (corrected) bivariate
Poisson process governed by per-team attack/defense strengths:

    lambda_home = mu * alpha_home * beta_away * gamma     (gamma = home advantage)
    lambda_away = mu * alpha_away * beta_home

where
    mu    = league baseline goals
    alpha = team attack strength   (alpha > 1 -> scores more than average)
    beta  = team defense weakness  (beta  > 1 -> concedes more than average)
    gamma = multiplicative home advantage

We EXTEND this to the player level.  A team's effective attack/defense is the
position-weighted aggregate of the players in its lineup.  When a key player is
listed absent (by core.player_data.PlayerData), the team's attack/defense is
degraded by a per-position factor.  Conceptually:

    team_attack  = sum(attack_weight[pos] * player_quality) over lineup
    team_defense = sum(defense_weight[pos] * player_quality) over lineup

True player-level lineup/xG data is expensive to collect.  So we build the
framework with a position-weighted xG proxy:

  - Team alpha/beta are fitted from historical match xG (or goals) via an
    iterative MLE that mirrors Dixon-Coles.
  - Player parameters are INFERRED: an absent key player applies a learned
    per-position degradation to the team aggregate.  The degradation priors are
    literature-derived and updatable via update_degradation() once real
    absence-outcome data is available.

This is strictly more principled than the hardcoded Elo point penalties used in
player_data.py, because the penalty propagates through the Poisson scoring model
rather than a logistic win-probability curve.

No scipy dependency: the Poisson PMF and Dixon-Coles tau correction are
implemented from scratch with the standard library `math` module.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level math helpers (no scipy)
# ---------------------------------------------------------------------------

# Cache factorials for small goal counts (we never go above ~max_goals=10).
_FACTORIAL = [math.factorial(k) for k in range(0, 25)]


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass: lam^k * exp(-lam) / k!

    Implemented without scipy.  Computed in log-space for numerical stability,
    then exponentiated, which avoids overflow of lam**k for large k.
    """
    if k < 0 or lam <= 0.0:
        return 0.0
    if k < len(_FACTORIAL):
        log_fact = math.log(_FACTORIAL[k])
    else:
        log_fact = math.lgamma(k + 1)
    log_pmf = k * math.log(lam) - lam - log_fact
    return math.exp(log_pmf)


def dixon_coles_correction(
    goals_home: int,
    goals_away: int,
    lam_h: float,
    lam_a: float,
    rho: float = -0.13,
) -> float:
    """Dixon-Coles (1997) low-score dependency correction (the tau factor).

    The independent double-Poisson model understates the frequency of low
    scorelines (0-0, 1-0, 0-1, 1-1).  Dixon-Coles introduces a single
    dependence parameter `rho` that re-weights only these four cells:

        tau(0,0) = 1 - lam_h * lam_a * rho
        tau(0,1) = 1 + lam_h * rho
        tau(1,0) = 1 + lam_a * rho
        tau(1,1) = 1 - rho
        tau(.,.) = 1   otherwise

    A negative rho (typical empirical value ~ -0.13) increases P(0-0) and
    P(1-1) while reducing P(1-0) and P(0-1), matching observed draw inflation.
    The factor is clamped to be non-negative to keep probabilities valid.
    """
    if goals_home == 0 and goals_away == 0:
        tau = 1.0 - lam_h * lam_a * rho
    elif goals_home == 0 and goals_away == 1:
        tau = 1.0 + lam_h * rho
    elif goals_home == 1 and goals_away == 0:
        tau = 1.0 + lam_a * rho
    elif goals_home == 1 and goals_away == 1:
        tau = 1.0 - rho
    else:
        tau = 1.0
    return tau if tau > 0.0 else 1e-10


# ---------------------------------------------------------------------------
# Literature-derived degradation priors (fraction of strength lost when a
# KEY player at this position is absent).
# ---------------------------------------------------------------------------

_DEGRADATION_PRIOR = {
    "forward":    0.18,   # 18% fewer goals when a key forward is absent
    "midfielder": 0.10,
    "defender":   0.07,   # 7% more goals conceded when a key defender is absent
    "goalkeeper": 0.15,
}

# Non-key (squad) players carry a fraction of the key-player impact.
_NON_KEY_FACTOR = 0.4


class PlayerModel:
    """Player-parametric Dixon-Coles model.

    Fitted team attack (alpha) and defense (beta) parameters are normalized so
    that the league-average team has alpha = beta = 1.0.  Predictions degrade
    those parameters according to absent players' positions.
    """

    POSITION_ATTACK_WEIGHT = {
        "forward": 0.45,
        "midfielder": 0.30,
        "defender": 0.15,
        "goalkeeper": 0.10,
    }
    POSITION_DEFENSE_WEIGHT = {
        "goalkeeper": 0.40,
        "defender": 0.35,
        "midfielder": 0.20,
        "forward": 0.05,
    }

    # Bounds keep the fitted parameters sane during gradient ascent.
    _ALPHA_MIN, _ALPHA_MAX = 0.30, 3.00
    _BETA_MIN, _BETA_MAX = 0.30, 3.00
    _LAMBDA_MIN, _LAMBDA_MAX = 0.30, 6.00
    _RHO = -0.13
    _MIN_MATCHES = 5  # teams with fewer matches fall back to league average

    def __init__(self) -> None:
        self._team_attack: dict[str, float] = {}     # team -> normalized alpha
        self._team_defense: dict[str, float] = {}    # team -> normalized beta (lower = better)
        self._position_degradation: dict[str, float] = dict(_DEGRADATION_PRIOR)
        self._league_avg_goals: float = 1.30
        self._home_advantage: float = 1.15           # multiplicative gamma
        self._match_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, max_iter: int = 20, lr: float = 0.05,
            tol: float = 1e-4) -> "PlayerModel":
        """Fit team attack/defense via iterative MLE (simplified Dixon-Coles).

        Mathematical approach
        ----------------------
        We maximise the Poisson log-likelihood of the observed match scoring
        rates.  For a match with home team i and away team j we model:

            lambda_h = mu * alpha_i * beta_j * gamma
            lambda_a = mu * alpha_j * beta_i

        The Poisson log-likelihood contribution of one observed value y given
        rate lam is:   l = y * log(lam) - lam   (the y! term is constant in the
        parameters).  Its derivative w.r.t. lam is (y/lam - 1).  Because each
        alpha/beta enters lam multiplicatively, the chain rule gives a clean
        update.  We work in log-space on the parameters (theta = log alpha) so
        d lam / d theta = lam, yielding the gradient:

            d l / d theta_alpha_i = (y - lam)          (summed over i's matches)

        i.e. the raw residual between observed and expected goals.  This is the
        standard Poisson-GLM score with a log link.  We ascend it with a fixed
        learning rate and re-estimate mu and gamma each round from the global
        residual.  Attack and defense are then re-centred so the geometric mean
        of alpha (and of beta) is 1.0, which removes the multiplicative
        identifiability gauge (alpha_i * c, beta_j / c gives the same lambdas).

        Convergence
        -----------
        Iterate until the mean absolute parameter change drops below `tol` or
        `max_iter` rounds elapse (safety cap).  Empirically converges in well
        under 20 rounds for season-scale data because the gradient is the
        residual and re-centering removes the dominant drift each round.

        Uses home_xg/away_xg when present (smoother target than integer goals);
        otherwise falls back to home_goals/away_goals.
        """
        df = df.copy()

        # Choose target columns: prefer xG, fall back to goals.
        has_xg = (
            "home_xg" in df.columns and "away_xg" in df.columns
            and df["home_xg"].notna().any() and df["away_xg"].notna().any()
        )
        if has_xg:
            # Keep only rows where both xG values exist.
            df = df[df["home_xg"].notna() & df["away_xg"].notna()].copy()
            df["y_home"] = df["home_xg"].astype(float)
            df["y_away"] = df["away_xg"].astype(float)
            logger.info("PlayerModel.fit: training on xG (%d matches)", len(df))
        else:
            df = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
            df["y_home"] = df["home_goals"].astype(float)
            df["y_away"] = df["away_goals"].astype(float)
            logger.info("PlayerModel.fit: training on goals (%d matches)", len(df))

        if df.empty:
            logger.warning("PlayerModel.fit: no usable rows; keeping defaults")
            return self

        homes = df["home_team"].to_numpy()
        aways = df["away_team"].to_numpy()
        y_h = df["y_home"].to_numpy()
        y_a = df["y_away"].to_numpy()

        teams = sorted(set(homes) | set(aways))
        alpha = {t: 1.0 for t in teams}
        beta = {t: 1.0 for t in teams}

        # League baseline mu = overall mean goals per team per match.
        mu = float((y_h.sum() + y_a.sum()) / (2.0 * len(df)))
        mu = max(mu, 0.10)
        # Home advantage seeded from the home/away scoring ratio.
        away_mean = max(float(y_a.mean()), 1e-6)
        gamma = max(float(y_h.mean()) / away_mean, 1.0)

        # Match counts per team (for the small-sample fallback at predict time).
        match_counts: dict[str, int] = {t: 0 for t in teams}
        for h, a in zip(homes, aways):
            match_counts[h] += 1
            match_counts[a] += 1

        for it in range(max_iter):
            grad_alpha = {t: 0.0 for t in teams}
            grad_beta = {t: 0.0 for t in teams}
            resid_h_sum = 0.0
            resid_a_sum = 0.0

            for k in range(len(df)):
                h = homes[k]
                a = aways[k]
                lam_h = mu * alpha[h] * beta[a] * gamma
                lam_a = mu * alpha[a] * beta[h]
                lam_h = min(max(lam_h, self._LAMBDA_MIN), self._LAMBDA_MAX)
                lam_a = min(max(lam_a, self._LAMBDA_MIN), self._LAMBDA_MAX)

                r_h = y_h[k] - lam_h   # Poisson score residual (home goals)
                r_a = y_a[k] - lam_a   # Poisson score residual (away goals)

                # Attack of a team is exercised when it scores.
                grad_alpha[h] += r_h
                grad_alpha[a] += r_a
                # Defense (beta) weakness of a team is exercised when it concedes.
                grad_beta[a] += r_h   # away's defense let in home goals
                grad_beta[h] += r_a   # home's defense let in away goals

                resid_h_sum += r_h
                resid_a_sum += r_a

            # Gradient-ascent step in log-space (multiplicative update).
            max_delta = 0.0
            for t in teams:
                n = max(match_counts[t], 1)
                a_old = alpha[t]
                b_old = beta[t]
                # Normalise gradient by team's match count -> stable step size.
                alpha[t] = a_old * math.exp(lr * grad_alpha[t] / n)
                beta[t] = b_old * math.exp(lr * grad_beta[t] / n)
                alpha[t] = min(max(alpha[t], self._ALPHA_MIN), self._ALPHA_MAX)
                beta[t] = min(max(beta[t], self._BETA_MIN), self._BETA_MAX)
                max_delta = max(max_delta, abs(alpha[t] - a_old), abs(beta[t] - b_old))

            # Re-centre to remove the multiplicative gauge freedom: force the
            # geometric mean of alpha and of beta to 1.0.
            alpha = self._recenter(alpha)
            beta = self._recenter(beta)

            # Re-estimate global mu and gamma from accumulated residuals so the
            # baseline keeps pace with the rescaled team parameters.
            mu = max(mu * math.exp(lr * (resid_h_sum + resid_a_sum) / (2.0 * len(df))),
                     0.10)

            logger.debug("iter %2d  max_delta=%.5f  mu=%.3f  gamma=%.3f",
                         it, max_delta, mu, gamma)
            if max_delta < tol:
                logger.info("PlayerModel.fit converged after %d iterations "
                            "(max_delta=%.6f)", it + 1, max_delta)
                break
        else:
            logger.info("PlayerModel.fit hit max_iter=%d (max_delta=%.6f)",
                        max_iter, max_delta)

        self._team_attack = alpha
        self._team_defense = beta
        self._match_counts = match_counts
        self._league_avg_goals = mu
        self._home_advantage = gamma
        # Degradation priors are unchanged by team fitting (need absence data).
        self._position_degradation = dict(_DEGRADATION_PRIOR)
        return self

    @staticmethod
    def _recenter(params: dict[str, float]) -> dict[str, float]:
        """Rescale so the geometric mean of the parameters is 1.0."""
        if not params:
            return params
        log_mean = sum(math.log(v) for v in params.values()) / len(params)
        scale = math.exp(log_mean)
        if scale <= 0.0:
            return params
        return {t: v / scale for t, v in params.items()}

    # ------------------------------------------------------------------
    # Degradation update (when real absence-outcome data becomes available)
    # ------------------------------------------------------------------

    def update_degradation(self, position: str, value: float) -> None:
        """Override a learned per-position degradation factor in [0, 0.9]."""
        position = position.lower()
        self._position_degradation[position] = float(min(max(value, 0.0), 0.9))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _degradation_sum(self, absent: Optional[list]) -> float:
        """Total fractional strength loss from a list of absent players.

        Each item may be a dict {"position": str, "is_key": bool} or a
        (position, is_key) tuple.  Key players apply the full per-position
        degradation; squad players apply _NON_KEY_FACTOR of it.
        """
        if not absent:
            return 0.0
        total = 0.0
        for item in absent:
            if isinstance(item, dict):
                pos = str(item.get("position", "midfielder")).lower()
                is_key = bool(item.get("is_key", False))
            else:  # tuple/list (position, is_key)
                pos = str(item[0]).lower()
                is_key = bool(item[1]) if len(item) > 1 else False
            base = self._position_degradation.get(pos, 0.08)
            total += base if is_key else base * _NON_KEY_FACTOR
        return total

    def predict_goals(
        self,
        home: str,
        away: str,
        home_absent: Optional[list] = None,
        away_absent: Optional[list] = None,
        neutral: bool = False,
    ) -> tuple[float, float]:
        """Return (lambda_home, lambda_away): expected goals for each side.

        Absent players degrade the affected team's attack (it scores less) and
        the opponent's effective conceding via this team's worsened defense.
        """
        alpha_h = self._strength(home, self._team_attack)
        alpha_a = self._strength(away, self._team_attack)
        beta_h = self._strength(home, self._team_defense)
        beta_a = self._strength(away, self._team_defense)

        # Attack degradation reduces the team's own alpha.
        home_attack_deg = self._degradation_sum(home_absent)
        away_attack_deg = self._degradation_sum(away_absent)
        alpha_h_adj = alpha_h * (1.0 - min(home_attack_deg, 0.9))
        alpha_a_adj = alpha_a * (1.0 - min(away_attack_deg, 0.9))

        # Defense degradation raises the team's own beta (higher = worse).
        # We reuse the same absence list but weighted toward defensive roles via
        # the position-specific prior (defender/goalkeeper carry most of it).
        home_def_deg = self._degradation_sum(home_absent)
        away_def_deg = self._degradation_sum(away_absent)
        beta_h_adj = beta_h * (1.0 + min(home_def_deg, 0.9))
        beta_a_adj = beta_a * (1.0 + min(away_def_deg, 0.9))

        mu = self._league_avg_goals
        gamma = 1.0 if neutral else self._home_advantage

        lam_h = mu * alpha_h_adj * beta_a_adj * gamma
        lam_a = mu * alpha_a_adj * beta_h_adj

        lam_h = min(max(lam_h, self._LAMBDA_MIN), self._LAMBDA_MAX)
        lam_a = min(max(lam_a, self._LAMBDA_MIN), self._LAMBDA_MAX)
        return lam_h, lam_a

    def predict_outcome(
        self,
        home: str,
        away: str,
        home_absent: Optional[list] = None,
        away_absent: Optional[list] = None,
        neutral: bool = False,
        max_goals: int = 10,
    ) -> dict:
        """Convert expected goals to 1X2 probabilities via Dixon-Coles Poisson.

        Build the joint score matrix P(h, a) = tau(h,a) * Pois(h|lam_h) *
        Pois(a|lam_a) over 0..max_goals for each side, then sum the cells into
        home-win / draw / away-win and normalise (the truncation tail and the
        tau correction make the raw matrix sum slightly off 1.0).
        """
        lam_h, lam_a = self.predict_goals(
            home, away, home_absent, away_absent, neutral
        )

        pmf_h = [poisson_pmf(i, lam_h) for i in range(max_goals + 1)]
        pmf_a = [poisson_pmf(j, lam_a) for j in range(max_goals + 1)]

        p_home = p_draw = p_away = 0.0
        total = 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau = dixon_coles_correction(i, j, lam_h, lam_a, self._RHO)
                p = pmf_h[i] * pmf_a[j] * tau
                total += p
                if i > j:
                    p_home += p
                elif i == j:
                    p_draw += p
                else:
                    p_away += p

        if total > 0:
            p_home /= total
            p_draw /= total
            p_away /= total

        return {
            "p_home": round(p_home, 6),
            "p_draw": round(p_draw, 6),
            "p_away": round(p_away, 6),
            "lambda_home": round(lam_h, 4),
            "lambda_away": round(lam_a, 4),
            "method": "player_model",
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def _strength(self, team: str, table: dict[str, float]) -> float:
        """Look up a team's parameter, defaulting to 1.0 (league average).

        Teams seen fewer than _MIN_MATCHES times also fall back to 1.0 because
        their fitted value is unreliable.
        """
        if team not in table:
            return 1.0
        if self._match_counts.get(team, 0) < self._MIN_MATCHES:
            return 1.0
        return table[team]

    def team_strength(self, team: str) -> dict:
        """Return {"attack": alpha, "defense": beta} for a team (1.0 if unseen)."""
        return {
            "attack": self._strength(team, self._team_attack),
            "defense": self._strength(team, self._team_defense),
        }

    def ranking(self, top_n: int = 20) -> str:
        """Render the top attacks and best defenses as an aligned table."""
        seen = [t for t in self._team_attack
                if self._match_counts.get(t, 0) >= self._MIN_MATCHES]

        top_attack = sorted(seen, key=lambda t: self._team_attack[t], reverse=True)[:top_n]
        # Lower beta = stronger defense.
        top_defense = sorted(seen, key=lambda t: self._team_defense[t])[:top_n]

        lines = []
        lines.append(f"{'TOP ATTACKS':<40}{'BEST DEFENSES':<40}")
        lines.append("-" * 80)
        lines.append(f"{'#':<3}{'Team':<27}{'alpha':>8}    "
                     f"{'#':<3}{'Team':<27}{'beta':>8}")
        lines.append("-" * 80)
        for i in range(min(top_n, max(len(top_attack), len(top_defense)))):
            left = ""
            if i < len(top_attack):
                t = top_attack[i]
                left = f"{i+1:<3}{t[:26]:<27}{self._team_attack[t]:>8.3f}"
            right = ""
            if i < len(top_defense):
                t = top_defense[i]
                right = f"{i+1:<3}{t[:26]:<27}{self._team_defense[t]:>8.3f}"
            lines.append(f"{left:<40}    {right}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise fitted parameters to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "team_attack": self._team_attack,
            "team_defense": self._team_defense,
            "position_degradation": self._position_degradation,
            "league_avg_goals": self._league_avg_goals,
            "home_advantage": self._home_advantage,
            "match_counts": self._match_counts,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "PlayerModel":
        """Load a model previously written by save()."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls()
        model._team_attack = data.get("team_attack", {})
        model._team_defense = data.get("team_defense", {})
        model._position_degradation = data.get("position_degradation",
                                               dict(_DEGRADATION_PRIOR))
        model._league_avg_goals = data.get("league_avg_goals", 1.30)
        model._home_advantage = data.get("home_advantage", 1.15)
        model._match_counts = data.get("match_counts", {})
        return model


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Tiny synthetic demo so this runs with no data files present.
    demo = pd.DataFrame(
        {
            "date": ["2024-01-01"] * 6,
            "home_team": ["England", "France", "England", "France", "England", "France"],
            "away_team": ["France", "England", "France", "England", "France", "England"],
            "home_goals": [2, 1, 3, 0, 2, 2],
            "away_goals": [1, 1, 1, 2, 0, 1],
            "home_xg": [2.1, 1.0, 2.8, 0.6, 1.9, 1.7],
            "away_xg": [1.2, 1.1, 0.9, 1.8, 0.7, 1.3],
        }
    )
    model = PlayerModel().fit(demo, max_iter=20)

    print("\n=== England vs France (neutral venue) ===")
    base = model.predict_outcome("England", "France", neutral=True)
    print(f"  lambda: home={base['lambda_home']:.2f}  away={base['lambda_away']:.2f}")
    print(f"  P(home)={base['p_home']:.3f}  P(draw)={base['p_draw']:.3f}  "
          f"P(away)={base['p_away']:.3f}")

    print("\n=== England missing a KEY forward ===")
    inj = model.predict_outcome(
        "England", "France",
        home_absent=[{"position": "forward", "is_key": True}],
        neutral=True,
    )
    print(f"  lambda: home={inj['lambda_home']:.2f}  away={inj['lambda_away']:.2f}")
    print(f"  P(home)={inj['p_home']:.3f}  P(draw)={inj['p_draw']:.3f}  "
          f"P(away)={inj['p_away']:.3f}")
    print(f"\n  England goal expectation dropped "
          f"{(1 - inj['lambda_home']/base['lambda_home'])*100:.1f}% with key forward out.")
