"""Fatigue & travel model for the Apollo forecasting engine.

Computes Elo adjustments that account for two systematically-mispriced
effects in professional football:

1. **Congestion fatigue** — teams playing on short rest (e.g. 3 days after
   their last fixture) underperform their Elo expectation by roughly
   1.5 goals-equivalent.
2. **Travel fatigue** — long-distance away travel (>2000 km) compounds the
   congestion penalty for the visiting side.

The model is pure Python + numpy + pandas. The haversine great-circle
distance is implemented inline (no external geo dependency). All lookups
degrade gracefully: unknown cities contribute zero travel penalty and
teams with no prior fixture are treated as fully rested.

No side effects occur at import time; the schedule index is built explicitly
via :meth:`FatigueModel.build_schedule_index` and is fully serializable.
"""

from __future__ import annotations

import bisect
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EARTH_RADIUS_KM = 6371.0088


class FatigueModel:
    """Rest- and travel-aware Elo adjustment model.

    Typical usage::

        fm = FatigueModel()
        fm.build_schedule_index(matches_df)
        adj = fm.get_match_adjustments(
            "Liverpool", "Arsenal", "2024-12-26",
            home_city="Liverpool", away_city="London",
        )
        # apply adj["net_elo_delta"] to the home side's Elo before predicting
    """

    # ------------------------------------------------------------------ #
    # Static geography. (lat, lon) in decimal degrees.
    # ------------------------------------------------------------------ #
    CITY_COORDS: dict[str, tuple[float, float]] = {
        # England
        "London": (51.5074, -0.1278),
        "Manchester": (53.4808, -2.2426),
        "Liverpool": (53.4084, -2.9916),
        "Birmingham": (52.4862, -1.8904),
        "Newcastle": (54.9783, -1.6178),
        "Leeds": (53.8008, -1.5491),
        "Leicester": (52.6369, -1.1398),
        "Brighton": (50.8225, -0.1372),
        "Nottingham": (52.9548, -1.1581),
        "Southampton": (50.9097, -1.4044),
        # Scotland
        "Glasgow": (55.8642, -4.2518),
        "Edinburgh": (55.9533, -3.1883),
        # Spain
        "Madrid": (40.4168, -3.7038),
        "Barcelona": (41.3851, 2.1734),
        "Seville": (37.3891, -5.9845),
        "Valencia": (39.4699, -0.3763),
        "Bilbao": (43.2630, -2.9350),
        "Vigo": (42.2406, -8.7207),
        "San Sebastian": (43.3183, -1.9812),
        "Villarreal": (39.9440, -0.1014),
        # Germany
        "Munich": (48.1351, 11.5820),
        "Dortmund": (51.5136, 7.4653),
        "Berlin": (52.5200, 13.4050),
        "Frankfurt": (50.1109, 8.6821),
        "Hamburg": (53.5511, 9.9937),
        "Stuttgart": (48.7758, 9.1829),
        "Leverkusen": (51.0459, 7.0192),
        "Leipzig": (51.3397, 12.3731),
        "Gelsenkirchen": (51.5177, 7.0857),
        "Dusseldorf": (51.2277, 6.7735),
        "Bremen": (53.0793, 8.8017),
        "Cologne": (50.9375, 6.9603),
        "Monchengladbach": (51.1805, 6.4428),
        "Wolfsburg": (52.4227, 10.7865),
        "Freiburg": (47.9990, 7.8421),
        "Augsburg": (48.3705, 10.8978),
        "Hoffenheim": (49.2386, 8.8884),
        "Mainz": (49.9929, 8.2473),
        # Italy
        "Milan": (45.4642, 9.1900),
        "Rome": (41.9028, 12.4964),
        "Naples": (40.8518, 14.2681),
        "Turin": (45.0703, 7.6869),
        "Florence": (43.7696, 11.2558),
        "Bergamo": (45.6983, 9.6773),
        "Bologna": (44.4949, 11.3426),
        "Genoa": (44.4056, 8.9463),
        "Verona": (45.4384, 10.9916),
        # France
        "Paris": (48.8566, 2.3522),
        "Lyon": (45.7640, 4.8357),
        "Marseille": (43.2965, 5.3698),
        "Lille": (50.6292, 3.0573),
        "Monaco": (43.7384, 7.4246),
        "Nice": (43.7102, 7.2620),
        "Rennes": (48.1173, -1.6778),
        "Nantes": (47.2184, -1.5536),
        "Bordeaux": (44.8378, -0.5792),
        "Saint-Etienne": (45.4397, 4.3872),
        # Low Countries
        "Amsterdam": (52.3676, 4.9041),
        "Rotterdam": (51.9244, 4.4777),
        "Eindhoven": (51.4416, 5.4697),
        "Brussels": (50.8503, 4.3517),
        # Portugal
        "Lisbon": (38.7223, -9.1393),
        "Porto": (41.1579, -8.6291),
        "Braga": (41.5454, -8.4265),
        # Nordics
        "Copenhagen": (55.6761, 12.5683),
        "Stockholm": (59.3293, 18.0686),
        "Oslo": (59.9139, 10.7522),
        "Helsinki": (60.1699, 24.9384),
        # Central / Eastern Europe
        "Prague": (50.0755, 14.4378),
        "Vienna": (48.2082, 16.3738),
        "Zurich": (47.3769, 8.5417),
        "Bern": (46.9480, 7.4474),
        "Warsaw": (52.2297, 21.0122),
        "Bucharest": (44.4268, 26.1025),
        "Belgrade": (44.7866, 20.4489),
        "Zagreb": (45.8150, 15.9819),
        "Budapest": (47.4979, 19.0402),
        "Athens": (37.9838, 23.7275),
        # Russia / Ukraine / Turkey
        "Moscow": (55.7558, 37.6173),
        "St Petersburg": (59.9311, 30.3609),
        "Kyiv": (50.4501, 30.5234),
        "Istanbul": (41.0082, 28.9784),
        # Middle East / North Africa
        "Doha": (25.2854, 51.5310),
        "Cairo": (30.0444, 31.2357),
        "Casablanca": (33.5731, -7.5898),
        "Rabat": (34.0209, -6.8416),
        "Tunis": (36.8065, 10.1815),
        "Algiers": (36.7538, 3.0588),
        # Sub-Saharan Africa
        "Lagos": (6.5244, 3.3792),
        "Nairobi": (-1.2921, 36.8219),
        "Dakar": (14.7167, -17.4677),
        "Johannesburg": (-26.2041, 28.0473),
        # Americas
        "Buenos Aires": (-34.6037, -58.3816),
        "Sao Paulo": (-23.5505, -46.6333),
        "Rio de Janeiro": (-22.9068, -43.1729),
        "Mexico City": (19.4326, -99.1332),
        "New York": (40.7128, -74.0060),
        "Los Angeles": (34.0522, -118.2437),
        "Miami": (25.7617, -80.1918),
        "Toronto": (43.6532, -79.3832),
        # Asia / Pacific
        "Seoul": (37.5665, 126.9780),
        "Tokyo": (35.6762, 139.6503),
        "Sydney": (-33.8688, 151.2093),
        "Melbourne": (-37.8136, 144.9631),
    }

    # ------------------------------------------------------------------ #
    # Team -> home city. Used to resolve travel when callers do not pass
    # explicit city names. Covers the big-five leagues plus major national
    # sides (mapped to capital / main stadium city).
    # ------------------------------------------------------------------ #
    VENUE_CITY: dict[str, str] = {
        # ---- Premier League ----
        "Arsenal": "London",
        "Chelsea": "London",
        "Tottenham": "London",
        "Tottenham Hotspur": "London",
        "West Ham": "London",
        "West Ham United": "London",
        "Crystal Palace": "London",
        "Fulham": "London",
        "Brentford": "London",
        "Manchester United": "Manchester",
        "Manchester City": "Manchester",
        "Liverpool": "Liverpool",
        "Everton": "Liverpool",
        "Newcastle": "Newcastle",
        "Newcastle United": "Newcastle",
        "Aston Villa": "Birmingham",
        "Wolves": "Birmingham",
        "Wolverhampton": "Birmingham",
        "Leeds": "Leeds",
        "Leeds United": "Leeds",
        "Leicester": "Leicester",
        "Leicester City": "Leicester",
        "Brighton": "Brighton",
        "Nottingham Forest": "Nottingham",
        "Southampton": "Southampton",
        # ---- La Liga ----
        "Real Madrid": "Madrid",
        "Atletico Madrid": "Madrid",
        "Atletico Madrid CF": "Madrid",
        "Getafe": "Madrid",
        "Rayo Vallecano": "Madrid",
        "Barcelona": "Barcelona",
        "Espanyol": "Barcelona",
        "Sevilla": "Seville",
        "Real Betis": "Seville",
        "Valencia": "Valencia",
        "Athletic Bilbao": "Bilbao",
        "Athletic Club": "Bilbao",
        "Real Sociedad": "San Sebastian",
        "Celta Vigo": "Vigo",
        "Villarreal": "Villarreal",
        # ---- Bundesliga ----
        "Bayern Munich": "Munich",
        "Bayern": "Munich",
        "Borussia Dortmund": "Dortmund",
        "Dortmund": "Dortmund",
        "RB Leipzig": "Leipzig",
        "Bayer Leverkusen": "Leverkusen",
        "Leverkusen": "Leverkusen",
        "Schalke 04": "Gelsenkirchen",
        "Schalke": "Gelsenkirchen",
        "Borussia Monchengladbach": "Monchengladbach",
        "Eintracht Frankfurt": "Frankfurt",
        "Frankfurt": "Frankfurt",
        "VfB Stuttgart": "Stuttgart",
        "Stuttgart": "Stuttgart",
        "Werder Bremen": "Bremen",
        "Hamburg": "Hamburg",
        "Hamburger SV": "Hamburg",
        "FC Koln": "Cologne",
        "Koln": "Cologne",
        "Hertha Berlin": "Berlin",
        "Union Berlin": "Berlin",
        "Wolfsburg": "Wolfsburg",
        "Freiburg": "Freiburg",
        "Augsburg": "Augsburg",
        "Hoffenheim": "Hoffenheim",
        "Mainz": "Mainz",
        "Fortuna Dusseldorf": "Dusseldorf",
        # ---- Serie A ----
        "AC Milan": "Milan",
        "Inter": "Milan",
        "Inter Milan": "Milan",
        "Juventus": "Turin",
        "Torino": "Turin",
        "Roma": "Rome",
        "AS Roma": "Rome",
        "Lazio": "Rome",
        "Napoli": "Naples",
        "Fiorentina": "Florence",
        "Atalanta": "Bergamo",
        "Bologna": "Bologna",
        "Genoa": "Genoa",
        "Sampdoria": "Genoa",
        "Hellas Verona": "Verona",
        # ---- Ligue 1 ----
        "Paris Saint-Germain": "Paris",
        "PSG": "Paris",
        "Lyon": "Lyon",
        "Olympique Lyonnais": "Lyon",
        "Marseille": "Marseille",
        "Olympique Marseille": "Marseille",
        "Lille": "Lille",
        "Monaco": "Monaco",
        "AS Monaco": "Monaco",
        "Nice": "Nice",
        "Rennes": "Rennes",
        "Nantes": "Nantes",
        "Bordeaux": "Bordeaux",
        "Saint-Etienne": "Saint-Etienne",
        # ---- Other major European clubs ----
        "Ajax": "Amsterdam",
        "Feyenoord": "Rotterdam",
        "PSV": "Eindhoven",
        "PSV Eindhoven": "Eindhoven",
        "Anderlecht": "Brussels",
        "Club Brugge": "Brussels",
        "Benfica": "Lisbon",
        "Sporting CP": "Lisbon",
        "Sporting Lisbon": "Lisbon",
        "Porto": "Porto",
        "FC Porto": "Porto",
        "Braga": "Braga",
        "Celtic": "Glasgow",
        "Rangers": "Glasgow",
        "Galatasaray": "Istanbul",
        "Fenerbahce": "Istanbul",
        "Besiktas": "Istanbul",
        "Olympiacos": "Athens",
        "Panathinaikos": "Athens",
        "Red Star Belgrade": "Belgrade",
        "Dinamo Zagreb": "Zagreb",
        "Shakhtar Donetsk": "Kyiv",
        "Dynamo Kyiv": "Kyiv",
        "Zenit": "St Petersburg",
        "Zenit St Petersburg": "St Petersburg",
        "Spartak Moscow": "Moscow",
        "CSKA Moscow": "Moscow",
        # ---- Major national teams (capital / main venue city) ----
        "England": "London",
        "Scotland": "Glasgow",
        "Germany": "Berlin",
        "France": "Paris",
        "Spain": "Madrid",
        "Italy": "Rome",
        "Portugal": "Lisbon",
        "Netherlands": "Amsterdam",
        "Belgium": "Brussels",
        "Austria": "Vienna",
        "Switzerland": "Bern",
        "Poland": "Warsaw",
        "Czech Republic": "Prague",
        "Croatia": "Zagreb",
        "Serbia": "Belgrade",
        "Greece": "Athens",
        "Romania": "Bucharest",
        "Hungary": "Budapest",
        "Denmark": "Copenhagen",
        "Sweden": "Stockholm",
        "Norway": "Oslo",
        "Finland": "Helsinki",
        "Russia": "Moscow",
        "Ukraine": "Kyiv",
        "Turkey": "Istanbul",
        "Argentina": "Buenos Aires",
        "Brazil": "Sao Paulo",
        "Mexico": "Mexico City",
        "USA": "New York",
        "United States": "New York",
        "Canada": "Toronto",
        "Qatar": "Doha",
        "Egypt": "Cairo",
        "Morocco": "Rabat",
        "Tunisia": "Tunis",
        "Algeria": "Algiers",
        "Nigeria": "Lagos",
        "Kenya": "Nairobi",
        "Senegal": "Dakar",
        "South Africa": "Johannesburg",
        "South Korea": "Seoul",
        "Japan": "Tokyo",
        "Australia": "Sydney",
    }

    def __init__(self) -> None:
        # team -> ascending list of pandas.Timestamp match dates
        self._schedule: dict[str, list[pd.Timestamp]] = {}

    # ------------------------------------------------------------------ #
    # Schedule index
    # ------------------------------------------------------------------ #
    def build_schedule_index(self, df: pd.DataFrame) -> None:
        """Pre-compute a per-team, ascending-sorted list of match dates.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``date``, ``home_team`` and ``away_team`` columns.
            ``date`` is coerced to ``datetime64`` and rows with an
            unparseable date are dropped.
        """
        required = {"date", "home_team", "away_team"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"build_schedule_index missing columns: {sorted(missing)}")

        dates = pd.to_datetime(df["date"], errors="coerce")
        work = pd.DataFrame(
            {
                "date": dates,
                "home_team": df["home_team"].astype("object"),
                "away_team": df["away_team"].astype("object"),
            }
        ).dropna(subset=["date"])

        # Long format: one row per (team, date) appearance.
        home = work[["date", "home_team"]].rename(columns={"home_team": "team"})
        away = work[["date", "away_team"]].rename(columns={"away_team": "team"})
        long = pd.concat([home, away], ignore_index=True)
        long = long.dropna(subset=["team"])

        schedule: dict[str, list[pd.Timestamp]] = {}
        for team, group in long.groupby("team", sort=False):
            # Deduplicate (a team cannot play twice in a day) and sort ascending.
            uniq = sorted(set(group["date"].tolist()))
            schedule[str(team)] = uniq

        self._schedule = schedule
        logger.info("Built schedule index for %d teams.", len(schedule))

    # ------------------------------------------------------------------ #
    # Rest
    # ------------------------------------------------------------------ #
    def days_rest(self, team: str, match_date: str | pd.Timestamp) -> int:
        """Days since ``team``'s last match strictly *before* ``match_date``.

        Returns ``7`` (treated as fully rested) when the team has no prior
        fixture in the schedule index.
        """
        md = pd.Timestamp(match_date)
        if pd.isna(md):
            return 7
        # Normalize to calendar day so same-day intraday timestamps compare cleanly.
        md = md.normalize()

        dates = self._schedule.get(team)
        if not dates:
            return 7

        # Largest index whose date is strictly < md.
        idx = bisect.bisect_left(dates, md) - 1
        if idx < 0:
            return 7

        prev = pd.Timestamp(dates[idx]).normalize()
        delta = (md - prev).days
        if delta <= 0:
            # No genuinely prior fixture (only same-day / future entries).
            return 7
        return int(delta)

    # ------------------------------------------------------------------ #
    # Travel
    # ------------------------------------------------------------------ #
    def travel_km(self, home_city: str, away_city: str) -> float:
        """Great-circle distance (km) between two cities via the haversine
        formula. Returns ``0.0`` if either city is unknown or identical."""
        if not home_city or not away_city:
            return 0.0
        if home_city == away_city:
            return 0.0

        a = self.CITY_COORDS.get(home_city)
        b = self.CITY_COORDS.get(away_city)
        if a is None or b is None:
            return 0.0

        lat1, lon1 = a
        lat2, lon2 = b
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        h = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        )
        c = 2.0 * math.asin(min(1.0, math.sqrt(h)))
        return float(_EARTH_RADIUS_KM * c)

    # ------------------------------------------------------------------ #
    # Elo adjustment lookups
    # ------------------------------------------------------------------ #
    @staticmethod
    def rest_elo_adjustment(days: int) -> float:
        """Elo penalty for short rest (more negative = more fatigued)."""
        if days <= 2:
            return -35.0
        if days == 3:
            return -20.0
        if days == 4:
            return -10.0
        if days == 5:
            return -4.0
        if days == 6:
            return -1.0
        return 0.0  # 7+ days

    @staticmethod
    def travel_elo_adjustment(km: float) -> float:
        """Elo penalty for away travel distance (applied to the away side)."""
        if km < 500:
            return 0.0
        if km < 1500:
            return -5.0
        if km < 3000:
            return -12.0
        if km < 6000:
            return -22.0
        return -35.0

    # ------------------------------------------------------------------ #
    # Combined per-match adjustments
    # ------------------------------------------------------------------ #
    def _resolve_city(self, team: str, explicit: str) -> str:
        """Prefer an explicitly supplied city; otherwise fall back to the
        team's known home venue city. Returns ``""`` when unresolved."""
        if explicit:
            return explicit
        return self.VENUE_CITY.get(team, "")

    def get_match_adjustments(
        self,
        home: str,
        away: str,
        match_date: str,
        home_city: str = "",
        away_city: str = "",
    ) -> dict:
        """Compute rest + travel Elo adjustments for a single fixture.

        The travel penalty is applied to the away side only, reflecting that
        the visitors bear the journey to the host venue. ``net_elo_delta`` is
        the home-minus-away swing: a positive value means the home side is
        net-advantaged by the fatigue/travel situation.
        """
        home_rest = self.days_rest(home, match_date)
        away_rest = self.days_rest(away, match_date)

        h_city = self._resolve_city(home, home_city)
        a_city = self._resolve_city(away, away_city)
        # Away team travels from its own home city to the host (home) city.
        km = self.travel_km(h_city, a_city)

        home_elo_adj = self.rest_elo_adjustment(home_rest)
        away_elo_adj = self.rest_elo_adjustment(away_rest) + self.travel_elo_adjustment(km)

        net = home_elo_adj - away_elo_adj
        return {
            "home_rest_days": int(home_rest),
            "away_rest_days": int(away_rest),
            "travel_km": round(float(km), 2),
            "home_elo_adj": round(float(home_elo_adj), 2),
            "away_elo_adj": round(float(away_elo_adj), 2),
            "net_elo_delta": round(float(net), 2),
        }

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Persist the schedule index to a parquet file."""
        rows: list[tuple[str, pd.Timestamp]] = []
        for team, dates in self._schedule.items():
            for d in dates:
                rows.append((team, pd.Timestamp(d)))

        df = pd.DataFrame(rows, columns=["team", "date"])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        logger.info("Saved schedule index (%d rows) to %s", len(df), out)

    def load(self, path: str) -> None:
        """Restore the schedule index from a parquet file written by :meth:`save`."""
        df = pd.read_parquet(path)
        schedule: dict[str, list[pd.Timestamp]] = {}
        if not df.empty:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            for team, group in df.groupby("team", sort=False):
                schedule[str(team)] = sorted(set(group["date"].tolist()))
        self._schedule = schedule
        logger.info("Loaded schedule index for %d teams from %s", len(schedule), path)
