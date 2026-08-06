"""
odds_fetcher.py

Production odds fetcher with a 4-source merge strategy:
  1. The Odds API (free tier, 500 req/month — responses cached for 300 s)
  2. SofaScore via RapidAPI
  3. Betfair Exchange (session-based auth, no overround; preferred when merging)
  4. OddsChecker HTML scraper (competition-specific URL, last resort)

All HTTP calls use a shared requests.Session with automatic retry + exponential
backoff (3 retries, status codes 429/500/502/503/504).

Usage:
    fetcher = OddsFetcher()
    odds = fetcher.get_odds("Argentina", "France", "2026-06-15")
    all_today = fetcher.get_all_today("2026-06-15")
"""

import logging
import os
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.team_names import normalize_team
from core.value_finder import remove_overround

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 10  # seconds, applied to every outbound request

# Suffixes to strip when doing fuzzy team name matching
_STRIP_SUFFIXES = (
    "fc", "cf", "sc", "ac", "rc", "fk", "bk", "sk", "nk",
    "united", "city", "town", "rovers", "wanderers", "athletic",
    "albion", "hotspur", "county", "district",
)

# Map short competition keys to Odds API sport keys
_COMPETITION_MAP = {
    "wc2026": "soccer_fifa_world_cup",
    "epl": "soccer_epl",
    "laliga": "soccer_spain_la_liga",
    "seriea": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga",
    "ligue1": "soccer_france_ligue_1",
}

# ---------------------------------------------------------------------------
# HTTP session factory
# ---------------------------------------------------------------------------


def _make_session() -> requests.Session:
    """Return a requests.Session pre-configured with retry + backoff logic."""
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_team(name: str) -> str:
    """Resolve FIFA canonical name then lowercase/strip for fuzzy matching.

    Resolution order:
      1. normalize_team() — alias table lookup (handles "USA" -> "United States",
         "IR Iran" -> "IR Iran", "Côte d'Ivoire" -> "Cote d'Ivoire", etc.)
      2. Lowercase + remove punctuation.
      3. Strip common *club* suffixes (FC, United, City …).

    Step 3 is applied AFTER canonical resolution so that national-team names
    that happen to contain words like "united" ("United States", "United Arab
    Emirates") are already resolved to their canonical form before suffix
    stripping, making them immune to false truncation.
    """
    # Step 1 — canonical alias resolution (national teams)
    name = normalize_team(name.strip())
    # Step 2 — lowercase + remove punctuation
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    # Step 3 — strip club-only suffixes
    tokens = [t for t in name.split() if t not in _STRIP_SUFFIXES]
    return " ".join(tokens)


def _teams_match(a: str, b: str) -> bool:
    """Return True if two team name strings refer to the same team."""
    if a.lower().strip() == b.lower().strip():
        return True
    # Try canonical resolution on both sides first
    if normalize_team(a.strip()) == normalize_team(b.strip()):
        return True
    return _normalise_team(a) == _normalise_team(b)


def _build_record(
    home: str,
    away: str,
    date: str,
    source: str,
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    is_exchange: bool = False,
) -> dict:
    """Assemble the standard output dict, computing fair prices and overround."""
    raw_home = 1.0 / home_odds
    raw_draw = 1.0 / draw_odds
    raw_away = 1.0 / away_odds
    overround = raw_home + raw_draw + raw_away - 1.0

    if is_exchange:
        # Exchange odds are already fair — no margin to remove
        fair_home = raw_home
        fair_draw = raw_draw
        fair_away = raw_away
    else:
        fair_home, fair_draw, fair_away = remove_overround(home_odds, draw_odds, away_odds)

    return {
        "home": home,
        "away": away,
        "date": date,
        "source": source,
        "home_odds": round(home_odds, 4),
        "draw_odds": round(draw_odds, 4),
        "away_odds": round(away_odds, 4),
        "fair_home": round(fair_home, 6),
        "fair_draw": round(fair_draw, 6),
        "fair_away": round(fair_away, 6),
        "overround": round(overround, 6),
        "is_exchange": is_exchange,
    }


# ---------------------------------------------------------------------------
# Source 1 — The Odds API
# ---------------------------------------------------------------------------


class _OddsAPISource:
    """Fetches odds from The Odds API (https://api.the-odds-api.com/v4)."""

    BASE_URL = "https://api.the-odds-api.com/v4"

    _CACHE_TTL = 300  # seconds before a cached response is considered stale

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.getenv("ODDS_API_KEY")
        self._session = _make_session()
        # Cache fields: populated on first successful request
        self._cache: list[dict] = []
        self._cache_time: float = 0.0
        self._cache_key: str = ""  # "{competition}:{date}" used for invalidation

    def available(self) -> bool:
        return bool(self.api_key)

    def get_all(self, competition: str, date: str = "") -> list[dict]:
        """Return raw event list from The Odds API for *competition*.

        Caches the last response for ``_CACHE_TTL`` seconds.  The cache is
        invalidated whenever *competition* or *date* changes so stale data is
        never served across matches scheduled on different days.
        """
        if not self.available():
            logger.info("[OddsAPI] ODDS_API_KEY not set — skipping source.")
            return []

        cache_key = f"{competition}:{date}"
        now = time.monotonic()
        if (
            self._cache
            and self._cache_key == cache_key
            and (now - self._cache_time) < self._CACHE_TTL
        ):
            logger.debug("[OddsAPI] Returning cached response (%d events).", len(self._cache))
            return self._cache

        sport = _COMPETITION_MAP.get(competition, "soccer_fifa_world_cup")
        url = (
            f"{self.BASE_URL}/sports/{sport}/odds/"
            f"?apiKey={self.api_key}&regions=eu&markets=h2h&oddsFormat=decimal"
        )
        try:
            resp = self._session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            logger.info("[OddsAPI] Fetched %d events (fresh).", len(data))
            self._cache = data
            self._cache_time = now
            self._cache_key = cache_key
            return data
        except Exception as exc:
            logger.warning("[OddsAPI] Request failed: %s", exc)
            return []

    def find_match(
        self, home: str, away: str, date: str, competition: str
    ) -> Optional[dict]:
        events = self.get_all(competition, date)
        for event in events:
            if not _teams_match(event.get("home_team", ""), home):
                continue
            if not _teams_match(event.get("away_team", ""), away):
                continue
            # Extract h2h market
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    home_key = event["home_team"]
                    away_key = event["away_team"]
                    ho = outcomes.get(home_key)
                    ao = outcomes.get(away_key)
                    do_ = outcomes.get("Draw")
                    if ho and do_ and ao:
                        logger.info("[OddsAPI] Matched %s vs %s.", home, away)
                        return _build_record(home, away, date, "odds_api", ho, do_, ao)
        logger.info("[OddsAPI] No match found for %s vs %s.", home, away)
        return None

    def get_all_today(self, date: str, competition: str) -> list[dict]:
        events = self.get_all(competition, date)
        results = []
        for event in events:
            # Filter by date prefix (YYYY-MM-DD)
            commence = event.get("commence_time", "")
            if not commence.startswith(date):
                continue
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    ho = outcomes.get(event["home_team"])
                    ao = outcomes.get(event["away_team"])
                    do_ = outcomes.get("Draw")
                    if ho and do_ and ao:
                        results.append(
                            _build_record(
                                event["home_team"],
                                event["away_team"],
                                date,
                                "odds_api",
                                ho,
                                do_,
                                ao,
                            )
                        )
                    break  # first bookmaker with h2h is enough
                break
        logger.info("[OddsAPI] Returning %d today's matches.", len(results))
        return results


# ---------------------------------------------------------------------------
# Source 2 — SofaScore via RapidAPI
# ---------------------------------------------------------------------------


class _SofaScoreSource:
    """Fetches odds from SofaScore via RapidAPI."""

    BASE_URL = "https://sofascore.p.rapidapi.com"

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.getenv("SOFASCORE_API_KEY")
        self.api_host: str = os.getenv("SOFASCORE_API_HOST", "sofascore.p.rapidapi.com")
        self._session = _make_session()

    def available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict:
        return {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.api_host,
        }

    def _get_event_odds(self, event_id: int) -> Optional[tuple[float, float, float]]:
        """Return (home_odds, draw_odds, away_odds) for a SofaScore event ID."""
        # Try the most common RapidAPI/SofaScore odds endpoints
        endpoints = [
            f"{self.BASE_URL}/matches/get-odds?id={event_id}",
            f"{self.BASE_URL}/events/get-odds?id={event_id}",
            f"{self.BASE_URL}/matches/{event_id}/odds",
        ]
        for url in endpoints:
            try:
                resp = self._session.get(url, headers=self._headers(), timeout=REQUEST_TIMEOUT)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                # Parse normalised structure
                odds = self._parse_odds_payload(data)
                if odds:
                    return odds
            except Exception as exc:
                logger.debug("[SofaScore] Odds endpoint %s failed: %s", url, exc)
        return None

    @staticmethod
    def _parse_odds_payload(data: dict) -> Optional[tuple[float, float, float]]:
        """
        Attempt to extract (home, draw, away) decimal odds from various
        SofaScore API payload shapes encountered in the wild.
        """
        # Shape 1: {"data": {"odds": {"1": X, "X": Y, "2": Z}}}
        try:
            inner = data.get("data", data)
            odds_obj = inner.get("odds", {})
            ho = float(odds_obj.get("1") or odds_obj.get("home") or 0)
            do_ = float(odds_obj.get("X") or odds_obj.get("draw") or 0)
            ao = float(odds_obj.get("2") or odds_obj.get("away") or 0)
            if ho > 1.0 and do_ > 1.0 and ao > 1.0:
                return ho, do_, ao
        except (TypeError, ValueError, AttributeError):
            pass

        # Shape 2: list of market dicts
        try:
            markets = data.get("data", data).get("markets", [])
            for market in markets:
                choices = market.get("choices", [])
                if len(choices) == 3:
                    ho, do_, ao = (float(c.get("fractionalValue", 0)) for c in choices)
                    if ho > 1.0 and do_ > 1.0 and ao > 1.0:
                        return ho, do_, ao
        except (TypeError, ValueError, AttributeError):
            pass

        return None

    def _get_today_events(self, date: str) -> list[dict]:
        """Return SofaScore event stubs for the given date."""
        url = f"{self.BASE_URL}/matches/get-by-date?date={date}&sport=football"
        fallback_urls = [
            f"{self.BASE_URL}/events/get-by-date?date={date}&sport=football",
        ]
        for attempt_url in [url] + fallback_urls:
            try:
                resp = self._session.get(
                    attempt_url, headers=self._headers(), timeout=REQUEST_TIMEOUT
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                events = data.get("data", data)
                if isinstance(events, dict):
                    events = events.get("events", [])
                if isinstance(events, list) and events:
                    logger.info("[SofaScore] Got %d events for %s.", len(events), date)
                    return events
            except Exception as exc:
                logger.debug("[SofaScore] Event list endpoint failed: %s", exc)
        return []

    def find_match(
        self, home: str, away: str, date: str, _competition: str
    ) -> Optional[dict]:
        if not self.available():
            logger.info("[SofaScore] API key not set — skipping source.")
            return None

        events = self._get_today_events(date)
        for event in events:
            home_team = (
                event.get("homeTeam", {}).get("name", "")
                or event.get("home_team", "")
            )
            away_team = (
                event.get("awayTeam", {}).get("name", "")
                or event.get("away_team", "")
            )
            if not (_teams_match(home_team, home) and _teams_match(away_team, away)):
                continue

            event_id = event.get("id") or event.get("event_id")
            if not event_id:
                continue

            odds = self._get_event_odds(int(event_id))
            if odds:
                ho, do_, ao = odds
                logger.info("[SofaScore] Matched %s vs %s.", home, away)
                return _build_record(home, away, date, "sofascore", ho, do_, ao)

        logger.info("[SofaScore] No match found for %s vs %s.", home, away)
        return None

    def get_all_today(self, date: str, _competition: str) -> list[dict]:
        if not self.available():
            return []

        events = self._get_today_events(date)
        results = []
        for event in events:
            home_team = (
                event.get("homeTeam", {}).get("name", "")
                or event.get("home_team", "")
            )
            away_team = (
                event.get("awayTeam", {}).get("name", "")
                or event.get("away_team", "")
            )
            event_id = event.get("id") or event.get("event_id")
            if not (home_team and away_team and event_id):
                continue
            odds = self._get_event_odds(int(event_id))
            if odds:
                ho, do_, ao = odds
                results.append(
                    _build_record(home_team, away_team, date, "sofascore", ho, do_, ao)
                )
        logger.info("[SofaScore] Returning %d today's matches.", len(results))
        return results


# ---------------------------------------------------------------------------
# Source 3 — Betfair Exchange
# ---------------------------------------------------------------------------


class _BetfairSource:
    """
    Fetches Match Odds from the Betfair Exchange.
    Exchange prices have no overround — is_exchange=True in all records.
    Requires BETFAIR_USERNAME, BETFAIR_PASSWORD, and BETFAIR_APP_KEY in .env.
    """

    LOGIN_URL = "https://identitysso.betfair.com/api/login"
    API_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"

    def __init__(self) -> None:
        self.app_key: Optional[str] = os.getenv("BETFAIR_APP_KEY")
        self.username: Optional[str] = os.getenv("BETFAIR_USERNAME")
        self.password: Optional[str] = os.getenv("BETFAIR_PASSWORD")
        self._session_token: Optional[str] = None
        self._session = _make_session()

    def available(self) -> bool:
        return bool(self.app_key and self.username and self.password)

    def _login(self) -> bool:
        """Acquire a Betfair SSO session token. Returns True on success."""
        if self._session_token:
            return True
        try:
            resp = self._session.post(
                self.LOGIN_URL,
                data={"username": self.username, "password": self.password},
                headers={"X-Application": self.app_key, "Accept": "application/json"},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("status") == "SUCCESS":
                self._session_token = payload["token"]
                logger.info("[Betfair] Session acquired.")
                return True
            logger.warning("[Betfair] Login failed: %s", payload.get("error"))
        except Exception as exc:
            logger.warning("[Betfair] Login error: %s", exc)
        return False

    def _rpc(self, method: str, params: dict) -> Optional[dict]:
        """Execute a single Betfair JSON-RPC call."""
        if not self._session_token:
            return None
        headers = {
            "X-Application": self.app_key,
            "X-Authentication": self._session_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        body = [{"jsonrpc": "2.0", "method": f"SportsAPING/v1.0/{method}", "params": params, "id": 1}]
        try:
            resp = self._session.post(
                self.API_URL, json=body, headers=headers, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            results = resp.json()
            return results[0].get("result") if results else None
        except Exception as exc:
            logger.warning("[Betfair] RPC %s failed: %s", method, exc)
            return None

    def _fetch_soccer_markets(self, date: str) -> list[dict]:
        """Return Match Odds market catalogues for the given date."""
        from_dt = f"{date}T00:00:00Z"
        to_dt = f"{date}T23:59:59Z"
        result = self._rpc(
            "listMarketCatalogue",
            {
                "filter": {
                    "eventTypeIds": ["1"],  # Soccer
                    "marketTypeCodes": ["MATCH_ODDS"],
                    "marketStartTime": {"from": from_dt, "to": to_dt},
                },
                "marketProjection": ["RUNNER_DESCRIPTION", "EVENT"],
                "maxResults": "200",
            },
        )
        return result or []

    def _fetch_market_book(self, market_id: str) -> Optional[dict]:
        """Return best available price for each runner in a market."""
        result = self._rpc(
            "listMarketBook",
            {
                "marketIds": [market_id],
                "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
            },
        )
        if result and isinstance(result, list):
            return result[0]
        return None

    @staticmethod
    def _best_back_price(runner: dict) -> Optional[float]:
        """Extract best available back price from a runner dict."""
        try:
            backs = runner["ex"]["availableToBack"]
            if backs:
                return float(backs[0]["price"])
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        return None

    def _market_to_record(
        self, catalogue: dict, home: str, away: str, date: str
    ) -> Optional[dict]:
        market_id = catalogue.get("marketId")
        runners = catalogue.get("runners", [])
        # Runner descriptions: typically Home / Draw / Away in order
        if len(runners) < 3:
            return None

        book = self._fetch_market_book(market_id)
        if not book:
            return None

        runner_map: dict[int, str] = {
            r["selectionId"]: r.get("runnerName", "")
            for r in runners
        }
        prices: dict[str, float] = {}
        for book_runner in book.get("runners", []):
            sel_id = book_runner["selectionId"]
            name = runner_map.get(sel_id, "")
            price = self._best_back_price(book_runner)
            if name and price:
                prices[name.lower()] = price

        # Identify home, draw, away prices
        ho = do_ = ao = None
        for name, price in prices.items():
            if "draw" in name:
                do_ = price
            elif _teams_match(name, home):
                ho = price
            elif _teams_match(name, away):
                ao = price

        if ho and do_ and ao:
            return _build_record(home, away, date, "betfair", ho, do_, ao, is_exchange=True)
        return None

    def find_match(
        self, home: str, away: str, date: str, _competition: str
    ) -> Optional[dict]:
        if not self.available():
            logger.info("[Betfair] Credentials not set — skipping source.")
            return None
        if not self._login():
            return None

        catalogues = self._fetch_soccer_markets(date)
        for cat in catalogues:
            event = cat.get("event", {})
            event_name = event.get("name", "")
            # Event name typically "Home v Away" or "Home vs Away"
            if not (_teams_match_in_event(event_name, home) and _teams_match_in_event(event_name, away)):
                continue
            record = self._market_to_record(cat, home, away, date)
            if record:
                logger.info("[Betfair] Matched %s vs %s.", home, away)
                return record

        logger.info("[Betfair] No match found for %s vs %s.", home, away)
        return None

    def get_all_today(self, date: str, _competition: str) -> list[dict]:
        if not self.available():
            return []
        if not self._login():
            return []

        catalogues = self._fetch_soccer_markets(date)
        results = []
        for cat in catalogues:
            event = cat.get("event", {})
            event_name = event.get("name", "")
            parts = re.split(r"\s+v(?:s)?\s+", event_name, flags=re.IGNORECASE)
            if len(parts) != 2:
                continue
            home_name, away_name = parts[0].strip(), parts[1].strip()
            record = self._market_to_record(cat, home_name, away_name, date)
            if record:
                results.append(record)

        logger.info("[Betfair] Returning %d today's matches.", len(results))
        return results


def _teams_match_in_event(event_name: str, team: str) -> bool:
    """Return True if *team* appears in a Betfair event name string."""
    return _normalise_team(team) in _normalise_team(event_name)


# ---------------------------------------------------------------------------
# Source 4 — OddsChecker scraper (fallback)
# ---------------------------------------------------------------------------


class _OddsCheckerSource:
    """
    Last-resort scraper: parses match odds tables from oddschecker.com/football.
    No API key required.
    """

    BASE_URL = "https://www.oddschecker.com/football"

    _COMPETITION_URLS: dict[str, str] = {
        "wc2026": "https://www.oddschecker.com/football/world-cup",
        "epl": "https://www.oddschecker.com/football/english/premier-league",
        "laliga": "https://www.oddschecker.com/football/spanish/la-liga",
        "seriea": "https://www.oddschecker.com/football/italian/serie-a",
        "bundesliga": "https://www.oddschecker.com/football/german/bundesliga",
        "ligue1": "https://www.oddschecker.com/football/french/ligue-1",
    }

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-GB,en;q=0.9",
    }

    def __init__(self) -> None:
        self._session = _make_session()

    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            resp = self._session.get(
                url, headers=self._HEADERS, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            logger.warning("[OddsChecker] HTTP error fetching %s: %s", url, exc)
            return None

    def _parse_matches(self, html: str) -> list[dict]:
        """
        Parse match rows from OddsChecker football listing page.
        Each row contains team names and decimal odds columns.
        """
        soup = BeautifulSoup(html, "lxml")
        results = []

        # OddsChecker renders odds in <tr> rows with data attributes or
        # structured td cells. We look for rows containing exactly 3 odds
        # (home / draw / away) using best-price spans.
        for row in soup.select("tr.match-on, tr[data-event-name]"):
            try:
                event_name = (
                    row.get("data-event-name", "")
                    or row.select_one(".event-name, .match-name, .coupon-team-names")
                )
                if hasattr(event_name, "get_text"):
                    event_name = event_name.get_text(" ", strip=True)

                parts = re.split(r"\s+v(?:s)?\s+", str(event_name), flags=re.IGNORECASE)
                if len(parts) != 2:
                    continue
                home_name, away_name = parts[0].strip(), parts[1].strip()

                # Best odds are in <td class="best-odds"> or data-best-dig attributes
                odds_cells = row.select("[data-best-dig]")
                if len(odds_cells) < 3:
                    # Try alternative: select all cells and filter numeric ones
                    odds_cells = [
                        td for td in row.select("td")
                        if re.match(r"^\d+(\.\d+)?$", td.get_text(strip=True))
                    ]
                if len(odds_cells) < 3:
                    continue

                ho = float(odds_cells[0].get("data-best-dig") or odds_cells[0].get_text(strip=True))
                do_ = float(odds_cells[1].get("data-best-dig") or odds_cells[1].get_text(strip=True))
                ao = float(odds_cells[2].get("data-best-dig") or odds_cells[2].get_text(strip=True))

                if not (ho > 1.0 and do_ > 1.0 and ao > 1.0):
                    continue

                results.append({
                    "home": home_name,
                    "away": away_name,
                    "home_odds": ho,
                    "draw_odds": do_,
                    "away_odds": ao,
                })
            except (ValueError, AttributeError, IndexError):
                continue

        logger.info("[OddsChecker] Parsed %d matches from HTML.", len(results))
        return results

    def _get_all_raw(self, competition: str = "") -> list[dict]:
        url = self._COMPETITION_URLS.get(competition, self.BASE_URL)
        html = self._fetch_html(url)
        if not html:
            return []
        return self._parse_matches(html)

    def find_match(
        self, home: str, away: str, date: str, competition: str
    ) -> Optional[dict]:
        raw = self._get_all_raw(competition)
        for m in raw:
            if _teams_match(m["home"], home) and _teams_match(m["away"], away):
                logger.info("[OddsChecker] Matched %s vs %s.", home, away)
                return _build_record(
                    home, away, date, "oddschecker",
                    m["home_odds"], m["draw_odds"], m["away_odds"],
                )
        logger.info("[OddsChecker] No match found for %s vs %s.", home, away)
        return None

    def get_all_today(self, date: str, competition: str) -> list[dict]:
        raw = self._get_all_raw(competition)
        results = [
            _build_record(
                m["home"], m["away"], date, "oddschecker",
                m["home_odds"], m["draw_odds"], m["away_odds"],
            )
            for m in raw
        ]
        logger.info("[OddsChecker] Returning %d today's matches.", len(results))
        return results


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class OddsFetcher:
    """
    Multi-source odds fetcher.

    ``get_odds()`` walks sources in priority order and returns the first hit.
    ``get_all_today()`` queries **all** sources and merges results by
    (home, away) key, preferring Betfair Exchange records over bookmaker ones.

    Priority order:
        1. The Odds API  (if ODDS_API_KEY is set; responses cached 300 s)
        2. SofaScore via RapidAPI  (if SOFASCORE_API_KEY is set)
        3. Betfair Exchange  (if BETFAIR_APP_KEY + credentials are set)
        4. OddsChecker scraper  (always available, best-effort)

    All sources are wrapped in try/except — this class never raises.
    """

    def __init__(self) -> None:
        self._sources = [
            _OddsAPISource(),
            _SofaScoreSource(),
            _BetfairSource(),
            _OddsCheckerSource(),
        ]

    def get_odds(
        self,
        home: str,
        away: str,
        date: str,
        competition: str = "wc2026",
    ) -> Optional[dict]:
        """
        Return odds for a single match from the first source that has data.

        Args:
            home: Home team name (fuzzy matched against source data).
            away: Away team name.
            date: ISO date string, e.g. "2026-06-15".
            competition: Short key from _COMPETITION_MAP (default "wc2026").

        Returns:
            Odds dict or None if all sources fail.
        """
        for source in self._sources:
            try:
                result = source.find_match(home, away, date, competition)
                if result:
                    return result
            except Exception as exc:
                logger.warning(
                    "[OddsFetcher] Unexpected error in source %s: %s",
                    type(source).__name__,
                    exc,
                )
        logger.error(
            "[OddsFetcher] All sources exhausted for %s vs %s on %s.", home, away, date
        )
        return None

    def get_all_today(
        self,
        date: str,
        competition: str = "wc2026",
    ) -> list[dict]:
        """
        Return all available odds for the given date, merging results across
        all sources.

        Every source is queried regardless of how many matches earlier sources
        returned.  Matches are keyed by ``(home, away)`` (normalised).  When
        the same match appears in multiple sources, the exchange record
        (``is_exchange=True``, e.g. Betfair) is preferred over bookmaker
        records; among non-exchange sources the first source in priority order
        wins.

        Args:
            date: ISO date string, e.g. "2026-06-15".
            competition: Short key from _COMPETITION_MAP (default "wc2026").

        Returns:
            List of odds dicts (may be empty if all sources fail).
        """
        # merged dict keyed by normalised (home, away) tuple
        merged: dict[tuple[str, str], dict] = {}

        for source in self._sources:
            try:
                results = source.get_all_today(date, competition)
            except Exception as exc:
                logger.warning(
                    "[OddsFetcher] Unexpected error in source %s: %s",
                    type(source).__name__,
                    exc,
                )
                continue

            for record in results:
                key = (
                    _normalise_team(record["home"]),
                    _normalise_team(record["away"]),
                )
                existing = merged.get(key)
                if existing is None:
                    # First time we see this match — take it unconditionally.
                    merged[key] = record
                elif record.get("is_exchange") and not existing.get("is_exchange"):
                    # Exchange record beats any bookmaker record.
                    merged[key] = record
                # else: keep the existing record (first-source priority or exchange already present)

        if not merged:
            logger.error("[OddsFetcher] All sources exhausted for %s.", date)
        else:
            logger.info(
                "[OddsFetcher] Merged %d unique matches for %s.", len(merged), date
            )
        return list(merged.values())
