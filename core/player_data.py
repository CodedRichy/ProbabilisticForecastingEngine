"""
player_data.py

Fetches player availability (injuries, suspensions, doubtful) from free sources.
Returns a per-team Elo modifier to feed into match predictions.

Sources tried in priority order:
  1. ESPN public API  — team roster + injury flags (no auth required)
  2. TransferMarkt    — squad injury scrape (HTML, polite rate-limit)
  3. Daily JSON cache — reuses today's data without re-fetching

Elo modifier logic
------------------
Each confirmed absent player reduces the team's effective Elo rating:
  goalkeeper absent   → -40 pts
  key attacker absent → -30 pts (high xG/90 or top squad position)
  regular starter     → -15 pts
  squad player        → -5 pts
  doubtful            → half the above
  cap: -100 pts total per team

Usage:
    from core.player_data import PlayerData

    modifier = PlayerData.get_elo_modifier("England")      # e.g. -25.0
    report   = PlayerData.get_report("England")            # str for Telegram
    all_data = PlayerData.fetch_all(["England", "France"]) # bulk prefetch
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from core.team_names import normalize_team

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CACHE_DIR  = Path(__file__).parent.parent / "data" / "player_cache"
_CACHE_TTL  = 6 * 3600          # 6 hours — refresh intra-day for lineup changes
_ESPN_BASE  = "https://site.api.espn.com/apis/site/v2/sports/soccer"
_TM_BASE    = "https://www.transfermarkt.com"
_TIMEOUT    = 15

_ELO_WEIGHTS = {
    "goalkeeper":  40,
    "key_attack":  30,
    "starter":     15,
    "squad":        5,
}
_DOUBTFUL_FACTOR = 0.5
_MAX_MODIFIER    = -100.0

# ESPN slug for each competition we support
_ESPN_SLUG: dict[str, str] = {
    "wc2026":    "fifa.world",
    "ucl":       "uefa.champions",
    "epl":       "eng.1",
    "laliga":    "esp.1",
    "bundesliga":"ger.1",
    "seriea":    "ita.1",
    "ligue1":    "fra.1",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlayerStatus:
    name:     str
    position: str            # "goalkeeper", "defender", "midfielder", "forward"
    status:   str            # "injured", "suspended", "doubtful", "fit"
    reason:   str = ""       # "Knee", "Yellow card accumulation", etc.
    is_key:   bool = False   # true if top-3 xG contributor or captain


@dataclass
class TeamAvailability:
    team:    str
    players: list[PlayerStatus] = field(default_factory=list)
    source:  str = ""

    @property
    def absent(self) -> list[PlayerStatus]:
        return [p for p in self.players if p.status in ("injured", "suspended")]

    @property
    def doubtful(self) -> list[PlayerStatus]:
        return [p for p in self.players if p.status == "doubtful"]

    def elo_modifier(self) -> float:
        delta = 0.0
        for p in self.absent:
            delta += _player_impact(p, factor=1.0)
        for p in self.doubtful:
            delta += _player_impact(p, factor=_DOUBTFUL_FACTOR)
        return max(delta, _MAX_MODIFIER)

    def summary(self) -> str:
        absent_names   = [p.name for p in self.absent]
        doubtful_names = [p.name for p in self.doubtful]
        parts = []
        if absent_names:
            parts.append(f"OUT: {', '.join(absent_names)}")
        if doubtful_names:
            parts.append(f"Doubtful: {', '.join(doubtful_names)}")
        return " | ".join(parts) if parts else "Full squad available"


def _player_impact(p: PlayerStatus, factor: float = 1.0) -> float:
    if p.position == "goalkeeper":
        base = _ELO_WEIGHTS["goalkeeper"]
    elif p.is_key:
        base = _ELO_WEIGHTS["key_attack"]
    elif p.position == "forward" or p.position == "midfielder":
        base = _ELO_WEIGHTS["starter"]
    else:
        base = _ELO_WEIGHTS["squad"]
    return -(base * factor)


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "application/json, text/html;q=0.9",
        "Accept-Language": "en-GB,en;q=0.9",
    })
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


_SESSION = _session()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(team: str, date_str: str) -> Path:
    safe = re.sub(r"[^\w]", "_", team.lower())
    return _CACHE_DIR / date_str / f"{safe}.json"


def _load_cache(team: str, date_str: str) -> Optional[TeamAvailability]:
    path = _cache_path(team, date_str)
    if not path.exists():
        return None
    try:
        age = time.time() - path.stat().st_mtime
        if age > _CACHE_TTL:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        players = [PlayerStatus(**p) for p in data["players"]]
        return TeamAvailability(team=data["team"], players=players, source=data["source"])
    except Exception:
        return None


def _save_cache(avail: TeamAvailability, date_str: str) -> None:
    path = _cache_path(avail.team, date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "team":    avail.team,
        "source":  avail.source,
        "players": [
            {"name": p.name, "position": p.position, "status": p.status,
             "reason": p.reason, "is_key": p.is_key}
            for p in avail.players
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# ESPN scraper
# ---------------------------------------------------------------------------

def _espn_team_id(team_name: str, competition: str = "wc2026") -> Optional[str]:
    slug = _ESPN_SLUG.get(competition, "fifa.world")
    url  = f"{_ESPN_BASE}/{slug}/teams"
    try:
        resp = _SESSION.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        teams = resp.json().get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        norm = normalize_team(team_name).lower()
        for entry in teams:
            t = entry.get("team", {})
            names_to_try = [
                t.get("displayName", ""),
                t.get("shortDisplayName", ""),
                t.get("name", ""),
                t.get("abbreviation", ""),
            ]
            if any(normalize_team(n).lower() == norm for n in names_to_try if n):
                return str(t.get("id", ""))
    except Exception as exc:
        logger.debug("ESPN team ID lookup failed for %s: %s", team_name, exc)
    return None


def _espn_injuries(team_id: str, competition: str = "wc2026") -> list[PlayerStatus]:
    slug = _ESPN_SLUG.get(competition, "fifa.world")
    url  = f"https://sports.core.api.espn.com/v2/sports/soccer/leagues/{slug}/teams/{team_id}/injuries"
    players: list[PlayerStatus] = []
    try:
        resp = _SESSION.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        for item in items:
            athlete = item.get("athlete", {})
            injury  = item.get("injuries", [{}])[0] if item.get("injuries") else {}
            status_raw = injury.get("status", "").lower()
            if not status_raw:
                continue

            if "out" in status_raw or "injured" in status_raw:
                status = "injured"
            elif "suspend" in status_raw:
                status = "suspended"
            elif "doubt" in status_raw or "question" in status_raw:
                status = "doubtful"
            else:
                continue  # fit — skip

            pos_raw  = athlete.get("position", {}).get("name", "forward").lower()
            position = _map_position(pos_raw)
            name     = athlete.get("displayName", athlete.get("fullName", "Unknown"))
            reason   = injury.get("shortComment", injury.get("longComment", ""))

            players.append(PlayerStatus(
                name=name, position=position, status=status,
                reason=reason[:80], is_key=False,
            ))
    except Exception as exc:
        logger.debug("ESPN injuries fetch failed for team %s: %s", team_id, exc)
    return players


def fetch_espn(team_name: str, competition: str = "wc2026") -> Optional[TeamAvailability]:
    team_id = _espn_team_id(team_name, competition)
    if not team_id:
        logger.debug("ESPN: no team ID for %s", team_name)
        return None
    players = _espn_injuries(team_id, competition)
    if players is None:
        return None
    return TeamAvailability(team=team_name, players=players, source="espn")


# ---------------------------------------------------------------------------
# TransferMarkt scraper
# ---------------------------------------------------------------------------

# Maps our normalized name → TM URL slug + team_id
# Add more as needed; most national teams for WC2026 listed here.
_TM_NATIONAL_TEAMS: dict[str, tuple[str, int]] = {
    "Argentina":       ("argentina",     3437),
    "France":          ("frankreich",    3377),
    "Brazil":          ("brasilien",     3439),
    "England":         ("england",       3411),
    "Germany":         ("deutschland",   3376),
    "Spain":           ("spanien",       3375),
    "Portugal":        ("portugal",      3382),
    "Netherlands":     ("niederlande",   3378),
    "Italy":           ("italien",       3376),
    "Belgium":         ("belgien",       3396),
    "Croatia":         ("kroatien",      3398),
    "Morocco":         ("marokko",       3440),
    "Japan":           ("japan",         3435),
    "USA":             ("vereinigte-states-von-amerika", 3438),
    "Mexico":          ("mexiko",        3441),
    "Senegal":         ("senegal",       3461),
    "Ecuador":         ("ecuador",       3447),
    "Uruguay":         ("uruguay",       3443),
    "Colombia":        ("kolumbien",     3446),
    "South Korea":     ("korea-sud",     3436),
    "Australia":       ("australien",    3462),
    "Canada":          ("kanada",        3442),
    "Saudi Arabia":    ("saudi-arabien", 3454),
    "Iran":            ("iran",          3453),
    "Ghana":           ("ghana",         3458),
    "Cameroon":        ("kamerun",       3459),
    "Tunisia":         ("tunesien",      3460),
    "Nigeria":         ("nigeria",       3456),
    "Ivory Coast":     ("elfenbeinkuste",3457),
    "Poland":          ("polen",         3390),
    "Serbia":          ("serbien",       3399),
    "Denmark":         ("danemark",      3384),
    "Switzerland":     ("schweiz",       3393),
    "Austria":         ("osterreich",    3392),
    "Czech Republic":  ("tschechien",    3389),
    "Hungary":         ("ungarn",        3388),
    "Norway":          ("norwegen",      3383),
    "Sweden":          ("schweden",      3386),
    "Ukraine":         ("ukraine",       3402),
    "Turkey":          ("turkei",        3401),
    "Greece":          ("griechenland",  3400),
    "Romania":         ("rumanien",      3387),
    "Scotland":        ("schottland",    3412),
    "Wales":           ("wales",         3414),
    "New Zealand":     ("neuseeland",    3469),
    "Algeria":         ("algerien",      3455),
    "Costa Rica":      ("costa-rica",    3444),
    "Panama":          ("panama",        3445),
    "Honduras":        ("honduras",      3449),
    "Jamaica":         ("jamaika",       3450),
    "El Salvador":     ("el-salvador",   3451),
    "Qatar":           ("katar",         3466),
    "Iraq":            ("irak",          3452),
    "UAE":             ("vereinigte-arabische-emirate", 3465),
}

_TM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer":         "https://www.transfermarkt.com/",
}

_TM_SESSION = requests.Session()
_TM_SESSION.headers.update(_TM_HEADERS)


def _parse_tm_injuries(html: str, team_name: str) -> list[PlayerStatus]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed — pip install beautifulsoup4")
        return []

    soup = BeautifulSoup(html, "html.parser")
    players: list[PlayerStatus] = []

    # TM injury table has class "items" inside the injury section
    for row in soup.select("table.items tbody tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        # Player name is in <a> with class "spielprofil_tooltip"
        a_tag = row.select_one("a.spielprofil_tooltip")
        name  = a_tag.get_text(strip=True) if a_tag else cells[0].get_text(strip=True)
        if not name:
            continue

        # Position cell (usually 3rd or 4th)
        pos_text = cells[2].get_text(strip=True).lower() if len(cells) > 2 else ""
        position = _map_position(pos_text)

        # Injury reason / type (usually last cell)
        reason = cells[-1].get_text(strip=True)[:80]

        # Check if "Suspension" appears anywhere in the row text
        row_text = row.get_text(" ", strip=True).lower()
        if "suspen" in row_text or "yellow" in row_text or "red card" in row_text:
            status = "suspended"
        elif "doubtful" in row_text or "questionable" in row_text:
            status = "doubtful"
        else:
            status = "injured"

        players.append(PlayerStatus(
            name=name, position=position, status=status,
            reason=reason, is_key=False,
        ))

    return players


def fetch_transfermarkt(team_name: str) -> Optional[TeamAvailability]:
    norm  = normalize_team(team_name)
    entry = _TM_NATIONAL_TEAMS.get(norm) or _TM_NATIONAL_TEAMS.get(team_name)
    if not entry:
        logger.debug("TM: no slug configured for %s", team_name)
        return None

    slug, team_id = entry
    url = f"{_TM_BASE}/{slug}/verletzungen/verein/{team_id}"
    try:
        resp = _TM_SESSION.get(url, timeout=_TIMEOUT)
        if resp.status_code == 403:
            logger.debug("TM: 403 for %s — bot detection", team_name)
            return None
        resp.raise_for_status()
        players = _parse_tm_injuries(resp.text, team_name)
        time.sleep(2.0)  # be polite
        return TeamAvailability(team=team_name, players=players, source="transfermarkt")
    except Exception as exc:
        logger.debug("TM fetch failed for %s: %s", team_name, exc)
        return None


# ---------------------------------------------------------------------------
# Position normaliser
# ---------------------------------------------------------------------------

def _map_position(raw: str) -> str:
    raw = raw.lower()
    if any(w in raw for w in ("goal", "gk", "keeper", "portero", "portiere", "gardien")):
        return "goalkeeper"
    if any(w in raw for w in ("defend", "back", "cb", "lb", "rb", "sweeper")):
        return "defender"
    if any(w in raw for w in ("attack", "forward", "striker", "wing", "cf", "st", "lw", "rw")):
        return "forward"
    return "midfielder"


# ---------------------------------------------------------------------------
# Key-player flagging (simple heuristic)
# ---------------------------------------------------------------------------

def _flag_key_players(players: list[PlayerStatus]) -> None:
    """Mark the first forward/midfielder as 'key' — rough proxy for star attacker."""
    forwards = [p for p in players if p.position in ("forward", "midfielder")]
    for p in forwards[:2]:
        p.is_key = True


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

class PlayerData:
    """Static facade for fetching and caching player availability."""

    @staticmethod
    def get_availability(
        team_name: str,
        date_str: Optional[str] = None,
        competition: str = "wc2026",
    ) -> TeamAvailability:
        import datetime
        date_str = date_str or datetime.date.today().isoformat()

        # 1. Try cache
        cached = _load_cache(team_name, date_str)
        if cached is not None:
            logger.debug("Player cache hit: %s (%s)", team_name, date_str)
            return cached

        # 2. Try ESPN
        avail = fetch_espn(team_name, competition)

        # 3. Try TransferMarkt if ESPN gave nothing useful
        if avail is None or not avail.players:
            avail = fetch_transfermarkt(team_name)

        # 4. Empty fallback
        if avail is None:
            avail = TeamAvailability(team=team_name, players=[], source="none")

        _flag_key_players(avail.players)
        _save_cache(avail, date_str)
        return avail

    @staticmethod
    def get_elo_modifier(
        team_name: str,
        date_str: Optional[str] = None,
        competition: str = "wc2026",
    ) -> float:
        avail = PlayerData.get_availability(team_name, date_str, competition)
        return avail.elo_modifier()

    @staticmethod
    def get_report(
        team_name: str,
        date_str: Optional[str] = None,
        competition: str = "wc2026",
    ) -> str:
        avail = PlayerData.get_availability(team_name, date_str, competition)
        mod   = avail.elo_modifier()
        lines = [f"📋 {team_name} squad news (via {avail.source})"]
        lines.append(avail.summary())
        if mod < 0:
            lines.append(f"Elo impact: {mod:+.0f} pts")
        return "\n".join(lines)

    @staticmethod
    def fetch_all(
        teams: list[str],
        date_str: Optional[str] = None,
        competition: str = "wc2026",
    ) -> dict[str, TeamAvailability]:
        result = {}
        for team in teams:
            try:
                result[team] = PlayerData.get_availability(team, date_str, competition)
            except Exception as exc:
                logger.warning("PlayerData fetch failed for %s: %s", team, exc)
                result[team] = TeamAvailability(team=team, players=[], source="error")
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fetch player availability for a team")
    parser.add_argument("teams", nargs="+", help="Team names e.g. England France")
    parser.add_argument("--date",        default=None,    help="YYYY-MM-DD (default: today)")
    parser.add_argument("--competition", default="wc2026")
    args = parser.parse_args()

    for team in args.teams:
        avail = PlayerData.get_availability(team, args.date, args.competition)
        mod   = avail.elo_modifier()
        print(f"\n{'─'*50}")
        print(f"Team     : {avail.team}")
        print(f"Source   : {avail.source}")
        print(f"Absent   : {[p.name for p in avail.absent]}")
        print(f"Doubtful : {[p.name for p in avail.doubtful]}")
        print(f"Elo delta: {mod:+.1f}")
