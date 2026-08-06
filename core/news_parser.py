import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_BBC_SLUGS: dict[str, str] = {
    "manchester united": "manchester-united",
    "man united": "manchester-united",
    "man utd": "manchester-united",
    "manchester city": "manchester-city",
    "man city": "manchester-city",
    "arsenal": "arsenal",
    "chelsea": "chelsea",
    "liverpool": "liverpool",
    "tottenham": "tottenham-hotspur",
    "spurs": "tottenham-hotspur",
    "newcastle": "newcastle-united",
    "aston villa": "aston-villa",
    "west ham": "west-ham-united",
    "everton": "everton",
    "brighton": "brighton-hove-albion",
    "brentford": "brentford",
    "fulham": "fulham",
    "wolves": "wolverhampton-wanderers",
    "wolverhampton": "wolverhampton-wanderers",
    "nottingham forest": "nottingham-forest",
    "crystal palace": "crystal-palace",
    "bournemouth": "afc-bournemouth",
    "leicester": "leicester-city",
    "ipswich": "ipswich-town",
    "southampton": "southampton",
    "real madrid": "real-madrid",
    "barcelona": "barcelona",
    "atletico madrid": "atletico-madrid",
    "atletico": "atletico-madrid",
    "sevilla": "sevilla",
    "juventus": "juventus",
    "ac milan": "ac-milan",
    "inter milan": "inter-milan",
    "inter": "inter-milan",
    "napoli": "napoli",
    "bayern munich": "bayern-munich",
    "bayern": "bayern-munich",
    "borussia dortmund": "borussia-dortmund",
    "dortmund": "borussia-dortmund",
    "psg": "paris-saint-germain",
    "paris saint-germain": "paris-saint-germain",
    "paris saint germain": "paris-saint-germain",
}

_SKY_SLUGS: dict[str, str] = {
    "manchester united": "man-utd",
    "man united": "man-utd",
    "man utd": "man-utd",
    "manchester city": "man-city",
    "man city": "man-city",
    "arsenal": "arsenal",
    "chelsea": "chelsea",
    "liverpool": "liverpool",
    "tottenham": "tottenham",
    "spurs": "tottenham",
    "newcastle": "newcastle",
    "aston villa": "aston-villa",
    "west ham": "west-ham",
    "everton": "everton",
    "brighton": "brighton",
    "brentford": "brentford",
    "fulham": "fulham",
    "wolves": "wolves",
    "wolverhampton": "wolves",
    "nottingham forest": "nottm-forest",
    "crystal palace": "crystal-palace",
    "bournemouth": "bournemouth",
    "leicester": "leicester",
    "ipswich": "ipswich",
    "southampton": "southampton",
}

_ESPN_IDS: dict[str, int] = {
    "manchester united": 360,
    "man united": 360,
    "man utd": 360,
    "manchester city": 382,
    "man city": 382,
    "arsenal": 359,
    "chelsea": 363,
    "liverpool": 364,
    "tottenham": 367,
    "spurs": 367,
    "newcastle": 361,
    "aston villa": 1,
    "west ham": 371,
    "everton": 368,
    "brighton": 331,
    "real madrid": 86,
    "barcelona": 83,
    "atletico madrid": 1068,
    "atletico": 1068,
    "juventus": 111,
    "ac milan": 103,
    "inter milan": 110,
    "inter": 110,
    "napoli": 114,
    "bayern munich": 132,
    "bayern": 132,
    "borussia dortmund": 124,
    "dortmund": 124,
    "psg": 160,
    "paris saint-germain": 160,
    "paris saint germain": 160,
}

_STAR_PLAYER_PATTERNS = re.compile(
    r"\b(captain|star|key player|main striker|top scorer|ace|talisman|first choice)\b",
    re.IGNORECASE,
)

_OUT_PATTERNS = re.compile(
    r"(?P<player>[A-Z][a-z]+(?: [A-Z][a-z]+)+)\s+(?:is\s+)?(?:ruled out|injured|out|misses|missing|will miss|sidelined|unavailable)",
    re.IGNORECASE,
)

_DOUBTFUL_PATTERNS = re.compile(
    r"(?P<player>[A-Z][a-z]+(?: [A-Z][a-z]+)+)\s+(?:is\s+)?(?:doubtful|50-50|fifty-fifty|a doubt|fitness doubt|touch and go|rated doubtful)",
    re.IGNORECASE,
)

_AVAILABLE_PATTERNS = re.compile(
    r"(?P<player>[A-Z][a-z]+(?: [A-Z][a-z]+)+)\s+(?:is\s+)?(?:available|fit|returned|back|cleared|passed fit|in contention)",
    re.IGNORECASE,
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_MODEL = "gemma4:e4b"
_OLLAMA_PROMPT_TEMPLATE = (
    "Extract injury/lineup information from this football news text. "
    "Return JSON only, no explanation.\nText: {text}\n\n"
    'JSON format: {{"out": ["player names"], "doubtful": ["player names"], '
    '"attack_impact": -0.0 to -1.0, "defense_impact": -0.0 to -1.0}}'
)

_EMPTY_RESULT: dict = {
    "team": "",
    "source": "",
    "fetched_at": "",
    "out": [],
    "doubtful": [],
    "available": [],
    "key_absence": False,
    "attack_impact": 0.0,
    "defense_impact": 0.0,
    "confidence": 0.0,
    "raw_text": "",
}


def _normalize_team(team: str) -> str:
    return team.strip().lower()


def _fetch_url(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        if resp.status_code == 200:
            return resp.text
        return None
    except Exception as exc:
        logger.debug("fetch failed for %s: %s", url, exc)
        return None


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _query_ollama(text: str) -> Optional[dict]:
    prompt = _OLLAMA_PROMPT_TEMPLATE.format(text=text[:2000])
    payload = {
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }
    try:
        resp = requests.post(_OLLAMA_URL, json=payload, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        raw = data.get("response", "")
        parsed = json.loads(raw)
        return parsed
    except Exception as exc:
        logger.debug("ollama query failed: %s", exc)
        return None


def _regex_extract(text: str) -> dict:
    out_players = [m.group("player") for m in _OUT_PATTERNS.finditer(text)]
    doubtful_players = [m.group("player") for m in _DOUBTFUL_PATTERNS.finditer(text)]
    available_players = [m.group("player") for m in _AVAILABLE_PATTERNS.finditer(text)]

    out_players = list(dict.fromkeys(out_players))
    doubtful_players = list(dict.fromkeys(p for p in doubtful_players if p not in out_players))
    available_players = list(dict.fromkeys(p for p in available_players if p not in out_players and p not in doubtful_players))

    attack_impact = min(0.0, -0.15 * len(out_players) - 0.07 * len(doubtful_players))
    attack_impact = max(-1.0, attack_impact)
    defense_impact = attack_impact

    return {
        "out": out_players,
        "doubtful": doubtful_players,
        "available": available_players,
        "attack_impact": round(attack_impact, 3),
        "defense_impact": round(defense_impact, 3),
        "fallback_required": True,
    }


def _fetch_bbc(team_key: str) -> tuple[Optional[str], str]:
    slug = _BBC_SLUGS.get(team_key)
    if not slug:
        return None, ""
    url = f"https://www.bbc.com/sport/football/teams/{slug}"
    html = _fetch_url(url)
    if html:
        return _strip_html(html), url
    return None, url


def _fetch_sky(team_key: str) -> tuple[Optional[str], str]:
    slug = _SKY_SLUGS.get(team_key)
    if not slug:
        return None, ""
    url = f"https://www.skysports.com/{slug}-injuries"
    html = _fetch_url(url)
    if html:
        return _strip_html(html), url
    return None, url


def _fetch_espn(team_key: str) -> tuple[Optional[str], str]:
    espn_id = _ESPN_IDS.get(team_key)
    if not espn_id:
        url = f"https://www.espn.com/soccer/injuries"
        html = _fetch_url(url)
        if html:
            return _strip_html(html), url
        return None, url
    url = f"https://www.espn.com/soccer/team/injuries/_/id/{espn_id}"
    html = _fetch_url(url)
    if html:
        return _strip_html(html), url
    return None, url


def _detect_key_absence(out_players: list[str], text: str) -> bool:
    if not out_players:
        return False
    if _STAR_PLAYER_PATTERNS.search(text):
        return True
    return False


class NewsParser:
    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def parse(self, team: str) -> dict:
        team_key = _normalize_team(team)
        result = dict(_EMPTY_RESULT)
        result["team"] = team
        result["fetched_at"] = datetime.utcnow().isoformat()

        text, source_url = self._scrape(team_key)
        if not text:
            logger.warning("no text fetched for team=%s", team)
            result["confidence"] = 0.0
            return result

        result["source"] = source_url
        result["raw_text"] = text[:500]

        ollama_data = _query_ollama(text)
        if ollama_data and not ollama_data.get("fallback_required"):
            out = ollama_data.get("out", [])
            doubtful = ollama_data.get("doubtful", [])
            available = ollama_data.get("available", [])
            attack_impact = float(ollama_data.get("attack_impact", 0.0))
            defense_impact = float(ollama_data.get("defense_impact", 0.0))
            confidence = 0.85
        else:
            fallback = _regex_extract(text)
            out = fallback["out"]
            doubtful = fallback["doubtful"]
            available = fallback["available"]
            attack_impact = fallback["attack_impact"]
            defense_impact = fallback["defense_impact"]
            confidence = 0.45 if (out or doubtful) else 0.2

        result["out"] = out
        result["doubtful"] = doubtful
        result["available"] = available
        result["attack_impact"] = max(-1.0, min(0.0, attack_impact))
        result["defense_impact"] = max(-1.0, min(0.0, defense_impact))
        result["key_absence"] = _detect_key_absence(out, text)
        result["confidence"] = confidence

        return result

    def _scrape(self, team_key: str) -> tuple[Optional[str], str]:
        for fetch_fn in (_fetch_bbc, _fetch_sky, _fetch_espn):
            try:
                text, url = fetch_fn(team_key)
                if text and len(text) > 100:
                    return text, url
            except Exception as exc:
                logger.debug("scrape error in %s: %s", fetch_fn.__name__, exc)
        return None, ""


class TeamNewsCache:
    _TTL_MINUTES: int = 30

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}
        self._parser = NewsParser()

    def get(self, team: str, date: str) -> dict:
        cache_key = f"{_normalize_team(team)}|{date}"
        entry = self._cache.get(cache_key)
        if entry:
            fetched_at_str = entry.get("fetched_at", "")
            if fetched_at_str:
                try:
                    fetched_at = datetime.fromisoformat(fetched_at_str)
                    age = datetime.utcnow() - fetched_at
                    if age < timedelta(minutes=self._TTL_MINUTES):
                        return entry
                except ValueError:
                    pass
        fresh = self._parser.parse(team)
        self._cache[cache_key] = fresh
        return fresh

    def invalidate(self, team: str, date: str) -> None:
        cache_key = f"{_normalize_team(team)}|{date}"
        self._cache.pop(cache_key, None)


_cache = TeamNewsCache()


def parse_lineup_adjustments(home_team: str, away_team: str, date: str) -> dict:
    home_news = _cache.get(home_team, date)
    away_news = _cache.get(away_team, date)

    home_adj = _compute_adjustment(home_news, is_home=True)
    away_adj = _compute_adjustment(away_news, is_home=False)

    return {"home_adj": round(home_adj, 4), "away_adj": round(away_adj, 4)}


def _compute_adjustment(news: dict, is_home: bool) -> float:
    if not news or not news.get("out") and not news.get("key_absence"):
        return 0.0

    base_floor = -0.05 if is_home else -0.04
    base_ceil = -0.12 if is_home else -0.10

    attack_impact = news.get("attack_impact", 0.0)
    defense_impact = news.get("defense_impact", 0.0)
    combined = (attack_impact + defense_impact) / 2.0

    key_absence = news.get("key_absence", False)
    confidence = news.get("confidence", 0.5)

    if key_absence:
        raw_adj = base_ceil
    elif combined < -0.5:
        raw_adj = base_ceil * 0.75 + base_floor * 0.25
    elif combined < -0.25:
        raw_adj = (base_floor + base_ceil) / 2.0
    elif news.get("out"):
        raw_adj = base_floor
    else:
        return 0.0

    adj = raw_adj * confidence
    return max(base_ceil, min(0.0, adj))
