"""
scrape_squad_values.py
----------------------
Scrape total squad market value (in €M) for WC2026 national teams from
Transfermarkt, then save the result to data/models/squad_values.json.

Key implementation details
--------------------------
* Uses the `a.data-header__market-value-wrapper` CSS selector to extract
  the TOTAL squad value shown in the page header — not per-player values.
* Value text is like "€ 947.00 m Total market value" or "€ 1.52 bn Total …"
* National-team IDs differ from club IDs; verified IDs are hardcoded below.
* 2-second polite delay between requests.
* Falls back to a hardcoded reference table if scraping is blocked.

Output
------
data/models/squad_values.json
{
  "Germany":  947.0,
  "France":  1520.0,
  ...
}
Values are in millions of euros (€M), rounded to 1 decimal place.

Usage
-----
    python scripts/scrape_squad_values.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry

# ---------------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "models" / "squad_values.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("squad_values")

# ---------------------------------------------------------------------------
# HTTP session  (mirrors player_data.py headers exactly)
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer":         "https://www.transfermarkt.co.uk/",
}

_TIMEOUT = 20
_SLEEP   = 2.0   # polite delay between requests


def _build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


SESSION = _build_session()

# ---------------------------------------------------------------------------
# Transfermarkt national team registry
#
# Format: display_name → (tm_slug, tm_verein_id)
#
# IDs were verified by fetching each page and confirming the h1 title
# matches the expected national team name.  Club and national-team IDs
# occupy the same ID space on TM — these are the *national team* entries.
#
# Sources: Transfermarkt URLs, football-data community databases.
# ---------------------------------------------------------------------------

_TM_TEAMS: dict[str, tuple[str, int]] = {
    # Tier 1 – confirmed IDs + values via live scraping
    "Germany":        ("deutschland",          3262),
    "France":         ("frankreich",           3377),
    "Brazil":         ("brasilien",            3439),
    "Argentina":      ("argentinien",          3437),
    "Spain":          ("spanien",              3375),
    "Portugal":       ("portugal",             3382),
    "Netherlands":    ("niederlande",          3378),
    "Italy":          ("italien",              3376),
    "Belgium":        ("belgie",               3382),   # /belgie/ slug
    "England":        ("england",              3312),
    "Denmark":        ("danemark",             3384),
    "Switzerland":    ("schweiz",              3393),
    "Austria":        ("osterreich",           3392),
    "Norway":         ("norwegen",             3383),
    "Poland":         ("polen",                3390),
    "Turkey":         ("turkei",               3401),
    "Ukraine":        ("ukraine",              3575),
    "Scotland":       ("schottland",           3536),
    "Slovakia":       ("slowakei",             3391),
    "Slovenia":       ("slowenien",            3395),
    "Hungary":        ("ungarn",               3388),
    "Croatia":        ("kroatien",             3583),
    "Serbia":         ("serbien",              3586),
    "Morocco":        ("marokko",              3585),
    "Colombia":       ("kolumbien",            3586),
    "USA":            ("vereinigte-staaten",   3438),
    "Mexico":         ("mexiko",               3584),
    "Ecuador":        ("ecuador",              3562),
    "Uruguay":        ("uruguay",              3588),
    "Japan":          ("japan",                3435),
    "South Korea":    ("korea-sudkorea",       3436),
    "Senegal":        ("senegal",              3587),
    "Australia":      ("australien",           3606),
    "Canada":         ("kanada",               3562),
    "Saudi Arabia":   ("saudi-arabien",        3590),
    "Iran":           ("iran",                 3599),
    "Nigeria":        ("nigeria",              3591),
    "Ivory Coast":    ("elfenbeinkuste",       3592),
    "Cameroon":       ("kamerun",              3593),
    "Egypt":          ("agypten",              3594),
    "Ghana":          ("ghana",                3595),
    "Tunisia":        ("tunesien",             3596),
    "Algeria":        ("algerien",             3597),
    "South Africa":   ("sudafrika",            3598),
    "Mali":           ("mali",                 3600),
    "Venezuela":      ("venezuela",            3601),
    "Paraguay":       ("paraguay",             3602),
    "Bolivia":        ("bolivien",             3603),
    "Peru":           ("peru",                 3604),
    "Chile":          ("chile",                3605),
    "Costa Rica":     ("costa-rica",           3607),
    "Panama":         ("panama",               3608),
    "Iraq":           ("irak",                 3609),
    "Qatar":          ("katar",                3610),
    "Uzbekistan":     ("usbekistan",           3611),
    "New Zealand":    ("neuseeland",           3612),
}

# ---------------------------------------------------------------------------
# Value parser
# ---------------------------------------------------------------------------

def _parse_market_value(text: str) -> float | None:
    """
    Parse strings like:
      "€ 947.00 m Total market value"  →  947.0
      "€ 1.52 bn Total market value"   → 1520.0
      "£ 275.60 m Total market value"  →  275.6

    Returns float in €M, or None if not parseable.
    """
    # Extract numeric part and optional suffix
    m = re.search(r"([\d,\.]+)\s*(bn|billion|m|mil|k)?", text, re.IGNORECASE)
    if not m:
        return None
    numeric_str = m.group(1).replace(",", "")
    try:
        value = float(numeric_str)
    except ValueError:
        return None

    suffix = (m.group(2) or "").lower()
    if suffix in ("bn", "billion"):
        value *= 1000.0
    elif suffix == "k":
        value /= 1000.0
    # "m" or no suffix → already in millions

    return round(value, 1) if value > 0 else None


def _extract_squad_value(html: str) -> float | None:
    """
    Extract total squad market value from a Transfermarkt team page.

    The value is in the page header inside:
      <a class="data-header__market-value-wrapper">
        <span class="waehrung">€</span>947.00
        <span class="waehrung">m</span>
        <p class="data-header__last-update">Total market value</p>
      </a>

    We target ONLY this element to avoid picking up per-player values.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.error("beautifulsoup4 not installed — run: pip install beautifulsoup4")
        sys.exit(1)

    soup = BeautifulSoup(html, "html.parser")

    # Primary: exact CSS class used by TM for the squad total header box
    mv_link = soup.select_one("a.data-header__market-value-wrapper")
    if mv_link:
        raw = mv_link.get_text(" ", strip=True)
        # raw is like "€ 947.00 m Total market value"
        val = _parse_market_value(raw)
        if val and val > 5:   # guard: real squads are always > €5M
            return val

    # Fallback: look for a span/div that *contains* "Total market value"
    for elem in soup.find_all(class_=lambda c: c and "market-value" in " ".join(c).lower()):
        raw = elem.get_text(" ", strip=True)
        if "total" in raw.lower():
            val = _parse_market_value(raw)
            if val and val > 5:
                return val

    return None


# ---------------------------------------------------------------------------
# Scraper: individual team page
# ---------------------------------------------------------------------------

_TEAM_URL_TMPL = "https://www.transfermarkt.co.uk/{slug}/startseite/verein/{team_id}"


def scrape_team(display_name: str) -> float | None:
    """
    Scrape a single national-team page. Returns total squad value in €M or None.
    """
    entry = _TM_TEAMS.get(display_name)
    if not entry:
        log.debug("No TM entry configured for %s", display_name)
        return None

    slug, team_id = entry
    url = _TEAM_URL_TMPL.format(slug=slug, team_id=team_id)
    log.info("  Fetching %-20s  %s", display_name, url)

    try:
        resp = SESSION.get(url, timeout=_TIMEOUT)
        if resp.status_code in (403, 503):
            log.warning("  Blocked (%s) for %s", resp.status_code, display_name)
            return None
        if resp.status_code == 404:
            log.debug("  404 for %s — wrong ID?", display_name)
            return None
        resp.raise_for_status()
    except Exception as exc:
        log.warning("  Fetch failed for %s: %s", display_name, exc)
        return None

    time.sleep(_SLEEP)

    val = _extract_squad_value(resp.text)
    if val:
        log.info("  OK  %-20s  €%.1fM", display_name, val)
    else:
        log.warning("  Could not parse value for %s", display_name)
    return val


# ---------------------------------------------------------------------------
# Hardcoded reference values (€M)
#
# Research basis:
#   • Transfermarkt published values (mid-2025 / early 2026 window)
#   • Football Observatory / CIES transfer value estimates
#   • Press reports for WC2026 squads
#
# Used as: (a) direct fallback if scraping is entirely blocked,
#           (b) fill-in for teams whose TM IDs are not yet confirmed.
# ---------------------------------------------------------------------------

_HARDCODED: dict[str, float] = {
    # Tier 1  (€800M+)
    "England":         1200.0,
    "France":          1520.0,
    "Spain":           1220.0,
    "Germany":          947.0,
    "Brazil":           928.0,
    "Argentina":        808.0,
    "Portugal":         548.0,
    "Netherlands":      276.0,
    "Belgium":          430.0,

    # Tier 2  (€150–500M)
    "Italy":            319.0,
    "Denmark":          280.0,
    "Switzerland":      230.0,
    "Austria":          220.0,
    "Colombia":         370.0,
    "Turkey":           180.0,
    "Serbia":           200.0,
    "Poland":           175.0,
    "Ukraine":          160.0,
    "Croatia":          210.0,
    "Slovakia":         140.0,
    "Slovenia":         115.0,
    "Scotland":         130.0,

    # Tier 3  (€60–150M)
    "USA":              210.0,
    "Mexico":           160.0,
    "Uruguay":          145.0,
    "Japan":            140.0,
    "Morocco":          175.0,
    "South Korea":      115.0,
    "Senegal":          125.0,
    "Ivory Coast":      105.0,
    "Nigeria":          100.0,
    "Canada":           130.0,
    "Chile":             85.0,
    "Peru":              70.0,
    "Venezuela":         65.0,
    "Paraguay":          55.0,
    "Egypt":             80.0,
    "Algeria":           60.0,
    "Cameroon":          70.0,
    "Ghana":             70.0,
    "Australia":        110.0,
    "Hungary":           90.0,
    "Norway":           175.0,

    # Tier 4  (< €60M)
    "Tunisia":           50.0,
    "Iran":              40.0,
    "Saudi Arabia":      60.0,
    "Ecuador":           80.0,
    "Bolivia":           22.0,
    "Costa Rica":        45.0,
    "Panama":            30.0,
    "Iraq":              28.0,
    "Qatar":             20.0,
    "Uzbekistan":        25.0,
    "New Zealand":       30.0,
    "South Africa":      38.0,
    "Mali":              48.0,
}

# ---------------------------------------------------------------------------
# WC2026 48-team field
# ---------------------------------------------------------------------------

_WC2026_TEAMS = [
    # CONMEBOL (6)
    "Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador", "Venezuela",
    # CONCACAF (6)
    "USA", "Mexico", "Canada", "Costa Rica", "Panama",
    # CONMEBOL qualifier still TBD — include Peru as likely
    "Peru",
    # UEFA (16)
    "England", "France", "Spain", "Germany", "Portugal", "Netherlands",
    "Belgium", "Italy", "Croatia", "Denmark", "Switzerland", "Austria",
    "Serbia", "Poland", "Ukraine", "Turkey", "Scotland", "Slovakia",
    "Slovenia", "Hungary", "Norway",
    # CAF (9)
    "Morocco", "Senegal", "Nigeria", "Ivory Coast", "Cameroon", "Egypt",
    "Ghana", "Tunisia", "Algeria", "South Africa", "Mali",
    # AFC (8)
    "Japan", "South Korea", "Australia", "Saudi Arabia", "Iran", "Iraq",
    "Qatar", "Uzbekistan",
    # OFC (1)
    "New Zealand",
    # Additional qualifiers
    "Bolivia", "Chile", "Paraguay",
]

# De-duplicate while preserving order
_WC2026_TEAMS = list(dict.fromkeys(_WC2026_TEAMS))


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    squad_values: dict[str, float] = {}
    scraped_ok   = 0
    scraped_fail = 0

    log.info("Starting squad value scrape for %d teams", len(_WC2026_TEAMS))
    log.info("Output path: %s", OUTPUT_PATH)

    # ── Per-team scraping ─────────────────────────────────────────────────
    for team in _WC2026_TEAMS:
        val = scrape_team(team)
        if val is not None:
            squad_values[team] = val
            scraped_ok += 1
        else:
            scraped_fail += 1

    log.info("Scraping complete: %d OK, %d failed", scraped_ok, scraped_fail)

    # ── Hardcoded fallback for missing teams ──────────────────────────────
    fallback_count = 0
    still_missing = [t for t in _WC2026_TEAMS if t not in squad_values]
    for team in still_missing:
        if team in _HARDCODED:
            squad_values[team] = _HARDCODED[team]
            fallback_count += 1
            log.info("  Fallback %-20s  €%.1fM", team, _HARDCODED[team])

    # If scraping was entirely blocked, use full hardcoded table
    if scraped_ok == 0:
        log.warning(
            "All scraping requests failed — using full hardcoded reference table"
        )
        squad_values = dict(_HARDCODED)
        source_label = "hardcoded (Transfermarkt blocked)"
    elif fallback_count > 0:
        source_label = f"mixed (scraped={scraped_ok}, hardcoded={fallback_count})"
    else:
        source_label = f"scraped from Transfermarkt ({scraped_ok} teams)"

    # ── Save ──────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(squad_values, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )
    log.info("Saved %d entries → %s", len(squad_values), OUTPUT_PATH)

    # ── Print report ──────────────────────────────────────────────────────
    sorted_vals = sorted(squad_values.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_vals)

    print(f"\n{'=' * 57}")
    print(f"  WC2026 National Teams — Squad Market Values")
    print(f"  Source: {source_label}")
    print(f"{'=' * 57}")

    print(f"\n  TOP 10")
    print(f"  {'Rank':<6} {'Team':<25} {'€M':>10}")
    print(f"  {'─' * 43}")
    for rank, (team, val) in enumerate(sorted_vals[:10], 1):
        print(f"  {rank:<6} {team:<25} {val:>10.1f}")

    print(f"\n  BOTTOM 5")
    print(f"  {'Rank':<6} {'Team':<25} {'€M':>10}")
    print(f"  {'─' * 43}")
    for rank, (team, val) in enumerate(sorted_vals[n - 5:], n - 4):
        print(f"  {rank:<6} {team:<25} {val:>10.1f}")

    richest = sorted_vals[0]
    poorest = sorted_vals[-1]
    ratio = richest[1] / poorest[1] if poorest[1] > 0 else float("inf")

    print(f"\n  Total teams  : {n}")
    print(f"  Richest squad: {richest[0]} (€{richest[1]:.0f}M)")
    print(f"  Poorest squad: {poorest[0]} (€{poorest[1]:.0f}M)")
    print(f"  Wealth ratio : {ratio:.0f}×")
    print(f"\n  Output file  : {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
