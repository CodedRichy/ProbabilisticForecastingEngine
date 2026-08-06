"""
Fixtures fetcher for today's football matches.

Primary source: ESPN public API (no auth required)
  https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard

Fallback: returns an empty list with a warning written to stderr.

Usage (CLI):
    python -m core.fixtures_fetcher
    python -m core.fixtures_fetcher --date 2026-06-24
    python -m core.fixtures_fetcher --competition wc2026
"""

import datetime
import sys
from typing import Optional

import requests
import requests.exceptions

from core.team_names import TEAM_ALIASES, normalize_team


# ── ESPN API config ───────────────────────────────────────────────────

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"

# Mapping of competition slugs to ESPN endpoint path overrides (future-proof).
COMPETITION_ENDPOINTS: dict[str, str] = {
    "wc2026": ESPN_URL,
}

REQUEST_TIMEOUT = 15  # seconds


# ── Core fetch ────────────────────────────────────────────────────────

def get_today_fixtures(
    date: Optional[str] = None,
    competition: str = "wc2026",
) -> list[dict]:
    """
    Fetch football fixtures for *date* from the ESPN public scoreboard API.

    Parameters
    ----------
    date : str, optional
        ISO-8601 date string ``"YYYY-MM-DD"``.  Defaults to today (UTC).
    competition : str
        Competition identifier.  ``"wc2026"`` targets FIFA World Cup 2026.
        Other values are accepted for forward-compatibility but currently
        resolve to the same ESPN FIFA World endpoint.

    Returns
    -------
    list[dict]
        Each element has the keys::

            {
                "home":     str,   # normalized home team name
                "away":     str,   # normalized away team name
                "time_utc": str,   # kick-off time as "HH:MM" (UTC) or ""
                "group":    str,   # e.g. "Group A" or ""
                "stage":    str,   # e.g. "Group Stage", "Round of 16", …
                "neutral":  bool,  # True for neutral-venue fixtures (WC2026)
            }

        Returns ``[]`` (empty list) if the API is unreachable or returns no
        usable data.  A warning is written to *stderr* in that case.
    """
    if date is None:
        date = datetime.date.today().isoformat()

    # ESPN wants compact date: "20260624"
    espn_date = date.replace("-", "")

    url = COMPETITION_ENDPOINTS.get(competition, ESPN_URL)

    try:
        response = requests.get(
            url,
            params={"dates": espn_date},
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; ProbabilisticForecastingEngine/1.0)"
                ),
                "Accept": "application/json",
            },
        )
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException as exc:
        print(
            f"WARNING [fixtures_fetcher]: ESPN API unreachable for {date}: {exc}",
            file=sys.stderr,
        )
        return []
    except ValueError as exc:
        # JSON decode failure
        print(
            f"WARNING [fixtures_fetcher]: Could not parse ESPN JSON for {date}: {exc}",
            file=sys.stderr,
        )
        return []

    return _parse_espn_payload(payload, date)


# ── ESPN JSON parser ──────────────────────────────────────────────────

def _parse_espn_payload(payload: dict, date: str) -> list[dict]:
    """
    Extract fixture dicts from the raw ESPN scoreboard JSON.

    The ESPN structure (simplified)::

        {
          "events": [
            {
              "name": "Argentina vs France",
              "status": { "type": { "state": "pre" } },
              "competitions": [
                {
                  "date": "2026-06-24T18:00Z",
                  "notes": [{"type": "event", "headline": "Group C"}],
                  "competitors": [
                    {"homeAway": "home", "team": {"displayName": "Argentina"}},
                    {"homeAway": "away", "team": {"displayName": "France"}}
                  ]
                }
              ]
            }
          ]
        }
    """
    events = payload.get("events", [])
    if not events:
        return []

    fixtures: list[dict] = []

    for event in events:
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        # ESPN usually provides one competition entry per event
        comp = competitions[0]

        # ── Kick-off time ─────────────────────────────────────────────
        raw_date = comp.get("date", "")  # e.g. "2026-06-24T18:00Z"
        time_utc = _extract_time_utc(raw_date)

        # ── Competitors ───────────────────────────────────────────────
        competitors = comp.get("competitors", [])
        home_name = ""
        away_name = ""
        for competitor in competitors:
            side = competitor.get("homeAway", "")
            display = competitor.get("team", {}).get("displayName", "")
            if side == "home":
                home_name = normalize_team(display)
            elif side == "away":
                away_name = normalize_team(display)

        # Skip if we could not identify both teams
        if not home_name or not away_name:
            continue

        # ── Group / Stage ─────────────────────────────────────────────
        group, stage = _extract_group_stage(comp, event)

        fixtures.append(
            {
                "home":     home_name,
                "away":     away_name,
                "time_utc": time_utc,
                "group":    group,
                "stage":    stage,
                # All WC group/knockout matches are at neutral venues
                # (USA/Canada/Mexico host cities). Domestic league fixtures
                # fetched via other endpoints should override this to False.
                "neutral":  True,
            }
        )

    return fixtures


def _extract_time_utc(raw_date: str) -> str:
    """
    Parse an ISO-8601 datetime string and return ``"HH:MM"`` in UTC.

    Returns ``""`` on any parse failure.
    """
    if not raw_date:
        return ""
    # Try compact ISO format first, then fallback
    for fmt in ("%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            dt = datetime.datetime.strptime(raw_date[:26], fmt)
            return dt.strftime("%H:%M")
        except ValueError:
            continue
    # Handle timezone offset strings e.g. "2026-06-24T18:00:00+00:00"
    try:
        # Strip timezone portion and parse
        dt = datetime.datetime.fromisoformat(raw_date.rstrip("Z").split("+")[0])
        return dt.strftime("%H:%M")
    except ValueError:
        return ""


def _extract_group_stage(comp: dict, event: dict) -> tuple[str, str]:
    """
    Return ``(group, stage)`` strings from ESPN competition metadata.

    ESPN stores notes like::
        [{"type": "event", "headline": "Group C - FIFA World Cup"}]

    The ``season`` dict on the event may carry ``"slug"`` with values like
    ``"group-stage"``, ``"round-of-16"``, etc.
    """
    group = ""
    stage = ""

    # Notes array: first note typically contains the group/round info
    notes = comp.get("notes", [])
    for note in notes:
        headline = note.get("headline", "")
        if not headline:
            continue
        # e.g. "Group C - FIFA World Cup" or "Round of 16 - FIFA World Cup"
        segment = headline.split(" - ")[0].strip()
        if segment.lower().startswith("group"):
            group = segment        # "Group C"
            stage = "Group Stage"
        else:
            stage = segment       # "Round of 16", "Quarter-Final", etc.
        break

    # Try ESPN season.slug as secondary source for stage
    if not stage:
        season_slug = event.get("season", {}).get("slug", "")
        if season_slug:
            stage = season_slug.replace("-", " ").title()

    # Try ESPN status note
    if not stage:
        status_detail = (
            event.get("status", {})
                 .get("type", {})
                 .get("detail", "")
        )
        if status_detail:
            stage = status_detail

    return group, stage


# ── CLI ───────────────────────────────────────────────────────────────

def _print_table(fixtures: list[dict], date: str) -> None:
    """Pretty-print a fixture list as a plain text table."""
    if not fixtures:
        print(f"No fixtures found for {date}.")
        return

    print(f"\nFixtures for {date} (UTC)\n")
    header = f"  {'Home':<25}  {'Away':<25}  {'Time':>5}  {'Group':<10}  {'Stage'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for fix in fixtures:
        row = (
            f"  {fix['home']:<25}  {fix['away']:<25}"
            f"  {fix['time_utc']:>5}  {fix['group']:<10}  {fix['stage']}"
        )
        print(row)

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch today's football fixtures from ESPN.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date in YYYY-MM-DD format (default: today).",
    )
    parser.add_argument(
        "--competition",
        default="wc2026",
        help="Competition identifier (default: wc2026).",
    )
    args = parser.parse_args()

    target_date = args.date or datetime.date.today().isoformat()
    print(f"Fetching fixtures from ESPN for {target_date} [{args.competition}]...")

    results = get_today_fixtures(date=target_date, competition=args.competition)
    _print_table(results, target_date)
