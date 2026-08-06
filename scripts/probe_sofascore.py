"""
SofaScore API probe — discovers available endpoints and data fields.

Run:
    python apollo/probe_sofascore.py

Requires env vars:
    SOFASCORE_API_KEY   - RapidAPI key
    SOFASCORE_API_HOST  - sofascore.p.rapidapi.com
    SOFASCORE_API_URL   - https://sofascore.p.rapidapi.com
"""

import os
import json
import time
import sys
from datetime import date, timedelta
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

# Load .env from apollo/ or project root
for env_path in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        print(f"Loaded .env from {env_path}")
        break


def get_headers():
    key = os.environ.get("SOFASCORE_API_KEY")
    host = os.environ.get("SOFASCORE_API_HOST", "sofascore.p.rapidapi.com")
    if not key:
        print("ERROR: SOFASCORE_API_KEY not set")
        sys.exit(1)
    return {
        "x-rapidapi-key": key,
        "x-rapidapi-host": host,
        "Content-Type": "application/json",
    }


BASE_URL = os.environ.get("SOFASCORE_API_URL", "https://sofascore.p.rapidapi.com")
HEADERS = None
SLEEP = 0.25  # 4 req/s, well under 5/s limit


def call(path: str, params: dict = None) -> dict | None:
    url = f"{BASE_URL}/{path.lstrip('/')}"
    print(f"\n>>> GET {url} params={params}")
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        print(f"    status: {resp.status_code}")
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"    body: {resp.text[:300]}")
            return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None
    finally:
        time.sleep(SLEEP)


def find_xg(data, path=""):
    """Recursively scan JSON for xG-related fields."""
    hits = []
    if isinstance(data, dict):
        for k, v in data.items():
            full_key = f"{path}.{k}" if path else k
            if any(x in k.lower() for x in ["xg", "expected", "xgoal"]):
                hits.append((full_key, v))
            hits.extend(find_xg(v, full_key))
    elif isinstance(data, list):
        for i, item in enumerate(data[:3]):
            hits.extend(find_xg(item, f"{path}[{i}]"))
    return hits


def safe_print(s):
    print(s.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"))


def print_keys(data, depth=0, max_depth=3):
    """Print JSON structure without drowning in values."""
    if depth > max_depth:
        return
    if isinstance(data, dict):
        for k, v in data.items():
            indent = "  " * depth
            if isinstance(v, (dict, list)):
                safe_print(f"{indent}{k}: ({type(v).__name__})")
                print_keys(v, depth + 1, max_depth)
            else:
                safe_print(f"{indent}{k}: {repr(v)[:80]}")
    elif isinstance(data, list) and data:
        safe_print(f"{'  ' * depth}[list of {len(data)}]")
        print_keys(data[0], depth + 1, max_depth)


def try_endpoints(candidates: list[tuple[str, dict]]) -> tuple[str, dict] | None:
    """Try a list of (path, params) until one returns 200. Returns (path, response) or None."""
    for path, params in candidates:
        data = call(path, params)
        if data:
            return path, data
    return None


def discover_all_endpoints():
    """
    Probe known RapidAPI sofascore sections:
    categories, sports, teams, players, managers,
    tournaments, matches, stages
    Focus: football only.
    """
    global HEADERS
    HEADERS = get_headers()

    # Known good IDs
    TEAM_ID       = 17      # Man City (football)
    AWAY_TEAM_ID  = 42      # Arsenal
    TOURNAMENT_ID = 17      # Premier League
    MATCH_ID      = 12894027
    PLAYER_ID     = 794946
    MANAGER_ID    = 53463   # Pep Guardiola (from teams/detail earlier)
    SEASON_ID     = 52186   # EPL 2024-25
    ROUND         = 1
    SPORT_ID      = 1       # football

    # Endpoints to probe, grouped by section
    # Format: (path, params)
    candidates = [
        # ── top-level ─────────────────────────────────────────────
        ("search",                  {"query": "Manchester City", "sportId": SPORT_ID}),
        ("categories/list",         {"sportId": SPORT_ID}),
        ("categories/list-live",    {"sportId": SPORT_ID}),
        ("sports/list",             {}),

        # ── teams ─────────────────────────────────────────────────
        ("teams/detail",            {"teamId": TEAM_ID}),
        ("teams/seasons",           {"teamId": TEAM_ID}),
        ("teams/schedule",          {"teamId": TEAM_ID}),
        ("teams/schedule",          {"teamId": TEAM_ID, "seasonId": SEASON_ID}),
        ("teams/results",           {"teamId": TEAM_ID}),
        ("teams/players",           {"teamId": TEAM_ID}),
        ("teams/squad",             {"teamId": TEAM_ID}),
        ("teams/transfers",         {"teamId": TEAM_ID}),
        ("teams/statistics",        {"teamId": TEAM_ID, "tournamentId": TOURNAMENT_ID, "seasonId": SEASON_ID}),
        ("teams/form",              {"teamId": TEAM_ID}),
        ("teams/injuries",          {"teamId": TEAM_ID}),
        ("teams/near-games",        {"teamId": TEAM_ID}),
        ("teams/events",            {"teamId": TEAM_ID, "type": "last", "pageIndex": 0}),
        ("teams/events",            {"teamId": TEAM_ID, "type": "next", "pageIndex": 0}),
        ("teams/featured-events",   {"teamId": TEAM_ID}),
        ("teams/streaks",           {"teamId": TEAM_ID}),
        ("teams/tournaments",       {"teamId": TEAM_ID}),

        # ── players ───────────────────────────────────────────────
        ("players/detail",          {"playerId": PLAYER_ID}),
        ("players/statistics",      {"playerId": PLAYER_ID, "tournamentId": TOURNAMENT_ID, "seasonId": SEASON_ID}),
        ("players/recent-matches",  {"playerId": PLAYER_ID}),
        ("players/transfers",       {"playerId": PLAYER_ID}),
        ("players/national-team",   {"playerId": PLAYER_ID}),
        ("players/heatmap",         {"playerId": PLAYER_ID, "matchId": MATCH_ID}),
        ("players/characteristics", {"playerId": PLAYER_ID}),
        ("players/events",          {"playerId": PLAYER_ID, "type": "last", "pageIndex": 0}),

        # ── managers ──────────────────────────────────────────────
        ("managers/detail",         {"managerId": MANAGER_ID}),
        ("managers/events",         {"managerId": MANAGER_ID, "type": "last", "pageIndex": 0}),
        ("managers/statistics",     {"managerId": MANAGER_ID}),

        # ── tournaments ───────────────────────────────────────────
        ("tournaments/detail",      {"tournamentId": TOURNAMENT_ID}),
        ("tournaments/seasons",     {"tournamentId": TOURNAMENT_ID}),
        ("tournaments/standings",   {"tournamentId": TOURNAMENT_ID, "seasonId": SEASON_ID}),
        ("tournaments/standings",   {"tournamentId": TOURNAMENT_ID, "seasonId": SEASON_ID, "type": "total"}),
        ("tournaments/rounds",      {"tournamentId": TOURNAMENT_ID, "seasonId": SEASON_ID}),
        ("tournaments/events",      {"tournamentId": TOURNAMENT_ID, "seasonId": SEASON_ID, "round": ROUND}),
        ("tournaments/schedule",    {"tournamentId": TOURNAMENT_ID, "seasonId": SEASON_ID}),
        ("tournaments/top-players", {"tournamentId": TOURNAMENT_ID, "seasonId": SEASON_ID}),
        ("tournaments/top-teams",   {"tournamentId": TOURNAMENT_ID, "seasonId": SEASON_ID}),
        ("tournaments/info",        {"tournamentId": TOURNAMENT_ID, "seasonId": SEASON_ID}),
        ("tournaments/last-matches",{"tournamentId": TOURNAMENT_ID, "seasonId": SEASON_ID}),
        ("tournaments/next-matches",{"tournamentId": TOURNAMENT_ID, "seasonId": SEASON_ID}),
        ("tournaments/statistics",  {"tournamentId": TOURNAMENT_ID, "seasonId": SEASON_ID}),
        ("tournaments/list",        {"categoryId": 1}),  # categoryId=1 = England football

        # ── matches ───────────────────────────────────────────────
        ("matches/detail",          {"matchId": MATCH_ID}),
        ("matches/statistics",      {"matchId": MATCH_ID}),
        ("matches/lineups",         {"matchId": MATCH_ID}),
        ("matches/incidents",       {"matchId": MATCH_ID}),
        ("matches/odds",            {"matchId": MATCH_ID}),
        ("matches/h2h",             {"matchId": MATCH_ID}),
        ("matches/highlights",      {"matchId": MATCH_ID}),
        ("matches/ratings",         {"matchId": MATCH_ID}),
        ("matches/votes",           {"matchId": MATCH_ID}),
        ("matches/shotmap",         {"matchId": MATCH_ID}),
        ("matches/momentum",        {"matchId": MATCH_ID}),
        ("matches/list",            {"date": "2024-05-19"}),
        ("matches/list-by-date",    {"date": "2024-05-19", "sportId": SPORT_ID}),
        ("matches/list-featured",   {"sportId": SPORT_ID}),
        ("matches/list-live",       {"sportId": SPORT_ID}),
        ("matches/get-pregame-form",{"matchId": MATCH_ID}),

        # ── stages ────────────────────────────────────────────────
        ("stages/detail",           {"stageId": ROUND}),
        ("stages/standings",        {"stageId": ROUND}),
        ("stages/events",           {"stageId": ROUND}),
    ]

    working = []
    print("=" * 60)
    print(f"PROBING {len(candidates)} endpoints...")
    print("=" * 60)

    for path, params in candidates:
        data = call(path, params)
        if data:
            working.append((path, list(params.keys())))
            print(f"\n*** WORKS: {path} params={list(params.keys())} ***")
            print_keys(data, max_depth=2)
            fname = path.replace("/", "_") + ".json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"  Saved: {fname}")

    print("\n" + "=" * 60)
    print(f"FOUND {len(working)} working endpoints:")
    for path, param_keys in working:
        print(f"  {path}  ({', '.join(param_keys)})")

    return working


def main():
    global HEADERS
    HEADERS = get_headers()

    # Known working: teamId=17 = Manchester City, tournamentId=17 = Premier League
    TEAM_ID = 17
    TOURNAMENT_ID = 17

    print("=" * 60)
    print("STEP 1: Find team's recent matches -> get event ID")
    print("=" * 60)

    event_id = None
    result = try_endpoints([
        ("teams/matches", {"teamId": TEAM_ID}),
        ("teams/events",  {"teamId": TEAM_ID}),
        ("teams/last-matches", {"teamId": TEAM_ID}),
        ("teams/results", {"teamId": TEAM_ID}),
    ])
    if result:
        path, data = result
        print(f"\n  Worked: {path}")
        print_keys(data, max_depth=3)
        with open("sofascore_team_matches_raw.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # Extract first event ID
        for key in ["events", "matches", "results"]:
            items = data.get(key, [])
            if items:
                event_id = items[0].get("id")
                home = items[0].get("homeTeam", {}).get("name", "?")
                away = items[0].get("awayTeam", {}).get("name", "?")
                print(f"\n>>> Event {event_id}: {home} vs {away}")
                break

    print("\n" + "=" * 60)
    print("STEP 2: Tournament seasons → season ID")
    print("=" * 60)

    season_id = None
    result = try_endpoints([
        ("tournaments/seasons", {"tournamentId": TOURNAMENT_ID}),
        ("unique-tournament/seasons", {"uniqueTournamentId": TOURNAMENT_ID}),
        ("tournaments/detail",  {"tournamentId": TOURNAMENT_ID}),
    ])
    if result:
        path, data = result
        print(f"\n  Worked: {path}")
        print_keys(data, max_depth=3)
        seasons = data.get("seasons", [])
        if seasons:
            season_id = seasons[0].get("id")
            print(f">>> Season ID: {season_id}")

    print("\n" + "=" * 60)
    print("STEP 3: Tournament events/fixtures")
    print("=" * 60)

    if season_id:
        result = try_endpoints([
            ("tournaments/rounds", {"tournamentId": TOURNAMENT_ID, "seasonId": season_id}),
            ("tournaments/events", {"tournamentId": TOURNAMENT_ID, "seasonId": season_id}),
            ("tournaments/matches", {"tournamentId": TOURNAMENT_ID, "seasonId": season_id}),
        ])
        if result:
            path, data = result
            print(f"\n  Worked: {path}")
            print_keys(data, max_depth=2)
            if not event_id:
                for key in ["events", "matches", "rounds"]:
                    items = data.get(key, [])
                    if items:
                        event_id = items[0].get("id")
                        break

    if not event_id:
        # Last resort: known SofaScore EPL match IDs (2024-25 season range)
        print("\n  No event found via API — trying known EPL event IDs")
        for eid in [12894027, 12894028, 12894029, 12800000, 13000000]:
            result = try_endpoints([
                ("events/detail", {"eventId": eid}),
                ("matches/detail", {"matchId": eid}),
            ])
            if result:
                event_id = eid
                break

    print("\n" + "=" * 60)
    print(f"STEP 4: Statistics for event {event_id} — hunting for xG")
    print("=" * 60)

    stats = None
    if event_id:
        result = try_endpoints([
            ("events/statistics", {"eventId": event_id}),
            ("matches/statistics", {"matchId": event_id}),
            ("events/detail",     {"eventId": event_id}),
            ("matches/detail",    {"matchId": event_id}),
        ])
        if result:
            path, stats = result
            print(f"\n  Worked: {path}")
            print_keys(stats, max_depth=4)
            xg_hits = find_xg(stats)
            if xg_hits:
                print("\n*** xG FIELDS FOUND ***")
                for p, val in xg_hits:
                    print(f"  {p} = {val}")
            else:
                print("\n  No xG fields found")
            with open("sofascore_stats_raw.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            print("  Raw saved: sofascore_stats_raw.json")

    print("\n" + "=" * 60)
    print(f"STEP 5: Lineups for event {event_id}")
    print("=" * 60)

    if event_id:
        result = try_endpoints([
            ("events/lineups",  {"eventId": event_id}),
            ("matches/lineups", {"matchId": event_id}),
        ])
        if result:
            path, data = result
            print(f"\n  Worked: {path}")
            print_keys(data, max_depth=3)
            with open("sofascore_lineups_raw.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print("  Raw saved: sofascore_lineups_raw.json")

    print("\n" + "=" * 60)
    print("STEP 6: Discover other endpoints via common patterns")
    print("=" * 60)

    candidates = [
        ("teams/players",        {"teamId": TEAM_ID}),
        ("teams/statistics",     {"teamId": TEAM_ID}),
        ("teams/transfers",      {"teamId": TEAM_ID}),
        ("tournaments/standings",{"tournamentId": TOURNAMENT_ID, "seasonId": season_id or 52186}),
        ("tournaments/top-players", {"tournamentId": TOURNAMENT_ID, "seasonId": season_id or 52186}),
        ("players/detail",       {"playerId": 794946}),   # Kevin De Bruyne
    ]
    working = []
    for path, params in candidates:
        data = call(path, params)
        if data:
            working.append(path)
            print(f"  WORKS: {path}")
            print_keys(data, max_depth=2)

    print("\n" + "=" * 60)
    print("PROBE COMPLETE")
    print("=" * 60)
    print(f"Working endpoints found: {working}")
    print("Check sofascore_*_raw.json for full payloads.")
    print("Key question answered in STEP 4: xG present?")


if __name__ == "__main__":
    discover_all_endpoints()
