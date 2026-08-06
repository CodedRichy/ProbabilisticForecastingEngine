"""
Betfair API client for the Apollo football forecasting engine.

Set these in .env:
    BETFAIR_APP_KEY=<your_application_key>
    BETFAIR_SESSION_TOKEN=<your_session_token>

The session token is obtained by logging in via the Betfair Identity API
(https://identitysso-cert.betfair.com/api/certlogin) or the non-interactive
login endpoint. The app key is found in the Betfair Developer Tools portal.
"""

import sys
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

# Allow `from scripts.scrape_betfair_odds import get_betfair_prices` when the
# repo root is not on sys.path (e.g. when called from a sub-directory).
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BETFAIR_BASE_URL = "https://api.betfair.com/exchange/betting/rest/v1.0/"
DATA_DIR = Path(__file__).parent.parent / "data" / "live"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_headers() -> dict | None:
    """
    Build the Betfair auth headers from environment variables.
    Returns None (and logs a WARNING) if credentials are missing.
    """
    app_key = os.getenv("BETFAIR_APP_KEY", "").strip()
    session_token = os.getenv("BETFAIR_SESSION_TOKEN", "").strip()

    if not app_key or not session_token:
        logger.warning(
            "BETFAIR_APP_KEY or BETFAIR_SESSION_TOKEN is not set in the environment. "
            "Add them to your .env file. Returning empty results."
        )
        return None

    return {
        "X-Application": app_key,
        "X-Authentication": session_token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _post(endpoint: str, headers: dict, body: dict) -> list | dict | None:
    """
    POST to a Betfair REST endpoint and return the parsed JSON response.
    Returns None on any HTTP or parsing error.
    """
    url = BETFAIR_BASE_URL + endpoint
    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP error calling %s: %s — response: %s", url, exc, exc.response.text if exc.response else "n/a")
    except requests.exceptions.ConnectionError as exc:
        logger.error("Connection error calling %s: %s", url, exc)
    except requests.exceptions.Timeout:
        logger.error("Request timed out calling %s", url)
    except requests.exceptions.RequestException as exc:
        logger.error("Unexpected request error calling %s: %s", url, exc)
    except ValueError as exc:
        logger.error("Failed to parse JSON response from %s: %s", url, exc)
    return None


# ---------------------------------------------------------------------------
# API Operations
# ---------------------------------------------------------------------------


def list_football_markets(date_str: str) -> list[dict]:
    """
    Call listMarketCatalogue for SOCCER / MATCH_ODDS markets on *date_str*.

    Parameters
    ----------
    date_str : str
        ISO date in ``YYYY-MM-DD`` format (e.g. ``"2025-09-14"``).
        Markets whose scheduled start time falls within that calendar day
        (UTC) are returned.

    Returns
    -------
    list[dict]
        Raw market catalogue entries from Betfair, sorted by matched volume
        (descending). Empty list on error or missing credentials.
    """
    headers = _get_headers()
    if headers is None:
        return []

    # Build inclusive UTC time range for the requested date
    try:
        day_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        logger.error("Invalid date_str '%s'. Expected YYYY-MM-DD format.", date_str)
        return []

    from datetime import timedelta
    day_end = day_start + timedelta(days=1)

    body = {
        "filter": {
            "eventTypeIds": ["1"],           # 1 = Soccer on Betfair
            "marketTypeCodes": ["MATCH_ODDS"],
            "marketStartTime": {
                "from": day_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": day_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        },
        "marketProjection": [
            "EVENT",
            "MARKET_START_TIME",
            "RUNNER_DESCRIPTION",
            "MARKET_DESCRIPTION",
        ],
        "sort": "MAXIMUM_TRADED",
        "maxResults": "200",
        "locale": "en",
    }

    logger.info("Fetching football markets for %s …", date_str)
    result = _post("listMarketCatalogue/", headers, body)

    if result is None:
        logger.error("listMarketCatalogue returned no data for %s.", date_str)
        return []

    if not isinstance(result, list):
        logger.error("Unexpected listMarketCatalogue response shape: %s", type(result))
        return []

    logger.info("Found %d markets for %s.", len(result), date_str)
    return result


def list_market_book(market_ids: list[str]) -> list[dict]:
    """
    Call listMarketBook to retrieve best available back/lay prices.

    Parameters
    ----------
    market_ids : list[str]
        Betfair market IDs (e.g. ``["1.234567890"]``).

    Returns
    -------
    list[dict]
        Raw market book entries. Empty list on error or missing credentials.
    """
    if not market_ids:
        return []

    headers = _get_headers()
    if headers is None:
        return []

    body = {
        "marketIds": market_ids,
        "priceProjection": {
            "priceData": ["SP_TRADED", "EX_BEST_OFFERS"],
            "exBestOffersOverrides": {
                "bestPricesDepth": 3,
                "rollupModel": "STAKE",
                "rollupLimit": 0,
            },
            "virtualise": False,
        },
        "orderProjection": "EXECUTABLE",
        "matchProjection": "NO_ROLLUP",
    }

    logger.info("Fetching market book for %d market(s) …", len(market_ids))
    result = _post("listMarketBook/", headers, body)

    if result is None:
        logger.error("listMarketBook returned no data.")
        return []

    if not isinstance(result, list):
        logger.error("Unexpected listMarketBook response shape: %s", type(result))
        return []

    return result


def convert_price_to_prob(price: float) -> float:
    """
    Convert a Betfair decimal exchange price to an implied probability.

    On a betting exchange there is no bookmaker overround, so the raw
    conversion ``1 / price`` gives the market-implied probability directly.

    Parameters
    ----------
    price : float
        Betfair decimal odds (must be >= 1.0).

    Returns
    -------
    float
        Implied probability in [0, 1]. Returns 0.0 for invalid inputs.
    """
    if not price or price < 1.0:
        return 0.0
    return 1.0 / price


# ---------------------------------------------------------------------------
# Public orchestration function
# ---------------------------------------------------------------------------


def get_betfair_prices(date_str: str = None) -> list[dict]:
    """
    Fetch Betfair MATCH_ODDS prices for football markets on *date_str* and
    persist them to ``data/live/betfair_odds_{date}.json``.

    Parameters
    ----------
    date_str : str, optional
        Target date in ``YYYY-MM-DD`` format. Defaults to today (UTC).

    Returns
    -------
    list[dict]
        Each entry contains::

            {
                "market_id":   str,
                "home_team":   str,
                "away_team":   str,
                "home_prob":   float,
                "draw_prob":   float,
                "away_prob":   float,
                "home_price":  float,
                "draw_price":  float,
                "away_price":  float,
                "event_time":  str,   # ISO-8601 UTC
            }

        Returns an empty list on error or missing credentials.
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # --- Step 1: fetch market catalogue ---------------------------------
    catalogue = list_football_markets(date_str)
    if not catalogue:
        logger.warning("No markets found for %s — returning empty list.", date_str)
        return []

    # Build a lookup: market_id -> catalogue entry
    catalogue_map: dict[str, dict] = {m["marketId"]: m for m in catalogue}
    market_ids = list(catalogue_map.keys())

    # --- Step 2: fetch market book (prices) -----------------------------
    books = list_market_book(market_ids)
    if not books:
        logger.warning("listMarketBook returned empty for %d markets.", len(market_ids))
        return []

    # Build a lookup: market_id -> book entry
    book_map: dict[str, dict] = {b["marketId"]: b for b in books}

    # --- Step 3: parse and combine -------------------------------------
    results: list[dict] = []

    for market_id, cat_entry in catalogue_map.items():
        book_entry = book_map.get(market_id)
        if book_entry is None:
            logger.debug("No book data for market %s — skipping.", market_id)
            continue

        # Extract runner descriptions from catalogue
        runners_meta: dict[int, str] = {}
        for runner in cat_entry.get("runners", []):
            runners_meta[runner["selectionId"]] = runner.get("runnerName", "Unknown")

        # Betfair MATCH_ODDS runners: Home (sortPriority=1), Draw (2), Away (3)
        # Sort runners by sortPriority to get deterministic Home/Draw/Away order
        cat_runners_sorted = sorted(
            cat_entry.get("runners", []),
            key=lambda r: r.get("sortPriority", 99),
        )

        if len(cat_runners_sorted) < 3:
            logger.debug(
                "Market %s has fewer than 3 runners (%d) — skipping.",
                market_id,
                len(cat_runners_sorted),
            )
            continue

        home_sel_id = cat_runners_sorted[0]["selectionId"]
        draw_sel_id = cat_runners_sorted[1]["selectionId"]
        away_sel_id = cat_runners_sorted[2]["selectionId"]

        home_team = runners_meta.get(home_sel_id, "Home")
        away_team = runners_meta.get(away_sel_id, "Away")

        # Extract best available back price for each runner from book
        def _best_back_price(sel_id: int) -> float:
            for runner in book_entry.get("runners", []):
                if runner["selectionId"] == sel_id:
                    available = (
                        runner.get("ex", {}).get("availableToBack", [])
                    )
                    if available:
                        return float(available[0].get("price", 0.0))
            return 0.0

        home_price = _best_back_price(home_sel_id)
        draw_price = _best_back_price(draw_sel_id)
        away_price = _best_back_price(away_sel_id)

        home_prob = convert_price_to_prob(home_price)
        draw_prob = convert_price_to_prob(draw_price)
        away_prob = convert_price_to_prob(away_price)

        # Event time
        event_time = cat_entry.get("marketStartTime", "")
        if not event_time:
            try:
                event_time = cat_entry["event"]["openDate"]
            except (KeyError, TypeError):
                event_time = ""

        results.append(
            {
                "market_id": market_id,
                "home_team": home_team,
                "away_team": away_team,
                "home_prob": round(home_prob, 6),
                "draw_prob": round(draw_prob, 6),
                "away_prob": round(away_prob, 6),
                "home_price": home_price,
                "draw_price": draw_price,
                "away_price": away_price,
                "event_time": event_time,
            }
        )

    logger.info("Parsed %d match-odds markets for %s.", len(results), date_str)

    # --- Step 4: persist to disk ----------------------------------------
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DATA_DIR / f"betfair_odds_{date_str}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        logger.info("Saved %d records to %s", len(results), out_path)
    except OSError as exc:
        logger.error("Failed to write output file: %s", exc)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape Betfair MATCH_ODDS prices for football markets."
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Target date (default: today UTC).",
    )
    args = parser.parse_args()

    prices = get_betfair_prices(date_str=args.date)

    if not prices:
        print("No prices returned. Check credentials and date.")
    else:
        print(f"\nRetrieved {len(prices)} markets:\n")
        for entry in prices:
            print(
                f"  {entry['event_time'][:16]}  "
                f"{entry['home_team']} vs {entry['away_team']}"
                f"  |  Home {entry['home_price']:.2f} ({entry['home_prob']:.1%})"
                f"  Draw {entry['draw_price']:.2f} ({entry['draw_prob']:.1%})"
                f"  Away {entry['away_price']:.2f} ({entry['away_prob']:.1%})"
                f"  [market: {entry['market_id']}]"
            )
        print(f"\nData saved to data/live/betfair_odds_{args.date or 'today'}.json")
