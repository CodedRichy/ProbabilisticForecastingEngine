"""
weather_model.py
----------------
Weather/conditions fetcher for WC2026 match venues using Open-Meteo (free, no API key required).

Usage:
    from core.weather_model import get_match_weather, list_venues

    weather = get_match_weather("MetLife Stadium", "2026-06-15T20:00:00")
    print(weather)
"""

import logging
import requests
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Venue registry — WC2026 host stadiums with (latitude, longitude)
# ---------------------------------------------------------------------------
VENUE_COORDS: dict[str, tuple[float, float]] = {
    "MetLife Stadium":   (40.8135, -74.0745),
    "Rose Bowl":         (34.1614, -118.1676),
    "AT&T Stadium":      (32.7473, -97.0945),
    "Estadio Azteca":    (19.3029, -99.1505),
    "BC Place":          (49.2767, -123.1117),
    "Levi's Stadium":    (37.4033, -121.9694),
    "Hard Rock Stadium": (25.9580, -80.2389),
}

# Open-Meteo hourly forecast endpoint (no API key required)
_OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&hourly=temperature_2m,wind_speed_10m,precipitation_probability"
    "&forecast_days=1"
)

_WIND_FLAG_THRESHOLD_KMH = 30.0  # affects aerial play above this speed


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def list_venues() -> list[str]:
    """Return a sorted list of all supported venue names."""
    return sorted(VENUE_COORDS.keys())


def get_conditions_label(temp_c: float) -> str:
    """
    Map a temperature (Celsius) to a human-readable conditions label.

    Thresholds:
        < 10 C  → "cold"
        < 25 C  → "ideal"
        25–30 C → "warm"
        > 30 C  → "hot"
    """
    if temp_c < 10.0:
        return "cold"
    if temp_c < 25.0:
        return "ideal"
    if temp_c <= 30.0:
        return "warm"
    return "hot"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_venue(venue: str) -> tuple[str, tuple[float, float]]:
    """
    Resolve a venue name (case-insensitive / substring) to its coordinates.

    Resolution order:
        1. Exact match
        2. Case-insensitive exact match
        3. Case-insensitive substring match (first hit wins)

    Raises:
        ValueError: if no match is found, listing all valid venue names.
    """
    # 1. Exact match
    if venue in VENUE_COORDS:
        return venue, VENUE_COORDS[venue]

    # 2. Case-insensitive exact match
    venue_lower = venue.strip().lower()
    for name, coords in VENUE_COORDS.items():
        if name.lower() == venue_lower:
            return name, coords

    # 3. Case-insensitive substring match
    candidates: list[tuple[str, tuple[float, float]]] = []
    for name, coords in VENUE_COORDS.items():
        if venue_lower in name.lower():
            candidates.append((name, coords))

    if len(candidates) == 1:
        logger.warning(
            "Venue %r resolved via substring match to %r.", venue, candidates[0][0]
        )
        return candidates[0]

    if len(candidates) > 1:
        names = [c[0] for c in candidates]
        raise ValueError(
            f"Venue {venue!r} is ambiguous — matched: {names}. "
            f"Valid venues: {list_venues()}"
        )

    raise ValueError(
        f"Venue {venue!r} not found. Valid venues: {list_venues()}"
    )


def _closest_hour_index(times: list[str], target_dt: datetime) -> int:
    """
    Return the index in *times* whose datetime is closest to *target_dt*.

    *times* is a list of ISO 8601 strings as returned by Open-Meteo
    (e.g. ["2026-06-15T00:00", "2026-06-15T01:00", ...]).
    """
    best_idx = 0
    best_delta = float("inf")

    for idx, t_str in enumerate(times):
        # Open-Meteo omits seconds — parse robustly
        try:
            slot_dt = datetime.fromisoformat(t_str)
        except ValueError:
            logger.warning("Could not parse time slot %r, skipping.", t_str)
            continue

        # Compare as naive datetimes (both are UTC/local-agnostic here)
        delta = abs((slot_dt - target_dt.replace(tzinfo=None)).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best_idx = idx

    return best_idx


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def get_match_weather(venue: str, match_datetime_utc: str) -> dict:
    """
    Fetch forecast weather conditions for a WC2026 match venue at a given time.

    Parameters
    ----------
    venue : str
        Venue name. Matched case-insensitively (and by substring) against
        the known VENUE_COORDS registry.
    match_datetime_utc : str
        ISO 8601 datetime string for the match kick-off time in UTC,
        e.g. ``"2026-06-15T20:00:00"``.

    Returns
    -------
    dict with keys:
        - ``temperature_c``     (float)  : Temperature at match time in °C.
        - ``wind_speed_kmh``    (float)  : Wind speed in km/h.
        - ``precipitation_pct`` (int)    : Precipitation probability 0–100.
        - ``conditions``        (str)    : One of "cold", "ideal", "warm", "hot".
        - ``wind_flag``         (bool)   : True if wind > 30 km/h.

    Raises
    ------
    ValueError
        If *venue* cannot be resolved to a known stadium.
    RuntimeError
        If the Open-Meteo API request fails or returns unexpected data.
    """
    # --- Resolve venue -------------------------------------------------------
    resolved_name, (lat, lon) = _resolve_venue(venue)
    logger.debug("Resolved venue %r → %r (%.4f, %.4f)", venue, resolved_name, lat, lon)

    # --- Parse target datetime -----------------------------------------------
    try:
        target_dt = datetime.fromisoformat(match_datetime_utc)
    except ValueError as exc:
        raise ValueError(
            f"Invalid match_datetime_utc {match_datetime_utc!r}. "
            "Expected ISO 8601 format, e.g. '2026-06-15T20:00:00'."
        ) from exc

    # --- Fetch forecast from Open-Meteo --------------------------------------
    url = _OPEN_METEO_URL.format(lat=lat, lon=lon)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Open-Meteo API request failed for venue {resolved_name!r}: {exc}"
        ) from exc

    try:
        data = response.json()
        hourly = data["hourly"]
        times: list[str] = hourly["time"]
        temps: list[Optional[float]] = hourly["temperature_2m"]
        winds: list[Optional[float]] = hourly["wind_speed_10m"]
        precips: list[Optional[int]] = hourly["precipitation_probability"]
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected response structure from Open-Meteo: {exc}"
        ) from exc

    # --- Find the closest hourly slot ----------------------------------------
    idx = _closest_hour_index(times, target_dt)
    logger.debug(
        "Match time %s → closest slot index %d (%s)", match_datetime_utc, idx, times[idx]
    )

    # --- Extract values (guard against None/missing) -------------------------
    raw_temp = temps[idx]
    raw_wind = winds[idx]
    raw_precip = precips[idx]

    if raw_temp is None:
        logger.warning("temperature_2m is None at slot %d, defaulting to 20.0 °C.", idx)
        raw_temp = 20.0
    if raw_wind is None:
        logger.warning("wind_speed_10m is None at slot %d, defaulting to 0.0 km/h.", idx)
        raw_wind = 0.0
    if raw_precip is None:
        logger.warning(
            "precipitation_probability is None at slot %d, defaulting to 0.", idx
        )
        raw_precip = 0

    temperature_c = float(raw_temp)
    wind_speed_kmh = float(raw_wind)
    precipitation_pct = int(raw_precip)

    return {
        "temperature_c":     temperature_c,
        "wind_speed_kmh":    wind_speed_kmh,
        "precipitation_pct": precipitation_pct,
        "conditions":        get_conditions_label(temperature_c),
        "wind_flag":         wind_speed_kmh > _WIND_FLAG_THRESHOLD_KMH,
    }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_venue = "MetLife Stadium"
    test_datetime = datetime.utcnow().strftime("%Y-%m-%dT20:00:00")

    print(f"Fetching weather for '{test_venue}' at {test_datetime} UTC ...\n")

    try:
        result = get_match_weather(test_venue, test_datetime)
        print("Weather result:")
        for key, value in result.items():
            print(f"  {key:<22} = {value}")
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
