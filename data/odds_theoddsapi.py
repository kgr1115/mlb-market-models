"""
The Odds API (the-odds-api.com) scraper — DraftKings source.

The free tier grants 500 credits/month. One call with
`markets=h2h,spreads,totals` costs 3 credits, and `regions` + `bookmakers`
are free filters. We refresh at five fixed times per day (ET):

    09:00, 12:00, 15:00, 17:00, 21:00

which is 5 calls * 3 credits * 30 days = 450 credits/month — under budget
with headroom for manual retries.

Design:
  - The caller (`fetch_dk_via_theoddsapi_cached`) is safe to call on every
    poll; it only hits the network when the most recent "window" boundary
    has passed since the last successful fetch.
  - The most recent payload is persisted to disk (`data/cache/theoddsapi_dk.json`)
    alongside a fetched_at_utc marker, so restarts don't waste credits.
  - All parsed prices go through the same `OddsSnapshot` dataclass as the
    other scrapers — so downstream code (OddsCache, build_market_data,
    shop_event) doesn't know or care where the DK numbers came from.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib import error, parse, request

from .odds_models import OddsBook, OddsSnapshot, make_event_id
from .team_names import try_normalize_team

log = logging.getLogger(__name__)

# Refresh windows in Eastern Time — when ET passes any of these and our
# cache was written before that window, we refetch. Order-independent but
# we keep them sorted for readability.
_WINDOWS_ET = (
    (9, 0),
    (12, 0),
    (15, 0),
    (17, 0),
    (21, 0),
)

# Endpoint — `baseball_mlb` is the sport key; h2h=moneyline, spreads=run line,
# totals=over/under. Bookmakers filter is free and trims the payload.
_BASE_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
_PARAMS = {
    "regions": "us",
    "markets": "h2h,spreads,totals",
    "oddsFormat": "decimal",
    "bookmakers": "draftkings",
    "dateFormat": "iso",
}

# Cache location — mirrors the existing dk_sample.json convention (project
# root) and gets gitignored via data/cache/ rule in .gitignore.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CACHE_DIR = _ROOT / "data" / "cache"
_CACHE_FILE = _CACHE_DIR / "theoddsapi_dk.json"

# In-process flag so we don't re-read disk cache on every poll.
_LAST_FETCH_AT_UTC: Optional[datetime] = None

_DOTENV_LOADED = False


def _ensure_dotenv_loaded() -> None:
    """Load .env once per process, best-effort.

    We defer the import so the module still works if python-dotenv isn't
    installed — the key just has to live in a real env var instead.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        log.debug("python-dotenv not installed; using os.environ directly")
        return
    env_path = _ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        log.info("Loaded env from %s", env_path)


def _get_api_key() -> Optional[str]:
    _ensure_dotenv_loaded()
    k = os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY")
    return k.strip() if k else None


# ---- window / cache logic ---------------------------------------------

def _now_et() -> datetime:
    """Current time in America/New_York."""
    try:
        from zoneinfo import ZoneInfo  # Py 3.9+
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:  # noqa: BLE001
        # Fallback: assume ET = UTC-5 (close enough for windowing edge cases)
        from datetime import timedelta
        return (datetime.now(timezone.utc) - timedelta(hours=5)).replace(
            tzinfo=timezone.utc
        )


def _most_recent_window_et(now_et: datetime) -> datetime:
    """Return the datetime of the most recent window boundary that has
    already passed today in ET. If the clock hasn't yet reached the first
    window, returns yesterday's last window.
    """
    today = now_et.replace(microsecond=0, second=0)
    candidates = [today.replace(hour=h, minute=m) for h, m in _WINDOWS_ET]
    passed = [c for c in candidates if c <= now_et]
    if passed:
        return passed[-1]
    # Before 9 AM today → use last window of the previous day.
    from datetime import timedelta
    yesterday = (now_et - timedelta(days=1)).replace(
        hour=_WINDOWS_ET[-1][0], minute=_WINDOWS_ET[-1][1],
        second=0, microsecond=0,
    )
    return yesterday


def _load_cache() -> tuple[Optional[list[dict]], Optional[datetime]]:
    """Return (payload, fetched_at_utc) if cache exists and is parseable."""
    if not _CACHE_FILE.exists():
        return None, None
    try:
        blob = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("theoddsapi cache unreadable (%s); will refetch", e)
        return None, None
    payload = blob.get("payload")
    ts = blob.get("fetched_at_utc")
    if not isinstance(payload, list) or not isinstance(ts, str):
        return None, None
    try:
        fetched_at = datetime.fromisoformat(ts)
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    except ValueError:
        return None, None
    return payload, fetched_at


def _save_cache(payload: list[dict], fetched_at_utc: datetime) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    blob = {
        "fetched_at_utc": fetched_at_utc.isoformat(),
        "payload": payload,
    }
    _CACHE_FILE.write_text(
        json.dumps(blob, separators=(",", ":")),
        encoding="utf-8",
    )


def _is_stale(fetched_at_utc: Optional[datetime]) -> bool:
    """True if we've crossed a window boundary since the last fetch."""
    if fetched_at_utc is None:
        return True
    now_et = _now_et()
    window = _most_recent_window_et(now_et)
    # Convert our stored UTC timestamp to the same tz as `window` for
    # a like-for-like comparison.
    try:
        fetched_local = fetched_at_utc.astimezone(window.tzinfo)
    except Exception:  # noqa: BLE001
        fetched_local = fetched_at_utc
    return fetched_local < window


# ---- HTTP + parse -----------------------------------------------------

def _http_get(url: str, timeout: float = 15.0) -> str:
    """Plain urllib GET — no cookies, no anti-bot, the-odds-api is a
    normal REST API."""
    req = request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "bbp-odds/1.0"},
    )
    with request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def _decimal_to_american(dec: float) -> Optional[int]:
    try:
        d = float(dec)
    except (TypeError, ValueError):
        return None
    if d <= 1.0:
        return None
    if d >= 2.0:
        return int(round((d - 1.0) * 100))
    return int(round(-100.0 / (d - 1.0)))


def _fetch_live(api_key: str, timeout: float = 15.0) -> list[dict]:
    """One round-trip to the-odds-api. Raises on non-200."""
    params = dict(_PARAMS)
    params["apiKey"] = api_key
    url = f"{_BASE_URL}?{parse.urlencode(params)}"
    # Don't log the full URL — it contains the key.
    log.warning(
        "theoddsapi: fetching %s (markets=%s, bookmakers=%s)",
        _BASE_URL, params["markets"], params["bookmakers"],
    )
    t0 = time.monotonic()
    try:
        body = _http_get(url, timeout=timeout)
    except error.HTTPError as e:
        # Surface the response body for easier diagnosis (rate-limit, bad key, etc)
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:  # noqa: BLE001
            err_body = ""
        raise RuntimeError(
            f"theoddsapi HTTP {e.code}: {e.reason} — {err_body}"
        ) from e
    dur = time.monotonic() - t0
    data = json.loads(body)
    if not isinstance(data, list):
        raise RuntimeError(
            f"theoddsapi: unexpected payload type {type(data).__name__}"
        )
    log.warning("theoddsapi: got %d events in %.2fs", len(data), dur)
    return data


def _parse_dk_snapshots(payload: list[dict]) -> list[OddsSnapshot]:
    """Turn the the-odds-api payload into DraftKings OddsSnapshots.

    One event in the payload becomes zero or one snapshot — we skip any
    event where DK isn't posted or team names don't normalize.
    """
    polled_at = datetime.now(timezone.utc)
    out: list[OddsSnapshot] = []
    for ev in payload:
        if not isinstance(ev, dict):
            continue
        home_raw = ev.get("home_team") or ""
        away_raw = ev.get("away_team") or ""
        home = try_normalize_team(home_raw)
        away = try_normalize_team(away_raw)
        if not home or not away:
            log.debug(
                "theoddsapi: unknown teams away=%r home=%r",
                away_raw, home_raw,
            )
            continue

        commence = ev.get("commence_time") or ""
        try:
            start_utc = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            if start_utc.tzinfo is None:
                start_utc = start_utc.replace(tzinfo=timezone.utc)
            start_utc = start_utc.astimezone(timezone.utc)
        except ValueError:
            start_utc = datetime.now(timezone.utc)

        dk_book = None
        for bk in ev.get("bookmakers") or []:
            if (bk.get("key") or "").lower() == "draftkings":
                dk_book = bk
                break
        if dk_book is None:
            continue

        home_ml = away_ml = None
        home_rl_line = home_rl_odds = away_rl_odds = None
        total_line = over_odds = under_odds = None

        # Totals: multiple point buckets are possible; pick the one whose
        # juice is closest to -110/-110 (the "main" number).
        totals_by_point: dict[float, dict[str, int]] = {}

        for mkt in dk_book.get("markets") or []:
            mkey = (mkt.get("key") or "").lower()
            outcomes = mkt.get("outcomes") or []

            if mkey == "h2h":
                for oc in outcomes:
                    name = try_normalize_team(oc.get("name") or "")
                    am = _decimal_to_american(oc.get("price"))
                    if am is None:
                        continue
                    if name == home:
                        home_ml = am
                    elif name == away:
                        away_ml = am

            elif mkey == "spreads":
                for oc in outcomes:
                    name = try_normalize_team(oc.get("name") or "")
                    point = oc.get("point")
                    try:
                        pt = float(point) if point is not None else None
                    except (TypeError, ValueError):
                        pt = None
                    if pt is None or abs(abs(pt) - 1.5) > 0.01:
                        continue
                    am = _decimal_to_american(oc.get("price"))
                    if am is None:
                        continue
                    if name == home:
                        home_rl_line = pt
                        home_rl_odds = am
                    elif name == away:
                        away_rl_odds = am

            elif mkey == "totals":
                for oc in outcomes:
                    side = (oc.get("name") or "").strip().lower()
                    if side not in ("over", "under"):
                        continue
                    point = oc.get("point")
                    try:
                        pt = float(point) if point is not None else None
                    except (TypeError, ValueError):
                        pt = None
                    if pt is None:
                        continue
                    am = _decimal_to_american(oc.get("price"))
                    if am is None:
                        continue
                    totals_by_point.setdefault(pt, {})[side] = am

        if totals_by_point:
            best = None
            best_score = 1e9
            for pt, sides in totals_by_point.items():
                if "over" not in sides or "under" not in sides:
                    continue
                score = abs(sides["over"] + 110) + abs(sides["under"] + 110)
                if score < best_score:
                    best_score = score
                    best = pt
            if best is not None:
                total_line = best
                over_odds = totals_by_point[best]["over"]
                under_odds = totals_by_point[best]["under"]

        event_id = make_event_id(start_utc, away, home)
        out.append(OddsSnapshot(
            book=OddsBook.DRAFTKINGS,
            event_id=event_id,
            home_team=home,
            away_team=away,
            game_time_utc=start_utc,
            home_ml=home_ml,
            away_ml=away_ml,
            home_rl_line=home_rl_line,
            home_rl_odds=home_rl_odds,
            away_rl_odds=away_rl_odds,
            total_line=total_line,
            over_odds=over_odds,
            under_odds=under_odds,
            polled_at_utc=polled_at,
            native_event_id=str(ev.get("id") or ""),
        ))
    return out


# ---- public entry point -----------------------------------------------

def fetch_dk_via_theoddsapi_cached(force: bool = False) -> list[OddsSnapshot]:
    """Return DK OddsSnapshots, using the-odds-api with a 5-window cache.

    - If `force=True`, always hits the API regardless of cache freshness.
    - If no API key is configured, returns [] (caller falls back).
    - If the API call fails, falls back to the stale cache when possible,
      else returns [] (caller falls back).
    """
    global _LAST_FETCH_AT_UTC

    api_key = _get_api_key()
    if not api_key:
        log.info("theoddsapi: ODDS_API_KEY not set — skipping")
        return []

    # Use in-memory marker first, then disk cache — the-odds-api response
    # isn't small and re-parsing is cheap, but re-reading disk every 10
    # minutes is wasteful so we cache the parsed result lazily.
    payload, fetched_at = _load_cache()

    need_refresh = force or _is_stale(fetched_at)
    if need_refresh:
        try:
            payload = _fetch_live(api_key)
            fetched_at = datetime.now(timezone.utc)
            _save_cache(payload, fetched_at)
            _LAST_FETCH_AT_UTC = fetched_at
        except Exception as e:  # noqa: BLE001
            log.warning("theoddsapi: live fetch failed (%s)", e)
            if payload is None:
                return []
            log.warning(
                "theoddsapi: using stale cache from %s",
                fetched_at.isoformat() if fetched_at else "?",
            )
    else:
        log.info(
            "theoddsapi: using cached payload from %s (next window boundary pending)",
            fetched_at.isoformat() if fetched_at else "?",
        )

    if not payload:
        return []
    return _parse_dk_snapshots(payload)


# Alias for symmetry with the other scrapers — importers can write
#     from data.odds_theoddsapi import fetch_theoddsapi_dk_snapshots
def fetch_theoddsapi_dk_snapshots() -> list[OddsSnapshot]:
    return fetch_dk_via_theoddsapi_cached()
