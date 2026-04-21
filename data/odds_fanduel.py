"""
FanDuel odds scraper.

FanDuel Sportsbook serves its own web UI from a per-state subdomain:

    https://sbapi.{state}.sportsbook.fanduel.com/api/

The endpoint we hit is:

    GET /api/content-managed-page
        ?page=SPORT&eventTypeId=7511
        &pbHorizontal=false&_ak=FhMFpcPWXMeyZxOx
        &timezone=America/New_York

We walk markets of type MONEY_LINE, MATCH_HANDICAP_(2-WAY) (run line),
and TOTAL_POINTS_(OVER/UNDER), pick the standard line only (no
alternates), and emit one OddsSnapshot per event.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional
from urllib import error, request

from .odds_models import OddsBook, OddsSnapshot, make_event_id
from .team_names import try_normalize_team

log = logging.getLogger(__name__)

_AFFILIATE_KEY = "FhMFpcPWXMeyZxOx"

_DEFAULT_STATE = os.environ.get("BBP_FANDUEL_STATE", "nj").lower()
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://sportsbook.fanduel.com/",
    "Origin": "https://sportsbook.fanduel.com",
}

_STATES_TO_TRY = [_DEFAULT_STATE, "nj", "pa", "ny", "mi", "co", "va", "il", "in", "oh"]

_MLB_EVENT_TYPE_ID = 7511


def _url_for(state: str) -> str:
    return (
        f"https://sbapi.{state}.sportsbook.fanduel.com/api/content-managed-page"
        f"?page=SPORT&eventTypeId={_MLB_EVENT_TYPE_ID}"
        f"&pbHorizontal=false&_ak={_AFFILIATE_KEY}"
        f"&timezone=America/New_York"
    )


def _http_get_json(url: str, timeout: float = 10.0) -> Any:
    req = request.Request(url, headers=_HEADERS)
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _try_fetch() -> Optional[dict]:
    seen: set[str] = set()
    for st in _STATES_TO_TRY:
        if st in seen:
            continue
        seen.add(st)
        try:
            return _http_get_json(_url_for(st))
        except (error.URLError, error.HTTPError, TimeoutError,
                json.JSONDecodeError) as e:
            log.debug("FanDuel fetch for state=%s failed: %s", st, e)
            continue
    log.warning("FanDuel: no state subdomain returned a usable payload")
    return None


def _parse_utc(s: str) -> datetime:
    s = s.rstrip("Z")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _american_from_runner(runner: dict) -> Optional[int]:
    o = runner.get("winRunnerOdds") or {}
    ado = o.get("americanDisplayOdds") or {}
    ai = ado.get("americanOddsInt")
    if ai is None:
        ai = ado.get("americanOdds")
    if ai is not None:
        try:
            return int(ai)
        except (TypeError, ValueError):
            pass
    true_odds = o.get("trueOdds") or {}
    dec = (true_odds.get("decimalOdds") or {}).get("decimalOdds")
    if dec is not None:
        try:
            dec_f = float(dec)
            if dec_f <= 1.0:
                return 0
            if dec_f >= 2.0:
                return int(round((dec_f - 1.0) * 100))
            return int(round(-100.0 / (dec_f - 1.0)))
        except (TypeError, ValueError):
            pass
    return None


def _runner_handicap(runner: dict) -> Optional[float]:
    h = runner.get("handicap")
    if h is None:
        h = runner.get("line")
    if h is None:
        return None
    try:
        return float(h)
    except (TypeError, ValueError):
        return None


def _index_attachments(payload: dict) -> tuple[dict, dict, dict]:
    att = payload.get("attachments") or {}
    return att.get("events") or {}, att.get("markets") or {}, att.get("runners") or {}


def _market_type(m: dict) -> str:
    return (m.get("marketType") or "").upper()


_ML_TYPES = ("MONEY_LINE", "MATCH_WINNER")
_RL_TYPES = ("MATCH_HANDICAP_(2-WAY)", "SPREAD_BETTING", "HANDICAP", "MATCH_HANDICAP")
_TOTAL_TYPES = ("TOTAL_POINTS_(OVER/UNDER)", "TOTAL_POINTS", "MATCH_TOTAL", "TOTAL",
                "TOTAL_RUNS_(OVER/UNDER)")


def _strip_pitcher_annotation(team_raw: str) -> str:
    i = team_raw.find(" (")
    return team_raw[:i].strip() if i >= 0 else team_raw.strip()


def _runner_side(runner: dict, home: str, away: str) -> Optional[str]:
    name = (runner.get("runnerName") or "").strip()
    low = name.lower()
    if low.startswith("over") or low == "o":
        return "over"
    if low.startswith("under") or low == "u":
        return "under"
    norm = try_normalize_team(name)
    if norm == home:
        return "home"
    if norm == away:
        return "away"
    return None


def fetch_fanduel_snapshots() -> list[OddsSnapshot]:
    payload = _try_fetch()
    if not payload:
        return []

    events, markets, runners = _index_attachments(payload)
    # Note: FD inlines runners inside each market.runners list, so
    # attachments.runners is often empty — don't gate on it.
    if not events or not markets:
        return []

    mk_by_event: dict[str, dict[str, list[dict]]] = {}
    for m in markets.values():
        ev_id = m.get("eventId")
        if ev_id is None:
            continue
        mname = (m.get("marketName") or "").lower()
        if "alt" in mname:
            continue
        mtype = _market_type(m)
        if mtype in _ML_TYPES:
            kind = "ml"
        elif mtype in _RL_TYPES:
            kind = "rl"
        elif mtype in _TOTAL_TYPES:
            kind = "total"
        else:
            continue
        ev_str = str(ev_id)
        mk_by_event.setdefault(ev_str, {}).setdefault(kind, []).append(m)

    polled_at = datetime.now(timezone.utc)
    out: list[OddsSnapshot] = []

    for ev_id, ev in events.items():
        ev_id_s = str(ev.get("eventId") or ev_id)
        name = ev.get("name") or ""
        if " @ " not in name:
            continue
        away_raw, home_raw = name.split(" @ ", 1)
        away_raw = _strip_pitcher_annotation(away_raw)
        home_raw = _strip_pitcher_annotation(home_raw)
        home = try_normalize_team(home_raw)
        away = try_normalize_team(away_raw)
        if not home or not away:
            log.debug("FD: unknown teams %r / %r", away_raw, home_raw)
            continue

        open_date = ev.get("openDate") or ev.get("startTime") or ""
        try:
            start_utc = _parse_utc(open_date)
        except Exception:
            start_utc = datetime.now(timezone.utc)

        event_id = make_event_id(start_utc, away, home)
        mks = mk_by_event.get(ev_id_s, {})

        def _runners_of(kind: str) -> list[dict]:
            rs: list[dict] = []
            for m in mks.get(kind, []):
                for rid in (m.get("runners") or []):
                    if isinstance(rid, dict):
                        rs.append(rid)
                    else:
                        rd = runners.get(str(rid)) or runners.get(rid)
                        if rd is not None:
                            rs.append(rd)
            return rs

        home_ml = away_ml = None
        for r in _runners_of("ml"):
            side = _runner_side(r, home, away)
            if side == "home":
                home_ml = _american_from_runner(r) or home_ml
            elif side == "away":
                away_ml = _american_from_runner(r) or away_ml

        home_rl_line = home_rl_odds = away_rl_odds = None
        for r in _runners_of("rl"):
            side = _runner_side(r, home, away)
            if side not in ("home", "away"):
                continue
            h = _runner_handicap(r)
            if h is None or abs(abs(h) - 1.5) > 0.01:
                continue
            price = _american_from_runner(r)
            if price is None:
                continue
            if side == "home":
                home_rl_line = h
                home_rl_odds = price
            else:
                away_rl_odds = price

        total_line = over_odds = under_odds = None
        by_line: dict[float, dict[str, int]] = {}
        for r in _runners_of("total"):
            side = _runner_side(r, home, away)
            if side not in ("over", "under"):
                continue
            h = _runner_handicap(r)
            if h is None:
                continue
            price = _american_from_runner(r)
            if price is None:
                continue
            by_line.setdefault(float(h), {})[side] = price
        best = None
        best_score = 1e9
        for l, sides in by_line.items():
            if "over" not in sides or "under" not in sides:
                continue
            score = abs(sides["over"] + 110) + abs(sides["under"] + 110)
            if score < best_score:
                best_score = score
                best = l
        if best is not None:
            total_line = best
            over_odds = by_line[best]["over"]
            under_odds = by_line[best]["under"]

        out.append(OddsSnapshot(
            book=OddsBook.FANDUEL,
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
            native_event_id=ev_id_s,
        ))

    return out
