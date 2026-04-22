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

from .odds_models import OddsBook, OddsSnapshot, make_event_id, pick_main_run_line
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

# FanDuel serves many MLB markets under the same marketType strings above —
# "1st 5 Innings Money Line", "Race to 3 Runs", "Innings 1-3 Total", team
# totals, derivative/period markets, etc. These are NOT the main full-game
# lines but they carry marketType=MONEY_LINE / MATCH_HANDICAP / TOTAL_POINTS.
# If we don't filter them out, the parse loop overwrites the main line with
# whatever derivative market happens to be processed last — yielding nonsense
# like a HOU ML of -100000 or an LAD/SF total of 9.5 when the real main is
# 7.5. Any market whose name contains one of these substrings is excluded.
_NON_MAIN_MARKET_SUBSTRINGS = (
    "alt",            # alternate lines (both total and RL)
    "1st",            # 1st 5 Innings, 1st Inning, etc.
    "first",          # First 5 Innings, First Inning, First Team to Score
    "inning",         # Innings 1-3, Innings 1-5, Inning-specific markets
    "race to",        # Race to 3 Runs, Race to 5 Runs
    "team total",     # Team Total Runs markets
    "team to",        # First Team to Score, Last Team to Score
    "both teams",     # Both Teams to Score
    "no runs",        # No Runs in the 1st
    "odd/even",       # Odd/Even Total Runs
    "exactly",        # Exact Runs
    "highest",        # Highest Scoring Inning/Half
    "to win series",  # Series winner futures
    "series",         # Series-level markets
    "margin",         # Winning Margin
    "shutout",        # Team to Record a Shutout
    "f5",             # explicit F5 labeling
)


def _is_non_main_market(mname: str) -> bool:
    m = (mname or "").lower()
    return any(sub in m for sub in _NON_MAIN_MARKET_SUBSTRINGS)


# FanDuel events can be in-play, just-settled, or scheduled. Our app wants
# pre-game main-line prices only — the live/settled ones produce garbage like
# a team shown at -100000 ML in the top of the 9th or an Over 8.5 at +270
# because the real total has already been crushed. The FD payload exposes
# state via a few different fields depending on the response shape; this
# helper returns True if ANY of them say the event isn't a clean pre-game.
def _event_is_prematch(ev: dict) -> bool:
    # Explicit boolean — FD serves this on most event objects.
    if ev.get("inPlay") is True:
        return False
    # String state fields. We whitelist only pre-match values; anything else
    # (STARTED, LIVE, FINISHED, SETTLED, CLOSED, SUSPENDED) is rejected.
    for key in ("eventState", "state", "status", "eventStatus"):
        v = ev.get(key)
        if v is None:
            continue
        s = str(v).strip().upper()
        if s in {"PREMATCH", "PRE_MATCH", "PRE-MATCH", "SCHEDULED",
                 "NOT_STARTED", "UPCOMING", "OPEN", ""}:
            continue
        # Anything else (LIVE, STARTED, FINISHED, SETTLED, SUSPENDED) rejects.
        return False
    # openDate in the future is also a good signal, but not all events carry
    # a trustworthy one here — don't gate on it, just fall through.
    return True


# Market-level status check. Even for a pre-match event, individual markets
# can be CLOSED / SUSPENDED, in which case their odds are stale/garbage.
def _market_is_active(m: dict) -> bool:
    for key in ("marketStatus", "status", "state"):
        v = m.get(key)
        if v is None:
            continue
        s = str(v).strip().upper()
        if s not in {"ACTIVE", "OPEN", "TRADING", "AVAILABLE", ""}:
            return False
    return True


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

    # Pre-filter events: drop anything that isn't a clean pre-match. Live
    # and just-settled events produce garbage main-line prices (e.g. HOU
    # ML at -100000 in the top of the 9th). We filter at the event level
    # so every market tied to that event is excluded too.
    prematch_event_ids: set[str] = set()
    skipped_live = 0
    for ev_id, ev in events.items():
        if _event_is_prematch(ev):
            prematch_event_ids.add(str(ev.get("eventId") or ev_id))
        else:
            skipped_live += 1
    if skipped_live:
        log.info("FanDuel: skipped %d in-play/settled events", skipped_live)

    mk_by_event: dict[str, dict[str, list[dict]]] = {}
    for m in markets.values():
        ev_id = m.get("eventId")
        if ev_id is None:
            continue
        ev_str = str(ev_id)
        if ev_str not in prematch_event_ids:
            continue
        if not _market_is_active(m):
            continue
        mname = (m.get("marketName") or "").lower()
        if _is_non_main_market(mname):
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

        # FanDuel sometimes publishes both the main Run Line (fav -1.5 /
        # dog +1.5) and a "reverse" RL (fav +1.5 / dog -1.5) under the
        # same MATCH_HANDICAP market type. The old loop took the last
        # runner and could latch onto the reverse, inverting the UI.
        # Collect every ±1.5 candidate per side, then pick the main.
        home_rl_cands: list[tuple[float, int]] = []
        away_rl_cands: list[tuple[float, int]] = []
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
                home_rl_cands.append((h, price))
            else:
                away_rl_cands.append((h, price))
        home_rl_line, home_rl_odds, away_rl_odds = pick_main_run_line(
            home_rl_cands, away_rl_cands, home_ml, away_ml,
        )

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
