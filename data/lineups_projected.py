"""
Projected lineup scraper — earlier than MLB posts official.

RotoWire publishes hand-curated projected lineups several hours before
teams officially announce. We scrape their daily-lineups page and return
TeamLineup objects tagged source=LINEUP_PROJECTED, which predict_all()
downweights slightly via input_certainty relative to LINEUP_CONFIRMED.

Contract:
    get_projected_lineups(date=None) -> dict[event_id, dict]
        returns {"home": TeamLineup, "away": TeamLineup}

If the page structure changes (which it will occasionally) or the
scraper can't reach RotoWire, the function returns {} — callers treat
that as "no projected lineups available" and fall back to implied
top-9-by-PA, so nothing breaks.

Note: RotoWire allows scraping for personal use but explicitly forbids
resale of scraped data. This feed is for the user's private predictor.
"""
from __future__ import annotations

import html
import logging
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from .lineups_models import (
    LineupSlot, LINEUP_PROJECTED, TeamLineup,
)
from .team_names import try_normalize_team

log = logging.getLogger(__name__)

_BASE_URL = "https://www.rotowire.com/baseball/daily-lineups.php"
_REQUEST_TIMEOUT = 12.0
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
    })
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            raw = resp.read()
            # RotoWire returns gzip sometimes
            if resp.info().get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            return raw.decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        log.warning("RotoWire fetch failed: %s", e)
        return ""


# --- HTML parsing helpers ----------------------------------------------------
# RotoWire's markup has per-game blocks that look roughly like:
#
#   <div class="lineup is-mlb">
#     <div class="lineup__abbr">NYY @ LAD  7:10 PM ET</div>
#     <div class="lineup__box">
#       <ul class="lineup__list is-visit">
#         <li class="lineup__player"><a ... title="Aaron Judge">CF Judge</a></li>
#         ...
#       </ul>
#       <ul class="lineup__list is-home">...</ul>
#     </div>
#     <div class="lineup__status">Projected</div>  OR "Confirmed"
#   </div>
#
# Their exact class names drift every few months, so we parse loosely with
# regex rather than a brittle DOM walk.

_GAME_BLOCK_RE = re.compile(
    r'<div[^>]*class="[^"]*\blineup\b[^"]*"[^>]*>.*?</div>\s*</div>',
    re.DOTALL | re.IGNORECASE,
)
_TEAM_ABBR_RE = re.compile(
    r'class="[^"]*lineup__team\b[^"]*(?:is-visit|is-home)[^"]*"[^>]*>'
    r'\s*<[^>]+>\s*([A-Z]{2,3})',
    re.IGNORECASE,
)
_VISIT_LIST_RE = re.compile(
    r'<ul[^>]*class="[^"]*lineup__list\b[^"]*\bis-visit\b[^"]*"[^>]*>(.*?)</ul>',
    re.DOTALL | re.IGNORECASE,
)
_HOME_LIST_RE = re.compile(
    r'<ul[^>]*class="[^"]*lineup__list\b[^"]*\bis-home\b[^"]*"[^>]*>(.*?)</ul>',
    re.DOTALL | re.IGNORECASE,
)
_PLAYER_RE = re.compile(
    r'<li[^>]*class="[^"]*lineup__player\b[^"]*"[^>]*>.*?'
    r'<a[^>]*href="([^"]+)"[^>]*>(?:\s*<span[^>]*>([A-Z0-9-]+)</span>)?\s*'
    r'([^<]+?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_STATUS_RE = re.compile(
    r'class="[^"]*lineup__status\b[^"]*"[^>]*>\s*([A-Za-z]+)', re.IGNORECASE,
)


def _fg_to_canonical(abbr: str) -> Optional[str]:
    """RotoWire abbrs are close to standard — try direct normalize first."""
    # RotoWire uses "CHC", "CWS", "LAD", etc. try_normalize_team handles many.
    return try_normalize_team(abbr)


def _parse_players(block_html: str) -> list[LineupSlot]:
    slots: list[LineupSlot] = []
    for i, m in enumerate(_PLAYER_RE.finditer(block_html), start=1):
        pos = (m.group(2) or "").strip()
        name = html.unescape(m.group(3)).strip()
        # Player ID: RotoWire uses slugs, not MLB IDs. Use the slug itself
        # as a placeholder — caller cross-references by name if needed.
        href = m.group(1) or ""
        pid = href.rsplit("-", 1)[-1].split(".")[0] if href else name
        slots.append(LineupSlot(
            order=i,
            player_id=f"rw:{pid}",
            name=name,
            position=pos,
            bats="R",  # RotoWire doesn't expose handedness in the list view
        ))
    return slots


def get_projected_lineups(date_yyyy_mm_dd: Optional[str] = None
                          ) -> dict[str, dict[str, TeamLineup]]:
    """Scrape RotoWire for projected lineups.

    Returns a dict keyed on a synthetic key "YYYY-MM-DD|AWAY|HOME"
    (NOT the canonical event_id, since RotoWire doesn't publish game
    times reliably on the list page). The lineups_aggregator joins
    these to real event_ids by matching (date, away, home).
    """
    url = _BASE_URL
    if date_yyyy_mm_dd:
        url = f"{_BASE_URL}?date={date_yyyy_mm_dd}"
    markup = _fetch_html(url)
    if not markup:
        return {}

    date_key = date_yyyy_mm_dd or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out: dict[str, dict[str, TeamLineup]] = {}

    blocks = _GAME_BLOCK_RE.findall(markup)
    log.info("RotoWire: %d game blocks found on page", len(blocks))
    # NOTE: RotoWire's CSS class names drift every few months. If this
    # function starts returning 0 lineups while MLB's schedule clearly
    # has games, re-inspect the page markup and update the regexes at
    # the top of this file. The MLB Stats API confirmed lineups are the
    # authoritative source, so this falling back to empty is safe — it
    # just means no pre-official lineup preview until MLB posts official.

    for block in blocks:
        abbrs = _TEAM_ABBR_RE.findall(block)
        if len(abbrs) < 2:
            continue
        visit = _fg_to_canonical(abbrs[0])
        home = _fg_to_canonical(abbrs[1])
        if not visit or not home:
            continue

        visit_list = _VISIT_LIST_RE.search(block)
        home_list = _HOME_LIST_RE.search(block)
        if not visit_list or not home_list:
            continue

        visit_slots = _parse_players(visit_list.group(1))
        home_slots = _parse_players(home_list.group(1))
        if not visit_slots or not home_slots:
            continue

        status_m = _STATUS_RE.search(block)
        status_word = (status_m.group(1).lower() if status_m else "projected")
        # We return everything as "projected" even when RotoWire labels it
        # "confirmed" — the MLB Stats API is the one source of truth for
        # confirmed lineups; this avoids two sources disagreeing.
        source_tag = LINEUP_PROJECTED
        _ = status_word

        key = f"{date_key}|{visit}|{home}"
        out[key] = {
            "away": TeamLineup(team=visit, source=source_tag, slots=visit_slots),
            "home": TeamLineup(team=home, source=source_tag, slots=home_slots),
        }

    log.info("RotoWire: parsed %d projected lineups", len(out))
    return out
