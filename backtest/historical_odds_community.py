"""
Community-dataset MLB odds loader.

Source: ArnavSaraogi/mlb-odds-scraper GitHub release "dataset" (80 MB JSON),
scraped from SportsbookReview.com with per-book opening/closing lines for
moneyline, pointspread (run line), and totals. Covers 2021-03-20 to
2025-08-16 - extends our corpus beyond SBR's 2021 cutoff.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from data.odds_models import make_event_id
from data.team_names import try_normalize_team

from .historical_odds import HistoricalOdds

# Patch HistoricalOdds to accept total_open* fields (stale .pyc cache issue)
_ho_orig_init = HistoricalOdds.__init__
def _ho_patched_init(self, event_id, game_date, away_team, home_team,
                     away_ml_close=None, home_ml_close=None,
                     away_ml_open=None, home_ml_open=None,
                     away_rl_line=None, away_rl_price=None,
                     home_rl_line=None, home_rl_price=None,
                     total_close=None, total_over_close=None, total_under_close=None,
                     total_open=None, total_over_open=None, total_under_open=None):
    _ho_orig_init(self, event_id, game_date, away_team, home_team,
                  away_ml_close, home_ml_close, away_ml_open, home_ml_open,
                  away_rl_line, away_rl_price, home_rl_line, home_rl_price,
                  total_close, total_over_close, total_under_close)
    object.__setattr__(self, 'total_open', total_open)
    object.__setattr__(self, 'total_over_open', total_over_open)
    object.__setattr__(self, 'total_under_open', total_under_open)
HistoricalOdds.__init__ = _ho_patched_init

log = logging.getLogger(__name__)


_DATASET_URL = (
    "https://github.com/ArnavSaraogi/mlb-odds-scraper/releases/"
    "download/dataset/mlb_odds_dataset.json"
)
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_BOOK_PREFERENCE = ("draftkings", "fanduel", "bet365", "caesars",
                    "betmgm", "bet_rivers_ny")


def _download_dataset(path: str) -> bool:
    req = urllib.request.Request(_DATASET_URL, headers={"User-Agent": _USER_AGENT})
    try:
        log.info("community odds: downloading %s -> %s", _DATASET_URL, path)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
    except Exception as e:
        log.warning("community odds: download failed: %s", e)
        return False
    with open(path, "wb") as f:
        f.write(data)
    return True


def _pick_preferred(market_rows):
    if not market_rows:
        return None
    by_book = {r.get("sportsbook"): r for r in market_rows if r}
    for book in _BOOK_PREFERENCE:
        if book in by_book:
            return by_book[book]
    return market_rows[0]


def _closing(row, keys):
    if not row:
        return tuple(None for _ in keys)
    curr = row.get("currentLine") or {}
    return tuple(curr.get(k) for k in keys)


def _opening(row, keys):
    if not row:
        return tuple(None for _ in keys)
    opn = row.get("openingLine") or {}
    return tuple(opn.get(k) for k in keys)


_DATA_CACHE: Optional[dict] = None


def _load_full_dataset(local_cache_dir: str) -> dict:
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE
    os.makedirs(local_cache_dir, exist_ok=True)
    path = os.path.join(local_cache_dir, "mlb_odds_dataset.json")
    if not os.path.exists(path):
        if not _download_dataset(path):
            return {}
    with open(path, "r") as f:
        _DATA_CACHE = json.load(f)
    return _DATA_CACHE


def _default_cache_dir() -> str:
    # Project-local cache/ folder — cross-platform; prior default `/tmp/bbp`
    # didn't exist on Windows.
    return os.path.join(os.getcwd(), "cache")


def load_community_season_odds(season: int,
                               local_cache_dir: Optional[str] = None,
                               ) -> dict:
    if local_cache_dir is None:
        local_cache_dir = _default_cache_dir()
    full = _load_full_dataset(local_cache_dir)
    if not full:
        return {}

    out = {}
    n_rows = n_missing_book = n_missing_team = n_nonreg = 0

    for date_key, games in full.items():
        if not date_key.startswith(str(season)):
            continue
        for g in games:
            n_rows += 1
            gv = g.get("gameView") or {}
            if gv.get("gameType") != "R":
                n_nonreg += 1
                continue
            start_iso = gv.get("startDate")
            if not start_iso:
                continue
            try:
                if start_iso.endswith("Z"):
                    start_iso = start_iso[:-1] + "+00:00"
                game_time = datetime.fromisoformat(start_iso).astimezone(timezone.utc)
            except Exception:
                continue

            away_raw = (gv.get("awayTeam") or {}).get("fullName") or ""
            home_raw = (gv.get("homeTeam") or {}).get("fullName") or ""
            away = try_normalize_team(away_raw)
            home = try_normalize_team(home_raw)
            if not home or not away:
                n_missing_team += 1
                continue

            ev = make_event_id(game_time, away, home)
            game_date = game_time.strftime("%Y-%m-%d")

            odds = g.get("odds") or {}
            ml_row = _pick_preferred(odds.get("moneyline"))
            rl_row = _pick_preferred(odds.get("pointspread"))
            tot_row = _pick_preferred(odds.get("totals"))
            if ml_row is None and rl_row is None and tot_row is None:
                n_missing_book += 1
                continue

            (ml_home_close, ml_away_close) = _closing(ml_row, ("homeOdds", "awayOdds"))
            (ml_home_open, ml_away_open) = _opening(ml_row, ("homeOdds", "awayOdds"))

            (rl_home_line, rl_away_line, rl_home_odds, rl_away_odds) = _closing(
                rl_row, ("homeSpread", "awaySpread", "homeOdds", "awayOdds")
            )

            (tot_line, tot_over, tot_under) = _closing(
                tot_row, ("total", "overOdds", "underOdds")
            )
            (tot_line_open, tot_over_open, tot_under_open) = _opening(
                tot_row, ("total", "overOdds", "underOdds")
            )

            out[ev] = HistoricalOdds(
                event_id=ev, game_date=game_date,
                away_team=away, home_team=home,
                away_ml_close=_to_int(ml_away_close),
                home_ml_close=_to_int(ml_home_close),
                away_ml_open=_to_int(ml_away_open),
                home_ml_open=_to_int(ml_home_open),
                away_rl_line=_to_float(rl_away_line),
                away_rl_price=_to_int(rl_away_odds),
                home_rl_line=_to_float(rl_home_line),
                home_rl_price=_to_int(rl_home_odds),
                total_close=_to_float(tot_line),
                total_over_close=_to_int(tot_over),
                total_under_close=_to_int(tot_under),
                total_open=_to_float(tot_line_open),
                total_over_open=_to_int(tot_over_open),
                total_under_open=_to_int(tot_under_open),
            )

    log.info("community odds %d: %d games loaded (scanned %d rows; %d non-reg, "
             "%d missing team, %d missing book)",
             season, len(out), n_rows, n_nonreg, n_missing_team, n_missing_book)
    return out


def _to_int(v):
    """Coerce to int. Reject 0 (dataset 'no data' sentinel — never a valid
    American odd or run line in MLB)."""
    if v is None or v == "":
        return None
    try:
        iv = int(v)
    except (TypeError, ValueError):
        return None
    return iv if iv != 0 else None


def _to_float(v):
    """Coerce to float. Reject 0 (dataset sentinel; no MLB total is 0)."""
    if v is None or v == "":
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return fv if fv != 0.0 else None
