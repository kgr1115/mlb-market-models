"""
Parse Sportsbook Reviews Online MLB odds XLSX.

Each season's file has one row per team per game (two rows per game,
Visitor then Home). Columns:

    Date Rot VH Team Pitcher
    1st..9th Final
    Open Close RunLine RLprice OpenOU OpenOUprice CloseOU CloseOUprice

Date is encoded as an integer like 401 (April 01) or 1001 (October 01);
year is implicit. Team is a 3-letter abbreviation. Prices are American.

We merge the two rows per game into one HistoricalOdds record keyed on
our canonical event_id (so it joins directly against HistoricalGame).
"""
from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import openpyxl

from data.odds_models import make_event_id
from data.team_names import FG_ABBR_TO_CANONICAL

log = logging.getLogger(__name__)

_SBR_URL_TEMPLATE = (
    "https://www.sportsbookreviewsonline.com/wp-content/uploads/"
    "sportsbookreviewsonline_com_737/mlb-odds-{season}.xlsx"
)
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# SBR uses its own abbreviations that don't always match our FG map.
# Override the ones that differ.
_SBR_ABBR_OVERRIDES = {
    "CUB": "Chicago Cubs",
    "CWS": "Chicago White Sox",
    "CHW": "Chicago White Sox",
    "KAN": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LOS": "Los Angeles Dodgers",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "SDG": "San Diego Padres",
    "SFO": "San Francisco Giants",
    "STL": "St. Louis Cardinals",
    "TAM": "Tampa Bay Rays",
    "WAS": "Washington Nationals",
    "ARI": "Arizona Diamondbacks",
}


def _sbr_team_to_canonical(abbr: str) -> Optional[str]:
    if not abbr:
        return None
    key = abbr.strip().upper()
    if key in _SBR_ABBR_OVERRIDES:
        return _SBR_ABBR_OVERRIDES[key]
    return FG_ABBR_TO_CANONICAL.get(key)


def _decode_sbr_date(code, season: int) -> Optional[str]:
    """Decode an SBR date code like 401 or 1001 into YYYY-MM-DD."""
    try:
        n = int(code)
    except (ValueError, TypeError):
        return None
    if n <= 0:
        return None
    # 3-digit: Mdd, 4-digit: MMdd
    s = str(n)
    if len(s) == 3:
        month = int(s[0])
        day = int(s[1:])
    elif len(s) == 4:
        month = int(s[:2])
        day = int(s[2:])
    else:
        return None
    # MLB seasons often start in March (spring training ended). Also
    # season files typically run through the playoffs (Oct/Nov). If
    # month is small and the date is late in the file, bump to next year.
    year = season
    # Trust the season for all calendar months in year.
    try:
        return f"{year:04d}-{month:02d}-{day:02d}"
    except ValueError:
        return None


@dataclass
class HistoricalOdds:
    event_id: str
    game_date: str            # YYYY-MM-DD

    away_team: str            # canonical
    home_team: str

    # Moneyline (American)
    away_ml_close: Optional[int] = None
    home_ml_close: Optional[int] = None
    away_ml_open: Optional[int] = None
    home_ml_open: Optional[int] = None

    # Run line (closing only)
    away_rl_line: Optional[float] = None
    away_rl_price: Optional[int] = None
    home_rl_line: Optional[float] = None
    home_rl_price: Optional[int] = None

    # Totals (closing)
    total_close: Optional[float] = None
    total_over_close: Optional[int] = None
    total_under_close: Optional[int] = None

    # Totals (opening — populated from community dataset; SBR 2018-19 has no opens)
    total_open: Optional[float] = None
    total_over_open: Optional[int] = None
    total_under_open: Optional[int] = None


# Monkey patch: .pyc cache contains old dataclass without total_open* fields.
# Patch __init__ to accept them at runtime.
_historical_odds_original_init = HistoricalOdds.__init__

def _historical_odds_patched_init(self, event_id, game_date, away_team, home_team,
                                  away_ml_close=None, home_ml_close=None,
                                  away_ml_open=None, home_ml_open=None,
                                  away_rl_line=None, away_rl_price=None,
                                  home_rl_line=None, home_rl_price=None,
                                  total_close=None, total_over_close=None, total_under_close=None,
                                  total_open=None, total_over_open=None, total_under_open=None):
    _historical_odds_original_init(self, event_id, game_date, away_team, home_team,
                                   away_ml_close, home_ml_close,
                                   away_ml_open, home_ml_open,
                                   away_rl_line, away_rl_price,
                                   home_rl_line, home_rl_price,
                                   total_close, total_over_close, total_under_close)
    object.__setattr__(self, 'total_open', total_open)
    object.__setattr__(self, 'total_over_open', total_over_open)
    object.__setattr__(self, 'total_under_open', total_under_open)

HistoricalOdds.__init__ = _historical_odds_patched_init


def _download_sbr(season: int, path: str) -> bool:
    url = _SBR_URL_TEMPLATE.format(season=season)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        log.warning("SBR download failed for %d: %s", season, e)
        return False
    with open(path, "wb") as f:
        f.write(data)
    return True


def load_sbr_season_odds(season: int,
                         local_cache_dir: str = "/tmp/bbp"
                         ) -> dict[str, HistoricalOdds]:
    """Return a dict event_id -> HistoricalOdds for the entire season."""
    os.makedirs(local_cache_dir, exist_ok=True)
    path = os.path.join(local_cache_dir, f"mlb-odds-{season}.xlsx")
    if not os.path.exists(path):
        if not _download_sbr(season, path):
            return {}

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = next(rows, None)
    if not header:
        return {}

    # Map header to column index for resilience.
    # Normalize headers so "Run Line" and "RunLine" map to the same key.
    # SBR uses spaces in headers through 2019 and removed them starting 2020/2021.
    _ALIASES = {
        "Run Line": "RunLine",
        "Open OU":  "OpenOU",
        "Close OU": "CloseOU",
    }
    col: dict[str, int] = {}
    for i, h in enumerate(header):
        if not h:
            continue
        key = _ALIASES.get(h, h)
        col[key] = i
    need = ("Date", "VH", "Team", "Final", "Open", "Close",
            "RunLine", "OpenOU", "CloseOU")
    missing = [c for c in need if c not in col]
    if missing:
        log.warning("SBR %d: missing expected columns %s", season, missing)

    out: dict[str, HistoricalOdds] = {}
    pending = None  # the visitor row, waiting for the home row

    for r in rows:
        if not r or r[col.get("Date", 0)] is None:
            continue
        date_code = r[col["Date"]]
        vh = (r[col["VH"]] or "").strip().upper()
        team_abbr = (r[col["Team"]] or "").strip().upper()
        team = _sbr_team_to_canonical(team_abbr)
        if not team or vh not in ("V", "H"):
            continue

        def _i(c_name):
            v = r[col[c_name]] if c_name in col else None
            try:
                return int(v) if v is not None and v != "" else None
            except (TypeError, ValueError):
                return None

        def _f(c_name):
            v = r[col[c_name]] if c_name in col else None
            try:
                return float(v) if v is not None and v != "" else None
            except (TypeError, ValueError):
                return None

        ml_open = _i("Open")
        ml_close = _i("Close")
        rl_line = _f("RunLine")
        # The RL price is in the column immediately after RunLine
        rl_price_idx = col["RunLine"] + 1 if "RunLine" in col else None
        rl_price = None
        if rl_price_idx is not None and rl_price_idx < len(r):
            try:
                rl_price = int(r[rl_price_idx]) if r[rl_price_idx] is not None else None
            except (TypeError, ValueError):
                rl_price = None

        # Totals closing
        tot_close = _f("CloseOU")
        tot_price_idx = col["CloseOU"] + 1 if "CloseOU" in col else None
        tot_price = None
        if tot_price_idx is not None and tot_price_idx < len(r):
            try:
                tot_price = int(r[tot_price_idx]) if r[tot_price_idx] is not None else None
            except (TypeError, ValueError):
                tot_price = None

        row = {
            "date_code": date_code, "team": team, "vh": vh,
            "ml_open": ml_open, "ml_close": ml_close,
            "rl_line": rl_line, "rl_price": rl_price,
            "tot_close": tot_close, "tot_price": tot_price,
        }

        if vh == "V":
            pending = row
        elif vh == "H" and pending is not None:
            # Merge; make sure dates match
            if pending["date_code"] != row["date_code"]:
                pending = row
                continue

            date_str = _decode_sbr_date(row["date_code"], season)
            if not date_str:
                pending = None
                continue

            # Build a game_time_utc - SBR doesn't have time, so use noon UTC
            # on that date. The event_id join accepts either; make_event_id
            try:
                game_time = datetime.fromisoformat(
                    f"{date_str}T12:00:00+00:00"
                ).astimezone(timezone.utc)
            except ValueError:
                pending = None
                continue

            ev = make_event_id(game_time, pending["team"], row["team"])

            tot_over = tot_under = None
            if row["tot_price"] is not None:
                tot_over = row["tot_price"]
                tot_under = row["tot_price"]

            out[ev] = HistoricalOdds(
                event_id=ev, game_date=date_str,
                away_team=pending["team"], home_team=row["team"],
                away_ml_open=pending["ml_open"],
                home_ml_open=row["ml_open"],
                away_ml_close=pending["ml_close"],
                home_ml_close=row["ml_close"],
                away_rl_line=pending["rl_line"],
                away_rl_price=pending["rl_price"],
                home_rl_line=row["rl_line"],
                home_rl_price=row["rl_price"],
                total_close=row["tot_close"],
                total_over_close=tot_over,
                total_under_close=tot_under,
            )
            pending = None

    log.info("SBR %d: loaded %d games with closing lines", season, len(out))
    return out
