"""
Load every MLB regular-season game for a given season from the MLB Stats API.

Returns HistoricalGame objects keyed on our canonical event_id so we can
join against the SBR closing-lines feed. Cached to SQLite; re-runs skip
the network.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Optional

from data.odds_models import make_event_id
from data.team_names import try_normalize_team

log = logging.getLogger(__name__)

_SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?sportId=1&season={season}&gameType=R"
    "&hydrate=probablePitcher,linescore"
)
_USER_AGENT = "BBP-Backtest/0.1"
_REQUEST_TIMEOUT = 30.0


@dataclass
class HistoricalGame:
    event_id: str
    game_pk: int
    game_date: str          # "YYYY-MM-DD" (local game date)
    game_time_utc: datetime
    home_team: str          # canonical
    away_team: str
    home_runs: int
    away_runs: int
    home_starter: str = ""
    away_starter: str = ""
    status: str = "Final"


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _db_path() -> str:
    # Default path: project-local cache/ dir (cross-platform; /tmp doesn't
    # exist on Windows). Override via env var if you want it elsewhere.
    override = os.environ.get("BBP_BACKTEST_CACHE")
    if override:
        return override
    cache_dir = os.path.join(os.getcwd(), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "bbp_backtest.sqlite")


def _init_cache(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_games (
            event_id     TEXT PRIMARY KEY,
            game_pk      INTEGER,
            season       INTEGER,
            game_date    TEXT,
            game_time_utc TEXT,
            home_team    TEXT,
            away_team    TEXT,
            home_runs    INTEGER,
            away_runs    INTEGER,
            home_starter TEXT,
            away_starter TEXT,
            status       TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hg_season ON historical_games(season)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hg_date ON historical_games(game_date)")
    conn.commit()


def load_season_games(season: int, use_cache: bool = True
                      ) -> list[HistoricalGame]:
    """Return every regular-season game for `season` as HistoricalGame.

    Pulls from cache first; if empty, hits MLB Stats API for the whole
    season in one call (~6-8 MB JSON) and upserts rows.
    """
    db = sqlite3.connect(_db_path())
    _init_cache(db)

    if use_cache:
        rows = db.execute(
            "SELECT event_id, game_pk, game_date, game_time_utc, "
            "home_team, away_team, home_runs, away_runs, "
            "home_starter, away_starter, status "
            "FROM historical_games WHERE season=? "
            "AND status='Final' ORDER BY game_date",
            (season,),
        ).fetchall()
        if rows:
            log.info("historical_games: loaded %d games from cache (season %d)",
                     len(rows), season)
            out = []
            for r in rows:
                try:
                    game_time = datetime.fromisoformat(r[3])
                except Exception:
                    game_time = datetime.now(timezone.utc)
                out.append(HistoricalGame(
                    event_id=r[0], game_pk=r[1], game_date=r[2],
                    game_time_utc=game_time,
                    home_team=r[4], away_team=r[5],
                    home_runs=r[6], away_runs=r[7],
                    home_starter=r[8] or "", away_starter=r[9] or "",
                    status=r[10] or "Final",
                ))
            db.close()
            return out

    log.info("historical_games: fetching season %d from MLB Stats API", season)
    data = _fetch_json(_SCHEDULE_URL.format(season=season))

    inserted = 0
    games: list[HistoricalGame] = []
    for day in data.get("dates", []):
        game_date_str = day.get("date", "")
        for g in day.get("games", []):
            if g.get("gameType") != "R":
                continue
            status = (g.get("status", {}) or {}).get("detailedState", "")
            if status != "Final":
                continue
            teams = g.get("teams", {}) or {}
            home_raw = (teams.get("home", {}).get("team", {}) or {}).get("name", "")
            away_raw = (teams.get("away", {}).get("team", {}) or {}).get("name", "")
            home = try_normalize_team(home_raw)
            away = try_normalize_team(away_raw)
            if not home or not away:
                continue

            # Parse game time
            try:
                gt = g.get("gameDate", "")
                if gt.endswith("Z"):
                    gt = gt[:-1] + "+00:00"
                game_time = datetime.fromisoformat(gt).astimezone(timezone.utc)
            except Exception:
                continue

            home_score = teams["home"].get("score")
            away_score = teams["away"].get("score")
            if home_score is None or away_score is None:
                continue

            home_sp = ((teams["home"].get("probablePitcher") or {}).get("fullName")
                       or "")
            away_sp = ((teams["away"].get("probablePitcher") or {}).get("fullName")
                       or "")

            ev = make_event_id(game_time, away, home)
            games.append(HistoricalGame(
                event_id=ev, game_pk=int(g["gamePk"]),
                game_date=game_date_str, game_time_utc=game_time,
                home_team=home, away_team=away,
                home_runs=int(home_score), away_runs=int(away_score),
                home_starter=home_sp, away_starter=away_sp,
                status=status,
            ))

            db.execute("""
                INSERT OR REPLACE INTO historical_games
                (event_id, game_pk, season, game_date, game_time_utc,
                 home_team, away_team, home_runs, away_runs,
                 home_starter, away_starter, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ev, int(g["gamePk"]), season, game_date_str,
                  game_time.isoformat(), home, away,
                  int(home_score), int(away_score),
                  home_sp, away_sp, status))
            inserted += 1

    db.commit()
    db.close()
    log.info("historical_games: cached %d finalized games for %d", inserted, season)
    return games
