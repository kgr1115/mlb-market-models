"""
SQLite cache for projection pulls.

One row per (source, player_id, fetched_date). We keep history so you
can see how a player's projection moved across the season (e.g. a
starter who got hurt will see their IP projection collapse overnight).
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .projections_models import (
    HitterProjection,
    PitcherProjection,
    ProjectionSource,
)


DEFAULT_CACHE_PATH = Path(
    os.environ.get("BBP_CACHE_PATH", "bbp_cache.sqlite")
).resolve()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS hitter_projections (
    source          TEXT NOT NULL,
    player_id       TEXT NOT NULL,
    fetched_date    TEXT NOT NULL,
    name            TEXT NOT NULL,
    team            TEXT,
    bats            TEXT,
    pa              REAL, ab REAL,
    avg             REAL, obp REAL, slg REAL, iso REAL,
    woba            REAL, xwoba REAL,
    bb_pct          REAL, k_pct REAL, barrel_pct REAL,
    wrc_plus        REAL, bsr REAL, war REAL,
    fetched_at_utc  TEXT NOT NULL,
    PRIMARY KEY (source, player_id, fetched_date)
);

CREATE TABLE IF NOT EXISTS pitcher_projections (
    source          TEXT NOT NULL,
    player_id       TEXT NOT NULL,
    fetched_date    TEXT NOT NULL,
    name            TEXT NOT NULL,
    team            TEXT,
    throws          TEXT,
    gs              REAL, g REAL, ip REAL,
    era             REAL, fip REAL, xfip REAL, siera REAL,
    k_pct           REAL, bb_pct REAL, k_bb_pct REAL,
    hr9             REAL, whip REAL,
    xwoba_against   REAL, csw_pct REAL,
    war             REAL,
    fetched_at_utc  TEXT NOT NULL,
    PRIMARY KEY (source, player_id, fetched_date)
);

CREATE INDEX IF NOT EXISTS idx_hit_proj_team
    ON hitter_projections(team, source, fetched_date);
CREATE INDEX IF NOT EXISTS idx_pit_proj_team
    ON pitcher_projections(team, source, fetched_date);
"""


class ProjectionsCache:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_CACHE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ----- writes ---------------------------------------------------------

    def upsert_hitters(self, projections: Iterable[HitterProjection],
                       fetched_date: Optional[date] = None) -> int:
        fd = (fetched_date or datetime.now(timezone.utc).date()).isoformat()
        rows = []
        for p in projections:
            rows.append((
                p.source.value, p.player_id, fd, p.name, p.team, p.bats,
                p.pa, p.ab, p.avg, p.obp, p.slg, p.iso, p.woba, p.xwoba,
                p.bb_pct, p.k_pct, p.barrel_pct, p.wrc_plus, p.bsr, p.war,
                p.fetched_at_utc.isoformat(),
            ))
        if not rows:
            return 0
        with self._conn() as c:
            c.executemany(
                """
                INSERT OR REPLACE INTO hitter_projections
                (source, player_id, fetched_date, name, team, bats,
                 pa, ab, avg, obp, slg, iso, woba, xwoba,
                 bb_pct, k_pct, barrel_pct, wrc_plus, bsr, war,
                 fetched_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def upsert_pitchers(self, projections: Iterable[PitcherProjection],
                        fetched_date: Optional[date] = None) -> int:
        fd = (fetched_date or datetime.now(timezone.utc).date()).isoformat()
        rows = []
        for p in projections:
            rows.append((
                p.source.value, p.player_id, fd, p.name, p.team, p.throws,
                p.gs, p.g, p.ip, p.era, p.fip, p.xfip, p.siera,
                p.k_pct, p.bb_pct, p.k_bb_pct, p.hr9, p.whip,
                p.xwoba_against, p.csw_pct, p.war,
                p.fetched_at_utc.isoformat(),
            ))
        if not rows:
            return 0
        with self._conn() as c:
            c.executemany(
                """
                INSERT OR REPLACE INTO pitcher_projections
                (source, player_id, fetched_date, name, team, throws,
                 gs, g, ip, era, fip, xfip, siera,
                 k_pct, bb_pct, k_bb_pct, hr9, whip,
                 xwoba_against, csw_pct, war,
                 fetched_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    # ----- reads ----------------------------------------------------------

    def latest_hitters(self, source: ProjectionSource,
                       team: Optional[str] = None) -> list[HitterProjection]:
        q = """
            SELECT source, player_id, fetched_date, name, team, bats,
                   pa, ab, avg, obp, slg, iso, woba, xwoba,
                   bb_pct, k_pct, barrel_pct, wrc_plus, bsr, war,
                   fetched_at_utc
            FROM hitter_projections
            WHERE source = ?
              AND fetched_date = (
                  SELECT MAX(fetched_date) FROM hitter_projections
                  WHERE source = ?
                    {team_clause}
              )
              {team_filter}
        """
        params: list = [source.value, source.value]
        team_clause = ""
        team_filter = ""
        if team is not None:
            team_clause = "AND team = ?"
            team_filter = "AND team = ?"
            params.extend([team, team])
        q = q.format(team_clause=team_clause, team_filter=team_filter)
        with self._conn() as c:
            rows = c.execute(q, params).fetchall()
        return [self._hitter_from_row(r) for r in rows]

    def latest_pitchers(self, source: ProjectionSource,
                        team: Optional[str] = None) -> list[PitcherProjection]:
        q = """
            SELECT source, player_id, fetched_date, name, team, throws,
                   gs, g, ip, era, fip, xfip, siera,
                   k_pct, bb_pct, k_bb_pct, hr9, whip,
                   xwoba_against, csw_pct, war,
                   fetched_at_utc
            FROM pitcher_projections
            WHERE source = ?
              AND fetched_date = (
                  SELECT MAX(fetched_date) FROM pitcher_projections
                  WHERE source = ?
                    {team_clause}
              )
              {team_filter}
        """
        params: list = [source.value, source.value]
        team_clause = ""
        team_filter = ""
        if team is not None:
            team_clause = "AND team = ?"
            team_filter = "AND team = ?"
            params.extend([team, team])
        q = q.format(team_clause=team_clause, team_filter=team_filter)
        with self._conn() as c:
            rows = c.execute(q, params).fetchall()
        return [self._pitcher_from_row(r) for r in rows]

    @staticmethod
    def _hitter_from_row(r: tuple) -> HitterProjection:
        (src, pid, _fd, name, team, bats, pa, ab, avg, obp, slg, iso, woba, xwoba,
         bb, k, barrel, wrc, bsr, war, fetched) = r
        return HitterProjection(
            source=ProjectionSource(src), player_id=pid, name=name, team=team,
            bats=bats or "R",
            pa=pa or 0.0, ab=ab or 0.0,
            avg=avg or 0.0, obp=obp or 0.0, slg=slg or 0.0, iso=iso or 0.0,
            woba=woba or 0.320, xwoba=xwoba,
            bb_pct=bb or 0.085, k_pct=k or 0.225, barrel_pct=barrel,
            wrc_plus=wrc or 100.0, bsr=bsr or 0.0, war=war,
            fetched_at_utc=datetime.fromisoformat(fetched),
        )

    @staticmethod
    def _pitcher_from_row(r: tuple) -> PitcherProjection:
        (src, pid, _fd, name, team, throws, gs, g, ip, era, fip, xfip, siera,
         k, bb, kbb, hr9, whip, xwoba_a, csw, war, fetched) = r
        return PitcherProjection(
            source=ProjectionSource(src), player_id=pid, name=name, team=team,
            throws=throws or "R",
            gs=gs or 0.0, g=g or 0.0, ip=ip or 0.0,
            era=era or 4.20, fip=fip or 4.10, xfip=xfip or 4.10, siera=siera or 4.05,
            k_pct=k or 0.225, bb_pct=bb or 0.085, k_bb_pct=kbb or 0.135,
            hr9=hr9 or 1.15, whip=whip or 1.28,
            xwoba_against=xwoba_a, csw_pct=csw, war=war,
            fetched_at_utc=datetime.fromisoformat(fetched),
        )

    def stats(self) -> dict:
        with self._conn() as c:
            h = c.execute("SELECT COUNT(*) FROM hitter_projections").fetchone()[0]
            p = c.execute("SELECT COUNT(*) FROM pitcher_projections").fetchone()[0]
        return {"hitter_rows": h, "pitcher_rows": p, "path": str(self.path)}
