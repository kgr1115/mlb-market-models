"""
Inspect every cached snapshot for one game across all books.

Usage:
    python scripts/inspect_game.py HOU CLE
    python scripts/inspect_game.py LAD SF 2026-04-21
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python scripts/inspect_game.py AWAY_ABBR HOME_ABBR [YYYY-MM-DD]")
        return 2
    away_abbr = sys.argv[1].upper()
    home_abbr = sys.argv[2].upper()
    day = sys.argv[3] if len(sys.argv) >= 4 else datetime.now(timezone.utc).strftime("%Y-%m-%d")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.team_names import FG_ABBR_TO_CANONICAL

    away = FG_ABBR_TO_CANONICAL.get(away_abbr)
    home = FG_ABBR_TO_CANONICAL.get(home_abbr)
    if not away or not home:
        print(f"Unknown abbrs: {away_abbr} / {home_abbr}")
        return 1
    print(f"Looking for {away} @ {home} (any UTC date overlapping {day})")

    conn = sqlite3.connect("bbp_cache.sqlite")
    # Show every row for this team-pair across ±1 day.
    cur = conn.execute(
        """
        SELECT book, event_id, game_time_utc, polled_at_utc,
               home_ml, away_ml,
               home_rl_line, home_rl_odds, away_rl_odds,
               total_line, over_odds, under_odds
        FROM odds_snapshots
        WHERE home_team = ? AND away_team = ?
        ORDER BY polled_at_utc DESC
        LIMIT 20
        """,
        (home, away),
    )
    rows = cur.fetchall()
    if not rows:
        print(" no rows in cache for that team pair")
    for r in rows:
        book, ev, gt, pt, hml, aml, hrl_line, hrl_o, arl_o, tl, oo, uo = r
        print(f"  [{book:10}] event={ev}")
        print(f"    game_time={gt}  polled={pt}")
        print(f"    ML: {away_abbr}={aml}  {home_abbr}={hml}")
        print(f"    RL: line={hrl_line}  {home_abbr}={hrl_o}  {away_abbr}={arl_o}")
        print(f"    TOT: line={tl}  O={oo}  U={uo}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
