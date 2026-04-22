"""
End-to-end diagnostic for today's HOU @ CLE card.

Answers:
  1. Is HOU @ CLE in today's FD fetch?
  2. Under what (home_team, away_team) names is FD storing every MLB game
     that was polled today?  (So we can see if HOU / CLE shows up at all.)
  3. What event_ids exist in the cache for today (April 21) across all books?
  4. What does the live-data layer actually hand to the UI for HOU @ CLE?

Run from the project root with the server STOPPED:
    python scripts/diag_hou_cle.py
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.odds_fanduel import fetch_fanduel_snapshots


def main() -> int:
    # -------- 1. today's FD fetch, look for HOU or CLE --------
    print("=" * 70)
    print("1. What FanDuel returns right now")
    print("=" * 70)
    snaps = fetch_fanduel_snapshots()
    print(f"Total snapshots: {len(snaps)}")
    print("\nAll (away @ home) in today's FD fetch:")
    for s in snaps:
        star = "  <---- HOU/CLE" if ("Astros" in s.away_team or "Astros" in s.home_team
                                      or "Guardians" in s.away_team
                                      or "Guardians" in s.home_team) else ""
        # OddsSnapshot fields: use _ml not _ml_odds here (snapshot dataclass,
        # not MarketData). Keep these the same.
        print(f"  {s.away_team:28} @ {s.home_team:28}  "
              f"ML={s.away_ml}/{s.home_ml}  TOT={s.total_line}  "
              f"game_utc={s.game_time_utc}{star}")

    # -------- 2. cache contents for today (both UTC dates) --------
    print("\n" + "=" * 70)
    print("2. Cache contents for 2026-04-21 and 2026-04-22 (all books)")
    print("=" * 70)
    conn = sqlite3.connect("bbp_cache.sqlite")
    for prefix in ("2026-04-21", "2026-04-22"):
        print(f"\n  game_time_utc prefix = {prefix!r}")
        rows = conn.execute(
            """
            SELECT book, event_id, away_team, home_team,
                   home_ml, away_ml, total_line, MAX(polled_at_utc)
            FROM odds_snapshots
            WHERE game_time_utc LIKE ?
            GROUP BY book, event_id
            ORDER BY event_id, book
            """,
            (f"{prefix}%",),
        ).fetchall()
        if not rows:
            print("    (no rows)")
            continue
        for r in rows:
            book, ev, away, home, hml, aml, tl, last = r
            star = "  <---- HOU/CLE" if ("Houston" in (away or "")
                                         or "Cleveland" in (home or "")) else ""
            print(f"    [{book:10}] {ev}  away={away!r} home={home!r}"
                  f"  ML={aml}/{hml} TOT={tl}  last_poll={last}{star}")
    conn.close()

    # -------- 3. ask live_data what it would hand the UI --------
    print("\n" + "=" * 70)
    print("3. What the live-data layer produces for HOU @ CLE today")
    print("=" * 70)
    try:
        from web.backend import live_data, mlb_data
        today = mlb_data.today_et()
        print(f"today_et = {today}")
        slate = live_data.warm_slate(today, force_odds=False)
        # Find the HOU @ CLE game in today's schedule.
        hou_game = None
        for ev_id, gs in slate.schedule.items():
            if (getattr(gs, "away_abbr", "") == "HOU"
                    and getattr(gs, "home_abbr", "") == "CLE"):
                hou_game = gs
                break
        if not hou_game:
            # fallback: search any schedule-like attrs
            print("  Could not find HOU @ CLE in slate.schedule")
            print("  Keys in slate.schedule:")
            for k in list(slate.schedule.keys())[:5]:
                print(f"    {k}")
        else:
            print(f"  Found schedule entry: {hou_game}")
        # Probe the per-book live builder with a constructed game dict.
        game_stub = {
            "date": today.isoformat(),
            "raw": {"away": {"abbr": "HOU"}, "home": {"abbr": "CLE"}},
        }
        pb = live_data.build_per_book_markets_live(game_stub, slate)
        print(f"\n  build_per_book_markets_live returned keys: {list(pb.keys())}")
        for book, md in pb.items():
            side = "HOME -1.5" if md.home_is_rl_favorite else "AWAY -1.5"
            print(f"    {book}: ML away/home = {md.away_ml_odds}/{md.home_ml_odds}  "
                  f"RL({side}) away/home = {md.away_rl_odds}/{md.home_rl_odds}  "
                  f"TOT={md.total_line} O={md.over_odds} U={md.under_odds}")
        ev = live_data.odds_event_id_from_api_game(game_stub)
        print(f"\n  odds_event_id_from_api_game -> {ev}")

        # -------- 4. direct cache lookup by event_id we'd actually use --------
        print("\n  direct cache lookup for event_id that live-data uses:")
        conn2 = sqlite3.connect("bbp_cache.sqlite")
        for book in ("fanduel", "draftkings", "pinnacle"):
            r = conn2.execute(
                """SELECT polled_at_utc, game_time_utc,
                          home_ml, away_ml, total_line, over_odds, under_odds
                   FROM odds_snapshots
                   WHERE book=? AND event_id=?
                   ORDER BY polled_at_utc DESC
                   LIMIT 3""",
                (book, ev),
            ).fetchall()
            if not r:
                print(f"    [{book:10}] NO ROWS for event_id={ev!r}")
            else:
                for row in r:
                    print(f"    [{book:10}] {row}")
        # Also see what build_market_live's ±1 day fallback would surface.
        print("\n  ±1 day fallback candidates (what fallback logic sees):")
        for book in ("fanduel", "draftkings", "pinnacle"):
            r = conn2.execute(
                """SELECT event_id, polled_at_utc, home_ml, away_ml,
                          total_line, over_odds, under_odds
                   FROM odds_snapshots
                   WHERE book=?
                     AND home_team='Cleveland Guardians'
                     AND away_team='Houston Astros'
                   ORDER BY polled_at_utc DESC
                   LIMIT 3""",
                (book,),
            ).fetchall()
            if not r:
                print(f"    [{book:10}] no HOU@CLE rows at all")
            else:
                for row in r:
                    print(f"    [{book:10}] {row}")
        conn2.close()
    except Exception as e:
        print(f"  [error] {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
