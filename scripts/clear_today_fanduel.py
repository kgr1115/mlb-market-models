"""
One-off: delete today's FanDuel snapshots from the odds cache so the next
poller iteration has to write fresh ones (using whatever filter logic is
currently in data/odds_fanduel.py).

Usage from project root:
    python scripts/clear_today_fanduel.py
    python scripts/clear_today_fanduel.py 2026-04-21       # explicit date
    python scripts/clear_today_fanduel.py --book fanduel   # different book
    python scripts/clear_today_fanduel.py --all-books
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", default=None,
                    help="YYYY-MM-DD UTC; defaults to today UTC")
    ap.add_argument("--book", default="fanduel",
                    help="book column value to target (default: fanduel)")
    ap.add_argument("--all-books", action="store_true",
                    help="delete rows for every book on the given date")
    ap.add_argument("--db", default="bbp_cache.sqlite",
                    help="path to the sqlite cache (default: bbp_cache.sqlite)")
    args = ap.parse_args()

    d = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    db = Path(args.db).resolve()
    if not db.exists():
        print(f"[error] cache db not found: {db}", file=sys.stderr)
        return 1

    # MLB games that start late in ET (e.g. 9 PM PT) have game_time_utc the
    # NEXT UTC day, so "today" in ET spans two UTC prefixes. We also clear
    # yesterday-UTC in case a morning run is trying to wipe last night's slate.
    from datetime import date as _date, timedelta as _td
    base = _date.fromisoformat(d)
    prefixes = [
        (base - _td(days=1)).isoformat(),
        base.isoformat(),
        (base + _td(days=1)).isoformat(),
    ]

    conn = sqlite3.connect(db)
    try:
        total = 0
        for prefix in prefixes:
            like = f"{prefix}%"
            if args.all_books:
                cur = conn.execute(
                    "DELETE FROM odds_snapshots WHERE game_time_utc LIKE ?",
                    (like,),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM odds_snapshots "
                    "WHERE book = ? AND game_time_utc LIKE ?",
                    (args.book, like),
                )
            total += cur.rowcount
        conn.commit()
        print(f"deleted {total} rows  (prefixes={prefixes}, "
              f"book={'ALL' if args.all_books else args.book}, db={db})")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
