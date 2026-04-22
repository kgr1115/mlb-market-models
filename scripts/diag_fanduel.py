"""
FanDuel diagnostic — runs one live FD fetch and prints:
  - total events in the FD payload
  - how many were pre-match vs live/settled according to our filter
  - a sample of the events that got rejected (and why)
  - a sample of the pre-match events that DID make it through, with
    the main-line ML / RL / Total we parsed for each
  - finally, how many FD rows currently exist in bbp_cache.sqlite for today

Run from project root:
    python scripts/diag_fanduel.py
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make sure we import project modules when run from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import odds_fanduel as fd  # noqa: E402
from data.odds_fanduel import (  # noqa: E402
    _event_is_prematch, _is_non_main_market, _market_is_active,
    _market_type, _ML_TYPES, _RL_TYPES, _TOTAL_TYPES,
    _index_attachments, _try_fetch,
)


def main() -> int:
    payload = _try_fetch()
    if not payload:
        print("FD fetch returned None — no payload (all state subdomains "
              "failed). Network issue or FD blocked us.")
        return 1

    events, markets, _runners = _index_attachments(payload)
    print(f"events in payload:  {len(events)}")
    print(f"markets in payload: {len(markets)}")

    # Classify every event.
    prematch: list[tuple[str, dict]] = []
    live: list[tuple[str, dict]] = []
    for ev_id, ev in events.items():
        (prematch if _event_is_prematch(ev) else live).append((ev_id, ev))
    print(f"events classified prematch: {len(prematch)}")
    print(f"events classified live/settled: {len(live)}")

    # Show some live-ish events (first few) with which state field flagged them.
    print("\n--- sample rejected (live/settled) events ---")
    for ev_id, ev in live[:5]:
        reasons = []
        if ev.get("inPlay") is True:
            reasons.append("inPlay=true")
        for key in ("eventState", "state", "status", "eventStatus"):
            v = ev.get(key)
            if v is None:
                continue
            reasons.append(f"{key}={v!r}")
        print(f"  {ev.get('name')!r:55} -> {', '.join(reasons) or '(no state flags matched)'}")

    # Show some sample *passing* events and the fields they expose so we can
    # confirm our whitelist is actually what's happening (vs. no state at all).
    print("\n--- sample accepted (prematch) events ---")
    for ev_id, ev in prematch[:5]:
        flags = []
        for key in ("inPlay", "eventState", "state", "status",
                    "eventStatus", "openDate"):
            if key in ev:
                flags.append(f"{key}={ev.get(key)!r}")
        print(f"  {ev.get('name')!r:55} -> {', '.join(flags)}")

    # For each prematch event, group its markets by our kind and show the
    # main-market names we'd actually ingest.
    prematch_ids = {str(ev.get("eventId") or ev_id) for ev_id, ev in prematch}
    kept: dict[str, dict[str, list[str]]] = {}
    rejected_markets: list[tuple[str, str, str]] = []  # (event, name, reason)
    for m in markets.values():
        ev_id = m.get("eventId")
        if ev_id is None:
            continue
        ev_str = str(ev_id)
        if ev_str not in prematch_ids:
            continue
        mname = (m.get("marketName") or "")
        if _is_non_main_market(mname):
            rejected_markets.append((ev_str, mname, "non-main name"))
            continue
        if not _market_is_active(m):
            rejected_markets.append((ev_str, mname, "market inactive"))
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
        kept.setdefault(ev_str, {}).setdefault(kind, []).append(mname)

    print(f"\n--- first 3 prematch events' kept markets ---")
    shown = 0
    for ev_id, ev in prematch:
        ev_str = str(ev.get("eventId") or ev_id)
        keep = kept.get(ev_str, {})
        if not keep:
            continue
        print(f"  {ev.get('name')!r}")
        for k, names in keep.items():
            print(f"     {k:5} -> {names}")
        shown += 1
        if shown >= 3:
            break

    # Bonus: run the full fetcher and see how many snapshots it produces.
    print("\n--- running full fetch_fanduel_snapshots() ---")
    snaps = fd.fetch_fanduel_snapshots()
    print(f"snapshots produced: {len(snaps)}")
    if snaps:
        s = snaps[0]
        print(f"  sample: {s.away_team} @ {s.home_team}  "
              f"ML={s.away_ml}/{s.home_ml}  "
              f"RL line={s.home_rl_line} odds={s.away_rl_odds}/{s.home_rl_odds}  "
              f"Total={s.total_line} O={s.over_odds} U={s.under_odds}")

    # Cache state.
    db = Path("bbp_cache.sqlite")
    if db.exists():
        conn = sqlite3.connect(db)
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        like = f"{today_utc}%"
        n = conn.execute(
            "SELECT COUNT(*) FROM odds_snapshots "
            "WHERE book='fanduel' AND game_time_utc LIKE ?",
            (like,),
        ).fetchone()[0]
        last = conn.execute(
            "SELECT MAX(polled_at_utc) FROM odds_snapshots "
            "WHERE book='fanduel'"
        ).fetchone()[0]
        conn.close()
        print(f"\ncache: {n} fanduel rows for today "
              f"({today_utc}), last poll = {last}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
