# MLB Betting Predictor — Web UI

A dark sportsbook-style dashboard that sits on top of the three predictors in
`predictors/` (moneyline, run line, totals). Stack is FastAPI + React (via CDN,
so no node build step is required).

## Run

```bash
pip install -r web/requirements.txt
python web/run.py
```

By default the launcher binds `0.0.0.0` so the app is reachable from other
devices on your LAN. It prints both URLs on startup:

```
  Local      →  http://127.0.0.1:8000
  Network    →  http://192.168.1.42:8000     # open this on your phone, tablet, etc.
```

Flags:

```
python web/run.py --local         # bind 127.0.0.1 only (no LAN exposure)
python web/run.py --port 9000     # change port
python web/run.py --no-open       # don't auto-open browser
```

**Windows firewall:** the first LAN launch may prompt Windows Defender
Firewall. Allow Python on **Private networks only** (not Public) so your
phone/laptop can reach the app without exposing it to coffee-shop Wi-Fi.

**Security note:** there is no auth. Only expose on networks you trust, or
stick with `--local`.

## Layout

```
web/
  backend/
    api.py          FastAPI endpoints + mock game generator
  frontend/
    index.html      single-file React app (Tailwind + Recharts from CDN)
  run.py            launcher
  requirements.txt
```

## Tabs

**Daily Picks** — card grid of today's games. Each card shows the ML / RL /
Totals pick with model probability, vig-stripped implied probability, edge,
EV/\$1, and a 0-100 confidence bar (LOW / LEAN / MEDIUM / HIGH). Filter by
confidence band, sort by confidence / edge / first pitch.

**Game Detail** — open any card. Full matchup header, all three predictions
side-by-side with their confidence bars and factor breakdown, plus raw team
stats (SP, bullpen, offense, defense) for both sides.

**Backtest** — KPI strip (bets, win%, units, ROI, bankroll), equity curve,
by-market table, by-confidence-band bar chart. Reads from
`/api/backtest/summary`, which currently returns mock data until the real
backtest finishes running.

## Endpoints

```
GET /api/health
GET /api/games/today?d=YYYY-MM-DD
GET /api/games/{game_id}            # game_id format: YYYY-MM-DD-AWAY@HOME
GET /api/backtest/summary
```

## Wiring up live data

`backend/api.py` has a `generate_todays_games` mock. Replace its guts with
real pulls (MLB Stats API for the slate + starters, pybaseball/FanGraphs for
pitcher/hitter stats, Open-Meteo for weather, The Odds API for markets). The
dataclasses in `predictors/shared.py` are the only shape you need to produce —
the three `predict_*` functions and everything the UI renders downstream will
keep working unchanged.

When the real backtest output is available, point `_mock_backtest` at the
results file (CSV or JSON) and drop the mock generator.
