# MLB Betting Predictor — Data Sources Reference

This document maps every indicator in the Run Line, Moneyline, and Totals frameworks to a recommended **free** data source, with backup options, access method, and integration notes. Focused on sources that are accurate, maintained, and either free outright or have a free tier generous enough for a single desktop app.

---

## 1. The "Stack" at a Glance

For 80%+ of the sabermetric inputs, the fastest path is the Python library **`pybaseball`**, which is a free, actively maintained wrapper around three authoritative sources:

- **Baseball Savant** (MLB's official Statcast portal) — xwOBA, Barrel%, launch/exit-velo, OAA, park factors, pitch-level data
- **FanGraphs** — SIERA, xFIP, wRC+, FIP, K-BB%, CSW%, BsR, DRS, framing runs, leaderboards with date filters (for 30-day rolling form)
- **Baseball Reference** — traditional stats, splits, game logs, umpire assignments

Install with `pip install pybaseball`. No API key required for any of the three underlying sources.

For the remaining inputs — weather, odds, umpire run impact, confirmed lineups — you bolt on a handful of free APIs described below.

---

## 2. Starting Pitcher Stats (SIERA, xFIP, K-BB%, CSW%, xwOBA-against, IP/GS, rolling form, splits, TTO)

**Primary: FanGraphs (via `pybaseball.pitching_stats`)**
SIERA, xFIP, K-BB%, FIP, CSW% (listed as `CSW%` in the plate-discipline section), IP, GS, and most rate stats are native FanGraphs columns. Date-bounded leaderboards support 30-day rolling windows directly (`pitching_stats(start_season, end_season, qual=0)` and you can filter by date range with `pitching_stats_range(start_dt, end_dt)`).

**Primary: Baseball Savant (via `pybaseball.statcast_pitcher`)**
xwOBA-against, xERA, whiff%, CSW components (called-strike% + whiff%), Barrel%-against, and pitch-level data. For Statcast leaderboards use `pybaseball.statcast_pitcher_expected_stats()`.

**Splits & TTO (Times Through the Order):**
- Baseball Reference splits pages — `pybaseball.pitching_stats_bref()` plus direct URL like `baseball-reference.com/players/split.fcgi` gives vs L/R, TTO1/2/3, home/away
- Baseball Savant: filter leaderboard by batter handedness for Statcast splits

**Backup:** Brooks Baseball (brooksbaseball.net) for PITCHf/x-era pitch types; MLB Stats API `statsapi.mlb.com/api/v1/people/{id}/stats` for official counting stats.

---

## 3. Bullpen Stats (bullpen FIP, high-leverage K%, last-3-day usage, WPA, shutdown/meltdown, rest)

**Primary: FanGraphs bullpen leaderboard (via `pybaseball.team_pitching`)**
Team bullpen splits include FIP, K%, BB%, WPA, SD, MD, ERA, and "Clutch." Date-bounded via `team_pitching(start_season, end_season)` with filter.

**Last-3-day usage & rest (pitch counts, back-to-backs):**
- **MLB Stats API** (free, no key): `statsapi.mlb.com/api/v1/schedule?teamId={id}&startDate=...&endDate=...&hydrate=probablePitcher,linescore,boxscore` returns every pitcher's appearances including pitches thrown. This is how you build a "days of rest" and "pitches in last 3 days" table per reliever.
- Python wrapper **`MLB-StatsAPI`** (`pip install MLB-StatsAPI`) makes this cleaner.

**High-leverage K%:** FanGraphs splits tab exposes high/medium/low leverage — scrape via `pybaseball` or direct HTML.

**Backup:** Baseball Reference bullpen usage pages (game-by-game reliever logs).

---

## 4. Offense (lineup-weighted wRC+, wOBA/xwOBA, ISO, Barrel%, vs LHP/RHP, OBP, top-of-order OBP, team K%)

**Primary: FanGraphs batting leaderboard (via `pybaseball.batting_stats` / `batting_stats_range`)**
wRC+, wOBA, ISO, OBP, K%, BB%, BsR — all present. Date-bounded leaderboards give 30-day rolling form. Handedness splits live on the FanGraphs splits leaderboards (`pybaseball.batting_stats_bref()` plus splits pages).

**Primary: Baseball Savant (via `pybaseball.statcast_batter_expected_stats`)**
xwOBA, Barrel%, HardHit%, max EV.

**Lineup-weighted aggregation:**
Once you have per-hitter wRC+/wOBA, multiply each starter in tonight's confirmed lineup by their weighted contribution. This requires pulling **confirmed lineups** — see Section 11.

**Backup:** MLB Stats API player stats endpoint for official counting stats.

---

## 5. Defense & Baserunning (OAA, DRS, catcher framing runs, BsR)

**OAA (Outs Above Average) — Primary: Baseball Savant**
`baseballsavant.mlb.com/leaderboard/outs_above_average` — position-level and team-aggregated. Scrape with `pybaseball` or direct CSV download.

**DRS (Defensive Runs Saved) — Primary: FanGraphs**
Team and player DRS available on fielding leaderboard. Sourced from Sports Info Solutions but republished free on FanGraphs.

**Catcher Framing Runs — Primary: Baseball Savant**
`baseballsavant.mlb.com/catcher_framing` — per-catcher framing runs, broken out by zone. Most accurate public framing metric.

**BsR (Baserunning Runs) — Primary: FanGraphs batting leaderboard (already included there).**

---

## 6. Park Factors (run & HR factors, handedness splits, roof status, altitude)

**Primary: Baseball Savant park factors**
`baseballsavant.mlb.com/leaderboard/statcast-park-factors` — 1-year and 3-year rolling factors for R, HR, 1B, 2B, 3B, broken out by batter handedness. Best current source.

**Primary (backup & longer history): FanGraphs "Guts!" park factors**
`fangraphs.com/guts.aspx?type=pf&teamid=0&season=YYYY` — multi-year handedness-split park factors. Good for smoothing when Statcast 1-yr is noisy.

**Roof / altitude / dimensions:** static data. Put it in a local JSON/CSV in the app. (Coors = 5,200 ft; retractable roofs at CHC? No. Retractable: ARI, HOU, MIA, MIL, SEA, TEX, TOR. Fixed dome: TB.)

**Backup:** Baseball Reference park factor pages (`baseball-reference.com/leagues/majors/{year}-park-factors.shtml`).

---

## 7. Weather (wind dir/speed, temp, humidity, pressure, precipitation)

**Primary: Open-Meteo (free, no API key, generous rate limits)**
`api.open-meteo.com/v1/forecast?latitude=...&longitude=...&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation_probability&temperature_unit=fahrenheit&wind_speed_unit=mph`
Excellent forecast accuracy for the 24-48 hr window that matters for betting. No auth, commercial use allowed.

**Backup: NOAA / National Weather Service API** (`api.weather.gov`, US-only, free, no key)
Official US forecasts. Slightly more awkward API (two-step: get grid point, then get forecast) but authoritative.

**Backup: OpenWeatherMap** — free tier 1,000 calls/day, requires key.

**Stadium coordinates:** keep a local lookup table of each park's lat/lon. The wind-direction-relative-to-field-orientation calculation also requires the **field bearing** (home-plate-to-center-field compass heading). Orientations are fixed per park; store once. Then project wind vector onto that bearing to get the true "blowing out / blowing in / cross" component, which is what actually matters for HRs.

---

## 8. Umpire Data (plate umpire R/G, called-strike rate, umpire × pitcher interaction)

**Umpire crew assignment for tonight:** MLB Stats API `statsapi.mlb.com/api/v1/schedule?gamePk={id}&hydrate=officials` returns the crew by position (plate, 1B, 2B, 3B).

**Primary: Umpire Scorecards (umpscorecards.com)**
Per-umpire accuracy, consistency, and favor metrics. Has a public website and a free API-ish JSON (`umpscorecards.com/umpires/...`) — scrape-able. Derived from Statcast, so data is accurate.

**Primary (baseline rates): Baseball Savant umpire leaderboard**
`baseballsavant.mlb.com/umpire-scorecard` (newer addition) — called-strike rate, zone accuracy.

**Primary (R/G per ump): Swish Analytics / EVAnalytics / self-compute**
A free path: pull all games an umpire called (Retrosheet or MLB Stats API), compute R/G and K/BB rate yourself. This is more robust than trusting a single scraped site.

**Backup: Retrosheet** — full historical umpire game logs, free, bulk CSV downloads.

---

## 9. Market Data — Odds, Line Moves, Public %, Closing-Line Value

This is the hardest category to get fully free.

**Primary: The Odds API (the-odds-api.com)**
Free tier: 500 requests/month. Covers MLB moneyline, run line, totals across most US books including Pinnacle. Enough for ~15-20 pulls per game day if you pull snapshots instead of live polling. Paid tiers are cheap ($25-40/mo) if you outgrow it.

**Backup: OddsPortal / SportsbookReview scraping**
SBR (sportsbookreview.com) has free line history pages. Scraping is allowed in moderation; respect robots.txt and rate-limit.

**Public % tickets vs. money:**
- **Action Network public betting page** — free to view, paywall for API; scrape sparingly
- **VSIN** — free public betting splits, daily articles
- **Covers.com consensus** — free, scrape-able

**RLM, steam, CLV:**
- Compute yourself from The Odds API snapshots + sportsbook-by-sportsbook comparison
- "Closing line value" requires capturing the close — snapshot at game start

**Sharp book reference price: Pinnacle.** Its line is widely treated as the "true" probability; The Odds API includes it.

---

## 10. Schedules, Game State, Probable Pitchers

**Primary: MLB Stats API** (`statsapi.mlb.com`, official, free, no key)
- `/api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=probablePitcher,linescore,team,venue,weather,officials`
- Returns everything: teams, venue, probable starters, weather (MLB-reported, but trust your own weather API more), officials, game status.
- Python wrapper: `MLB-StatsAPI` (`pip install MLB-StatsAPI`) or `statsapi` library.

**Backup:** ESPN unofficial API, RotoWire scraped schedule.

---

## 11. Confirmed Lineups

**Primary: MLB Stats API**
`statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live` returns lineups once posted (typically 1-4 hours pre-first-pitch). Poll every 10-15 min on game day.

**Primary (earlier projected lineups): RotoWire**
`rotowire.com/baseball/daily-lineups.php` — projected lineups often available the night before, marked "projected" vs "official." Scrape-friendly HTML.

**Backup:** Lineups.com, Rotogrinders, Baseball Press — all free, all scrape-friendly. Cross-reference for confidence in the projection.

---

## 12. Rest, Travel, Schedule Stress

Derived entirely from MLB Stats API schedule data:

- **Rest days:** count days since team's last game.
- **Travel miles:** geodesic distance between last venue and tonight's venue; store venue lat/lon locally.
- **Schedule stress:** games-in-last-N-days, timezones crossed, getaway-day flag, doubleheader flag — all computable.

No external source needed beyond the schedule feed.

---

## 13. NRFI / YRFI First-Inning Model Inputs

- **Starter first-inning history:** query Statcast game logs, filter `inning == 1`, aggregate.
- **Top-of-order OBP:** from the confirmed lineup (positions 1-3) × FanGraphs OBP.
- **First-inning total run environment:** compute from historical first-inning R/G (available via Retrosheet or Statcast game-state queries).

Free throughout — all derivable from `pybaseball` + MLB Stats API.

---

## 14. Recommended Install / Dependency List

```
pip install pybaseball MLB-StatsAPI requests pandas numpy
```

Optional:
```
pip install beautifulsoup4 lxml      # for any HTML scraping (RotoWire, SBR, etc.)
pip install python-dateutil pytz     # timezone math for travel/rest
```

API keys needed (all have free tiers):
- **The Odds API** — sign up for free key at the-odds-api.com
- **OpenWeatherMap** (optional backup) — free key at openweathermap.org

No key needed for: Open-Meteo, MLB Stats API, Baseball Savant, FanGraphs, Baseball Reference, NOAA.

---

## 15. Caching & Freshness Strategy

Most of these indicators don't need live polling. Recommended refresh cadence:

| Data | Refresh |
|---|---|
| Season-to-date leaderboards (SIERA, wRC+, FIP, etc.) | Daily, post-midnight ET |
| 30-day rolling splits | Daily |
| Park factors | Weekly |
| Schedule / probable pitchers | Hourly on game day |
| Confirmed lineups | Every 10-15 min, starting 4 hr pre-first-pitch |
| Bullpen rest / last-3-day usage | After each completed game |
| Weather | Every 2 hr day-of; hourly in last 3 hr pre-first-pitch |
| Odds / line moves | Every 15-30 min; more often near close if the free-tier budget allows |
| Umpire assignment | Once at lineup release |

Cache everything locally (SQLite works fine for a desktop app) so you're not re-hitting sources you already have.

---

## 16. One-Line Summary per Source

- **`pybaseball` (Python)** — wrapper for Savant + FanGraphs + Baseball Reference; gets you ~80% of the sabermetric inputs free
- **MLB Stats API** — official schedule, lineups, probable pitchers, umpires, game state, box scores; free, no key
- **Baseball Savant** — Statcast-native: xwOBA, Barrel%, OAA, framing runs, park factors
- **FanGraphs** — SIERA, xFIP, wRC+, DRS, BsR; leaderboards with date filters
- **Open-Meteo** — free, no-key weather forecasts
- **The Odds API** — free-tier MLB odds across books (500 req/mo)
- **Umpire Scorecards** — public umpire-performance metrics
- **Retrosheet** — deep historical backfill, free bulk CSV

That set covers every indicator in the Run Line, Moneyline, and Totals frameworks without any paid subscription.
