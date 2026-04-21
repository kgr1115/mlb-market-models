# MLB Betting Predictor

A Python + FastAPI + React project that predicts daily MLB bets across three markets — **Moneyline**, **Run Line**, and **Totals (Over/Under)** — scored with per-pick model probability, market-implied probability, edge, EV, and a 0–100 confidence bar.

The project was built end-to-end with AI assistance: research → feature design → implementation → backtest harness → honest diagnostics → UI. The repository is as much a record of the *process* — what worked, what didn't, and why — as it is a model.

---

## Current status (2026-04-20)

This is a research project, not a shipped product.

- **Run Line** shows a positive out-of-sample test ROI after a targeted feature-wiring fix (details in *Iteration history* below). Numbers from the 2018–22 train / 2023–25 test split: `+23.6% ROI` at the current gate (n=4,115) and `+31.2% ROI` tuned (n=1,407). Still needs verification through the full engine path before any real-money use.
- **Moneyline** and **Totals** have **negative** out-of-sample test ROI at every threshold combination we grid-searched. The feature set does not produce information that DraftKings closing lines haven't already priced in. They're kept in the repo as working scaffolding for future feature work, not as picks to follow.
- The web UI renders predictions and a backtest dashboard, but the live-data pipeline still falls back to a mock-game generator in several places. The *shape* is production-quality; the *data wiring* is partial.

"Useful as a framework, honest about its edge" is the accurate one-line summary.

---

## What it does

1. **Ingests** — starters, lineups, team stats, weather, umpires, and odds from free sources (`data/`).
2. **Normalizes** — everything lands in a small set of dataclasses (`TeamStats`, `GameContext`, `MarketData`) so all three predictors consume one object graph.
3. **Predicts** — three market-specific models (`predictors/moneyline.py`, `run_line.py`, `totals.py`) each compute a weighted sum over ~5–8 feature families, then convert to a probability and compare against the vig-stripped market line.
4. **Scores confidence** — a single `confidence_score(edge, family_agreement, input_certainty, variance_penalty)` function in `predictors/shared.py` maps to a 0–100 bar with LOW / LEAN / MEDIUM / HIGH labels.
5. **Backtests** — historical games × historical odds × the same predictor code path, written to per-season JSON files for the web UI.
6. **Serves** — a FastAPI backend at `web/backend/api.py` and a single-file React frontend at `web/frontend/index.html` render daily picks, per-game detail, and the backtest dashboard.

---

## Project layout

```
BaseballBettingPredictor/
├── predictors/                  Three market predictors + shared types
│   ├── shared.py                Dataclasses, league averages, confidence math
│   ├── moneyline.py             7-family weighted model → win probability
│   ├── run_line.py              5-family model → P(margin ≥ 2) via normal CDF
│   ├── totals.py                8-family model → Poisson run projection
│   ├── team_totals.py, f5.py, nrfi.py   Soft / secondary markets
│   ├── slow_features.py         Catcher framing × ump, travel shock, etc.
│   └── narrow_gate.py           Per-market edge/confidence gatekeeping
│
├── data/                        Ingestion + caching
│   ├── lineups_*.py             MLB Stats API: confirmed & projected lineups, injuries
│   ├── odds_*.py                The Odds API + book-specific scrapers + SQLite cache
│   ├── projections_*.py         FanGraphs leaderboards, DRA, Monte Carlo, rollups
│   ├── rolling_stats.py         30-day rolling FanGraphs/Savant pulls via pybaseball
│   ├── weather.py               Open-Meteo forecasts
│   └── weather_history.py       Open-Meteo archive API for backtesting
│
├── backtest/                    Historical replay harness
│   ├── engine.py                Load games, build inputs, run predictors, grade, report
│   ├── historical_games.py      Retrosheet parser
│   ├── historical_odds.py       SBR XLSX parser (2018/2019/2021)
│   ├── historical_odds_community.py  DraftKings closing lines 2021–2025
│   ├── historical_stats.py      Prior-year baseline stats
│   ├── report.py                Convert results → /api/backtest/summary JSON
│   └── ungated_predictions.csv  Every raw pick across 2018–2025 (for feature research)
│
├── bet_selection/               Kelly sizing, ranker, bankroll, slip builder
├── models/                      Experimental: closing-line and opening-line value models
│
├── web/                         FastAPI backend + React frontend + launcher
│   ├── backend/api.py           Endpoints: /games/today, /games/{id}, /backtest/summary
│   ├── backend/live_data.py     Background poller for live odds + lineups
│   ├── backend/mlb_data.py      Live schedule + starter fetch
│   ├── backend/backtest_results_*.json   Pre-computed per-season backtest output
│   ├── frontend/index.html      Single-file React app (Tailwind + Recharts via CDN)
│   ├── run.py                   Launcher (LAN-exposed by default, --local to restrict)
│   ├── README.md                Web-specific quick start
│   └── requirements.txt
│
├── predict.py                   Demo: builds a sample Yankees @ Dodgers matchup
├── run_backtest_2024_2025.py    Recent-seasons backtest driver
├── run_multi_backtest.py        Full 2018–2025 backtest driver
├── tune_thresholds.py           Train/test grid search over (min_edge, min_conf)
├── analyze_families.py          Per-family correlation + agree/disagree-ROI diagnostic
├── collect_predictions.py       Ungated prediction collector for feature research
│
├── data_sources.md              Free-source mapping for every indicator used
├── .env.example                 Template: ODDS_API_KEY, HOST, PORT, cache paths
└── *.docx                       Research docs per market + backtest report
```

---

## The three predictors

Every predictor takes the same four inputs and returns the same `PredictionResult`:

```python
PredictionResult = predict_moneyline(home: TeamStats, away: TeamStats,
                                     ctx: GameContext, market: MarketData)
```

Fields on `PredictionResult`: `market`, `pick`, `odds`, `model_prob`, `implied_prob`, `edge`, `expected_value_per_unit`, `confidence` (0–100), `confidence_label`, and a `detail` dict for UI display.

### Moneyline — 7 families

Weighted family scores from the code in `predictors/moneyline.py`:

| Family        | Weight | Inputs |
| ------------- | ------ | ------ |
| Pitcher       | 35%    | SIERA, xFIP, K-BB%, CSW%, xwOBA-against, IP/GS, rolling ERA |
| Bullpen       | 20%    | FIP, high-leverage K%, shutdown%, meltdown% |
| Offense       | 20%    | wRC+, wOBA, xwOBA, ISO, Barrel%, K% |
| Defense       | 7%     | OAA, DRS, catcher framing, BsR |
| Situational   | 8%     | Form (last 10/20), rest days, travel miles |
| Environment   | 5%     | Park run factor, wind, temperature, humidity |
| Market        | 5%     | Line movement, public ticket/money split |

The combined score is mapped through a logistic to a home-win probability, with a **heavy-favorite penalty** (documented in `project_moneyline_framework.md`) because large favorites historically under-perform their model-implied probabilities on closing lines.

### Run Line — 5 families

Run line is inherently about *margin*, not win probability. `predictors/run_line.py` builds a weighted family diff, converts to expected margin (`FAMILY_DIFF_TO_RUNS = 0.55`), applies a home-field lift of +0.18 runs, then computes `P(margin ≥ 2)` via a normal CDF with a 4-run margin standard deviation.

| Family   | Weight |
| -------- | ------ |
| Pitcher  | 30%    |
| Bullpen  | 20%    |
| Offense  | 25%    |
| Context  | 15%    |
| Market   | 10%    |

The context family emphasizes slugging × park HR factor, wind direction × ISO differential, umpire runs/game, and whether lineups are confirmed. The market family uses an opener-vs-current line-movement fallback when public-ticket data isn't available — this single wiring fix is what made RL the one market with positive test ROI (see *Iteration history*).

### Totals (Over/Under) — 8 families

`predictors/totals.py` projects an expected total via the weighted family sum, then models the total as `Normal(μ, σ=3.3)` and computes `P(total > line)` against the vig-stripped market.

| Family   | Weight |
| -------- | ------ |
| Pitcher  | 25%    |
| Bullpen  | 15%    |
| Offense  | 20%    |
| Park     | 15%    |
| Weather  | 12%    |
| Umpire   | 5%     |
| Pace     | 3%     |
| Market   | 5%     |

Park and weather are first-class inputs — Totals is the market where environmental factors move the line most. There's a **Coors overlay**: when the venue is Coors Field, the model adds +0.4 runs and widens σ by +0.6 to reflect the altitude's systematic effect on both mean and variance.

### Confidence scoring

A single function in `predictors/shared.py`:

```python
confidence_score(edge, family_agreement, input_certainty,
                 variance_penalty=0.0, extra_penalty=0.0) → (0–100, label)
```

- **Edge** — raw size of model prob minus implied prob.
- **Family agreement** — fraction of families whose sign matches the chosen side.
- **Input certainty** — how many of the required inputs are confirmed (starters, lineups, weather inside the forecast window) vs. projected.
- **Variance penalty** — market-specific stddev; e.g. one-run RL games are ~28% of the corpus, so RL gets a heavier penalty.
- **Extra penalty** — situational dings for heavy favorites, short-rest bullpens, Coors, etc.

The 0–100 score bins into LOW / LEAN / MEDIUM / HIGH.

---

## Data sources

All free. See `data_sources.md` for the full mapping.

| Source              | Used for                                                           | Cost |
| ------------------- | ------------------------------------------------------------------ | ---- |
| MLB Stats API       | Schedule, probable pitchers, umpires, confirmed lineups, game state | Free |
| pybaseball          | SIERA, xFIP, wRC+, K-BB%, xwOBA, ISO, Barrel%, OAA, DRS, park factors | Free |
| Baseball Savant     | Statcast xwOBA, Barrel%, OAA, framing runs (via pybaseball)        | Free |
| FanGraphs           | Leaderboards with date filters (via pybaseball)                    | Free |
| The Odds API        | Moneyline, run line, totals across books                           | Free tier (500/mo) |
| Open-Meteo          | Weather forecasts + historical archive                             | Free, no key |
| Umpire Scorecards   | Umpire tendencies (runs/game, K-zone)                              | Free |

Odds are cached in SQLite (`bbp_cache.sqlite`, keyed by `event_id + timestamp`) and the weather archive is cached to `data/cache/weather_history.json` so backtests don't repeatedly hit the Open-Meteo API.

---

## The backtest

`backtest/engine.py` is the single entry point for historical replay. It:

1. Loads historical games from Retrosheet (`backtest/historical_games.py`).
2. Loads historical odds — SBR XLSX for 2018/2019/2021 (`historical_odds.py`) and a community-scraped DraftKings dataset for 2021–2025 (`historical_odds_community.py`).
3. Optionally prewarms weather from the Open-Meteo archive when `run_backtest(..., with_weather=True)`.
4. For each game, builds `TeamStats` + `GameContext` + `MarketData` and runs the same three `predict_*` functions that serve live picks.
5. Grades each pick against the actual outcome and produces per-market ROI, per-confidence-band breakdowns, and an equity curve.
6. Writes JSON under `web/backend/` so the `/api/backtest/summary` endpoint can render the dashboard without re-running the loop.

Two drivers:

```bash
python run_backtest_2024_2025.py     # recent seasons only
python run_multi_backtest.py         # full 2018–2025
```

Two research utilities, used heavily during the iteration history below:

```bash
python collect_predictions.py        # dump every raw pick across all seasons → CSV
python tune_thresholds.py            # train/test grid search over (min_edge, min_conf)
python analyze_families.py           # per-family correlation + agree/disagree-ROI
```

---

## Backtest numbers

Pulled verbatim from `web/backend/backtest_results*.json`. These are the **gated** bets — picks that cleared `MIN_EDGE` and the per-market confidence threshold — not the full universe of predictions. Assumes flat 1-unit staking on each bet; the Kelly-sized version in `bet_selection/` is separate.

Two important caveats before reading the numbers:

1. **These are pooled across 2018–2025** — in-sample and out-of-sample mixed. The honest out-of-sample picture is the train/test split in the *Iteration history* section below. Moneyline and Totals look less bad here than they do there.
2. **Run Line benefits from the 2026-04-20 market-family fix** across the full corpus. The fix is legitimate (it was validated on a proper 2018–22 / 2023–25 split before these JSONs were regenerated), but the +20% pooled RL ROI should be read with that in mind.

### Pooled results (2018-03-29 → 2025-08-17)

| Scope              | Bets   | Win%  | Units | ROI%   |
| ------------------ | ------ | ----- | ----- | ------ |
| **All markets**    | 13,010 | 58.9% | +1,458.8 | **+11.21%** |
| Moneyline          | 4,533  | 47.6% | −113.2   | −2.50% |
| **Run Line**       | 7,907  | 66.1% | +1,616.5 | **+20.44%** |
| Totals             | 570    | 48.0% | −44.5    | −7.81% |

| Confidence | Bets  | Win%  | ROI%    |
| ---------- | ----- | ----- | ------- |
| MEDIUM     | 6,706 | 55.3% | +5.12%  |
| **HIGH**   | 6,286 | 62.7% | **+17.74%** |

LOW and LEAN are 0-bet — the gates filter them out.

### Per-season breakdown

| Year | Bets  | Win%  | Pooled ROI | ML ROI (n)     | RL ROI (n)     | Totals ROI (n) |
| ---- | ----- | ----- | ---------- | -------------- | -------------- | -------------- |
| 2018 | 1,141 | 53.9% | +2.73%     | +5.71% (554)   | +0.94% (495)   | −5.60% (92)    |
| 2019 | 1,214 | 54.1% | +1.54%     | +2.68% (620)   | +2.56% (497)   | −10.91% (97)   |
| 2021 | 2,281 | 57.5% | +10.28%    | −8.71% (830)   | +22.68% (1,357) | −1.02% (94)    |
| 2022 | 2,223 | 59.7% | +12.62%    | −4.47% (693)   | +22.26% (1,444) | −11.66% (86)   |
| 2023 | 2,207 | 62.1% | +16.47%    | +0.43% (701)   | +25.21% (1,473) | −33.09% (33)   |
| 2024 | 2,334 | 58.7% | +10.51%    | −10.88% (740)  | +22.88% (1,495) | −16.40% (99)   |
| 2025 | 1,610 | 62.7% | +17.72%    | +4.92% (395)   | +22.38% (1,146) | +13.56% (69)   |

Reading the table left-to-right across a season, the pattern is consistent: RL carries the pool, ML oscillates around break-even-to-negative, and Totals is noisy on small samples (86–99 bets most years, just 33 in 2023). 2025 is partial (through mid-August) and the only year every market was positive — too small a sample to conclude anything from.

The honest way to read this: the pooled +11.21% is mostly an RL story layered on a corpus where the RL market-family fix was wired in before these JSONs were generated. The **proper** bet-or-not decision is the train/test split in the next section, and it says ship RL only.

---

## Web app

```bash
pip install -r web/requirements.txt
python web/run.py                    # LAN-exposed (default), auto-opens browser
python web/run.py --local            # 127.0.0.1 only
python web/run.py --port 9000
```

The launcher detects the LAN IP and prints both a local and a network URL so the app is reachable from a phone on the same Wi-Fi.

**Stack:**

- **Backend:** FastAPI + Uvicorn. Single file at `web/backend/api.py`. Four endpoints: `/api/health`, `/api/games/today`, `/api/games/{game_id}` (where `game_id = YYYY-MM-DD-AWAY@HOME`), and `/api/backtest/summary`. A background poller (`live_data.py`) starts on app startup unless `BBP_DISABLE_POLLER=1`.
- **Frontend:** Single-file React 18 app loaded from CDNs (React, Recharts, Tailwind, Babel standalone). No build step. Three tabs: **Daily Picks** (filter by confidence, sort by confidence/edge/first pitch), **Game Detail** (full matchup + factor breakdown + raw team stats), and **Backtest** (KPI strip, equity curve, by-market and by-confidence-band charts).

More detail in `web/README.md`.

---

## Setup

Prereqs: Python 3.10+ (the repo has been run against both 3.10 and 3.14).

```bash
# 1. Clone and enter the repo
cd C:\Projects\BaseballBettingPredictor

# 2. Install Python deps
pip install -r web/requirements.txt
pip install pybaseball pandas numpy openpyxl requests python-dateutil pytz beautifulsoup4 MLB-StatsAPI

# 3. Set up .env
copy .env.example .env
# Edit .env to add your Odds API key (free tier at https://the-odds-api.com/)

# 4. Run the demo
python predict.py

# 5. Run the web app
python web/run.py
```

The demo (`predict.py`) builds a sample Yankees @ Dodgers matchup with realistic stats and prints all three predictions with their confidence scores — a fast sanity check that the predictor code path works end-to-end without any network dependencies.

---

## Iteration history

This is the part that may be most useful for a portfolio reader. The interesting work wasn't writing the features; it was figuring out *which features actually produced edge* on out-of-sample data, and being willing to publish negative findings instead of a polished story.

### 1. Initial result looked great — and was wrong

A narrow backtest on SBR closing lines for 2018, 2019, and 2021 produced a pooled ROI of **+1.78%**. Feature weights and confidence thresholds had been tuned on exactly that same corpus.

### 2. Expanding the corpus collapsed the edge

Adding DraftKings closing lines for 2021–2025 (via a community scraper) expanded the grading corpus to 9,979 bets across seven seasons. Pooled ROI moved from **+1.78% to −2.81%**. Per-season ROI made the pattern obvious:

| Year  | ROI      | Notes                                  |
| ----- | -------- | -------------------------------------- |
| 2018  | +3.73%   | SBR, juiced-ball era                   |
| 2019  | +2.07%   | SBR, juiced-ball era                   |
| 2021  | −7.74%   | Community DK (was +0.15% on SBR)        |
| 2022  | −4.31%   |                                        |
| 2023  | −1.08%   | Pitch-clock + shift-ban era begins     |
| 2024  | −9.79%   |                                        |
| 2025  | +3.43%   | Partial (through mid-August)           |

Two independent issues: thresholds had been tuned on the 2018–2021 sample, and DraftKings' "flipped run line" postings (ML favorite given +1.5 on near-pick'em games) had grown from 13% to 22% of the slate — and the expected-margin framework handled them badly.

### 3. Formal train/test split confirmed the overfit

Rather than tweak thresholds to chase the new numbers, we ran an *ungated* collection of every pick across 2018–2025 (40,971 rows / 34,152 graded bets via `collect_predictions.py`), split train = 2018/2019/2021/2022 and test = 2023/2024/2025, and grid-searched `(min_edge, min_confidence)` on the train set (`tune_thresholds.py`).

| Market    | Train ROI | Test ROI | Current-live test ROI | Null (bet all) test ROI |
| --------- | --------- | -------- | --------------------- | ----------------------- |
| Moneyline | +2.52%    | **−5.21%** | −3.39%                | −3.07%                  |
| Run line  | −2.38%    | **−7.61%** | −3.27%                | −3.74%                  |
| Totals    | +0.62%    | **−4.50%** | −5.69%                | −4.17%                  |

No threshold combination produced positive out-of-sample ROI on any market. The live totals filter was *worse* than betting everything. That's the fingerprint of overfit plus structural no-edge — threshold optimization can't create edge where none exists.

The conclusion logged in memory: *"Don't waste more effort on threshold sweeps, new confidence formulas, or bet-sizing schemes. The raw edge isn't there. Meaningful next steps would need to change features, not tuning."*

### 4. Per-family diagnostics found which families were silent or anti-predictive

`analyze_families.py` computes, for each family, the Pearson correlation of its side-aligned score with the outcome, plus the agree-ROI minus disagree-ROI on the train set.

- **Moneyline:** pitcher, offense, and bullpen carried the signal; defense and situational produced literal zeros because the backtest harness didn't populate them.
- **Run line:** pitcher was the *only* family with meaningful signal. Market was all zeros. RL was effectively a one-family model — which explained why it was the weakest market.
- **Totals:** pitcher and bullpen were **anti-signal** on DK closing lines (ROI deltas −3.79% and −6.03%). Closing lines already price in public pitcher/bullpen reputation, so when the model agrees that pitching is strong, it's piling onto sharp lines with no additional edge.

A hand-flip of the totals pitcher/bullpen signs was tested and **reverted** — it made the null and current-gate ROIs *worse*, only marginally improving the tuned cell. Lesson: per-family sign-flipping doesn't reliably translate into test wins. The real path for totals would be a calibrated regression, not a hand-tuned coefficient flip.

### 5. One targeted feature-wiring fix flipped Run Line from negative to positive

The RL market family had been inert — requiring public-ticket data that wasn't present in the historical corpus. A small fix in `predictors/run_line.py::market_score_rl` added a line-movement fallback: when a `openingLine` price is present in the community odds dataset, compute the movement direction and use it as a weak-but-real sharp-money proxy.

Same train/test split, same everything else:

| Market    | Null test ROI | Current-gate test ROI | Tuned test ROI                                |
| --------- | ------------- | --------------------- | --------------------------------------------- |
| ML        | −3.16%        | −3.32% (n=1,898)      | −3.64% (n=119) — overfit                      |
| **RL**    | **+17.49%**   | **+23.63% (n=4,115)** | **+31.21% (n=1,407)** — tuned (edge≥0.010, conf≥85) |
| Totals    | −3.88%        | −2.38% (n=156)        | −2.38% (n=156) — overfit                      |

A previously-zero family, when given even a weak non-zero signal, flipped 10–20% of the close-call RL picks toward the post-movement side, and those flipped picks won disproportionately on the test set. Per-family analysis confirmed: RL market family train correlation +0.066, agree-ROI +12.86% vs disagree-ROI +2.45%, a +10.4 pt delta over 2,903 agreements.

Important caveats:

- The same fallback now fires for totals, but **anti-signals** there (agree-ROI −2.31% vs disagree-ROI +2.36% on train). The sign-flipping finding from step 4 says we can't just flip it.
- The null RL test ROI of +17.49% on 5,725 bets is high enough that independent verification through the regular engine backtest path — not the collector — is still required before any live deployment.
- ML and Totals remain at defensive gates; both still show negative test ROI.

### 6. Infrastructure investments that paid off

- **Ungated prediction corpus** (`backtest/ungated_predictions.csv`) — dumping every raw pick keyed to (event, market) means we can replay any feature experiment without re-running the full backtest.
- **Historical weather client** (`data/weather_history.py`) — Open-Meteo archive with on-disk JSON cache, shares `VENUES` and `_classify_wind` with the live weather module for a single source of orientation truth.
- **`_BYPASS_GATES` flag** — kept wired in the predictors so the ungated collector can always be re-run as features change.
- **Self-documenting FIXMEs** — e.g. the umpire-per-game fallback in `live_data.py` now uses `_LEAGUE["ump_runs_per_game"]` with a FIXME pointing to the missing data source, rather than a silent hardcode.

---

## What this demonstrates

- **Designing a multi-market sabermetric pipeline** — three predictors, one normalized input graph, one confidence function, one web UI.
- **Writing a historical backtest that matches the live code path** — same `predict_*` functions grade the past and serve the present.
- **Doing the grown-up version of model evaluation** — train/test splits, per-family diagnostics, null baselines, rather than just pointing at a training-set ROI.
- **Publishing negative results** — the repo says "ML and Totals don't have edge" because that's what the data says.
- **Making targeted investments** — one feature-wiring fix produced the project's only positive out-of-sample ROI.
- **Using AI as a collaborator, not a magic wand** — research drafts, code generation, diagnostics, documentation, and editing were all AI-assisted, but the engineering decisions (what to measure, when to stop tuning, whether to ship a hand-flipped sign) were the part that actually mattered.

---

## How I worked with AI on this

I'm not a professional sabermetrician or quant — I built this as a way to learn by doing, with AI as a collaborator across every phase. The honest account of how the division of labor shook out:

### What AI was good at

- **Breadth of research.** For each market, I had AI survey the sabermetric literature for indicators with documented predictive value and draft a `*_Indicators_Research.docx` before any code was written. That's the kind of work that would take me a week of reading; it took an afternoon.
- **Turning design into a typed surface.** Once the family/weight structure was decided, AI drafted `predictors/shared.py` (dataclasses + league averages + math helpers) and the three `predict_*` modules in a consistent style. The code came out long but legible, and the "one normalized input graph feeds all three models" property was easy to preserve because a single pair of hands (well, one pair and a language model) wrote all of it.
- **Wiring the backtest harness.** Retrosheet parser, SBR/community odds loaders, per-season grading, JSON output, equity curves — all large blocks of mechanical code that AI produced quickly and accurately enough that the work was mostly review, not reimplementation.
- **Diagnostic scripts.** `collect_predictions.py`, `tune_thresholds.py`, `analyze_families.py` — each was a "build me a tool that answers this specific question" request. Short turnaround, correct the first time, reusable afterward.
- **Writing this README and the research docs.** The structure, tables, and prose were AI-drafted and I corrected the specifics against the code.

### What I had to do myself

- **Decide what to measure.** The most important call in the whole project was "stop tuning thresholds, run a proper train/test split." AI will happily keep iterating on whatever metric you're pointing at. Redirecting attention — *this number is flattering, I need a harder test* — is a judgment call that has to come from the human.
- **Kill overfit findings.** The initial +1.78% headline was real in the sense that the code produced it; it was also a trap. Expanding the corpus to include DraftKings 2021–2025 data was a decision to make the evaluation harder on purpose. Without that push, the project would have "shipped" a model with no out-of-sample edge.
- **Revert the sign-flip.** When per-family analysis showed totals pitcher/bullpen were anti-signal, the obvious move was to flip their signs. AI executed the flip cleanly. The resulting test-set numbers got *worse* on the null and current-gate ROIs, only marginally better on the tuned cell. Calling that experiment a failure and reverting — rather than trying to rationalize a partial improvement — was a human call.
- **Spot the one wire worth fixing.** The RL market family was inert in the backtest harness because the feature code required public-ticket data that the historical corpus didn't have. Reading the per-family diagnostic carefully enough to notice that "all zeros" meant "never fires, not never-predictive" was the realization that led to the opener-line-movement fallback — and the only positive out-of-sample result in the project.
- **Stay honest in the README.** The easy version of this document says "+11.21% ROI across 13,010 bets" and leaves it there. The real version has to also say "but the proper train/test split says ML and Totals have no edge, and the RL number deserves one more round of verification."

### How I'd describe the collaboration

AI compressed the implementation loop from weeks to hours, which let me spend the saved time on evaluation rather than typing. Evaluation is where the actual signal-to-noise work happens, and evaluation is where I was still the load-bearing participant. I don't think this project existed *because of* AI — the problem, the feature design, and the willingness to publish negative findings were mine — but it's several months further along than it would be without it.

If there's one portable lesson here, it's the one implied by the bullet above about killing overfit findings: **AI is extraordinary at producing more work; the scarce skill is knowing which piece of work actually answers the question.** That's the skill the project tried to exercise.

---

## Known limitations and next steps

- **Live-data wiring is partial.** `web/backend/api.py` falls back to a mock game generator in places where the full live pipeline (lineups × odds × weather × projections) isn't yet assembled. The dataclass contract is stable; the ingestion glue isn't all there.
- **No automated tests.** The ungated prediction CSV + train/test grid search acts as an integration harness, but there's no unit test suite yet.
- **Totals fix path is clear but not done.** (1) populate weather/umpire/pace/market in the backtest harness (4 of 8 families are currently zero), (2) fit family weights and signs from data via a calibrated regression rather than hand-chosen coefficients.
- **Moneyline remains flat.** The closing line absorbs pitcher/bullpen/offense reputation cleanly. Meaningful edge would probably require pre-pitch Savant features, bullpen availability granularity, or a switch from closing to opening lines.
- **No live execution** — by design. The bet-selection modules (`bet_selection/kelly.py`, `ranker.py`, `bankroll.py`, `slip.py`) produce a ranked, Kelly-sized slip, but nothing in the project places wagers with a sportsbook.

---

## Files you can read to dig deeper

- `data_sources.md` — every indicator mapped to a free source.
- `predictors/shared.py` — league averages, dataclasses, confidence math.
- `predictors/{moneyline,run_line,totals}.py` — each model is ~300–500 lines, heavily commented.
- `backtest/engine.py` — the historical replay loop.
- `tune_thresholds.py`, `analyze_families.py`, `collect_predictions.py` — the diagnostic scripts behind the iteration history.
- `Architecture_Expansion_v1.docx`, `Backtest_Report_2025_2026_v3.docx`, and the per-market research docs (`Moneyline_Indicators_Research.docx`, `RunLine_Indicators_Research.docx`, `Totals_Indicators_Research.docx`) — original design notes.

---

## License / use

Personal research project. No live-betting integration. Data-source terms of service apply to any of the scrapers in `data/`.
