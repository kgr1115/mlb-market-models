"""
FastAPI backend for the MLB Betting Predictor web UI.

Game schedule, team/pitcher matchups, game status, and final scores all come
from the free MLB Stats API (see mlb_data.py). Advanced stats (SIERA, wRC+,
etc.) are still seeded synthetically — they're deterministic per gamePk so
the same matchup always produces the same prediction. When the pybaseball
stats pipeline is wired up, replace `_synth_team_stats` with real pulls.

Endpoints
---------
GET /api/health
GET /api/games/today?d=YYYY-MM-DD       (defaults to today in US/Eastern)
GET /api/games/{game_id}                 game_id = YYYY-MM-DD-AWAY@HOME
GET /api/backtest/summary
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Make the project root importable so we can use `predictors`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictors import (  # noqa: E402
    PitcherStats, BullpenStats, OffenseStats, DefenseStats,
    TeamStats, GameContext, MarketData,
    predict_all, narrow_picks, surviving_only, gate_summary,
)
from bet_selection import (  # noqa: E402
    build_slip, BankrollPolicy,
)
from models.closing_line import (  # noqa: E402
    predict_closing_line_ml, predict_closing_line_total, predict_closing_line_rl,
    coefficients_loaded,
)
from data.line_shop import shop_all  # noqa: E402
from data.odds_cache import OddsCache  # noqa: E402
from web.backend import mlb_data  # noqa: E402
from web.backend import live_data  # noqa: E402

log = logging.getLogger("api")

app = FastAPI(title="MLB Betting Predictor", version="0.3.0-live")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _start_poller() -> None:
    # Opt-out via env for local dev where the network isn't available.
    if os.environ.get("BBP_DISABLE_POLLER") == "1":
        log.info("Background poller disabled (BBP_DISABLE_POLLER=1)")
        return
    try:
        live_data.start_background_poller()
    except Exception as e:
        log.exception("Failed to start background poller: %s", e)


@app.on_event("shutdown")
def _stop_poller() -> None:
    try:
        live_data.stop_background_poller()
    except Exception:
        pass


# =============================================================================
# Synthetic-but-stable advanced stats
# =============================================================================
# Each team and each pitcher gets deterministic stats seeded from its name/id.
# This gives consistent predictions until the real stats pipeline is wired in.

def _seed_rng(*keys: Any) -> random.Random:
    h = hashlib.md5("|".join(str(k) for k in keys).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


# Rough quality tiers by team (home-advantage-neutral, mid-season feel).
# Positive numbers = stronger team. Reset every season in real life.
_TEAM_QUALITY: dict[str, float] = {
    "LAD": 0.80, "NYY": 0.65, "ATL": 0.60, "HOU": 0.50, "PHI": 0.55,
    "BAL": 0.55, "SEA": 0.35, "MIN": 0.30, "MIL": 0.30, "BOS": 0.20,
    "TEX": 0.15, "SD": 0.20, "SDP": 0.20, "TOR": 0.20, "NYM": 0.25,
    "ARI": 0.10, "AZ": 0.10, "CLE": 0.35, "TB": 0.20, "TBR": 0.20,
    "STL": 0.05, "CHC": 0.10, "KC": 0.10, "KCR": 0.10, "CIN": 0.00,
    "DET": 0.10, "SF": 0.00, "SFG": 0.00, "LAA": -0.10, "ANA": -0.10,
    "WSH": -0.30, "WAS": -0.30, "MIA": -0.40, "PIT": -0.20,
    "OAK": -0.50, "ATH": -0.50,
    "CHW": -0.55, "CWS": -0.55, "COL": -0.45,
}


def _team_quality(abbr: str) -> float:
    return _TEAM_QUALITY.get(abbr.upper(), 0.0)


def _synth_pitcher(pitcher_name: str, pitcher_id: Any, hand: str,
                   team_quality: float) -> PitcherStats:
    rng = _seed_rng("P", pitcher_id or pitcher_name or "tbd")
    # Better teams tend to have slightly better probable starters
    tier = rng.uniform(-0.7, 0.7) + team_quality * 0.4
    siera = 4.05 - tier * 0.8 + rng.uniform(-0.25, 0.25)
    return PitcherStats(
        name=pitcher_name or "TBD",
        throws=hand if hand in ("L", "R") else "R",
        siera=siera,
        xfip=siera + rng.uniform(-0.15, 0.25),
        k_bb_pct=0.135 + tier * 0.04 + rng.uniform(-0.01, 0.015),
        csw_pct=0.285 + tier * 0.015 + rng.uniform(-0.008, 0.008),
        xwoba_against=0.320 - tier * 0.018 + rng.uniform(-0.008, 0.010),
        ip_per_gs=5.30 + tier * 0.35 + rng.uniform(-0.3, 0.3),
        rolling_30d_era=max(1.8, siera + rng.uniform(-0.5, 0.8)),
    )


def _synth_bullpen(team_abbr: str, quality: float) -> BullpenStats:
    rng = _seed_rng("BP", team_abbr)
    q = quality + rng.uniform(-0.3, 0.3)
    return BullpenStats(
        fip=4.00 - q * 0.50 + rng.uniform(-0.2, 0.2),
        hi_lev_k_pct=0.235 + q * 0.030 + rng.uniform(-0.008, 0.008),
        meltdown_pct=max(0.06, 0.14 - q * 0.035 + rng.uniform(-0.015, 0.015)),
        shutdown_pct=min(0.45, 0.30 + q * 0.045 + rng.uniform(-0.015, 0.015)),
        closer_pitches_last3d=rng.randint(0, 35),
        setup_pitches_last3d=rng.randint(0, 40),
        days_since_closer_used=rng.randint(0, 3),
    )


def _synth_offense(team_abbr: str, quality: float) -> OffenseStats:
    rng = _seed_rng("OFF", team_abbr)
    q = quality + rng.uniform(-0.25, 0.25)
    wrc = 100 + q * 16
    return OffenseStats(
        wrc_plus=wrc,
        wOBA=0.320 + (wrc - 100) / 100 * 0.030,
        xwOBA=0.320 + (wrc - 100) / 100 * 0.028 + rng.uniform(-0.004, 0.004),
        obp=0.320 + (wrc - 100) / 100 * 0.025,
        iso=0.165 + (wrc - 100) / 100 * 0.050,
        barrel_pct=0.080 + (wrc - 100) / 100 * 0.025,
        k_pct=max(0.17, 0.225 - (wrc - 100) / 100 * 0.012),
        top_of_order_obp=0.340 + (wrc - 100) / 100 * 0.030,
    )


def _synth_defense(team_abbr: str, quality: float) -> DefenseStats:
    rng = _seed_rng("DEF", team_abbr)
    q = quality + rng.uniform(-0.3, 0.3)
    return DefenseStats(
        oaa=q * 10 + rng.uniform(-4, 4),
        drs=q * 12 + rng.uniform(-5, 5),
        catcher_framing_runs=q * 4 + rng.uniform(-2, 2),
        bsr=q * 3 + rng.uniform(-2, 2),
    )


def _synth_form(team_abbr: str, quality: float) -> tuple[float, float]:
    rng = _seed_rng("FORM", team_abbr)
    l10 = 0.500 + quality * 0.15 + rng.uniform(-0.05, 0.05)
    l20 = 0.500 + quality * 0.10 + rng.uniform(-0.05, 0.05)
    return max(0.2, min(0.8, l10)), max(0.2, min(0.8, l20))


def _synth_team(side: dict, is_home: bool, game_pk: Any) -> TeamStats:
    abbr = side["abbr"]
    quality = _team_quality(abbr)
    l10, l20 = _synth_form(abbr, quality)
    rng = _seed_rng("G", game_pk, side["abbr"])
    return TeamStats(
        name=side["name"] or abbr,
        is_home=is_home,
        pitcher=_synth_pitcher(side["pitcher_name"], side["pitcher_id"],
                               side["pitcher_hand"], quality),
        bullpen=_synth_bullpen(abbr, quality),
        offense=_synth_offense(abbr, quality),
        defense=_synth_defense(abbr, quality),
        form_last10_win_pct=l10,
        form_last20_win_pct=l20,
        rest_days=rng.choice([0, 1, 1, 1, 2]),
        travel_miles_72h=0 if is_home else rng.choice([0, 400, 900, 1500, 2200]),
        lineup_confirmed=side["pitcher_name"] != "TBD",
        starter_confirmed=side["pitcher_name"] != "TBD",
    )


def _synth_context(g: dict) -> GameContext:
    home_abbr = g["home"]["abbr"]
    rng = _seed_rng("CTX", g.get("game_pk"), home_abbr)
    pf_run, pf_hr = mlb_data.get_park_factors(home_abbr)
    roof = "closed" if (mlb_data.is_domed(home_abbr) and rng.random() < 0.5) else "none"

    # Day/night from the UTC gameDate (hour < 19Z ~ day game)
    day_game = False
    iso = g.get("game_date_utc") or ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        day_game = dt.hour < 19
    except Exception:
        pass

    return GameContext(
        park_run_factor=pf_run, park_hr_factor=pf_hr,
        altitude_ft=5280 if home_abbr.upper() == "COL" else rng.choice([50, 500, 800]),
        roof_status=roof,
        wind_speed_mph=0 if roof == "closed" else rng.uniform(0, 15),
        wind_direction=rng.choice(["out", "in", "cross", "none"]) if roof != "closed" else "none",
        temperature_f=rng.uniform(58, 88),
        humidity_pct=rng.uniform(30, 80),
        precipitation_pct=rng.uniform(0, 20),
        ump_runs_per_game=rng.uniform(8.3, 9.3),
        ump_called_strike_rate=rng.uniform(0.48, 0.52),
        day_game=day_game,
        doubleheader=False,
        extra_innings_prev_game=False,
    )


def _synth_market(g: dict, home: TeamStats, away: TeamStats) -> MarketData:
    rng = _seed_rng("MKT", g.get("game_pk"))
    # Derive "home strength" from the synthetic stats so lines are internally consistent
    hs = (home.offense.wrc_plus - away.offense.wrc_plus) / 200.0
    hs += (away.pitcher.siera - home.pitcher.siera) / 10.0
    hs = max(-0.45, min(0.45, hs))

    if hs > 0.05:
        home_ml = -int(100 + hs * 250)
        away_ml = int(90 + hs * 220)
        home_is_fav = True
    elif hs < -0.05:
        away_ml = -int(100 + (-hs) * 250)
        home_ml = int(90 + (-hs) * 220)
        home_is_fav = False
    else:
        home_ml, away_ml, home_is_fav = -115, -105, True

    if home_is_fav:
        home_rl = rng.choice([+110, +120, +130, +140])
        away_rl = -(home_rl + rng.randint(5, 20))
    else:
        away_rl = rng.choice([+110, +120, +130, +140])
        home_rl = -(away_rl + rng.randint(5, 20))

    total = rng.choice([7.0, 7.5, 8.0, 8.5, 9.0, 9.5])
    return MarketData(
        home_ml_odds=home_ml, away_ml_odds=away_ml,
        opener_home_ml_odds=home_ml + rng.randint(-10, 10),
        home_rl_odds=home_rl, away_rl_odds=away_rl,
        home_is_rl_favorite=home_is_fav,
        total_line=total,
        over_odds=rng.choice([-115, -110, -105, +100]),
        under_odds=rng.choice([-115, -110, -105, +100]),
        opener_total=total + rng.choice([-0.5, 0.0, 0.0, 0.5]),
        public_ticket_pct_home=rng.uniform(0.35, 0.70),
        public_money_pct_home=rng.uniform(0.30, 0.75),
        public_ticket_pct_over=rng.uniform(0.40, 0.70),
        public_money_pct_over=rng.uniform(0.35, 0.70),
        steam_flag_home=rng.random() < 0.1,
        steam_flag_over=rng.random() < 0.1,
    )


# =============================================================================
# Pick grading — compares a prediction to the actual final score
# =============================================================================

def grade_pick(market: str, pick: str, home_score: Optional[int],
               away_score: Optional[int], total_line: float) -> Optional[str]:
    """Return 'win' | 'loss' | 'push' | 'no_bet' | None if ungradable."""
    if pick == "NO BET":
        return "no_bet"
    if home_score is None or away_score is None:
        return None

    if market == "moneyline":
        if pick.startswith("HOME"):
            return "win" if home_score > away_score else "loss"
        if pick.startswith("AWAY"):
            return "win" if away_score > home_score else "loss"

    if market == "run_line":
        margin = home_score - away_score
        if "HOME -1.5" in pick:
            return "win" if margin >= 2 else "loss"
        if "AWAY -1.5" in pick:
            return "win" if -margin >= 2 else "loss"
        if "HOME +1.5" in pick:
            return "win" if margin >= -1 else "loss"
        if "AWAY +1.5" in pick:
            return "win" if margin <= 1 else "loss"

    if market == "totals":
        total = home_score + away_score
        if "OVER" in pick:
            if total > total_line: return "win"
            if total < total_line: return "loss"
            return "push"
        if "UNDER" in pick:
            if total < total_line: return "win"
            if total > total_line: return "loss"
            return "push"

    return None


# =============================================================================
# Game loader — real schedule first, synthetic fallback
# =============================================================================

_games_cache: dict[str, list[dict]] = {}


_SLATE_CACHE_TTL = 60  # seconds — same as SLATE_TTL_SEC upstream
_cache_stamp: dict[str, float] = {}


def _load_games(game_date: date) -> list[dict]:
    """Try real MLB schedule; fall back to a small mock if the API is down.

    Markets, team stats, and context come from live_data when available
    and silently fall back to the synthetic seeders otherwise.
    """
    key = game_date.isoformat()
    now = time.time()
    last = _cache_stamp.get(key, 0.0)
    if key in _games_cache and (now - last) < _SLATE_CACHE_TTL:
        return _games_cache[key]

    try:
        raw = mlb_data.fetch_schedule(game_date)
    except Exception as e:
        log.warning("MLB Stats API unavailable (%s); using fallback slate", e)
        raw = []

    if not raw:
        raw = _fallback_slate(game_date)

    # Warm the live-data caches (odds, projections, lineups) for the slate.
    # Any failure here is tolerated — each builder falls back individually.
    try:
        slate = live_data.warm_slate(game_date)
    except Exception as e:
        log.exception("live_data.warm_slate failed: %s", e)
        slate = None

    games: list[dict] = []
    for g in raw:
        # Build a minimal "pre-game" dict that the live_data builders need.
        pre = {
            "id": mlb_data.game_pk_to_id(game_date, g["away"]["abbr"], g["home"]["abbr"]),
            "date": game_date.isoformat(),
            "raw": g,
        }

        # --- Teams -----------------------------------------------------
        home_team = away_team = None
        if slate is not None:
            try:
                home_team = live_data.build_team_stats_live(pre, slate, is_home=True)
                away_team = live_data.build_team_stats_live(pre, slate, is_home=False)
            except Exception as e:
                log.exception("live team stats failed for %s: %s", pre["id"], e)
        if home_team is None:
            home_team = _synth_team(g["home"], is_home=True, game_pk=g.get("game_pk"))
        if away_team is None:
            away_team = _synth_team(g["away"], is_home=False, game_pk=g.get("game_pk"))

        # --- Context ---------------------------------------------------
        ctx = None
        if slate is not None:
            try:
                pf_run, pf_hr = mlb_data.get_park_factors(g["home"]["abbr"])
                altitude = 5280 if g["home"]["abbr"].upper() == "COL" else 500
                ctx = live_data.build_context_live(
                    pre, slate, g["home"]["abbr"],
                    pf_run, pf_hr, altitude, mlb_data.is_domed(g["home"]["abbr"]),
                )
            except Exception as e:
                log.exception("live context failed for %s: %s", pre["id"], e)
        if ctx is None:
            ctx = _synth_context(g)

        # --- Market ----------------------------------------------------
        market = None
        per_book: dict = {}
        opening_per_book: dict = {}
        if slate is not None:
            try:
                market = live_data.build_market_live(pre, slate)
            except Exception as e:
                log.exception("live market failed for %s: %s", pre["id"], e)
            try:
                per_book = live_data.build_per_book_markets_live(pre, slate)
            except Exception as e:
                log.exception("per-book markets failed for %s: %s", pre["id"], e)
                per_book = {}
            try:
                opening_per_book = live_data.build_per_book_opening_markets_live(pre, slate)
            except Exception as e:
                log.exception("per-book opening markets failed for %s: %s", pre["id"], e)
                opening_per_book = {}
        if market is None:
            market = _synth_market(g, home_team, away_team)

        games.append({
            "id": pre["id"],
            "date": game_date.isoformat(),
            "raw": g,
            "home": home_team,
            "away": away_team,
            "ctx": ctx,
            "market": market,
            "per_book": per_book,          # {"draftkings": MarketData, ...} — latest/closing
            "opening_per_book": opening_per_book,  # same shape, earliest snapshots
        })

    _games_cache[key] = games
    _cache_stamp[key] = now
    return games


# Alias kept for historical call sites — endpoints use _games_for(target).
_games_for = _load_games


def _fallback_slate(game_date: date) -> list[dict]:
    """Tiny hand-crafted slate used only when the real API is unreachable.

    Deliberately labeled as OFFLINE so the UI can tell the user the schedule
    isn't live.
    """
    return [
        {
            "game_pk": f"offline-{game_date.isoformat()}-1",
            "game_date_utc": f"{game_date.isoformat()}T23:05:00Z",
            "status": "scheduled",
            "status_detail": "Scheduled (offline fallback)",
            "current_inning": None, "inning_half": None,
            "venue": "Offline",
            "home": {"abbr": "LAD", "name": "Dodgers", "team_id": None, "score": None,
                     "pitcher_name": "Yamamoto", "pitcher_id": None, "pitcher_hand": "R"},
            "away": {"abbr": "NYY", "name": "Yankees", "team_id": None, "score": None,
                     "pitcher_name": "Cole", "pitcher_id": None, "pitcher_hand": "R"},
        },
    ]


# =============================================================================
# Serialization
# =============================================================================

def _dc_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _dc_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dc_to_dict(v) for v in obj]
    return obj


def _round_pred(r) -> dict:
    return {
        "market": r.market,
        "pick": r.pick,
        "odds": r.odds,
        "model_prob": round(r.model_prob, 4),
        "implied_prob": round(r.implied_prob, 4),
        "edge": round(r.edge, 4),
        "ev_per_unit": round(r.expected_value_per_unit, 4),
        "confidence": round(r.confidence, 1),
        "confidence_label": r.confidence_label,
    }


def _book_market_dict(md) -> dict:
    """Serialize a MarketData into the subset the UI needs per book."""
    return {
        "home_ml": md.home_ml_odds,
        "away_ml": md.away_ml_odds,
        "total_line": md.total_line,
        "over_odds": md.over_odds,
        "under_odds": md.under_odds,
        "home_rl_odds": md.home_rl_odds,
        "away_rl_odds": md.away_rl_odds,
        "home_is_rl_favorite": md.home_is_rl_favorite,
    }


def _summarize_game(game: dict, include_detail: bool = False) -> dict:
    home, away = game["home"], game["away"]
    ctx, market, raw = game["ctx"], game["market"], game["raw"]
    results = predict_all(home, away, ctx, market)

    home_score = raw["home"].get("score")
    away_score = raw["away"].get("score")
    status = raw["status"]

    predictions: dict[str, dict] = {}
    for market_name, r in results.items():
        p = _round_pred(r)
        if include_detail:
            p["detail"] = _dc_to_dict(r.detail)
        if status == "final":
            p["result"] = grade_pick(market_name, r.pick, home_score, away_score,
                                     market.total_line)
        elif status == "live":
            # The underlying predictors are PREGAME — they price a brand-new
            # game from starting pitchers + team stats and don't know inning,
            # score, or outs. Once the game is in progress those numbers are
            # misleading (e.g. "HOME ML 55% / edge +52%" on a 6-1 game in the
            # bottom of the 9th). Neutralize the pick so the UI stops showing
            # HIGH-confidence bets on games that are already half over, but
            # keep the raw fields (detail, odds) so the details pane can
            # still explain what the pregame model thought.
            p["pick"] = "NO BET (game in progress)"
            p["odds"] = None
            p["edge"] = 0.0
            p["ev_per_unit"] = 0.0
            p["confidence"] = 0.0
            p["confidence_label"] = "LIVE"
        predictions[market_name] = p

    out = {
        "id": game["id"],
        "date": game["date"],
        "status": status,                          # 'scheduled' | 'live' | 'final'
        "status_detail": raw.get("status_detail", ""),
        "first_pitch_utc": raw.get("game_date_utc"),
        "venue": raw.get("venue", ""),
        "home": {
            "abbr": raw["home"]["abbr"],
            "name": home.name,
            "pitcher": home.pitcher.name,
            "pitcher_siera": round(home.pitcher.siera, 2),
            "pitcher_throws": home.pitcher.throws,
            "form_l10": round(home.form_last10_win_pct, 3),
            "lineup_confirmed": home.lineup_confirmed,
            "score": home_score,
        },
        "away": {
            "abbr": raw["away"]["abbr"],
            "name": away.name,
            "pitcher": away.pitcher.name,
            "pitcher_siera": round(away.pitcher.siera, 2),
            "pitcher_throws": away.pitcher.throws,
            "form_l10": round(away.form_last10_win_pct, 3),
            "lineup_confirmed": away.lineup_confirmed,
            "score": away_score,
        },
        "market": {
            "home_ml": market.home_ml_odds,
            "away_ml": market.away_ml_odds,
            "total_line": market.total_line,
            "over_odds": market.over_odds,
            "under_odds": market.under_odds,
            "home_rl_odds": market.home_rl_odds,
            "away_rl_odds": market.away_rl_odds,
            "home_is_rl_favorite": market.home_is_rl_favorite,
        },
        "park": {
            "run_factor": ctx.park_run_factor,
            "hr_factor": ctx.park_hr_factor,
            "roof": ctx.roof_status,
            "temp_f": round(ctx.temperature_f, 0),
            "wind_mph": round(ctx.wind_speed_mph, 0),
            "wind_dir": ctx.wind_direction,
        },
        "live": {
            "inning": raw.get("current_inning"),
            "half": raw.get("inning_half"),
        } if status == "live" else None,
        "predictions": predictions,
    }

    # --- Per-book lines + best-price recommendations -----------------------
    per_book = game.get("per_book") or {}
    books_out: dict[str, dict] = {}
    for b in ("draftkings", "fanduel", "pinnacle"):
        md = per_book.get(b)
        if md is not None:
            books_out[b] = _book_market_dict(md)
        else:
            books_out[b] = None  # signal "unavailable" to the UI
    out["books"] = books_out

    # Starting (opening) lines per book — useful for final games so the
    # UI can show "opened X, closed Y". We always emit this (even for
    # scheduled games) so the client can reuse the same rendering logic
    # to show line movement while the game is still on the board.
    opening_per_book = game.get("opening_per_book") or {}
    books_starting: dict[str, dict] = {}
    for b in ("draftkings", "fanduel", "pinnacle"):
        md = opening_per_book.get(b)
        if md is not None:
            books_starting[b] = _book_market_dict(md)
        else:
            books_starting[b] = None
    out["books_starting"] = books_starting

    # Consensus "starting market" — mirrors `market` above but with the
    # earliest known values. Prefer DK, fall back to FanDuel, then
    # Pinnacle. If no opener is known, fall back to whatever
    # opener-shaped fields live on the current MarketData
    # (`opener_home_ml_odds`, `opener_total`).
    start_md = None
    for b in ("draftkings", "fanduel", "pinnacle"):
        if opening_per_book.get(b) is not None:
            start_md = opening_per_book[b]
            break
    starting_market: dict[str, Any] = {}
    if start_md is not None:
        starting_market = _book_market_dict(start_md)
    else:
        # Best-effort reconstruction from whatever opener hints live on
        # the MarketData dataclass.
        starting_market = {
            "home_ml": getattr(market, "opener_home_ml_odds", None),
            "away_ml": None,
            "total_line": getattr(market, "opener_total", None),
            "over_odds": None,
            "under_odds": None,
            "home_rl_odds": None,
            "away_rl_odds": None,
            "home_is_rl_favorite": market.home_is_rl_favorite,
        }
    out["starting_market"] = starting_market

    # Recommendations only make sense for scheduled games; skip once the
    # game goes live/final (the pick is locked in at that point).
    if status == "scheduled" and per_book:
        try:
            recs = live_data.recommendations_from_predictions(
                results, per_book, books=("draftkings", "fanduel"),
            )
            out["recommendations"] = recs
        except Exception as e:
            log.exception("recommendations failed for %s: %s", game["id"], e)
            out["recommendations"] = []
    else:
        out["recommendations"] = []

    if include_detail:
        out["home_full"] = _dc_to_dict(home)
        out["away_full"] = _dc_to_dict(away)
        out["ctx_full"] = _dc_to_dict(ctx)
        out["market_full"] = _dc_to_dict(market)
    return out


# =============================================================================
# Backtest mock (unchanged — replace with real file when backtest finishes)
# =============================================================================

def _mock_backtest(seed: int = 7) -> dict:
    rng = random.Random(seed)
    markets = ["moneyline", "run_line", "totals"]
    by_market: dict[str, dict] = {}
    equity_curve = [{"date": (date.today() - timedelta(days=180 - i)).isoformat(),
                     "equity": 1000.0} for i in range(181)]
    running = 1000.0
    for i in range(1, len(equity_curve)):
        running += rng.gauss(2.2, 28)
        equity_curve[i]["equity"] = round(running, 2)

    for m in markets:
        n = rng.randint(180, 240)
        wins = int(n * rng.uniform(0.51, 0.56))
        units = round(rng.uniform(8.5, 28.0), 2) if m != "run_line" else round(rng.uniform(-2.0, 15.0), 2)
        by_market[m] = {
            "bets": n, "wins": wins, "losses": n - wins,
            "win_pct": round(wins / n, 3),
            "units_won": units, "roi_pct": round(units / n * 100, 2),
            "avg_edge_pct": round(rng.uniform(2.2, 4.1), 2),
            "avg_confidence": round(rng.uniform(55, 68), 1),
        }

    buckets = {}
    for label, win_rate, n_mult in [("LOW", 0.48, 0.8), ("LEAN", 0.52, 1.3),
                                     ("MEDIUM", 0.55, 1.5), ("HIGH", 0.61, 0.9)]:
        n = int(100 * n_mult)
        wins = int(n * (win_rate + rng.uniform(-0.02, 0.02)))
        buckets[label] = {"bets": n, "wins": wins,
                          "win_pct": round(wins / n, 3),
                          "roi_pct": round((win_rate - 0.524) * 100 + rng.uniform(-1.5, 1.5), 2)}

    total_bets = sum(v["bets"] for v in by_market.values())
    total_wins = sum(v["wins"] for v in by_market.values())
    total_units = sum(v["units_won"] for v in by_market.values())
    return {
        "period": {"start": (date.today() - timedelta(days=180)).isoformat(),
                   "end": date.today().isoformat(), "days": 180},
        "totals": {"bets": total_bets, "wins": total_wins,
                   "losses": total_bets - total_wins,
                   "win_pct": round(total_wins / total_bets, 3),
                   "units_won": round(total_units, 2),
                   "roi_pct": round(total_units / total_bets * 100, 2),
                   "starting_bankroll": 1000.0,
                   "ending_bankroll": equity_curve[-1]["equity"]},
        "by_market": by_market, "by_confidence": buckets,
        "equity_curve": equity_curve,
        "note": "Mock results while the real backtest is running. "
                "Replace with the real output file when available.",
    }


# =============================================================================
# Routes
# =============================================================================

@app.get("/api/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}


@app.get("/api/games/today")
def games_today(d: str | None = None):
    try:
        target = date.fromisoformat(d) if d else mlb_data.today_et()
    except ValueError:
        raise HTTPException(400, "date must be YYYY-MM-DD")
    games = _games_for(target)
    return {
        "date": target.isoformat(),
        "games": [_summarize_game(g) for g in games],
    }


@app.get("/api/games/{game_id}")
def game_detail(game_id: str):
    try:
        game_date = date.fromisoformat(game_id[:10])
    except ValueError:
        raise HTTPException(400, "bad game id")
    for g in _games_for(game_date):
        if g["id"] == game_id:
            return _summarize_game(g, include_detail=True)
    raise HTTPException(404, f"no game with id {game_id}")


# -----------------------------------------------------------------------------
# Backtest summary — loads the real results JSON written by run_backtest.py,
# and falls back to the mock if the file is missing/unreadable.
# -----------------------------------------------------------------------------
_BACKTEST_RESULTS_PATH = Path(__file__).resolve().parent / "backtest_results.json"


def _load_real_backtest() -> Optional[dict]:
    if not _BACKTEST_RESULTS_PATH.exists():
        return None
    try:
        with _BACKTEST_RESULTS_PATH.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("backtest_results.json unreadable: %s", e)
        return None


@app.get("/api/backtest/summary")
def backtest_summary():
    real = _load_real_backtest()
    if real is not None:
        return real
    return _mock_backtest()


# -----------------------------------------------------------------------------
# Gate / narrowing / bet-slip / line-shop / CLV endpoints
# -----------------------------------------------------------------------------

@app.get("/api/gate/summary")
def gate_status():
    """Current production gates per market — which markets are live,
    what edge and confidence thresholds apply, and why.
    """
    return {
        "gates": gate_summary(),
        "closing_line_model_active": coefficients_loaded(),
    }


@app.get("/api/today/slip")
def today_slip(d: str | None = None,
               bankroll: float = 10000.0,
               unit_size: float = 100.0,
               max_legs_per_event: int = 1,
               kelly_fraction: float = 0.25):
    """Produce today's Kelly-sized bet slip.

    Query params:
      - bankroll: dollars (default 10,000)
      - unit_size: dollars per unit (default 100)
      - max_legs_per_event: default 1 = one bet per game
      - kelly_fraction: default 0.25 (quarter Kelly)
    """
    try:
        target = date.fromisoformat(d) if d else mlb_data.today_et()
    except ValueError:
        raise HTTPException(400, "date must be YYYY-MM-DD")

    try:
        games = _games_for(target)
    except NameError:
        games = _load_games(target)

    # Collect predictions from every market on every game and pass
    # through the narrow_gate filter.
    policy = BankrollPolicy(
        bankroll=bankroll, kelly_fraction=kelly_fraction, unit_size=unit_size,
    )
    all_preds = []
    event_id_of: dict[int, str] = {}
    clv_by_event: dict[str, dict] = {}
    for game in games:
        home, away = game["home"], game["away"]
        ctx, market = game["ctx"], game["market"]
        results = predict_all(home, away, ctx, market)
        # Narrow: zero-out picks that don't clear the production gate
        narrowed = narrow_picks(list(results.values()))
        for r in narrowed:
            event_id_of[id(r)] = game["id"]
        all_preds.extend(narrowed)
        # CLV — only meaningful when coefficients are fitted
        if coefficients_loaded():
            clv_by_event[game["id"]] = {
                "moneyline": predict_closing_line_ml(home, away, ctx, market)
                              .predicted_close_home_prob,
                "totals": predict_closing_line_total(home, away, ctx, market)
                           .predicted_close_total,
            }

    slip = build_slip(
        picks=surviving_only(all_preds),
        event_id_of=event_id_of,
        policy=policy,
        shopped_markets=None,    # shopped prices already baked into MarketData
        predicted_closing_by_event=clv_by_event or None,
        max_legs_per_event=max_legs_per_event,
    )
    return {"date": target.isoformat(), **slip.to_dict(),
            "clv_active": coefficients_loaded()}


@app.get("/api/shop/today")
def shop_today(d: str | None = None):
    """Line-shop across every book we have in the cache for today.

    Returns each event with the best available price per side and any
    arbitrage / middle alerts.
    """
    try:
        target = date.fromisoformat(d) if d else mlb_data.today_et()
    except ValueError:
        raise HTTPException(400, "date must be YYYY-MM-DD")

    try:
        cache = OddsCache()
        shopped = shop_all(cache, target.isoformat())
        return {
            "date": target.isoformat(),
            "events": [sm.to_dict() for sm in shopped],
        }
    except Exception as e:
        log.warning("Line-shop failed: %s", e)
        return {"date": target.isoformat(), "events": [],
                "note": "Cache unavailable — no shopped lines to report."}


@app.get("/api/clv/today")
def clv_today(d: str | None = None):
    """Per-event predicted closing line movement — snipe / wait / neutral.

    Until the closing-line model coefficients have been fit offline, this
    endpoint returns `model_active=False` and zero-movement predictions
    for every market. Once coefficients exist the `direction` and
    `clv_edge` fields become actionable.
    """
    try:
        target = date.fromisoformat(d) if d else mlb_data.today_et()
    except ValueError:
        raise HTTPException(400, "date must be YYYY-MM-DD")
    try:
        games = _games_for(target)
    except NameError:
        games = _load_games(target)

    out = []
    for game in games:
        home, away, ctx, market = game["home"], game["away"], game["ctx"], game["market"]
        out.append({
            "id": game["id"],
            "moneyline": _clv_dc(predict_closing_line_ml(home, away, ctx, market)),
            "totals":    _clv_dc(predict_closing_line_total(home, away, ctx, market)),
            "run_line":  _clv_dc(predict_closing_line_rl(home, away, ctx, market)),
        })
    return {
        "date": target.isoformat(),
        "model_active": coefficients_loaded(),
        "events": out,
    }


def _clv_dc(pred) -> dict:
    return {
        "market": pred.market,
        "predicted_close_home_prob": pred.predicted_close_home_prob,
        "predicted_close_total": pred.predicted_close_total,
        "current_implied_prob": pred.current_implied_prob,
        "current_line": pred.current_line,
        "clv_edge": round(pred.clv_edge, 5),
        "direction": pred.direction,
        "coefficients_fitted": pred.coefficients_fitted,
    }


# -----------------------------------------------------------------------------
# Static frontend
# -----------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    if (FRONTEND_DIR / "static").exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(FRONTEND_DIR / "static")),
            name="static",
        )

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")
