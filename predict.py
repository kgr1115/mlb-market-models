"""
Demo / entry point for the MLB betting predictors.

Usage:
    python predict.py

This file builds a sample Yankees @ Dodgers matchup with realistic stats,
then runs all three models (Moneyline, Run Line, Totals) and prints the
picks, probabilities, edges, and confidence scores.

In production you'll wire this up to live pybaseball + MLB Stats API +
Open-Meteo + The Odds API feeds (see data_sources.md), but the predictor
signatures won't change — just populate the dataclasses and call the
three `predict_*` functions.
"""
from __future__ import annotations

from predictors import (
    PitcherStats, BullpenStats, OffenseStats, DefenseStats,
    TeamStats, GameContext, MarketData,
    predict_all,
)


def _fmt_pct(p):
    return f"{p*100:.1f}%"


def _fmt_edge(e):
    return f"{e*100:+.1f}%"


def print_result(r):
    print(f"\n=== {r.market.upper()} ===")
    print(f"Pick:              {r.pick}")
    if r.odds is not None:
        print(f"Model probability: {_fmt_pct(r.model_prob)}")
        print(f"Implied (no vig):  {_fmt_pct(r.implied_prob)}")
        print(f"Edge:              {_fmt_edge(r.edge)}")
        print(f"EV per $1 staked:  {r.expected_value_per_unit:+.3f}")
    print(f"Confidence:        {r.confidence:.1f}/100  [{r.confidence_label}]")


def sample_game():
    """Build a realistic sample game. Dodgers home vs. Yankees away."""
    # HOME: Dodgers, with a top-tier starter (~Cole-ish, borrowed for flavor)
    home = TeamStats(
        name="Dodgers",
        is_home=True,
        pitcher=PitcherStats(
            name="Yamamoto", throws="R",
            siera=3.20, xfip=3.35, k_bb_pct=0.21, csw_pct=0.315,
            xwoba_against=0.280, ip_per_gs=6.1, rolling_30d_era=2.75,
        ),
        bullpen=BullpenStats(
            fip=3.40, hi_lev_k_pct=0.30, meltdown_pct=0.09, shutdown_pct=0.40,
            closer_pitches_last3d=12, setup_pitches_last3d=18, days_since_closer_used=1,
        ),
        offense=OffenseStats(
            wrc_plus=118, wOBA=0.345, xwOBA=0.340, obp=0.340, iso=0.190,
            barrel_pct=0.095, k_pct=0.215, top_of_order_obp=0.365,
        ),
        defense=DefenseStats(
            oaa=12, drs=18, catcher_framing_runs=4, bsr=3,
        ),
        form_last10_win_pct=0.700, form_last20_win_pct=0.650,
        rest_days=1, travel_miles_72h=0,
        lineup_confirmed=True, starter_confirmed=True,
    )

    # AWAY: Yankees, very good lineup but lesser SP today
    away = TeamStats(
        name="Yankees",
        is_home=False,
        pitcher=PitcherStats(
            name="Stroman", throws="R",
            siera=4.25, xfip=4.30, k_bb_pct=0.11, csw_pct=0.275,
            xwoba_against=0.330, ip_per_gs=5.3, rolling_30d_era=4.60,
        ),
        bullpen=BullpenStats(
            fip=3.75, hi_lev_k_pct=0.27, meltdown_pct=0.12, shutdown_pct=0.32,
            closer_pitches_last3d=22, setup_pitches_last3d=30, days_since_closer_used=1,
        ),
        offense=OffenseStats(
            wrc_plus=122, wOBA=0.350, xwOBA=0.348, obp=0.345, iso=0.210,
            barrel_pct=0.105, k_pct=0.230, top_of_order_obp=0.380,
        ),
        defense=DefenseStats(
            oaa=-2, drs=5, catcher_framing_runs=2, bsr=-1,
        ),
        form_last10_win_pct=0.600, form_last20_win_pct=0.580,
        rest_days=1, travel_miles_72h=2200,  # cross-country trip
        lineup_confirmed=True, starter_confirmed=True,
    )

    ctx = GameContext(
        park_run_factor=0.97,   # Dodger Stadium slightly pitcher-friendly
        park_hr_factor=1.02,
        altitude_ft=500, roof_status="none",
        wind_speed_mph=6, wind_direction="out",   # breeze out to LF
        temperature_f=72, humidity_pct=55, precipitation_pct=0,
        ump_runs_per_game=8.9, ump_called_strike_rate=0.505,
        day_game=False, doubleheader=False, extra_innings_prev_game=False,
    )

    market = MarketData(
        home_ml_odds=-165, away_ml_odds=+140,
        opener_home_ml_odds=-155,
        home_rl_odds=+120, away_rl_odds=-145,
        home_is_rl_favorite=True,
        total_line=8.5, over_odds=-110, under_odds=-110,
        opener_total=8.5,
        public_ticket_pct_home=0.55, public_money_pct_home=0.62,
        public_ticket_pct_over=0.62, public_money_pct_over=0.58,
        steam_flag_home=False, steam_flag_over=False,
    )
    return home, away, ctx, market


def main():
    home, away, ctx, market = sample_game()
    print(f"{away.name} @ {home.name}")
    print(f"  SP: {away.pitcher.name} ({away.pitcher.siera:.2f} SIERA) vs "
          f"{home.pitcher.name} ({home.pitcher.siera:.2f} SIERA)")
    print(f"  ML: {away.name} {market.away_ml_odds:+d} / "
          f"{home.name} {market.home_ml_odds:+d}")
    print(f"  RL: {'home -1.5' if market.home_is_rl_favorite else 'away -1.5'}")
    print(f"  Total: {market.total_line}")
    print(f"  Park factor: {ctx.park_run_factor}   "
          f"Wind: {ctx.wind_speed_mph}mph {ctx.wind_direction}   "
          f"Temp: {ctx.temperature_f}F   "
          f"Ump R/G: {ctx.ump_runs_per_game}")

    results = predict_all(home, away, ctx, market)
    for r in results.values():
        print_result(r)

    print("\n--- Summary ---")
    for r in results.values():
        print(f"  {r.market.ljust(10)} {r.pick.ljust(28)} "
              f"conf={r.confidence:5.1f} [{r.confidence_label}]")


if __name__ == "__main__":
    main()
