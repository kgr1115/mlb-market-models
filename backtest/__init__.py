"""
Historical backtesting package.

Layout:
    historical_games.py             - MLB Stats API schedule + final scores
    historical_odds.py              - Sportsbook Reviews Online XLSX loader (2018-2021)
    historical_odds_community.py    - ArnavSaraogi community dataset (2021-2025)
    historical_stats.py             - Prior-year baseline team + pitcher stats
    engine.py                       - Main backtest loop (predict, grade, aggregate)
    report.py                       - Aggregate raw grades into the /api/backtest shape
"""
from .historical_games import load_season_games, HistoricalGame
from .historical_odds import load_sbr_season_odds, HistoricalOdds
from .historical_odds_community import load_community_season_odds
from .historical_stats import BaselineStats, load_baseline
from .engine import run_backtest, BacktestResults
from .report import to_api_shape, write_results_json

__all__ = [
    "HistoricalGame", "load_season_games",
    "HistoricalOdds", "load_sbr_season_odds", "load_community_season_odds",
    "BaselineStats", "load_baseline",
    "BacktestResults", "run_backtest", "to_api_shape", "write_results_json",
]
