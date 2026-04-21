"""
PECOTA-alt team win Monte Carlo.

Simulates a team's remaining schedule N times using blended per-game
win probabilities and reports the distribution of full-season wins.

Approach:
    1. For each remaining game, estimate a true single-game win prob
       from the team's blended per-game strength + opponent strength +
       home/away adjustment.
    2. Draw Bernoulli outcomes for every game, sum wins.
    3. Repeat N times, return mean + percentiles.

This is deliberately simple — it's not a pitch-by-pitch sim. The
value is converting season-level team strength into game-level win
probabilities with realistic variance, so you can answer:
    "Given their projection, what's the P(85+ wins) for this team?"

Pure Python / stdlib. Uses random.random() for portability; if numpy
is present it'll use it automatically for a ~10x speedup.
"""
from __future__ import annotations

import math
import random
from typing import Iterable, Optional

from .projections_models import TeamWinProjection

try:
    import numpy as _np   # optional; used for vectorized sim when available
except ImportError:       # pragma: no cover
    _np = None


# Home-field advantage in per-game win probability (empirical MLB ~0.04).
_HOME_FIELD_LIFT = 0.040


def pythag_win_prob(team_wrc_plus: float, team_runs_allowed_pct: float,
                    opp_wrc_plus: float, opp_runs_allowed_pct: float,
                    is_home: bool = True) -> float:
    """Estimate P(team beats opp) from simple offense-vs-defense components.

    team_runs_allowed_pct: team's run-prevention strength where 1.00 =
    league-average, higher = *worse* at preventing runs. So a strong
    pitching staff has value < 1.00.

    Math: convert each team's (offense, defense) into an expected runs
    scored in this matchup, then use a Bradley-Terry-like logistic on
    the run differential.
    """
    team_runs = (team_wrc_plus / 100.0) * (opp_runs_allowed_pct)
    opp_runs = (opp_wrc_plus / 100.0) * (team_runs_allowed_pct)
    # Typical per-game run expectancy ~4.5; scale so the logistic is
    # sensitive at realistic run-differential magnitudes.
    diff = (team_runs - opp_runs) * 1.7
    p = 1.0 / (1.0 + math.exp(-diff))
    if is_home:
        p = min(0.95, p + _HOME_FIELD_LIFT)
    else:
        p = max(0.05, p - _HOME_FIELD_LIFT)
    return max(0.01, min(0.99, p))


def _simulate_python(per_game_probs: list[float], iterations: int
                     ) -> list[int]:
    """Pure-Python fallback if numpy isn't available."""
    wins = [0] * iterations
    for i in range(iterations):
        w = 0
        for p in per_game_probs:
            if random.random() < p:
                w += 1
        wins[i] = w
    return wins


def _simulate_numpy(per_game_probs: list[float], iterations: int
                    ) -> list[int]:
    """Vectorized simulation using numpy — much faster for 10k iterations."""
    probs = _np.asarray(per_game_probs, dtype=_np.float64)
    # Draw (iterations, games) uniform random and threshold against prob row
    rolls = _np.random.random((iterations, len(probs)))
    # Broadcasting: each column j uses probs[j]
    outcomes = (rolls < probs).sum(axis=1)
    return outcomes.tolist()


def simulate_season(
    team: str,
    current_wins: int,
    per_game_probs: list[float],
    *,
    iterations: int = 10000,
    playoffs_threshold: Optional[int] = None,
    seed: Optional[int] = None,
) -> TeamWinProjection:
    """Run the simulator for one team.

    per_game_probs : one float per remaining game, order doesn't matter.
                     Build these externally via pythag_win_prob() or
                     by blending projection + rolling form.
    current_wins   : W the team already has in the bag — added to each
                     simulated total.
    playoffs_threshold : if supplied, computes P(full-season wins ≥ threshold).
    """
    if seed is not None:
        random.seed(seed)
        if _np is not None:
            _np.random.seed(seed)

    if not per_game_probs:
        return TeamWinProjection(
            team=team, games_remaining=0,
            mean_wins_ros=0.0,
            projected_full_season_wins=float(current_wins),
            win_dist_pctiles={p: current_wins for p in (10, 25, 50, 75, 90)},
            playoffs_prob=(1.0 if (playoffs_threshold is not None and
                                   current_wins >= playoffs_threshold) else 0.0),
            per_game_win_prob=0.500,
            iterations=iterations,
        )

    sim = _simulate_numpy if _np is not None else _simulate_python
    ros_wins = sim(per_game_probs, iterations)

    mean_wins_ros = sum(ros_wins) / iterations
    full_totals = [current_wins + w for w in ros_wins]

    # Percentiles
    full_sorted = sorted(full_totals)
    def pct(p: int) -> int:
        idx = max(0, min(iterations - 1, (p * iterations) // 100))
        return full_sorted[idx]
    pctiles = {p: pct(p) for p in (10, 25, 50, 75, 90)}

    playoffs_prob = None
    if playoffs_threshold is not None:
        hit = sum(1 for t in full_totals if t >= playoffs_threshold)
        playoffs_prob = hit / iterations

    return TeamWinProjection(
        team=team,
        games_remaining=len(per_game_probs),
        mean_wins_ros=round(mean_wins_ros, 1),
        projected_full_season_wins=round(current_wins + mean_wins_ros, 1),
        win_dist_pctiles=pctiles,
        playoffs_prob=(round(playoffs_prob, 3) if playoffs_prob is not None else None),
        per_game_win_prob=round(sum(per_game_probs) / len(per_game_probs), 4),
        iterations=iterations,
    )


def flat_schedule_probs(win_prob_home: float, win_prob_away: float,
                        n_home: int, n_away: int) -> list[float]:
    """Quick helper: build a per_game_probs list from summary numbers.

    Useful when you don't yet have a real remaining-schedule object —
    you can pass "average win prob at home" and "average win prob away"
    plus the raw home/away game counts.
    """
    return [win_prob_home] * n_home + [win_prob_away] * n_away
