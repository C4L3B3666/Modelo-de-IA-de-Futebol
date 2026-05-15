"""Dynamic Elo ratings adapted for football match prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from .poisson import validate_matches


@dataclass(init=False)
class EloRating:
    """Chronological Elo system with home advantage and goal-difference scaling."""

    base_rating: float = 1500.0
    k_factor: float = 24.0
    home_advantage: float = 65.0
    draw_value: float = 0.5
    decay_per_year: float = 0.92
    ratings: Dict[str, float] = field(default_factory=dict)
    last_played: Dict[str, pd.Timestamp] = field(default_factory=dict)

    def __init__(
        self,
        base_rating: float = 1500.0,
        k_factor: float = 24.0,
        home_advantage: float = 65.0,
        draw_value: float = 0.5,
        decay_per_year: float = 0.92,
        ratings: Optional[Dict[str, float]] = None,
        last_played: Optional[Dict[str, pd.Timestamp]] = None,
        base: Optional[float] = None,
        k: Optional[float] = None,
        home_adv: Optional[float] = None,
    ) -> None:
        # base/k/home_adv keep compatibility with the earlier project API.
        self.base_rating = float(base if base is not None else base_rating)
        self.k_factor = float(k if k is not None else k_factor)
        self.home_advantage = float(home_adv if home_adv is not None else home_advantage)
        self.draw_value = float(draw_value)
        self.decay_per_year = float(decay_per_year)
        self.ratings = dict(ratings or {})
        self.last_played = dict(last_played or {})

    def get(self, team: str) -> float:
        return float(self.ratings.get(team, self.base_rating))

    def expected_home_score(self, home_team: str, away_team: str) -> float:
        diff = (self.get(home_team) + self.home_advantage - self.get(away_team)) / 400.0
        return float(1.0 / (1.0 + 10.0 ** (-diff)))

    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        date: Optional[pd.Timestamp] = None,
    ) -> None:
        if date is not None:
            self._apply_time_decay(home_team, date)
            self._apply_time_decay(away_team, date)

        expected_home = self.expected_home_score(home_team, away_team)
        actual_home = 1.0 if home_goals > away_goals else self.draw_value if home_goals == away_goals else 0.0
        margin_multiplier = self._goal_difference_multiplier(home_goals, away_goals)
        delta = self.k_factor * margin_multiplier * (actual_home - expected_home)

        self.ratings[home_team] = self.get(home_team) + delta
        self.ratings[away_team] = self.get(away_team) - delta
        if date is not None:
            self.last_played[home_team] = pd.to_datetime(date)
            self.last_played[away_team] = pd.to_datetime(date)

    def fit(self, matches: pd.DataFrame) -> "EloRating":
        data = validate_matches(matches)
        for _, row in data.iterrows():
            self.update_match(
                row["home_team"],
                row["away_team"],
                int(row["home_goals"]),
                int(row["away_goals"]),
                row["date"],
            )
        return self

    def pre_match_ratings(self, matches: pd.DataFrame) -> pd.DataFrame:
        data = validate_matches(matches)
        rows = []
        for _, row in data.iterrows():
            rows.append(
                {
                    **row.to_dict(),
                    "home_elo": self.get(row["home_team"]),
                    "away_elo": self.get(row["away_team"]),
                    "home_elo_win_expectation": self.expected_home_score(row["home_team"], row["away_team"]),
                }
            )
            self.update_match(
                row["home_team"],
                row["away_team"],
                int(row["home_goals"]),
                int(row["away_goals"]),
                row["date"],
            )
        return pd.DataFrame(rows)

    def _apply_time_decay(self, team: str, date: pd.Timestamp) -> None:
        if team not in self.last_played:
            return
        years = max((pd.to_datetime(date) - self.last_played[team]).days, 0) / 365.25
        if years <= 0:
            return
        rating = self.get(team)
        self.ratings[team] = self.base_rating + (rating - self.base_rating) * (self.decay_per_year ** years)

    @staticmethod
    def _goal_difference_multiplier(home_goals: int, away_goals: int) -> float:
        margin = abs(int(home_goals) - int(away_goals))
        if margin <= 1:
            return 1.0
        return float(math.log(margin + 1.0) * 1.35)

    # Backward-compatible methods used by the previous trainer.
    def expected(self, home: str, away: str) -> float:
        return self.expected_home_score(home, away)

    def update(self, home: str, away: str, result: float) -> None:
        home_goals, away_goals = (1, 0) if result == 1.0 else (0, 0) if result == 0.5 else (0, 1)
        self.update_match(home, away, home_goals, away_goals)


__all__ = ["EloRating"]
