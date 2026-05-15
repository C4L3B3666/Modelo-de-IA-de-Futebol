"""Leakage-safe temporal feature engineering for historical football data."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd

from .poisson import validate_matches


@dataclass
class TeamState:
    goals_for: Deque[int]
    goals_against: Deque[int]
    home_for: Deque[int]
    home_against: Deque[int]
    away_for: Deque[int]
    away_against: Deque[int]
    last_date: Optional[pd.Timestamp] = None


class TemporalFeatureEngineer:
    """Builds pre-match features by scanning matches in chronological order."""

    def __init__(self, window: int = 5) -> None:
        self.window = window

    def transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        data = validate_matches(matches)
        states: Dict[str, TeamState] = defaultdict(self._new_state)
        rows: List[dict] = []

        for _, row in data.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            home_state = states[home]
            away_state = states[away]

            rows.append(
                {
                    **row.to_dict(),
                    "home_form_points": points_per_game(home_state.goals_for, home_state.goals_against),
                    "away_form_points": points_per_game(away_state.goals_for, away_state.goals_against),
                    "home_avg_for": mean_or_default(home_state.goals_for),
                    "home_avg_against": mean_or_default(home_state.goals_against),
                    "away_avg_for": mean_or_default(away_state.goals_for),
                    "away_avg_against": mean_or_default(away_state.goals_against),
                    "home_home_avg_for": mean_or_default(home_state.home_for),
                    "home_home_avg_against": mean_or_default(home_state.home_against),
                    "away_away_avg_for": mean_or_default(away_state.away_for),
                    "away_away_avg_against": mean_or_default(away_state.away_against),
                    "home_days_since": days_since(home_state.last_date, row["date"]),
                    "away_days_since": days_since(away_state.last_date, row["date"]),
                }
            )

            self._update_home(home_state, int(row["home_goals"]), int(row["away_goals"]), row["date"])
            self._update_away(away_state, int(row["away_goals"]), int(row["home_goals"]), row["date"])

        return pd.DataFrame(rows)

    def latest_team_features(self, matches: pd.DataFrame) -> Dict[str, dict]:
        featured = self.transform(matches)
        states: Dict[str, dict] = {}
        for _, row in featured.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            states[home] = {
                "avg_for": row["home_avg_for"],
                "avg_against": row["home_avg_against"],
                "home_avg_for": row["home_home_avg_for"],
                "home_avg_against": row["home_home_avg_against"],
            }
            states[away] = {
                "avg_for": row["away_avg_for"],
                "avg_against": row["away_avg_against"],
                "away_avg_for": row["away_away_avg_for"],
                "away_avg_against": row["away_away_avg_against"],
            }
        return states

    def _new_state(self) -> TeamState:
        return TeamState(
            goals_for=deque(maxlen=self.window),
            goals_against=deque(maxlen=self.window),
            home_for=deque(maxlen=self.window),
            home_against=deque(maxlen=self.window),
            away_for=deque(maxlen=self.window),
            away_against=deque(maxlen=self.window),
        )

    @staticmethod
    def _update_home(state: TeamState, goals_for: int, goals_against: int, date: pd.Timestamp) -> None:
        state.goals_for.append(goals_for)
        state.goals_against.append(goals_against)
        state.home_for.append(goals_for)
        state.home_against.append(goals_against)
        state.last_date = date

    @staticmethod
    def _update_away(state: TeamState, goals_for: int, goals_against: int, date: pd.Timestamp) -> None:
        state.goals_for.append(goals_for)
        state.goals_against.append(goals_against)
        state.away_for.append(goals_for)
        state.away_against.append(goals_against)
        state.last_date = date


def mean_or_default(values: Deque[int], default: float = 1.25) -> float:
    return float(np.mean(values)) if values else default


def points_per_game(goals_for: Deque[int], goals_against: Deque[int]) -> float:
    if not goals_for:
        return 1.0
    points = 0
    for scored, conceded in zip(goals_for, goals_against):
        points += 3 if scored > conceded else 1 if scored == conceded else 0
    return float(points / len(goals_for))


def days_since(last_date: Optional[pd.Timestamp], current_date: pd.Timestamp) -> float:
    if last_date is None or pd.isna(last_date):
        return float("nan")
    return float(max((current_date - last_date).days, 0))


__all__ = ["TemporalFeatureEngineer"]
