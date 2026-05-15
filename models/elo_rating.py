"""ELO rating system for football match forecasting.

This module provides a professional ELO-based rating engine with:
- dynamic rating updates per match,
- home advantage,
- expected result calculation,
- configurable sensitivity via K-factor,
- support for continuous seasons,
- future integration hooks for Poisson-based expected goals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class EloMatchResult:
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    season: int
    match_date: Optional[datetime]
    home_expected: float
    away_expected: float
    home_rating_delta: float
    away_rating_delta: float
    home_rating_before: float
    away_rating_before: float
    home_rating_after: float
    away_rating_after: float


class EloRatingSystem:
    """Professional ELO rating system for football.

    The rating system stores team ratings and match history across seasons.
    It can update ratings dynamically after each match and includes a configurable
    home advantage to reflect the tendency of home teams to perform better.
    """

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 20.0,
        home_advantage: float = 100.0,
        continuous_seasons: bool = True,
    ) -> None:
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.continuous_seasons = continuous_seasons
        self.ratings: pd.DataFrame = pd.DataFrame(
            columns=["team", "rating", "season", "last_updated"]
        )
        self.history: pd.DataFrame = pd.DataFrame(
            columns=[
                "season",
                "match_date",
                "home_team",
                "away_team",
                "home_rating_before",
                "away_rating_before",
                "home_expected",
                "away_expected",
                "home_goals",
                "away_goals",
                "home_rating_after",
                "away_rating_after",
                "home_rating_delta",
                "away_rating_delta",
            ]
        )

    def set_k_factor(self, k_factor: float) -> None:
        """Adjust the sensitivity of the Elo updates."""
        self.k_factor = float(k_factor)

    def set_home_advantage(self, home_advantage: float) -> None:
        """Adjust the home advantage in rating points."""
        self.home_advantage = float(home_advantage)

    def initialize_teams(self, teams: List[str], season: int) -> None:
        """Initialize ratings for a list of teams."""
        existing = set(self.ratings["team"]) if not self.ratings.empty else set()
        new_entries = []
        for team in teams:
            if team not in existing:
                new_entries.append(
                    {
                        "team": team,
                        "rating": self.initial_rating,
                        "season": season,
                        "last_updated": pd.NaT,
                    }
                )
        if new_entries:
            self.ratings = pd.concat(
                [self.ratings, pd.DataFrame(new_entries)],
                ignore_index=True,
            )

    def get_rating(self, team: str) -> float:
        """Return the latest rating for the specified team."""
        if team not in self.ratings["team"].values:
            self._add_team(team, season=0)
        return float(self.ratings.loc[self.ratings["team"] == team, "rating"].iloc[-1])

    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        season: int,
        match_date: Optional[datetime] = None,
    ) -> EloMatchResult:
        """Update ratings based on a match result."""
        self._ensure_team(home_team, season)
        self._ensure_team(away_team, season)

        home_rating_before = self.get_rating(home_team)
        away_rating_before = self.get_rating(away_team)

        home_expected, away_expected = self.expected_result(home_team, away_team)
        home_outcome = self._match_outcome(home_goals, away_goals)
        away_outcome = 1.0 - home_outcome

        home_delta = self.k_factor * (home_outcome - home_expected)
        away_delta = -home_delta

        home_rating_after = home_rating_before + home_delta
        away_rating_after = away_rating_before + away_delta

        self._set_rating(home_team, home_rating_after, season, match_date)
        self._set_rating(away_team, away_rating_after, season, match_date)

        result = EloMatchResult(
            home_team=home_team,
            away_team=away_team,
            home_goals=home_goals,
            away_goals=away_goals,
            season=season,
            match_date=match_date,
            home_expected=home_expected,
            away_expected=away_expected,
            home_rating_delta=float(home_delta),
            away_rating_delta=float(away_delta),
            home_rating_before=float(home_rating_before),
            away_rating_before=float(away_rating_before),
            home_rating_after=float(home_rating_after),
            away_rating_after=float(away_rating_after),
        )
        self._append_history(result)
        return result

    def expected_result(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Calculate expected probabilities for home win and away win."""
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        rating_diff = home_rating + self.home_advantage - away_rating
        home_expected = 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))
        away_expected = 1.0 - home_expected
        return float(home_expected), float(away_expected)

    def get_rating_history(self, team: Optional[str] = None) -> pd.DataFrame:
        """Return the full rating update history, optionally filtered by team."""
        if team:
            return self.history[self.history["home_team"].eq(team) | self.history["away_team"].eq(team)].copy()
        return self.history.copy()

    def get_current_ratings(self) -> pd.DataFrame:
        """Return the most recent rating for each team."""
        return (
            self.ratings.sort_values("last_updated")
            .drop_duplicates(subset=["team"], keep="last")
            .reset_index(drop=True)
        )

    def _match_outcome(self, home_goals: int, away_goals: int) -> float:
        if home_goals > away_goals:
            return 1.0
        if home_goals == away_goals:
            return 0.5
        return 0.0

    def _ensure_team(self, team: str, season: int) -> None:
        if team not in self.ratings["team"].values:
            self._add_team(team, season)

    def _add_team(self, team: str, season: int) -> None:
        self.ratings = pd.concat(
            [
                self.ratings,
                pd.DataFrame(
                    [
                        {
                            "team": team,
                            "rating": self.initial_rating,
                            "season": season,
                            "last_updated": pd.NaT,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    def _set_rating(self, team: str, rating: float, season: int, match_date: Optional[datetime]) -> None:
        self.ratings = pd.concat(
            [
                self.ratings,
                pd.DataFrame(
                    [
                        {
                            "team": team,
                            "rating": float(rating),
                            "season": season,
                            "last_updated": match_date,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    def _append_history(self, result: EloMatchResult) -> None:
        self.history = pd.concat(
            [
                self.history,
                pd.DataFrame(
                    [
                        {
                            "season": result.season,
                            "match_date": result.match_date,
                            "home_team": result.home_team,
                            "away_team": result.away_team,
                            "home_rating_before": result.home_rating_before,
                            "away_rating_before": result.away_rating_before,
                            "home_expected": result.home_expected,
                            "away_expected": result.away_expected,
                            "home_goals": result.home_goals,
                            "away_goals": result.away_goals,
                            "home_rating_after": result.home_rating_after,
                            "away_rating_after": result.away_rating_after,
                            "home_rating_delta": result.home_rating_delta,
                            "away_rating_delta": result.away_rating_delta,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )


__all__ = ["EloRatingSystem", "EloMatchResult"]
