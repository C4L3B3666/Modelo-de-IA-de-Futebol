"""Weighted Poisson baseline for football score forecasting.

The model uses only historical matches. It estimates league scoring rates, home
advantage, and team attack/defense strengths with exponential time decay so
recent matches influence predictions more than old matches.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


MATCH_COLUMNS = ["date", "home_team", "away_team", "home_goals", "away_goals"]


class PoissonModel:
    """Base independent Poisson model with weighted historical averages."""

    def __init__(
        self,
        max_goals: int = 10,
        half_life_days: float = 365.0,
        min_lambda: float = 0.05,
        smoothing: float = 0.08,
    ) -> None:
        self.max_goals = max_goals
        self.half_life_days = half_life_days
        self.min_lambda = min_lambda
        self.smoothing = smoothing
        self.fitted_ = False

    def fit(self, matches: pd.DataFrame, as_of_date: Optional[pd.Timestamp] = None) -> "PoissonModel":
        data = validate_matches(matches)
        self.as_of_date_ = pd.to_datetime(as_of_date) if as_of_date is not None else data["date"].max()
        data = data.loc[data["date"] <= self.as_of_date_].copy()
        if data.empty:
            raise ValueError("No historical matches available at or before as_of_date.")

        data["weight"] = temporal_weights(data["date"], self.as_of_date_, self.half_life_days)
        self.teams_ = sorted(pd.unique(data[["home_team", "away_team"]].values.ravel()))

        home_weight = data["weight"].sum()
        away_weight = data["weight"].sum()
        self.avg_home_goals_ = weighted_mean(data["home_goals"], data["weight"])
        self.avg_away_goals_ = weighted_mean(data["away_goals"], data["weight"])
        self.home_advantage_ = math.log(
            max(self.avg_home_goals_, self.min_lambda) / max(self.avg_away_goals_, self.min_lambda)
        )

        rows = []
        for team in self.teams_:
            home = data.loc[data["home_team"] == team]
            away = data.loc[data["away_team"] == team]
            attack_home = weighted_mean(home["home_goals"], home["weight"], self.avg_home_goals_)
            defense_home = weighted_mean(home["away_goals"], home["weight"], self.avg_away_goals_)
            attack_away = weighted_mean(away["away_goals"], away["weight"], self.avg_away_goals_)
            defense_away = weighted_mean(away["home_goals"], away["weight"], self.avg_home_goals_)

            rows.append(
                {
                    "team": team,
                    "attack_home": shrink_ratio(attack_home, self.avg_home_goals_, len(home), self.smoothing),
                    "defense_home": shrink_ratio(defense_home, self.avg_away_goals_, len(home), self.smoothing),
                    "attack_away": shrink_ratio(attack_away, self.avg_away_goals_, len(away), self.smoothing),
                    "defense_away": shrink_ratio(defense_away, self.avg_home_goals_, len(away), self.smoothing),
                }
            )

        self.strengths_ = pd.DataFrame(rows).set_index("team")
        self.fitted_ = True
        return self

    def expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        self._ensure_fitted()
        home = self._team_strength(home_team)
        away = self._team_strength(away_team)
        lambda_home = self.avg_home_goals_ * home["attack_home"] * away["defense_away"]
        lambda_away = self.avg_away_goals_ * away["attack_away"] * home["defense_home"]
        return max(float(lambda_home), self.min_lambda), max(float(lambda_away), self.min_lambda)

    def predict_expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Compatibility alias used by higher-level predictors."""
        return self.expected_goals(home_team, away_team)

    def score_matrix(
        self,
        home_team: str,
        away_team: str,
        max_goals: Optional[int] = None,
    ) -> pd.DataFrame:
        lambda_home, lambda_away = self.expected_goals(home_team, away_team)
        return independent_poisson_distribution(lambda_home, lambda_away, self.max_goals if max_goals is None else max_goals)

    def predict_score_probabilities(
        self,
        home_team: str,
        away_team: str,
        max_goals: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compatibility alias returning exact-score probabilities."""
        return self.score_matrix(home_team, away_team, max_goals)

    def outcome_probabilities(self, home_team: str, away_team: str, max_goals: Optional[int] = None) -> Dict[str, float]:
        distribution = self.score_matrix(home_team, away_team, max_goals)
        return outcome_probabilities_from_scores(distribution)

    def _team_strength(self, team: str) -> pd.Series:
        if team not in self.strengths_.index:
            raise KeyError(f"Unknown team: {team}")
        return self.strengths_.loc[team]

    def _ensure_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("PoissonModel is not fitted.")


def validate_matches(matches: pd.DataFrame, required: Iterable[str] = MATCH_COLUMNS) -> pd.DataFrame:
    missing = [column for column in required if column not in matches.columns]
    if missing:
        raise ValueError(f"Missing required match columns: {missing}")
    data = matches.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["home_team"] = data["home_team"].astype(str).str.strip()
    data["away_team"] = data["away_team"].astype(str).str.strip()
    data["home_goals"] = pd.to_numeric(data["home_goals"], errors="coerce")
    data["away_goals"] = pd.to_numeric(data["away_goals"], errors="coerce")
    data = data.dropna(subset=MATCH_COLUMNS)
    data["home_goals"] = data["home_goals"].astype(int)
    data["away_goals"] = data["away_goals"].astype(int)
    data = data.sort_values("date").reset_index(drop=True)
    if data.empty:
        raise ValueError("Match data is empty after validation.")
    if (data[["home_goals", "away_goals"]] < 0).any().any():
        raise ValueError("Goals must be non-negative.")
    return data


def temporal_weights(dates: pd.Series, as_of_date: pd.Timestamp, half_life_days: float) -> np.ndarray:
    if half_life_days <= 0:
        return np.ones(len(dates), dtype=float)
    age_days = (pd.to_datetime(as_of_date) - pd.to_datetime(dates)).dt.days.clip(lower=0)
    return np.power(0.5, age_days.to_numpy(dtype=float) / half_life_days)


def weighted_mean(values: pd.Series, weights: pd.Series | np.ndarray, default: float = 0.0) -> float:
    if len(values) == 0:
        return float(default)
    weights_array = np.asarray(weights, dtype=float)
    total = weights_array.sum()
    if total <= 0:
        return float(default)
    return float(np.dot(values.to_numpy(dtype=float), weights_array) / total)


def shrink_ratio(value: float, baseline: float, sample_size: int, smoothing: float) -> float:
    baseline = max(float(baseline), 1e-9)
    reliability = sample_size / (sample_size + max(smoothing * 100.0, 1.0))
    shrunk = reliability * value + (1.0 - reliability) * baseline
    return max(float(shrunk / baseline), 1e-6)


def poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-12)
    return float(math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)))


def independent_poisson_distribution(lambda_home: float, lambda_away: float, max_goals: int) -> pd.DataFrame:
    rows = []
    for home_goals in range(max_goals + 1):
        p_home = poisson_pmf(home_goals, lambda_home)
        for away_goals in range(max_goals + 1):
            rows.append(
                {
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "probability": p_home * poisson_pmf(away_goals, lambda_away),
                }
            )
    result = pd.DataFrame(rows)
    result["probability"] = result["probability"] / result["probability"].sum()
    return result.sort_values("probability", ascending=False).reset_index(drop=True)


def outcome_probabilities_from_scores(scores: pd.DataFrame) -> Dict[str, float]:
    return {
        "home_win": float(scores.loc[scores["home_goals"] > scores["away_goals"], "probability"].sum()),
        "draw": float(scores.loc[scores["home_goals"] == scores["away_goals"], "probability"].sum()),
        "away_win": float(scores.loc[scores["home_goals"] < scores["away_goals"], "probability"].sum()),
    }


__all__ = [
    "PoissonModel",
    "independent_poisson_distribution",
    "outcome_probabilities_from_scores",
    "poisson_pmf",
    "temporal_weights",
    "validate_matches",
]
