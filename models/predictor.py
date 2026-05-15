"""High-level offline football predictor and simple betting backtester."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .elo import EloRating
from .features import TemporalFeatureEngineer
from .poisson import PoissonModel, outcome_probabilities_from_scores, validate_matches


@dataclass
class Prediction:
    home_team: str
    away_team: str
    expected_goals_home: float
    expected_goals_away: float
    outcome_probabilities: Dict[str, float]
    over_under: Dict[str, float]
    btts: Dict[str, float]
    top_scores: list[dict]
    score_distribution: pd.DataFrame


class FootballPredictor:
    """Combines Dixon-Coles score probabilities with dynamic Elo context."""

    def __init__(
        self,
        model_type: str = "dixon_coles",
        max_goals: int = 10,
        half_life_days: float = 365.0,
        elo: Optional[EloRating] = None,
    ) -> None:
        self.model_type = model_type
        self.max_goals = max_goals
        self.half_life_days = half_life_days
        self.elo = elo or EloRating()
        self.features = TemporalFeatureEngineer()
        self.model: object
        self.fitted_ = False

    def fit(self, matches: pd.DataFrame, as_of_date: Optional[pd.Timestamp] = None) -> "FootballPredictor":
        data = validate_matches(matches)
        if as_of_date is not None:
            data = data.loc[data["date"] <= pd.to_datetime(as_of_date)].copy()
        if data.empty:
            raise ValueError("No training matches available.")

        self.training_data_ = data
        self.elo = EloRating(
            base_rating=self.elo.base_rating,
            k_factor=self.elo.k_factor,
            home_advantage=self.elo.home_advantage,
            decay_per_year=self.elo.decay_per_year,
        ).fit(data)

        if self.model_type == "poisson":
            self.model = PoissonModel(max_goals=self.max_goals, half_life_days=self.half_life_days).fit(data)
        elif self.model_type == "dixon_coles":
            from .dixon_coles import DixonColesModel

            self.model = DixonColesModel(max_goals=self.max_goals).fit(data)
        else:
            raise ValueError("model_type must be 'poisson' or 'dixon_coles'.")

        self.team_features_ = self.features.latest_team_features(data)
        self.fitted_ = True
        return self

    def predict(self, home_team: str, away_team: str, top_n: int = 8) -> Dict[str, object]:
        self._ensure_fitted()
        expected_home, expected_away = self.model.predict_expected_goals(home_team, away_team)
        scores = self.model.predict_score_probabilities(home_team, away_team, self.max_goals)
        scores = scores.copy()
        scores["probability"] = scores["probability"] / scores["probability"].sum()

        outcome = outcome_probabilities_from_scores(scores)
        return {
            "home_team": home_team,
            "away_team": away_team,
            "expected_goals": {
                "home": float(expected_home),
                "away": float(expected_away),
            },
            "probabilities": outcome,
            "elo": {
                "home_rating": self.elo.get(home_team),
                "away_rating": self.elo.get(away_team),
                "home_win_expectation": self.elo.expected_home_score(home_team, away_team),
            },
            "over_under": self._over_under(scores),
            "btts": self._btts(scores),
            "top_scores": scores.head(top_n).to_dict(orient="records"),
            "score_distribution": scores,
        }

    def predict_match(self, home: str, away: str, top_n: int = 8, **_: object) -> Dict[str, object]:
        """Compatibility wrapper for the previous project API."""
        return self.predict(home, away, top_n=top_n)

    def _over_under(self, scores: pd.DataFrame) -> Dict[str, float]:
        totals = scores["home_goals"] + scores["away_goals"]
        return {
            "over_1_5": float(scores.loc[totals > 1.5, "probability"].sum()),
            "under_1_5": float(scores.loc[totals < 1.5, "probability"].sum()),
            "over_2_5": float(scores.loc[totals > 2.5, "probability"].sum()),
            "under_2_5": float(scores.loc[totals < 2.5, "probability"].sum()),
            "over_3_5": float(scores.loc[totals > 3.5, "probability"].sum()),
            "under_3_5": float(scores.loc[totals < 3.5, "probability"].sum()),
        }

    @staticmethod
    def _btts(scores: pd.DataFrame) -> Dict[str, float]:
        yes = float(scores.loc[(scores["home_goals"] > 0) & (scores["away_goals"] > 0), "probability"].sum())
        return {"yes": yes, "no": 1.0 - yes}

    def _ensure_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("FootballPredictor is not fitted.")


class Backtester:
    """Walk-forward offline backtest with probabilistic accuracy and fixed stake ROI."""

    def __init__(
        self,
        model_type: str = "dixon_coles",
        max_goals: int = 8,
        stake: float = 1.0,
        min_train_matches: int = 100,
        retrain_every: int = 25,
    ) -> None:
        self.model_type = model_type
        self.max_goals = max_goals
        self.stake = stake
        self.min_train_matches = min_train_matches
        self.retrain_every = retrain_every

    def run(self, matches: pd.DataFrame, train_until: Optional[str | pd.Timestamp] = None) -> Dict[str, object]:
        data = validate_matches(matches)
        if train_until is not None:
            train_until_date = pd.to_datetime(train_until)
            start_index = int((data["date"] <= train_until_date).sum())
        else:
            start_index = self.min_train_matches
        if start_index < 2 or start_index >= len(data):
            raise ValueError("Backtest split leaves too few train or test matches.")

        predictor: Optional[FootballPredictor] = None
        rows = []
        profit = 0.0

        for idx in range(start_index, len(data)):
            if predictor is None or (idx - start_index) % self.retrain_every == 0:
                predictor = FootballPredictor(model_type=self.model_type, max_goals=self.max_goals)
                predictor.fit(data.iloc[:idx])

            row = data.iloc[idx]
            try:
                pred = predictor.predict(row["home_team"], row["away_team"])
            except KeyError:
                continue

            probs = pred["probabilities"]
            actual = match_outcome(int(row["home_goals"]), int(row["away_goals"]))
            predicted = max(probs, key=probs.get)
            actual_probability = max(float(probs[actual]), 1e-12)
            selected_probability = float(probs[predicted])
            decimal_odds = 1.0 / max(selected_probability, 1e-12)
            pnl = self.stake * (decimal_odds - 1.0) if predicted == actual else -self.stake
            profit += pnl

            rows.append(
                {
                    "date": row["date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "actual": actual,
                    "predicted": predicted,
                    "actual_probability": actual_probability,
                    "log_loss": -np.log(actual_probability),
                    "brier": brier_score(probs, actual),
                    "pnl": pnl,
                    "cumulative_profit": profit,
                }
            )

        results = pd.DataFrame(rows)
        total_staked = len(results) * self.stake
        return {
            "summary": {
                "matches": int(len(results)),
                "accuracy": float((results["actual"] == results["predicted"]).mean()) if len(results) else 0.0,
                "mean_log_loss": float(results["log_loss"].mean()) if len(results) else float("nan"),
                "mean_brier": float(results["brier"].mean()) if len(results) else float("nan"),
                "profit": float(profit),
                "roi": float(profit / total_staked) if total_staked else 0.0,
            },
            "predictions": results,
        }


def match_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals == away_goals:
        return "draw"
    return "away_win"


def brier_score(probabilities: Dict[str, float], actual: str) -> float:
    return float(sum((probabilities[key] - (1.0 if key == actual else 0.0)) ** 2 for key in probabilities))


# Compatibility alias for older code that imported Predictor.
Predictor = FootballPredictor


__all__ = ["Backtester", "FootballPredictor", "Prediction", "Predictor"]
