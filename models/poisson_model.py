"""Poisson goal model for football match prediction.

This module implements an object-oriented Poisson regression model tailored for football
score forecasting. It estimates offensive and defensive strengths for each team,
accounts for home advantage, and computes full scoreline probabilities.
"""

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd


class PoissonGoalModel:
    """Modelo de gols Poisson para futebol.

    O modelo estima forças ofensivas e defensivas de times em casa e fora, e
    utiliza uma vantagem de casa explícita para prever distribuições de gols.
    """

    def __init__(self, max_goals: int = 10, smoothing: float = 1e-8) -> None:
        self.max_goals = max_goals
        self.smoothing = smoothing
        self.fitted_ = False
        self.league_avg_home_goals_: float = 0.0
        self.league_avg_away_goals_: float = 0.0
        self.home_advantage_: float = 1.0
        self.team_strengths_: pd.DataFrame = pd.DataFrame()

    def fit(
        self,
        matches: pd.DataFrame,
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
    ) -> "PoissonGoalModel":
        """Ajusta o modelo a partir de um DataFrame de partidas.

        Args:
            matches: DataFrame contendo uma linha por partida.
            home_team_col: Nome da coluna do time da casa.
            away_team_col: Nome da coluna do time visitante.
            home_goals_col: Nome da coluna de gols do time da casa.
            away_goals_col: Nome da coluna de gols do time visitante.

        Returns:
            A instância ajustada do modelo.
        """
        self._validate_input(
            matches,
            [home_team_col, away_team_col, home_goals_col, away_goals_col],
        )

        self.league_avg_home_goals_ = float(matches[home_goals_col].mean())
        self.league_avg_away_goals_ = float(matches[away_goals_col].mean())
        self.home_advantage_ = self._compute_home_advantage(
            self.league_avg_home_goals_, self.league_avg_away_goals_
        )

        self.team_strengths_ = self._compute_team_strengths(
            matches,
            home_team_col,
            away_team_col,
            home_goals_col,
            away_goals_col,
        )

        self.fitted_ = True
        return self

    def predict_expected_goals(
        self, home_team: str, away_team: str
    ) -> Tuple[float, float]:
        """Calcula os gols esperados de home e away para uma partida.

        Args:
            home_team: Nome do time da casa.
            away_team: Nome do time visitante.

        Returns:
            Tupla com (home_expected_goals, away_expected_goals).
        """
        self._ensure_fitted()

        home_strength = self._get_team_strength(home_team)
        away_strength = self._get_team_strength(away_team)

        baseline = float(np.sqrt(self.league_avg_home_goals_ * self.league_avg_away_goals_))
        home_expected = (
            baseline
            * self.home_advantage_
            * home_strength["attack_home"]
            * away_strength["defense_away"]
        )
        away_expected = (
            baseline
            * home_strength["defense_home"]
            * away_strength["attack_away"]
        )

        return float(home_expected), float(away_expected)

    def predict_score_probabilities(
        self, home_team: str, away_team: str, max_goals: int | None = None
    ) -> pd.DataFrame:
        """Retorna a distribuição de probabilidades para placares possíveis.

        Args:
            home_team: Nome do time da casa.
            away_team: Nome do time visitante.
            max_goals: Máximo de gols considerados em cada eixo.

        Returns:
            DataFrame com colunas [home_goals, away_goals, probability].
        """
        self._ensure_fitted()
        max_goals = max_goals if max_goals is not None else self.max_goals

        home_expectation, away_expectation = self.predict_expected_goals(home_team, away_team)

        probabilities = []
        for home_goals in range(max_goals + 1):
            home_p = self._poisson_pmf(home_goals, home_expectation)
            for away_goals in range(max_goals + 1):
                away_p = self._poisson_pmf(away_goals, away_expectation)
                probabilities.append(
                    {
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "probability": float(home_p * away_p),
                    }
                )

        return pd.DataFrame(probabilities).sort_values(
            by=["probability"], ascending=False
        ).reset_index(drop=True)

    def get_team_strengths(self) -> pd.DataFrame:
        """Retorna o DataFrame de forças ofensivas e defensivas por time."""
        self._ensure_fitted()
        return self.team_strengths_.copy()

    def _compute_team_strengths(
        self,
        matches: pd.DataFrame,
        home_team_col: str,
        away_team_col: str,
        home_goals_col: str,
        away_goals_col: str,
    ) -> pd.DataFrame:
        """Calcula forças de ataque e defesa baseadas em médias por time."""
        home_for = matches.groupby(home_team_col)[home_goals_col].mean()
        home_against = matches.groupby(home_team_col)[away_goals_col].mean()
        away_for = matches.groupby(away_team_col)[away_goals_col].mean()
        away_against = matches.groupby(away_team_col)[home_goals_col].mean()

        strengths = pd.DataFrame(
            {
                "attack_home": home_for / (self.league_avg_home_goals_ + self.smoothing),
                "defense_home": home_against / (self.league_avg_away_goals_ + self.smoothing),
                "attack_away": away_for / (self.league_avg_away_goals_ + self.smoothing),
                "defense_away": away_against / (self.league_avg_home_goals_ + self.smoothing),
            }
        )

        strengths = strengths.fillna(1.0)
        strengths["attack_home"] = strengths["attack_home"].clip(lower=self.smoothing)
        strengths["attack_away"] = strengths["attack_away"].clip(lower=self.smoothing)
        strengths["defense_home"] = strengths["defense_home"].clip(lower=self.smoothing)
        strengths["defense_away"] = strengths["defense_away"].clip(lower=self.smoothing)

        return strengths

    def _compute_home_advantage(self, avg_home_goals: float, avg_away_goals: float) -> float:
        """Calcula a vantagem de casa como razão entre as médias de gols."""
        if avg_away_goals <= 0:
            return 1.0
        return float(avg_home_goals / avg_away_goals)

    def _validate_input(self, matches: pd.DataFrame, required_columns: list[str]) -> None:
        missing = [col for col in required_columns if col not in matches.columns]
        if missing:
            raise ValueError(f"DataFrame de partidas faltando colunas obrigatórias: {missing}")

    def _ensure_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("Modelo Poisson não foi ajustado. Chame fit() antes de prever.")

    @staticmethod
    def _poisson_pmf(k: int, lam: float) -> float:
        """Calcula a probabilidade Poisson de k gols com parâmetro lambda."""
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return float(math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)))


__all__ = ["PoissonGoalModel"]
