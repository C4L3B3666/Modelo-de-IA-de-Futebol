"""Dixon-Coles football model with low-score correction and Poisson integration.

This module implements a mathematically correct Dixon-Coles goal model with:
- maximum likelihood estimation of home/away attack and defense strengths,
- low-score correction for 0-0, 1-0, 0-1, and 1-1 matches,
- parameter optimization using SciPy,
- Poisson goal probability integration,
- object-oriented architecture with evaluation/backtesting helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except ImportError as exc:
    raise ImportError(
        "Dixon-Coles training requires scipy. Install scipy>=1.11.0."
    ) from exc


@dataclass
class DixonColesFitResult:
    success: bool
    fun: float
    n_iterations: int
    message: str
    x: np.ndarray


class DixonColesModel:
    """Dixon-Coles model for football scoreline forecasting.

    The model estimates separate home and away attack and defense strengths,
    includes a home advantage term, and applies a low-score correction factor
    to improve fit for match results with 0 or 1 goals.
    """

    def __init__(
        self,
        max_goals: int = 8,
        rho_bounds: Tuple[float, float] = (-0.99, 0.99),
        regularization: float = 1e-4,
        tolerance: float = 1e-8,
        verbose: bool = False,
    ) -> None:
        self.max_goals = max_goals
        self.rho_bounds = rho_bounds
        self.regularization = regularization
        self.tolerance = tolerance
        self.verbose = verbose

        self.fitted_ = False
        self.team_index_: Dict[str, int] = {}
        self.team_names_: List[str] = []
        self.home_attack_: np.ndarray = np.array([])
        self.away_attack_: np.ndarray = np.array([])
        self.home_defense_: np.ndarray = np.array([])
        self.away_defense_: np.ndarray = np.array([])
        self.home_advantage_: float = 0.0
        self.rho_: float = 0.0
        self.fit_result_: Optional[DixonColesFitResult] = None

    def fit(
        self,
        matches: pd.DataFrame,
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        method: str = "L-BFGS-B",
        maxiter: int = 1000,
    ) -> "DixonColesModel":
        """Fit the Dixon-Coles model to a DataFrame of matches."""
        self._validate_input(
            matches,
            [home_team_col, away_team_col, home_goals_col, away_goals_col],
        )

        matches = matches.copy()
        matches[home_team_col] = matches[home_team_col].astype(str)
        matches[away_team_col] = matches[away_team_col].astype(str)

        self.team_names_ = sorted(
            pd.unique(matches[[home_team_col, away_team_col]].values.ravel())
        )
        if len(self.team_names_) < 2:
            raise ValueError("É necessário pelo menos dois times para ajustar o modelo.")

        self.team_index_ = {team: idx for idx, team in enumerate(self.team_names_)}
        self.n_teams_ = len(self.team_names_)

        self.home_team_idx_ = matches[home_team_col].map(self.team_index_).to_numpy(dtype=int)
        self.away_team_idx_ = matches[away_team_col].map(self.team_index_).to_numpy(dtype=int)
        self.home_goals_ = matches[home_goals_col].to_numpy(dtype=int)
        self.away_goals_ = matches[away_goals_col].to_numpy(dtype=int)

        self.league_avg_home_goals_ = float(self.home_goals_.mean())
        self.league_avg_away_goals_ = float(self.away_goals_.mean())

        initial_params = self._initial_parameter_vector(matches)
        bounds = self._build_parameter_bounds()

        optimization = minimize(
            fun=self._negative_log_likelihood,
            x0=initial_params,
            method=method,
            bounds=bounds,
            options={"maxiter": maxiter, "disp": self.verbose, "gtol": self.tolerance},
        )

        self._unpack_parameters(optimization.x)
        self.fit_result_ = DixonColesFitResult(
            success=optimization.success,
            fun=float(optimization.fun),
            n_iterations=int(getattr(optimization, "nit", 0)),
            message=optimization.message,
            x=optimization.x,
        )
        self.fitted_ = optimization.success
        if not self.fitted_ and self.verbose:
            print("Dixon-Coles fit did not converge:", optimization.message)

        return self

    def predict_expected_goals(
        self, home_team: str, away_team: str
    ) -> Tuple[float, float]:
        """Return the expected home and away goals for a single fixture."""
        self._ensure_fitted()
        if home_team not in self.team_index_ or away_team not in self.team_index_:
            raise KeyError("Times desconhecidos para previsão de gols.")

        home_idx = self.team_index_[home_team]
        away_idx = self.team_index_[away_team]

        home_rate = math.exp(
            self.home_attack_[home_idx]
            + self.away_defense_[away_idx]
            + self.home_advantage_
        )
        away_rate = math.exp(
            self.away_attack_[away_idx] + self.home_defense_[home_idx]
        )
        return float(home_rate), float(away_rate)

    def predict_score_probabilities(
        self, home_team: str, away_team: str, max_goals: Optional[int] = None
    ) -> pd.DataFrame:
        """Return a DataFrame with probability for each plausible scoreline."""
        self._ensure_fitted()
        max_goals = self.max_goals if max_goals is None else max_goals
        home_exp, away_exp = self.predict_expected_goals(home_team, away_team)

        probabilities = []
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = self._score_probability(home_goals, away_goals, home_exp, away_exp)
                probabilities.append(
                    {
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "probability": float(prob),
                    }
                )

        distribution = pd.DataFrame(probabilities)
        distribution = distribution.sort_values(by=["probability"], ascending=False).reset_index(drop=True)
        return distribution

    def predict_outcome_probabilities(
        self, home_team: str, away_team: str, max_goals: Optional[int] = None
    ) -> Dict[str, float]:
        """Return probabilities for home win, draw, and away win."""
        distribution = self.predict_score_probabilities(home_team, away_team, max_goals)
        home_win = float(
            distribution.loc[distribution["home_goals"] > distribution["away_goals"], "probability"].sum()
        )
        draw = float(
            distribution.loc[distribution["home_goals"] == distribution["away_goals"], "probability"].sum()
        )
        away_win = float(
            distribution.loc[distribution["home_goals"] < distribution["away_goals"], "probability"].sum()
        )
        return {"home": home_win, "draw": draw, "away": away_win}

    def get_team_parameters(self) -> pd.DataFrame:
        """Return a DataFrame with the fitted attack and defense strengths."""
        self._ensure_fitted()
        return pd.DataFrame(
            {
                "team": self.team_names_,
                "home_attack": self.home_attack_,
                "away_attack": self.away_attack_,
                "home_defense": self.home_defense_,
                "away_defense": self.away_defense_,
            }
        )

    def evaluate(
        self,
        matches: pd.DataFrame,
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        max_goals: Optional[int] = None,
    ) -> pd.DataFrame:
        """Evaluate the fitted model on a set of matches and return prediction metrics."""
        self._ensure_fitted()
        self._validate_input(
            matches,
            [home_team_col, away_team_col, home_goals_col, away_goals_col],
        )

        rows = []
        max_goals = self.max_goals if max_goals is None else max_goals

        for _, row in matches.iterrows():
            home = str(row[home_team_col])
            away = str(row[away_team_col])
            home_goals = int(row[home_goals_col])
            away_goals = int(row[away_goals_col])

            home_exp, away_exp = self.predict_expected_goals(home, away)
            outcome_probs = self.predict_outcome_probabilities(home, away, max_goals)
            score_prob = self._score_probability(home_goals, away_goals, home_exp, away_exp)
            predicted_outcome = self._match_outcome_from_probabilities(outcome_probs)
            actual_outcome = self._match_outcome(home_goals, away_goals)

            rows.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "home_expected_goals": home_exp,
                    "away_expected_goals": away_exp,
                    "home_win_prob": outcome_probs["home"],
                    "draw_prob": outcome_probs["draw"],
                    "away_win_prob": outcome_probs["away"],
                    "score_prob": score_prob,
                    "predicted_outcome": predicted_outcome,
                    "actual_outcome": actual_outcome,
                }
            )

        results = pd.DataFrame(rows)
        results["log_loss"] = -np.log(results["score_prob"].clip(min=1e-15))
        results["prediction_error"] = np.where(
            results["predicted_outcome"] == results["actual_outcome"], 0.0, 1.0
        )
        return results

    def backtest(
        self,
        matches: pd.DataFrame,
        train_fraction: float = 0.7,
        date_col: Optional[str] = None,
        **fit_kwargs: Any,
    ) -> pd.DataFrame:
        """Run a simple training/test backtest split and return evaluation results."""
        if date_col is not None and date_col in matches.columns:
            matches = matches.sort_values(date_col)
        else:
            matches = matches.reset_index(drop=True)

        split = int(len(matches) * train_fraction)
        if split < 1 or split >= len(matches):
            raise ValueError("train_fraction deve estar entre 0 e 1, com pelo menos um jogo em cada conjunto.")

        training = matches.iloc[:split].reset_index(drop=True)
        testing = matches.iloc[split:].reset_index(drop=True)

        self.fit(training, **fit_kwargs)
        evaluation = self.evaluate(testing)
        return evaluation

    def _initial_parameter_vector(self, matches: pd.DataFrame) -> np.ndarray:
        team_count = self.n_teams_
        home_scored = matches.groupby("home_team")["home_goals"].mean()
        away_scored = matches.groupby("away_team")["away_goals"].mean()
        home_conceded = matches.groupby("home_team")["away_goals"].mean()
        away_conceded = matches.groupby("away_team")["home_goals"].mean()

        home_attack = np.array(
            [
                math.log((home_scored.get(team, 0.0) + 1e-3) / (self.league_avg_home_goals_ + 1e-3))
                for team in self.team_names_
            ]
        )
        away_attack = np.array(
            [
                math.log((away_scored.get(team, 0.0) + 1e-3) / (self.league_avg_away_goals_ + 1e-3))
                for team in self.team_names_
            ]
        )
        away_defense = np.array(
            [
                math.log((home_conceded.get(team, 0.0) + 1e-3) / (self.league_avg_home_goals_ + 1e-3))
                for team in self.team_names_
            ]
        )
        home_defense = np.array(
            [
                math.log((away_conceded.get(team, 0.0) + 1e-3) / (self.league_avg_away_goals_ + 1e-3))
                for team in self.team_names_
            ]
        )

        home_attack -= home_attack.mean()
        away_attack -= away_attack.mean()

        init_home_attack = home_attack[1:]
        init_away_attack = away_attack[1:]
        initial_vector = np.concatenate(
            [
                init_home_attack,
                init_away_attack,
                away_defense,
                home_defense,
                np.array([math.log(self.league_avg_home_goals_ / max(self.league_avg_away_goals_, 1e-3))]),
                np.array([0.0]),
            ]
        )
        return initial_vector

    def _build_parameter_bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        n = self.n_teams_
        bounds: List[Tuple[Optional[float], Optional[float]]] = []
        bounds.extend([(-5.0, 5.0)] * (n - 1))
        bounds.extend([(-5.0, 5.0)] * (n - 1))
        bounds.extend([(-5.0, 5.0)] * n)
        bounds.extend([(-5.0, 5.0)] * n)
        bounds.append((None, None))
        bounds.append(self.rho_bounds)
        return bounds

    def _unpack_parameters(self, x: np.ndarray) -> None:
        n = self.n_teams_
        idx = 0
        home_attack = np.zeros(n, dtype=float)
        away_attack = np.zeros(n, dtype=float)
        home_attack[1:] = x[idx : idx + n - 1]
        home_attack[0] = -home_attack[1:].sum()
        idx += n - 1

        away_attack[1:] = x[idx : idx + n - 1]
        away_attack[0] = -away_attack[1:].sum()
        idx += n - 1

        away_defense = x[idx : idx + n]
        idx += n
        home_defense = x[idx : idx + n]
        idx += n

        home_advantage = float(x[idx]); idx += 1
        rho = float(x[idx])

        self.home_attack_ = home_attack
        self.away_attack_ = away_attack
        self.away_defense_ = away_defense
        self.home_defense_ = home_defense
        self.home_advantage_ = home_advantage
        self.rho_ = rho

    def _negative_log_likelihood(self, x: np.ndarray) -> float:
        n = self.n_teams_
        idx = 0

        home_attack = np.zeros(n, dtype=float)
        away_attack = np.zeros(n, dtype=float)
        home_attack[1:] = x[idx : idx + n - 1]
        home_attack[0] = -home_attack[1:].sum()
        idx += n - 1

        away_attack[1:] = x[idx : idx + n - 1]
        away_attack[0] = -away_attack[1:].sum()
        idx += n - 1

        away_defense = x[idx : idx + n]
        idx += n
        home_defense = x[idx : idx + n]
        idx += n

        home_advantage = float(x[idx]); idx += 1
        rho = float(x[idx])

        home_rates = np.exp(
            home_attack[self.home_team_idx_]
            + away_defense[self.away_team_idx_]
            + home_advantage
        )
        away_rates = np.exp(
            away_attack[self.away_team_idx_]
            + home_defense[self.home_team_idx_]
        )

        log_p_home = self._poisson_logpmf(self.home_goals_, home_rates)
        log_p_away = self._poisson_logpmf(self.away_goals_, away_rates)
        log_tau = self._dixon_coles_log_tau(
            self.home_goals_, self.away_goals_, home_rates, away_rates, rho
        )

        log_likelihood = np.sum(log_p_home + log_p_away + log_tau)
        penalty = self.regularization * (
            np.sum(home_attack[1:] ** 2)
            + np.sum(away_attack[1:] ** 2)
            + np.sum(home_defense ** 2)
            + np.sum(away_defense ** 2)
        )
        return float(-log_likelihood + penalty)

    def _dixon_coles_log_tau(
        self,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        home_rates: np.ndarray,
        away_rates: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        tau = np.ones_like(home_rates, dtype=float)
        mask00 = (home_goals == 0) & (away_goals == 0)
        mask01 = (home_goals == 0) & (away_goals == 1)
        mask10 = (home_goals == 1) & (away_goals == 0)
        mask11 = (home_goals == 1) & (away_goals == 1)

        tau[mask00] = 1.0 - rho * home_rates[mask00] * away_rates[mask00]
        tau[mask01] = 1.0 + rho * home_rates[mask01]
        tau[mask10] = 1.0 + rho * away_rates[mask10]
        tau[mask11] = 1.0 - rho

        if np.any(tau <= 0):
            return np.full_like(tau, -1e8)

        return np.log(tau)

    def _score_probability(
        self, home_goals: int, away_goals: int, home_rate: float, away_rate: float
    ) -> float:
        base = self._poisson_pmf(home_goals, home_rate) * self._poisson_pmf(away_goals, away_rate)
        correction = self._dixon_coles_tau_scalar(home_goals, away_goals, home_rate, away_rate, self.rho_)
        return float(base * correction)

    def _dixon_coles_tau_scalar(
        self, home_goals: int, away_goals: int, home_rate: float, away_rate: float, rho: float
    ) -> float:
        if home_goals == 0 and away_goals == 0:
            return max(1.0 - rho * home_rate * away_rate, 0.0)
        if home_goals == 0 and away_goals == 1:
            return 1.0 + rho * home_rate
        if home_goals == 1 and away_goals == 0:
            return 1.0 + rho * away_rate
        if home_goals == 1 and away_goals == 1:
            return 1.0 - rho
        return 1.0

    @staticmethod
    def _poisson_pmf(k: int, lam: float) -> float:
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return float(math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)))

    @staticmethod
    def _poisson_logpmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
        lam = np.asarray(lam, dtype=float)
        k = np.asarray(k, dtype=int)
        result = np.where(
            lam <= 0,
            np.where(k == 0, 0.0, -1e8),
            -lam + k * np.log(lam) - np.log(np.array([math.gamma(val + 1) for val in k], dtype=float)),
        )
        return result

    @staticmethod
    def _match_outcome(home_goals: int, away_goals: int) -> str:
        if home_goals > away_goals:
            return "home"
        if home_goals == away_goals:
            return "draw"
        return "away"

    @staticmethod
    def _match_outcome_from_probabilities(outcome_probs: Dict[str, float]) -> str:
        if outcome_probs["home"] >= outcome_probs["draw"] and outcome_probs["home"] >= outcome_probs["away"]:
            return "home"
        if outcome_probs["draw"] >= outcome_probs["away"]:
            return "draw"
        return "away"

    @staticmethod
    def _validate_input(matches: pd.DataFrame, required_columns: List[str]) -> None:
        missing = [col for col in required_columns if col not in matches.columns]
        if missing:
            raise ValueError(f"DataFrame de partidas faltando colunas obrigatórias: {missing}")

    def _ensure_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("Modelo Dixon-Coles não foi ajustado. Chame fit() antes de prever.")


__all__ = ["DixonColesModel", "DixonColesFitResult"]
