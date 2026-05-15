"""Entrypoint profissional para previsão de placares de futebol.

O aplicativo carrega partidas de um CSV, ajusta um modelo Poisson, faz a previsão
para um confronto e imprime probabilidades e placares prováveis.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from models.poisson_model import PoissonGoalModel


DEFAULT_DATA_PATH = Path("data/real_matches.csv")
DEFAULT_MAX_GOALS = 6


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina um modelo de previsão de futebol e gera probabilidades de placares."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Caminho para o CSV de partidas com colunas home_team, away_team, home_goals, away_goals.",
    )
    parser.add_argument(
        "--home-team",
        default=None,
        help="Nome do time da casa para previsão. Se omitido, usa o primeiro confronto do CSV.",
    )
    parser.add_argument(
        "--away-team",
        default=None,
        help="Nome do time visitante para previsão. Se omitido, usa o primeiro confronto do CSV.",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=DEFAULT_MAX_GOALS,
        help="Máximo de gols considerados na distribuição de placares.",
    )
    parser.add_argument(
        "--top-scores",
        type=int,
        default=10,
        help="Número de placares mais prováveis a exibir.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nível de log para execução.",
    )
    return parser.parse_args()


def load_match_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        alternate = Path("data/real_matches.csv") if path != Path("data/real_matches.csv") else Path("data/raw/real_matches.csv")
        if alternate.exists():
            path = alternate
        else:
            raise FileNotFoundError(f"Arquivo de dados não encontrado: {path}")

    df = pd.read_csv(path)
    required_columns = ["home_team", "away_team", "home_goals", "away_goals"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"CSV de partidas falta as seguintes colunas obrigatórias: {missing}"
        )

    return df


def format_probability(value: float) -> str:
    return f"{value * 100:.2f}%"


def display_score_probabilities(probabilities: pd.DataFrame, top_n: int) -> None:
    top_scores = probabilities.head(top_n)
    print("\nPlacar prováveis:")
    print("Home  Away  Probabilidade")
    print("----- ----- ------------")
    for _, row in top_scores.iterrows():
        print(
            f"{int(row['home_goals']):>4} {int(row['away_goals']):>6} {row['probability'] * 100:>11.2f}%"
        )


def compute_outcome_probabilities(probabilities: pd.DataFrame) -> Dict[str, float]:
    return {
        "Vitória casa": float(
            probabilities.loc[probabilities["home_goals"] > probabilities["away_goals"], "probability"].sum()
        ),
        "Empate": float(
            probabilities.loc[probabilities["home_goals"] == probabilities["away_goals"], "probability"].sum()
        ),
        "Vitória fora": float(
            probabilities.loc[probabilities["home_goals"] < probabilities["away_goals"], "probability"].sum()
        ),
    }


def main() -> int:
    args = parse_arguments()
    setup_logging(args.log_level)
    logger = logging.getLogger("app")
    logger.info("Carregando dados de %s", args.data)

    matches = load_match_data(args.data)
    logger.info("Dados carregados: %d partidas", len(matches))

    model = PoissonGoalModel(max_goals=args.max_goals)
    model.fit(matches)
    logger.info("Modelo Poisson ajustado com sucesso.")

    home_team = args.home_team
    away_team = args.away_team
    if home_team is None or away_team is None:
        first_match = matches.iloc[0]
        home_team = first_match["home_team"] if home_team is None else home_team
        away_team = first_match["away_team"] if away_team is None else away_team
        logger.info(
            "Nenhum time informado. Usando confronto padrão do CSV: %s x %s",
            home_team,
            away_team,
        )

    home_expected, away_expected = model.predict_expected_goals(
        home_team, away_team
    )
    probabilities = model.predict_score_probabilities(
        home_team, away_team, max_goals=args.max_goals
    )

    print("\nPrevisão para:")
    print(f"  Casa : {args.home_team}")
    print(f"  Fora : {args.away_team}")
    print(f"  Gols esperados (casa) : {home_expected:.3f}")
    print(f"  Gols esperados (fora) : {away_expected:.3f}")

    outcome_prob = compute_outcome_probabilities(probabilities)
    print("\nProbabilidades de resultado:")
    for label, value in outcome_prob.items():
        print(f"  {label:14}: {format_probability(value)}")

    display_score_probabilities(probabilities, args.top_scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
