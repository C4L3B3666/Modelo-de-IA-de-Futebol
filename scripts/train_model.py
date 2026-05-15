"""Train and persist the offline football prediction model.

This script uses only historical CSV data. It performs a chronological
train/test split, reports probabilistic metrics, then fits the final model on
all available history and saves it for later inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.predictor import Backtester, FootballPredictor
from models.poisson import validate_matches


DEFAULT_DATA_PATH = Path("data/real_matches.csv")
DEFAULT_MODEL_PATH = Path("models/artifacts/football_predictor.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the offline football predictor.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model", choices=["poisson", "dixon_coles"], default="poisson")
    parser.add_argument("--max-goals", type=int, default=8)
    parser.add_argument(
        "--train-until",
        default=None,
        help="Optional split date for validation. Example: 2025-01-01.",
    )
    parser.add_argument("--min-train-matches", type=int, default=300)
    parser.add_argument("--retrain-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matches = validate_matches(pd.read_csv(args.data))

    backtester = Backtester(
        model_type=args.model,
        max_goals=args.max_goals,
        min_train_matches=args.min_train_matches,
        retrain_every=args.retrain_every,
    )
    backtest = backtester.run(matches, train_until=args.train_until)
    summary = backtest["summary"]

    predictor = FootballPredictor(model_type=args.model, max_goals=args.max_goals)
    predictor.fit(matches)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "predictor": predictor,
        "metadata": {
            "model_type": args.model,
            "max_goals": args.max_goals,
            "data_path": str(args.data),
            "matches": int(len(matches)),
            "first_date": str(matches["date"].min().date()),
            "last_date": str(matches["date"].max().date()),
            "teams": int(len(pd.unique(matches[["home_team", "away_team"]].values.ravel()))),
            "backtest": summary,
        },
    }
    joblib.dump(artifact, args.output)

    print("Modelo treinado e salvo com sucesso.")
    print(f"Arquivo: {args.output}")
    print(json.dumps(artifact["metadata"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
