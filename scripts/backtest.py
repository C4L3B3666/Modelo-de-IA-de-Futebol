"""Offline walk-forward backtest for the football predictor."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from models.predictor import Backtester
from models.poisson import validate_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline historical backtest.")
    parser.add_argument("--data", type=Path, default=Path("data/real_matches.csv"))
    parser.add_argument("--train-until", default=None, help="Date T. Train up to T and predict matches after T.")
    parser.add_argument("--model", choices=["poisson", "dixon_coles"], default="poisson")
    parser.add_argument("--max-goals", type=int, default=8)
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument("--min-train-matches", type=int, default=100)
    parser.add_argument("--retrain-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matches = validate_matches(pd.read_csv(args.data))
    backtester = Backtester(
        model_type=args.model,
        max_goals=args.max_goals,
        stake=args.stake,
        min_train_matches=args.min_train_matches,
        retrain_every=args.retrain_every,
    )
    result = backtester.run(matches, train_until=args.train_until)
    summary = result["summary"]
    print("Backtest summary")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
