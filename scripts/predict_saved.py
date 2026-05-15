"""Run inference from a persisted trained predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


DEFAULT_MODEL_PATH = Path("models/artifacts/football_predictor.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a match using a saved model artifact.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--home-team", required=True)
    parser.add_argument("--away-team", required=True)
    parser.add_argument("--top-scores", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = joblib.load(args.model_path)
    predictor = artifact["predictor"]
    prediction = predictor.predict(args.home_team, args.away_team, top_n=args.top_scores)

    print(f"{args.home_team} x {args.away_team}")
    print(
        "xG:",
        f"{prediction['expected_goals']['home']:.3f}",
        "-",
        f"{prediction['expected_goals']['away']:.3f}",
    )
    print("Probabilidades:")
    for key, value in prediction["probabilities"].items():
        print(f"  {key}: {value * 100:.2f}%")
    print("Placares mais prováveis:")
    for row in prediction["top_scores"]:
        print(
            f"  {int(row['home_goals'])}-{int(row['away_goals'])}: "
            f"{row['probability'] * 100:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
