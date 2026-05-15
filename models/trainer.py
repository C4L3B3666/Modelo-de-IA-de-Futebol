"""Treinador de modelo com XGBoost e engenharia de features para futebol."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .elo import EloRating


@dataclass
class ModelTrainer:
    elo_k: float = 20.0
    elo_base: int = 1500
    n_rolling: int = 5

    def fit(self, matches: pd.DataFrame, save_dir: Optional[Path] = None) -> Dict[str, object]:
        df = matches.sort_values("date").reset_index(drop=True).copy()

        # compute league home advantage
        league_adv = df.groupby("league")["home_goals"].mean() - df.groupby("league")["away_goals"].mean()
        league_adv = league_adv.to_dict()

        # add Elo ratings before each match
        elo = EloRating(base=self.elo_base, k=self.elo_k)
        elo_home = []
        elo_away = []
        results = []
        for _, row in df.iterrows():
            h = row["home_team"]
            a = row["away_team"]
            elo_home.append(elo.get(h))
            elo_away.append(elo.get(a))
            # result
            if row["home_goals"] > row["away_goals"]:
                res = 1.0
            elif row["home_goals"] == row["away_goals"]:
                res = 0.5
            else:
                res = 0.0
            results.append(res)
            elo.update(h, a, res)

        df["elo_home"] = elo_home
        df["elo_away"] = elo_away

        # rolling features: goals for/against per team
        df["home_goals_for_rolling"] = 0.0
        df["home_goals_against_rolling"] = 0.0
        df["away_goals_for_rolling"] = 0.0
        df["away_goals_against_rolling"] = 0.0

        # precompute per-team rolling using groupby + apply
        teams = pd.unique(df["home_team"].tolist() + df["away_team"].tolist())
        df_indexed = df.copy()
        df_indexed["match_id"] = np.arange(len(df_indexed))
        team_stats = {t: [] for t in teams}

        # build per-row features by scanning chronologically
        last_match_date: Dict[str, pd.Timestamp] = {}
        days_since_home = []
        days_since_away = []

        # maintain deque-like lists for last n matches per team
        recent_for: Dict[str, list] = {t: [] for t in teams}
        recent_against: Dict[str, list] = {t: [] for t in teams}

        for _, row in df.iterrows():
            h = row["home_team"]
            a = row["away_team"]
            date = row.get("date")

            # days since
            if h in last_match_date and pd.notnull(date):
                days_since_home.append((date - last_match_date[h]).days)
            else:
                days_since_home.append(None)
            if a in last_match_date and pd.notnull(date):
                days_since_away.append((date - last_match_date[a]).days)
            else:
                days_since_away.append(None)

            # rolling averages
            hf = np.mean(recent_for[h][-self.n_rolling:]) if recent_for[h] else np.nan
            ha = np.mean(recent_against[h][-self.n_rolling:]) if recent_against[h] else np.nan
            af = np.mean(recent_for[a][-self.n_rolling:]) if recent_for[a] else np.nan
            aa = np.mean(recent_against[a][-self.n_rolling:]) if recent_against[a] else np.nan

            df.loc[df_indexed.index[df_indexed["match_id"] == row.name], "home_goals_for_rolling"] = hf
            df.loc[df_indexed.index[df_indexed["match_id"] == row.name], "home_goals_against_rolling"] = ha
            df.loc[df_indexed.index[df_indexed["match_id"] == row.name], "away_goals_for_rolling"] = af
            df.loc[df_indexed.index[df_indexed["match_id"] == row.name], "away_goals_against_rolling"] = aa

            # after recording features, append current match results
            recent_for[h].append(row["home_goals"])
            recent_against[h].append(row["away_goals"])
            recent_for[a].append(row["away_goals"])
            recent_against[a].append(row["home_goals"])

            last_match_date[h] = date
            last_match_date[a] = date

        df["days_since_home"] = pd.Series(days_since_home)
        df["days_since_away"] = pd.Series(days_since_away)

        # feature assembly
        features = [
            "elo_home",
            "elo_away",
            "home_goals_for_rolling",
            "home_goals_against_rolling",
            "away_goals_for_rolling",
            "away_goals_against_rolling",
            "days_since_home",
            "days_since_away",
        ]

        X = df[features].copy()
        # Impute simple NaNs
        X = X.fillna(X.mean())

        # label: 0 home, 1 draw, 2 away
        y = df.apply(lambda r: 0 if r["home_goals"] > r["away_goals"] else (1 if r["home_goals"] == r["away_goals"] else 2), axis=1)

        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "xgb",
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        use_label_encoder=False,
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        verbosity=0,
                    ),
                ),
            ]
        )

        clf.fit(X, y)

        result = {
            "model": clf,
            "league_adv": league_adv,
            "team_states": {},
        }

        # save latest team states for inference
        team_states = {}
        for t in teams:
            team_states[t] = {
                "elo": elo.get(t),
                "recent_for": recent_for.get(t, [])[-self.n_rolling:],
                "recent_against": recent_against.get(t, [])[-self.n_rolling:],
                "last_date": last_match_date.get(t),
            }
        result["team_states"] = team_states

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            dump(clf, save_dir / "xgb_model.joblib")
            dump(league_adv, save_dir / "league_adv.joblib")
            dump(team_states, save_dir / "team_states.joblib")

        return result
