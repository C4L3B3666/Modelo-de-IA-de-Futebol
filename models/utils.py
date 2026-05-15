"""Utilitários para carregamento e pré-processamento de partidas JSON/CSV."""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd


def load_json_matches(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_json(p, orient="records")
        # flatten nested json structures
        df_flat = pd.json_normalize(df.to_dict(orient="records"))
        frames.append(df_flat)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    return normalize_matches(df)


def normalize_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Common column normalization
    col_map = {
        "fixture.date": "date",
        "league.name": "league",
        "teams.home.name": "home_team",
        "teams.away.name": "away_team",
        "teams.home.goals": "home_goals",
        "teams.away.goals": "away_goals",
    }
    for src, dst in col_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # Fallbacks for CSV-style columns
    if "date" not in df.columns and "match_date" in df.columns:
        df["date"] = df["match_date"]

    # Parse dates
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Ensure goals as ints
    for g in ("home_goals", "away_goals"):
        if g in df.columns:
            df[g] = pd.to_numeric(df[g], errors="coerce")
            df[g] = df[g].fillna(0).astype(int)

    # Simple team name normalization
    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].astype(str).str.strip()
    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].astype(str).str.strip()

    # Keep only relevant columns if available
    keep = [c for c in ("date", "league", "home_team", "away_team", "home_goals", "away_goals") if c in df.columns]
    return df[keep]


def list_json_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("**/*.json") if p.is_file()])


def get_team_history(df: pd.DataFrame, team: str) -> pd.DataFrame:
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    return df.loc[mask].sort_values("date").reset_index(drop=True)
