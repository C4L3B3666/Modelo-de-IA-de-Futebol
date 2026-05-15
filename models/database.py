from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_DB_PATH = Path("data/football.db")


def connect_database(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), detect_types=sqlite3.PARSE_DECLTYPES)
    connection.row_factory = sqlite3.Row
    _initialize_schema(connection)
    return connection


def _initialize_schema(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER,
            league_id INTEGER NOT NULL,
            league_name TEXT,
            season INTEGER NOT NULL,
            round TEXT,
            date TEXT NOT NULL,
            status TEXT,
            home_team_id INTEGER,
            home_team_name TEXT,
            away_team_id INTEGER,
            away_team_name TEXT,
            home_goals INTEGER,
            away_goals INTEGER,
            winner TEXT,
            halftime_home INTEGER,
            halftime_away INTEGER,
            fulltime_home INTEGER,
            fulltime_away INTEGER,
            extratime_home INTEGER,
            extratime_away INTEGER,
            penalty_home INTEGER,
            penalty_away INTEGER,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_matches_fixture ON matches(fixture_id)"
    )
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_matches_matchkey ON matches(league_id, season, date, home_team_id, away_team_id)"
    )
    connection.commit()


def upsert_matches(connection: sqlite3.Connection, matches: Iterable[Dict[str, object]]) -> int:
    cursor = connection.cursor()
    query = """
        INSERT OR REPLACE INTO matches (
            fixture_id,
            league_id,
            league_name,
            season,
            round,
            date,
            status,
            home_team_id,
            home_team_name,
            away_team_id,
            away_team_name,
            home_goals,
            away_goals,
            winner,
            halftime_home,
            halftime_away,
            fulltime_home,
            fulltime_away,
            extratime_home,
            extratime_away,
            penalty_home,
            penalty_away,
            updated_at
        ) VALUES (
            :fixture_id,
            :league_id,
            :league_name,
            :season,
            :round,
            :date,
            :status,
            :home_team_id,
            :home_team_name,
            :away_team_id,
            :away_team_name,
            :home_goals,
            :away_goals,
            :winner,
            :halftime_home,
            :halftime_away,
            :fulltime_home,
            :fulltime_away,
            :extratime_home,
            :extratime_away,
            :penalty_home,
            :penalty_away,
            :updated_at
        )
    """
    row_count = 0
    now = datetime.utcnow().isoformat(sep=" ", timespec="seconds")

    for match in matches:
        parameters = dict(match)
        parameters["updated_at"] = now
        cursor.execute(query, parameters)
        row_count += 1

    connection.commit()
    return row_count


def get_last_update(connection: sqlite3.Connection, league_id: Optional[int] = None) -> Optional[str]:
    cursor = connection.cursor()
    if league_id is None:
        cursor.execute("SELECT MAX(updated_at) AS last_update FROM matches")
    else:
        cursor.execute(
            "SELECT MAX(updated_at) AS last_update FROM matches WHERE league_id = ?",
            (league_id,),
        )
    row = cursor.fetchone()
    return row["last_update"] if row is not None else None


def fetch_matches_for_league(
    connection: sqlite3.Connection,
    league_id: int,
    season: Optional[int] = None,
) -> List[sqlite3.Row]:
    cursor = connection.cursor()
    if season is None:
        cursor.execute("SELECT * FROM matches WHERE league_id = ? ORDER BY date", (league_id,))
    else:
        cursor.execute(
            "SELECT * FROM matches WHERE league_id = ? AND season = ? ORDER BY date",
            (league_id, season),
        )
    return cursor.fetchall()
