"""Fetch real football match feeds from API-Football and save them as structured CSV and SQLite.

This module is designed for professional ingestion:
- loads API credentials from .env / environment variables
- supports multiple leagues and seasons
- deduplicates by fixture ID
- saves a combined CSV to data/real_matches.csv
- keeps a SQLite store for future expansion
- includes robust API error handling and structured logs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.database import DEFAULT_DB_PATH, connect_database, upsert_matches

API_HOST = "v3.football.api-sports.io"
API_BASE_URL = f"https://{API_HOST}"
DEFAULT_OUTPUT_PATH = Path("data/real_matches.csv")
DEFAULT_TIMEOUT = 10
DEFAULT_LOG_LEVEL = "INFO"


def load_environment() -> None:
    load_dotenv()


def setup_logging(level: str = DEFAULT_LOG_LEVEL) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_league_ids(values: Optional[List[str]]) -> List[int]:
    league_ids: List[int] = []
    if not values:
        return league_ids

    for value in values:
        if not value:
            continue
        for token in value.split(","):
            token = token.strip()
            if token:
                league_ids.append(int(token))

    return league_ids


def get_api_key(cli_key: Optional[str]) -> str:
    if cli_key:
        return cli_key

    api_key = (
        os.getenv("API_FOOTBALL_KEY")
        or os.getenv("APISPORTS_KEY")
        or os.getenv("FOOTBALL_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "API key not found. Set API_FOOTBALL_KEY, APISPORTS_KEY, or FOOTBALL_API_KEY in .env or environment."
        )
    return api_key


def create_session() -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "x-apisports-host": API_HOST,
            "Accept": "application/json",
        }
    )
    return session


def parse_iso_date(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return date.fromisoformat(value).isoformat()


def fetch_matches(
    session: requests.Session,
    api_key: str,
    league_id: int,
    season: int,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    skip_plan_errors: bool = False,
) -> List[Dict[str, Any]]:
    logger = logging.getLogger("fetch_matches")
    matches: List[Dict[str, Any]] = []

    params: Dict[str, Any] = {
        "league": league_id,
        "season": season,
    }
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    logger.debug(
        "Requesting fixtures league=%s season=%s from=%s to=%s",
        league_id,
        season,
        from_date,
        to_date,
    )

    response = session.get(
        f"{API_BASE_URL}/fixtures",
        headers={"x-apisports-key": api_key},
        params=params,
        timeout=DEFAULT_TIMEOUT,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.error(
            "API-Football HTTP error: %s %s",
            response.status_code,
            response.text,
        )
        raise RuntimeError(
            f"API-Football request failed with status {response.status_code}"
        ) from exc
    except requests.RequestException as exc:
        logger.exception("API-Football request exception")
        raise RuntimeError("Network error while fetching matches") from exc

    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected API response: payload is not a JSON object")

    errors = payload.get("errors")
    if errors:
        logger.error("API-Football returned errors: %s", errors)
        err_text = str(errors).lower()
        # Se a resposta indicar limitação do plano (ex: Free plans do not have access),
        # pule esta liga/temporada se o usuário tiver solicitado via flag.
        if (
            ("free plan" in err_text
            or "free plans" in err_text
            or "do not have access" in err_text
            or "not have access" in err_text
            or "access to this season" in err_text)
            and skip_plan_errors
        ):
            logger.warning(
                "Acesso negado por restrição de plano para league=%s season=%s: %s",
                league_id,
                season,
                errors,
            )
            return []

        # Senão, trate como erro fatal para que a execução pare e o usuário veja o problema.
        raise RuntimeError("API-Football returned error payload")

    response_data = payload.get("response")
    if response_data is None:
        raise RuntimeError("API-Football response missing 'response' field")

    matches.extend(normalize_match(item) for item in response_data)

    logger.info("Fetched %d matches for league %s season %s", len(matches), league_id, season)
    return matches


def normalize_match(match_payload: Dict[str, Any]) -> Dict[str, Any]:
    fixture = match_payload.get("fixture", {}) or {}
    league = match_payload.get("league", {}) or {}
    teams = match_payload.get("teams", {}) or {}
    goals = match_payload.get("goals", {}) or {}
    score = match_payload.get("score", {}) or {}

    return {
        "fixture_id": fixture.get("id"),
        "league_id": league.get("id"),
        "league_name": league.get("name"),
        "season": league.get("season"),
        "round": league.get("round"),
        "date": fixture.get("date"),
        "status": fixture.get("status", {}).get("short"),
        "home_team_id": teams.get("home", {}).get("id"),
        "home_team_name": teams.get("home", {}).get("name"),
        "home_team": teams.get("home", {}).get("name"),
        "away_team_id": teams.get("away", {}).get("id"),
        "away_team_name": teams.get("away", {}).get("name"),
        "away_team": teams.get("away", {}).get("name"),
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
        "winner": score.get("winner"),
        "halftime_home": score.get("halftime", {}).get("home"),
        "halftime_away": score.get("halftime", {}).get("away"),
        "fulltime_home": score.get("fulltime", {}).get("home"),
        "fulltime_away": score.get("fulltime", {}).get("away"),
        "extratime_home": score.get("extratime", {}).get("home"),
        "extratime_away": score.get("extratime", {}).get("away"),
        "penalty_home": score.get("penalty", {}).get("home"),
        "penalty_away": score.get("penalty", {}).get("away"),
    }


def normalize_matches(matches: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(matches)
    if df.empty:
        return df

    df = df.convert_dtypes()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "status" in df.columns:
        df = df[df["status"].isin(["FT", "AET", "PEN"])]
    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    df = df.sort_values(by=["league_id", "season", "date", "fixture_id"])
    df = df.drop_duplicates(subset=["fixture_id"], keep="last")
    return df


def save_matches_to_csv(matches: pd.DataFrame, output_path: Path) -> None:
    logger = logging.getLogger("fetch_matches")
    output_path = build_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if matches.empty:
        logger.warning("Nenhum registro para salvar em %s", output_path)
        return

    if output_path.exists() and output_path.stat().st_size > 0:
        existing = pd.read_csv(output_path)
        matches = pd.concat([existing, matches], ignore_index=True, sort=False)

    if "fixture_id" in matches.columns:
        matches = matches.drop_duplicates(subset=["fixture_id"], keep="last")
    else:
        matches = matches.drop_duplicates(
            subset=["date", "home_team", "away_team", "home_goals", "away_goals"],
            keep="last",
        )

    if "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
        matches = matches.sort_values(["date", "home_team", "away_team"])

    logger.info("Salvando %d partidas acumuladas em %s", len(matches), output_path)
    matches.to_csv(output_path, index=False, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Coleta partidas reais da API-Football e salva uma exportação CSV. "
            "Use .env para carregar API_KEY e configure liga/temporada conforme necessário."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--league-ids",
        action="append",
        default=None,
        help=(
            "Identificadores de liga separados por vírgula ou em múltiplos argumentos. "
            "Exemplo: --league-ids 39,140"
        ),
    )
    parser.add_argument(
        "--season",
        type=int,
        default=os.getenv("API_FOOTBALL_SEASON"),
        help="Temporada a ser consultada (por exemplo, 2024).",
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default=os.getenv("API_FOOTBALL_FROM_DATE"),
        help="Data inicial no formato YYYY-MM-DD.",
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default=os.getenv("API_FOOTBALL_TO_DATE"),
        help="Data final no formato YYYY-MM-DD.",
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Atualiza o intervalo diário em torno da data atual.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Caminho do CSV de saída ou diretório onde será gravado real_matches.csv.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Chave da API-Football. Alternativamente, use .env e API_FOOTBALL_KEY.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nível de log para execução.",
    )

    parser.add_argument(
        "--skip-plan-errors",
        action="store_true",
        default=False,
        help=(
            "Quando definido, fará com que a execução pule ligas/temporadas bloqueadas "
            "pelo plano (ex.: mensagem 'Free plans do not have access') em vez de abortar."
        ),
    )

    args = parser.parse_args()
    league_ids = parse_league_ids(args.league_ids)
    if not league_ids:
        env_leagues = os.getenv("API_FOOTBALL_LEAGUE_IDS")
        league_ids = parse_league_ids([env_leagues]) if env_leagues else []

    if not league_ids:
        parser.error("Obrigatório informar --league-ids ou API_FOOTBALL_LEAGUE_IDS.")

    if args.season is None:
        parser.error("Obrigatório informar --season ou API_FOOTBALL_SEASON.")

    args.league_ids = sorted(set(league_ids))
    return args


def build_output_path(base_output: Path) -> Path:
    if base_output.is_dir():
        return base_output / DEFAULT_OUTPUT_PATH.name

    if base_output.suffix == "":
        return base_output.with_suffix(".csv")

    return base_output


def build_date_range(args: argparse.Namespace) -> tuple[Optional[str], Optional[str]]:
    if args.daily:
        today = date.today()
        return (
            (today - timedelta(days=3)).isoformat(),
            (today + timedelta(days=14)).isoformat(),
        )

    return parse_iso_date(args.from_date), parse_iso_date(args.to_date)


def main() -> int:
    load_environment()
    args = parse_arguments()
    setup_logging(args.log_level)
    logger = logging.getLogger("fetch_matches")

    from_date, to_date = build_date_range(args)
    logger.info(
        "Iniciando ingestão: leagues=%s season=%s from=%s to=%s",
        args.league_ids,
        args.season,
        from_date,
        to_date,
    )

    api_key = get_api_key(args.api_key)
    session = create_session()
    connection = connect_database(DEFAULT_DB_PATH)

    all_frames: List[pd.DataFrame] = []
    total_saved = 0

    try:
        for league_id in args.league_ids:
            try:
                league_matches = fetch_matches(
                    session,
                    api_key,
                    league_id,
                    args.season,
                    from_date=from_date,
                    to_date=to_date,
                    skip_plan_errors=args.skip_plan_errors,
                )
            except RuntimeError as exc:
                logger.error("Erro ao buscar partidas para liga %s: %s", league_id, exc)
                continue

            if not league_matches:
                logger.warning("Liga %s retornou 0 partidas", league_id)
                continue

            df = normalize_matches(league_matches)
            if df.empty:
                logger.warning("Liga %s não gerou um DataFrame válido", league_id)
                continue

            saved = upsert_matches(connection, df.to_dict("records"))
            total_saved += saved
            all_frames.append(df)
            logger.info(
                "Liga %s: %s partidas carregadas, %s registros salvos/atualizados.",
                league_id,
                len(df),
                saved,
            )

        if all_frames:
            combined = pd.concat(all_frames, ignore_index=True)
            combined = combined.drop_duplicates(subset=["fixture_id"])
            save_matches_to_csv(combined, args.output)
        else:
            logger.warning("Nenhuma partida processada; nenhum CSV será gerado.")

        logger.info(
            "Ingestão completa. Ligas processadas=%s, registros únicos=%s, registros salvos=%s.",
            len(args.league_ids),
            sum(len(frame) for frame in all_frames),
            total_saved,
        )
        return 0

    except Exception as exc:
        logger.exception("Falha ao executar ingestão de partidas: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
