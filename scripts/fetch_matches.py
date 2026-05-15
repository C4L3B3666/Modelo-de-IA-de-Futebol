import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_HOST = "v3.football.api-sports.io"
API_BASE_URL = f"https://{API_HOST}/v3"
DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_TIMEOUT = 10


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_api_key(cli_key: Optional[str]) -> str:
    if cli_key:
        return cli_key

    api_key = os.getenv("API_FOOTBALL_KEY") or os.getenv("APISPORTS_KEY")
    if not api_key:
        raise RuntimeError(
            "API key not found. Set the API_FOOTBALL_KEY environment variable or pass --api-key."
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
    session.headers.update({
        "x-apisports-host": API_HOST,
        "Accept": "application/json",
    })
    return session


def fetch_matches(
    session: requests.Session,
    api_key: str,
    league_id: int,
    season: int,
) -> List[Dict[str, Any]]:
    logger = logging.getLogger("fetch_matches")
    matches: List[Dict[str, Any]] = []
    page = 1

    while True:
        logger.info("Fetching matches for league=%s season=%s page=%s", league_id, season, page)
        response = session.get(
            f"{API_BASE_URL}/fixtures",
            headers={"x-apisports-key": api_key},
            params={"league": league_id, "season": season, "page": page},
            timeout=DEFAULT_TIMEOUT,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            logger.error(
                "API request failed: %s - %s",
                response.status_code,
                response.text,
            )
            raise RuntimeError("Falha na requisição à API-Football") from error

        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Resposta inesperada da API: formato inválido")

        if payload.get("errors"):
            logger.error("API returned errors: %s", payload["errors"])
            raise RuntimeError("A API retornou erros ao buscar partidas")

        response_data = payload.get("response")
        if response_data is None:
            raise RuntimeError("Resposta da API não contém campo 'response'")

        matches.extend([normalize_match(item) for item in response_data])

        paging = payload.get("paging") or {}
        current_page = paging.get("current", page)
        total_pages = paging.get("total", page)

        if current_page >= total_pages:
            break

        page += 1

    logger.info("Total matches fetched: %s", len(matches))
    return matches


def normalize_match(match_payload: Dict[str, Any]) -> Dict[str, Any]:
    fixture = match_payload.get("fixture", {})
    league = match_payload.get("league", {})
    teams = match_payload.get("teams", {})
    goals = match_payload.get("goals", {})
    score = match_payload.get("score", {})

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
        "away_team_id": teams.get("away", {}).get("id"),
        "away_team_name": teams.get("away", {}).get("name"),
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
        "winner": match_payload.get("score", {}).get("winner"),
        "halftime_home": score.get("halftime", {}).get("home"),
        "halftime_away": score.get("halftime", {}).get("away"),
        "fulltime_home": score.get("fulltime", {}).get("home"),
        "fulltime_away": score.get("fulltime", {}).get("away"),
        "extratime_home": score.get("extratime", {}).get("home"),
        "extratime_away": score.get("extratime", {}).get("away"),
        "penalty_home": score.get("penalty", {}).get("home"),
        "penalty_away": score.get("penalty", {}).get("away"),
    }


def save_matches_to_csv(matches: List[Dict[str, Any]], output_path: Path) -> None:
    logger = logging.getLogger("fetch_matches")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not matches:
        logger.warning("Nenhuma partida para salvar em %s", output_path)
        return

    fieldnames = list(matches[0].keys())
    logger.info("Gravando %s partidas em %s", len(matches), output_path)

    try:
        with output_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matches)
    except OSError as error:
        logger.error("Falha ao salvar CSV: %s", error)
        raise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Coleta partidas de futebol da API-Football e salva em CSV. "
            "Use --league-id e --season ou configure as variáveis de ambiente "
            "API_FOOTBALL_LEAGUE_ID e API_FOOTBALL_SEASON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--league-id",
        type=int,
        default=os.getenv("API_FOOTBALL_LEAGUE_ID"),
        help="ID da liga no API-Football (por exemplo, 39 para Premier League).",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=os.getenv("API_FOOTBALL_SEASON"),
        help="Temporada que será consultada (por exemplo, 2024).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Caminho do arquivo CSV de saída. Se for pasta, o nome será gerado automaticamente.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Chave de API-Football. Alternativamente, use a variável de ambiente API_FOOTBALL_KEY.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nível de log para execução.",
    )

    args = parser.parse_args()
    if args.league_id is None:
        parser.error(
            "argument --league-id is required (or set API_FOOTBALL_LEAGUE_ID)"
        )
    if args.season is None:
        parser.error(
            "argument --season is required (or set API_FOOTBALL_SEASON)"
        )

    return args


def build_output_path(base_output: Path, league_id: int, season: int) -> Path:
    if base_output.is_dir() or str(base_output).endswith("/") or str(base_output).endswith('\\'):
        filename = f"matches_league_{league_id}_{season}.csv"
        return base_output / filename

    if base_output.suffix.lower() != ".csv":
        base_output = base_output.with_suffix(".csv")

    return base_output


def main() -> int:
    args = parse_arguments()
    setup_logging(args.log_level)
    logger = logging.getLogger("fetch_matches")
    logger.info("Iniciando coleta de partidas: league=%s season=%s", args.league_id, args.season)

    try:
        api_key = load_api_key(args.api_key)
        session = create_session()
        matches = fetch_matches(session, api_key, args.league_id, args.season)

        output_path = build_output_path(args.output, args.league_id, args.season)
        save_matches_to_csv(matches, output_path)

        logger.info("Coleta finalizada com sucesso.")
        return 0
    except Exception as error:
        logger.exception("Erro durante a coleta de partidas: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
