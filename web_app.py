"""Local browser dashboard for the offline football predictor.

Run:
    python web_app.py

Then open:
    http://127.0.0.1:8000
"""

from __future__ import annotations

from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, urlparse

import pandas as pd

from app import compute_outcome_probabilities, load_match_data
from models.poisson_model import PoissonGoalModel


DATA_PATH = Path("data/real_matches.csv")
MAX_GOALS = 6
DEFAULT_LIMIT = 100


class PredictorState:
    def __init__(self, data_path: Path = DATA_PATH) -> None:
        self.data_path = data_path
        self.matches = load_match_data(data_path)
        self.model = PoissonGoalModel(max_goals=MAX_GOALS)
        self.model.fit(self.matches)
        self.teams = sorted(self.model.get_team_strengths().index.tolist())

    def predict(self, home_team: str, away_team: str) -> dict:
        home_expected, away_expected = self.model.predict_expected_goals(home_team, away_team)
        scores = self.model.predict_score_probabilities(home_team, away_team, max_goals=MAX_GOALS)
        return {
            "home_expected": home_expected,
            "away_expected": away_expected,
            "outcomes": compute_outcome_probabilities(scores),
            "scores": scores.head(8),
        }


STATE = PredictorState()


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/":
            self.send_error(404)
            return

        query = parse_qs(parsed.query)
        home_team = query.get("home_team", [STATE.teams[0]])[0]
        away_team = query.get("away_team", [STATE.teams[1] if len(STATE.teams) > 1 else STATE.teams[0]])[0]
        limit = parse_limit(query.get("limit", [str(DEFAULT_LIMIT)])[0])

        html = render_dashboard(home_team, away_team, limit)
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_limit(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_LIMIT
    return max(parsed, 0)


def render_dashboard(home_team: str, away_team: str, limit: int) -> str:
    prediction_html = render_prediction(home_team, away_team)
    teams_options_home = render_team_options(home_team)
    teams_options_away = render_team_options(away_team)
    matches_html = render_matches_table(limit)
    total_matches = len(STATE.matches)
    shown_matches = total_matches if limit == 0 else min(limit, total_matches)

    return f"""<!doctype html>
<html lang="pt">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Football AI Model</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #5b6773;
      --line: #d8dee6;
      --accent: #0b6bcb;
      --accent-dark: #064f99;
      --good: #0b7a53;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      font-size: 15px;
    }}
    header {{
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 18px 24px;
    }}
    h1 {{ margin: 0; font-size: 22px; letter-spacing: 0; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 22px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      margin-bottom: 18px;
      padding: 18px;
    }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    form {{
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(180px, 1fr) 120px auto;
      gap: 12px;
      align-items: end;
    }}
    label {{ display: grid; gap: 6px; color: var(--muted); font-size: 13px; }}
    select, input, button {{
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 0 10px;
      font: inherit;
      background: #fff;
    }}
    button {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      cursor: pointer;
      padding: 0 16px;
    }}
    button:hover {{ background: var(--accent-dark); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      background: #fbfcfd;
    }}
    .metric span {{ color: var(--muted); display: block; font-size: 12px; margin-bottom: 6px; }}
    .metric strong {{ font-size: 20px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .table-wrap {{ overflow-x: auto; }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 760px) {{
      form {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Football AI Model</h1>
  </header>
  <main>
    <section>
      <h2>Previsão</h2>
      <form method="get" action="/">
        <label>Casa<select name="home_team">{teams_options_home}</select></label>
        <label>Fora<select name="away_team">{teams_options_away}</select></label>
        <label>Jogos<input name="limit" type="number" min="0" value="{limit}"></label>
        <button type="submit">Atualizar</button>
      </form>
    </section>
    {prediction_html}
    <section>
      <h2>Jogos no CSV</h2>
      <p class="muted">Mostrando {shown_matches} de {total_matches} jogos. Use 0 no campo Jogos para mostrar todos.</p>
      {matches_html}
    </section>
  </main>
</body>
</html>"""


def render_prediction(home_team: str, away_team: str) -> str:
    try:
        prediction = STATE.predict(home_team, away_team)
    except KeyError as exc:
        return f"<section><h2>Erro</h2><p>{escape(str(exc))}</p></section>"

    outcomes = prediction["outcomes"]
    scores = prediction["scores"]
    score_rows = "\n".join(
        "<tr>"
        f"<td>{int(row.home_goals)}-{int(row.away_goals)}</td>"
        f"<td>{format_pct(row.probability)}</td>"
        "</tr>"
        for row in scores.itertuples()
    )
    return f"""<section>
  <h2>{escape(home_team)} x {escape(away_team)}</h2>
  <div class="metrics">
    <div class="metric"><span>xG casa</span><strong>{prediction['home_expected']:.2f}</strong></div>
    <div class="metric"><span>xG fora</span><strong>{prediction['away_expected']:.2f}</strong></div>
    <div class="metric"><span>Casa</span><strong>{format_pct(outcomes['Vitória casa'])}</strong></div>
    <div class="metric"><span>Empate</span><strong>{format_pct(outcomes['Empate'])}</strong></div>
    <div class="metric"><span>Fora</span><strong>{format_pct(outcomes['Vitória fora'])}</strong></div>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Placar</th><th>Probabilidade</th></tr></thead>
      <tbody>{score_rows}</tbody>
    </table>
  </div>
</section>"""


def render_matches_table(limit: int) -> str:
    matches = STATE.matches.copy()
    if "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
        matches = matches.sort_values("date", ascending=False)
    if limit > 0:
        matches = matches.head(limit)

    rows = []
    for row in matches.itertuples():
        date = getattr(row, "date", "")
        date_text = "" if pd.isna(date) else str(date)[:10]
        home = str(getattr(row, "home_team"))
        away = str(getattr(row, "away_team"))
        url = f"/?home_team={quote_plus(home)}&away_team={quote_plus(away)}&limit={limit}"
        rows.append(
            "<tr>"
            f"<td>{escape(date_text)}</td>"
            f"<td><a href=\"{url}\">{escape(home)}</a></td>"
            f"<td><a href=\"{url}\">{escape(away)}</a></td>"
            f"<td>{int(getattr(row, 'home_goals'))}-{int(getattr(row, 'away_goals'))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Data</th><th>Casa</th>'
        '<th>Fora</th><th>Resultado</th></tr></thead><tbody>'
        + "\n".join(rows)
        + "</tbody></table></div>"
    )


def render_team_options(selected: str) -> str:
    return "\n".join(
        f'<option value="{escape(team)}"{" selected" if team == selected else ""}>{escape(team)}</option>'
        for team in STATE.teams
    )


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> int:
    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard disponível em http://{host}:{port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
