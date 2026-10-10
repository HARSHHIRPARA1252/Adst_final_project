from pathlib import Path
import sys
from io import BytesIO

import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as bgmi_app
from src.bmsd_source import parse_progression_routes, parse_stage_format, parse_stage_groups, stage_groups_to_tables


@pytest.fixture()
def client():
    bgmi_app.app.config.update(TESTING=True)
    return bgmi_app.app.test_client()


def test_pages_render(client):
    for path in ("/", "/dashboard", "/live", "/mvp", "/predict", "/compare", "/leaderboard", "/history", "/team-analytics", "/model-analytics", "/about"):
        response = client.get(path, follow_redirects=True)
        assert response.status_code == 200, path


def test_health_and_stats_endpoints(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["historical_rows"] == len(bgmi_app.HISTORY)
    assert response.json["model_loaded"] is True
    assert bgmi_app.MVP_MODEL is not None
    assert client.get("/api/stats?tournament=BMSD").json["stats"]["matches"] == bgmi_app._summary_stats(bgmi_app._filter_history("BMSD"))["matches"]
    live_page = client.get("/live")
    assert live_page.status_code == 200
    assert b"Choose bracket and match" in live_page.data
    assert b"18 / 18 complete" in live_page.data or b"complete" in live_page.data
    assert b"Upper Bracket Survival" in live_page.data
    assert b"Semi Finals" in live_page.data
    assert b"Teams by stage" in live_page.data
    assert b"Meta Ninza" in live_page.data
    assert b"Lower Bracket Survival" in live_page.data
    assert b"Watch official live stream" in live_page.data
    assert b"youtube.com/@kraftonindiaesports/live" in live_page.data
    assert b"youtube.com/embed/live_stream" not in live_page.data
    assert b"NinjaBoi" in client.get("/mvp").data


def test_single_team_prediction_api_and_validation(client):
    good = client.post("/api/predict", json={"team": "GODL", "avg_points": 12, "avg_kills": 5})
    assert good.status_code == 200
    assert 0 <= good.json["winning_probability"] <= 1
    bad = client.post("/api/predict", json={"team": "GODL", "avg_points": -1})
    assert bad.status_code == 400
    assert client.post("/api/predict", json={"avg_points": 10}).status_code == 400


def test_lobby_prediction_is_normalized_and_rejects_bad_lobbies(client):
    result = client.post("/api/predict-match", json={"teams": ["GODL", "OUTRGE"]})
    assert result.status_code == 200
    assert sum(row["winning_probability"] for row in result.json["results"]) == pytest.approx(1.0)
    assert client.post("/api/predict-match", json={"teams": ["GODL"]}).status_code == 400


def test_grand_finals_predictions_start_at_match_one_then_advance(monkeypatch):
    teams = [f"Finalist {number:02d}" for number in range(1, 17)]
    stage_format = pd.DataFrame([{
        "stage": "Grand Finals", "sequence": 8, "total_matches": 18,
        "groups_division": "Smash Rules",
    }])
    section = {
        "_stage_key": "grand finals", "sequence": 8,
        "completed_matches": 0, "total_matches": 18,
        "known_teams": 16, "expected_teams": 16, "teams": teams,
    }
    monkeypatch.setattr(bgmi_app, "HISTORY", bgmi_app.HISTORY.copy())

    before_finals = bgmi_app._live_stage_lobbies(pd.DataFrame(), stage_format, [section])
    finals_forecasts = [row for row in before_finals if row["stage"] == "Grand Finals"]
    assert [row["match"] for row in finals_forecasts] == list(range(1, 19))
    assert finals_forecasts[0]["group"] == ""

    final_row = {column: np.nan for column in bgmi_app.HISTORY.columns}
    final_row.update({
        "tournament": "BMSD", "stage": "Grand Finals", "match_id": 1,
        "match_order": int(pd.to_numeric(bgmi_app.HISTORY["match_order"], errors="coerce").max()) + 1,
        "team": teams[0], "team_key": teams[0].casefold(),
    })
    monkeypatch.setattr(
        bgmi_app,
        "HISTORY",
        pd.concat([bgmi_app.HISTORY, pd.DataFrame([final_row])], ignore_index=True),
    )
    section["completed_matches"] = 1
    after_match_one = bgmi_app._live_stage_lobbies(pd.DataFrame(), stage_format, [section])
    finals_forecasts = [row for row in after_match_one if row["stage"] == "Grand Finals"]
    assert [row["match"] for row in finals_forecasts] == list(range(2, 19))


def test_grand_finals_fixture_has_no_group_label():
    fixtures = pd.DataFrame([{
        "source_match_id": "gf-match-1", "overall_match": 100,
        "match": 1, "stage": "Grand Finals", "group": "Group GF",
        "map": "Erangel", "status": "SCHEDULED", "scheduled_at": "2026-10-16",
        "team": "Team Alpha",
    }, {
        "source_match_id": "gf-match-1", "overall_match": 100,
        "match": 1, "stage": "Grand Finals", "group": "Group GF",
        "map": "Erangel", "status": "SCHEDULED", "scheduled_at": "2026-10-16",
        "team": "Team Beta",
    }])
    lobbies = bgmi_app._upcoming_lobbies(fixtures)
    assert len(lobbies) == 1
    assert lobbies[0]["group"] == ""


def test_grand_finals_scoreboard_waits_for_16_and_uses_actual_scores(monkeypatch):
    teams = [f"Finalist {number:02d}" for number in range(1, 17)]
    monkeypatch.setattr(bgmi_app, "HISTORY", bgmi_app.HISTORY.iloc[0:0].copy())
    initial = bgmi_app._grand_finals_scoreboard(teams)
    assert len(initial) == 16
    assert all(row["matches"] == 0 and row["points"] == 0 for row in initial)
    assert bgmi_app._grand_finals_scoreboard(teams[:15]) == []

    scores = pd.DataFrame([{
        "tournament": "BMSD", "stage": "Grand Finals", "match_id": 1,
        "match_order": 67, "team": teams[0], "team_key": teams[0].casefold(),
        "points": 18, "kills": 8, "win": 1, "placement_points": 10, "rank": 1,
    }])
    monkeypatch.setattr(bgmi_app, "HISTORY", scores)
    updated = bgmi_app._grand_finals_scoreboard(teams)
    assert updated[0]["team"] == teams[0]
    assert updated[0]["matches"] == 1
    assert updated[0]["points"] == 18
    assert updated[0]["finishes"] == 8
    assert updated[0]["wwcds"] == 1


def test_live_page_shows_final_score_and_finalist_mvp_after_qualification(client, monkeypatch):
    teams = [f"Finalist {number:02d}" for number in range(1, 17)]
    monkeypatch.setattr(bgmi_app, "_bmsd_stage_sections", lambda _formats: [{
        "stage": "Grand Finals", "sequence": 8, "_stage_key": "grand finals",
        "groups": [{"group": "Teams", "teams": teams}], "teams": teams,
        "known_teams": 16, "expected_teams": 16, "pending_count": 0,
        "completed_matches": 0, "total_matches": 18, "completion_pct": 0,
        "status": "Not started", "division": "Smash Rules", "forecasts": [],
    }])
    monkeypatch.setattr(bgmi_app, "_live_stage_lobbies", lambda *_args: [{
        "id": "final-match-1", "overall_match": 67, "match": 1,
        "stage": "Grand Finals", "group": "", "map": "", "scheduled_at": "",
        "teams": teams, "inferred": True, "stage_total": 18,
        "completed_in_stage": 0, "as_of_match": 66, "predictions": [],
    }])
    monkeypatch.setattr(bgmi_app, "_grand_finals_smash_state", lambda _match: {
        "match": 1, "target_score": None, "leader_after_match_12": None,
        "eligible_teams": [], "champion": None,
    })
    monkeypatch.setattr(bgmi_app, "_next_match_mvp_candidates", lambda _lobby: [{
        "player": "Finalist Star", "team": teams[0], "matches": 12,
        "avg_finishes": 3.0, "recent3_finishes": 4.0, "predicted_finishes": 4.2,
    }])

    response = client.get("/live")
    assert response.status_code == 200
    assert b"16-team final scoreboard" in response.data
    assert b"Grand Finals MVP" in response.data
    assert b"Finalist Star" in response.data
    assert b"Finalist 16" in response.data
    assert b"Grand Finals</span><h3>Teams</h3>" not in response.data


def test_grand_finals_smash_target_and_eligibility_are_match_aware(monkeypatch):
    rows = []
    for match_number in range(1, 13):
        rows.extend([
            {"tournament": "BMSD", "stage": "Grand Finals", "match_id": match_number, "match_order": match_number, "team": "Team Alpha", "points": 10, "win": int(match_number == 1)},
            {"tournament": "BMSD", "stage": "Grand Finals", "match_id": match_number, "match_order": match_number, "team": "Team Beta", "points": 9.75, "win": 0},
        ])
    history = pd.DataFrame(rows)
    monkeypatch.setattr(bgmi_app, "HISTORY", history)

    before_match_13 = bgmi_app._grand_finals_smash_state(13)
    assert before_match_13["leader_after_match_12"] == pytest.approx(120)
    assert before_match_13["target_score"] == pytest.approx(130)
    assert before_match_13["eligible_teams"] == []

    match_13 = pd.DataFrame([{
        "tournament": "BMSD", "stage": "Grand Finals", "match_id": 13,
        "match_order": 13, "team": "Team Beta", "points": 13, "win": 1,
    }])
    monkeypatch.setattr(bgmi_app, "HISTORY", pd.concat([history, match_13], ignore_index=True))
    before_match_14 = bgmi_app._grand_finals_smash_state(14)
    assert before_match_14["eligible_teams"] == ["Team Beta"]
    assert before_match_14["champion"] is None

    match_14 = pd.DataFrame([{
        "tournament": "BMSD", "stage": "Grand Finals", "match_id": 14,
        "match_order": 14, "team": "Team Beta", "points": 15, "win": 1,
    }])
    monkeypatch.setattr(bgmi_app, "HISTORY", pd.concat([history, match_13, match_14], ignore_index=True))
    before_match_15 = bgmi_app._grand_finals_smash_state(15)
    assert before_match_15["champion"] == "Team Beta"
    assert before_match_15["championship_match"] == 14


def test_bmsd_csv_template_and_bad_upload(client):
    template = client.get("/live/template.csv")
    assert template.status_code == 200
    assert b"wwcd" in template.data
    bad = client.post("/live/import", data={"scorecards": (b"team,match,wwcd\nA,1,1\n", "bad.csv")})
    assert bad.status_code == 302
    assert "error=" in bad.headers["Location"]


def test_prediction_form_and_diagnostic_images_render(client):
    page = client.get("/predict", follow_redirects=True)
    assert page.status_code == 200
    assert b"Team win prediction model" in page.data
    assert b"Choose bracket and match" in page.data
    for filename in ("model_comparison.png", "roc_curve.png", "calibration_curve.png", "feature_importance.png"):
        image = client.get(f"/model-report/{filename}")
        assert image.status_code == 200
        assert image.mimetype == "image/png"


def test_bmsd_import_keeps_earlier_matches(client, tmp_path, monkeypatch):
    existing = bgmi_app.HISTORY[bgmi_app.HISTORY["tournament"].astype(str).str.casefold().eq("bmsd")].copy()
    monkeypatch.setattr(bgmi_app, "DATA_DIR", tmp_path)
    monkeypatch.setattr(bgmi_app, "HISTORY", bgmi_app.HISTORY.copy())
    new_match = (
        "date,overall_match,match,stage,group,map,team,rank,wwcd,elim_points,place_points,total_points\n"
        "2026-10-03,62,8,Upper Bracket Survival,UB,Erangel,TEAMZ,1,1,8,10,18\n"
        "2026-10-03,62,8,Upper Bracket Survival,UB,Erangel,TEAMY,2,0,5,6,11\n"
    ).encode()
    response = client.post("/live/import", data={"scorecards": (BytesIO(new_match), "match.csv")})
    assert response.status_code == 302
    assert "uploaded=1" in response.headers["Location"], response.headers["Location"]
    stored = pd.read_csv(tmp_path / "bmsd.csv")
    new_data = bgmi_app.normalize_match_data(pd.read_csv(BytesIO(new_match)), "BMSD", default_year=2026)
    replaced = set(new_data["match_key"].astype(str))
    retained = existing[~existing["match_key"].astype(str).isin(replaced)]
    assert stored["match_key"].nunique() == retained["match_key"].nunique() + new_data["match_key"].nunique()
    assert len(stored) == len(retained) + len(new_data)


def test_source_stage_and_progression_parsers():
    import json

    source_groups = [{
        "stageName": "Upper Bracket Survival",
        "matches": [{
            "id": "fixture-1", "matchNumber": 1, "overallMatchNumber": 61,
            "format": "Match 1 (Group UB)", "status": "COMPLETED",
            "teamResults": [
                {"rank": 1, "wwcd": True, "elimsPoints": 8, "placePoints": 10, "totalPoints": 18, "team": {"name": "Blue Army", "shortName": "BAR"}},
                {"rank": 2, "wwcd": False, "elimsPoints": 5, "placePoints": 6, "totalPoints": 11, "team": {"name": "Team Alpha", "shortName": "ALP"}},
            ],
        }],
    }]
    chunk = '{"stageGroups":' + json.dumps(source_groups) + '}'
    html = '<script>self.__next_f.push([1,' + json.dumps(chunk) + '])</script>'
    parsed = parse_stage_groups(html)
    scores, fixtures = stage_groups_to_tables(parsed, checked_at="2026-10-02")
    assert len(scores) == 2
    assert set(fixtures["team"]) == {"Blue Army", "Team Alpha"}
    assert set(scores["overall_match"]) == {61}

    format_html = '<script>self.__next_f.push([1,' + json.dumps('{"stages":[{"name":"Upper Bracket Survival","sequence":4,"totalMatches":18,"teamsCount":16,"groupsDivision":"Single Lobby","groups":{"UB Team":[{"teamName":"Team A","source":{"rank":1,"group":"Group A","stage":"Week 3"}}]},"rules":[{"thresholdRank":"Top 8","destination":"Semi Finals"}]}]}') + '])</script>'
    format_rows = parse_stage_format(format_html, checked_at="2026-10-02")
    assert format_rows.iloc[0]["stage"] == "Upper Bracket Survival"
    assert format_rows.iloc[0]["total_matches"] == 18
    assert '"destination": "Semi Finals"' in format_rows.iloc[0]["rules_json"]

    progression_html = (
        '<section id="route-week-3-group-a-"><h3>Week 3 (Group A)</h3>'
        '<span class="ed-chip">Top 4 → Grand Finals</span>'
        '<div title="Week 1"></div><div title="Semi Finals"></div>'
        '<h4>Meta Ninza</h4><h4>iQOO 8BIT</h4></section>'
    )
    routes = parse_progression_routes(progression_html)
    assert routes == [{
        "route_id": "route-week-3-group-a-",
        "name": "Week 3 (Group A)",
        "advancement": "Top 4 → Grand Finals",
        "teams": ["Meta Ninza", "iQOO 8BIT"],
        "stages": ["Week 1", "Semi Finals"],
    }]


def test_live_refresh_saves_scorecards_groups_and_finalists(client, tmp_path, monkeypatch):
    raw = pd.DataFrame([
        {"date": "2026-10-03", "overall_match": 65, "match": 1, "stage": "Lower Bracket Survival", "group": "LB", "map": "Erangel", "team": "TEAM A", "rank": 1, "wwcd": True, "elim_points": 7, "place_points": 10, "total_points": 17},
        {"date": "2026-10-03", "overall_match": 65, "match": 1, "stage": "Lower Bracket Survival", "group": "LB", "map": "Erangel", "team": "TEAM B", "rank": 2, "wwcd": False, "elim_points": 5, "place_points": 6, "total_points": 11},
    ])
    fixtures = pd.DataFrame([
        {"date": "2026-10-03", "overall_match": 66, "match": 1, "source_match_id": "fixture-66", "stage": "Lower Bracket Survival", "group": "LB", "map": "Erangel", "status": "SCHEDULED", "scheduled_at": "2026-10-03T10:00:00Z", "source_checked": "2026-10-02", "team": "TEAM A"},
        {"date": "2026-10-03", "overall_match": 66, "match": 1, "source_match_id": "fixture-66", "stage": "Lower Bracket Survival", "group": "LB", "map": "Erangel", "status": "SCHEDULED", "scheduled_at": "2026-10-03T10:00:00Z", "source_checked": "2026-10-02", "team": "TEAM B"},
        {"date": "2026-10-04", "overall_match": 67, "match": 2, "source_match_id": "fixture-67", "stage": "Semi Finals", "group": "SF", "map": "Erangel", "status": "SCHEDULED", "scheduled_at": "2026-10-04T10:00:00Z", "source_checked": "2026-10-02", "team": "TEAM C"},
        {"date": "2026-10-04", "overall_match": 67, "match": 2, "source_match_id": "fixture-67", "stage": "Semi Finals", "group": "SF", "map": "Erangel", "status": "SCHEDULED", "scheduled_at": "2026-10-04T10:00:00Z", "source_checked": "2026-10-02", "team": "TEAM D"},
    ])
    monkeypatch.setattr(bgmi_app, "DATA_DIR", tmp_path)
    monkeypatch.setattr(bgmi_app, "FIXTURES_PATH", tmp_path / "raw" / "bmsd_fixtures.csv")
    monkeypatch.setattr(bgmi_app, "PROGRESSION_PATH", tmp_path / "raw" / "bmsd_progression.csv")
    monkeypatch.setattr(bgmi_app, "STAGE_FORMAT_PATH", tmp_path / "raw" / "bmsd_stage_format.csv")
    monkeypatch.setattr(bgmi_app, "PLAYER_STATS_PATH", tmp_path / "raw" / "bmsd_player_stats.csv")
    monkeypatch.setattr(bgmi_app, "PLAYER_MATCH_STATS_PATH", tmp_path / "raw" / "bmsd_player_match_stats.csv")
    monkeypatch.setattr(bgmi_app, "HISTORY", bgmi_app.HISTORY.copy())
    monkeypatch.setattr(bgmi_app, "fetch_bmsd_source", lambda: (raw, fixtures))
    monkeypatch.setattr(bgmi_app, "fetch_bmsd_format", lambda: pd.DataFrame([{
        "stage": "Lower Bracket Survival", "stage_id": "stage-lb", "sequence": 5,
        "start_date": "2026-10-04", "end_date": "2026-10-06", "total_matches": 2,
        "matches_per_day": 1, "teams_count": 2, "groups_division": "Single Lobby",
        "format_type": "Survival Stage", "groups_json": '{"LB Team":[{"teamName":"TEAM A"},{"teamName":"TEAM B"}]}',
        "rules_json": "[]", "source_checked": "2026-10-02", "source_url": "https://example.test/format",
    }]))
    monkeypatch.setattr(bgmi_app, "fetch_bmsd_progression", lambda: [{
        "route_id": "route-test", "name": "Semi Finals", "advancement": "Top 4 → Grand Finals", "teams": ["TEAM A", "TEAM B"],
    }])
    monkeypatch.setattr(bgmi_app, "fetch_bmsd_player_stats", lambda: (
        pd.DataFrame([{"player_id": "p1", "player": "Fragger", "team": "TEAM A", "matches_played": 2, "total_elims": 8, "avg_elims": 4, "max_elims": 5, "five_plus_elims_matches": 1, "source_checked": "2026-10-02"}]),
        pd.DataFrame([{"player_id": "p1", "player": "Fragger", "team": "TEAM A", "source_match_id": "m1", "stage": "Lower Bracket Survival", "elims": 5}]),
    ))

    response = client.post("/live/refresh")
    assert response.status_code == 302
    assert "refreshed=1" in response.headers["Location"]
    saved = client.get("/live")
    assert b"Lower Bracket Survival" in saved.data
    assert b"TEAM A" in saved.data
    assert b"Remaining match estimates" in saved.data
    assert b"TEAM C" in saved.data
    assert b"Grand Finals" in saved.data
    assert b"Relative lobby estimate" in saved.data


def test_mvp_page_uses_stage_matched_player_performance(client, tmp_path, monkeypatch):
    summary_path = tmp_path / "player_summary.csv"
    match_path = tmp_path / "player_matches.csv"
    pd.DataFrame([{
        "player_id": "p1", "player": "Frag One", "team": "Team A", "matches_played": 2,
        "total_elims": 8, "avg_elims": 4, "max_elims": 5, "five_plus_elims_matches": 1,
        "source_checked": "2026-10-02",
    }]).to_csv(summary_path, index=False)
    pd.DataFrame([{
        "player_id": "p1", "player": "Frag One", "team": "Team A", "source_match_id": "m1",
        "stage": "Week 3", "elims": 5,
    }]).to_csv(match_path, index=False)
    monkeypatch.setattr(bgmi_app, "PLAYER_STATS_PATH", summary_path)
    monkeypatch.setattr(bgmi_app, "PLAYER_MATCH_STATS_PATH", match_path)

    all_page = client.get("/mvp")
    assert all_page.status_code == 200
    assert b"Frag One" in all_page.data
    assert b"not a calibrated probability" in all_page.data
    stage_page = client.get("/mvp?stage=Week%203")
    assert stage_page.status_code == 200
    assert b"Week 3" in stage_page.data
    assert b"5.00" in stage_page.data


def test_next_match_mvp_candidates_exclude_the_match_being_predicted(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame([
        {"source_match_id": "m1", "overall_match": 1, "status": "COMPLETED"},
        {"source_match_id": "m2", "overall_match": 2, "status": "COMPLETED"},
    ]).to_csv(raw_dir / "bmsd_fixtures.csv", index=False)
    pd.DataFrame([
        {"player_id": "p1", "player": "Star", "team": "Team A", "source_match_id": "m1", "elims": 3},
        {"player_id": "p1", "player": "Star", "team": "Team A", "source_match_id": "m2", "elims": 99},
    ]).to_csv(tmp_path / "player_matches.csv", index=False)
    monkeypatch.setattr(bgmi_app, "DATA_DIR", tmp_path)
    monkeypatch.setattr(bgmi_app, "PLAYER_MATCH_STATS_PATH", tmp_path / "player_matches.csv")

    result = bgmi_app._next_match_mvp_candidates({
        "overall_match": 2, "teams": ["Team A"],
    })
    assert result[0]["player"] == "Star"
    assert result[0]["matches"] == 1
    assert result[0]["avg_finishes"] == pytest.approx(3.0)
    assert result[0]["recent3_finishes"] == pytest.approx(3.0)
    assert np.isfinite(result[0]["predicted_finishes"])
