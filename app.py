# """Flask application for the BGMI match win probability project."""

# from __future__ import annotations

# from pathlib import Path
# import io
# import json
# import logging
# import re
# import time
# import threading

# import joblib
# import numpy as np
# import pandas as pd
# from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
# from sklearn.exceptions import NotFittedError

# from src.data_preprocessing import (
#     CANONICAL_COLUMNS,
#     canonical_team_key,
#     load_all_historical_data,
#     normalize_match_data,
#     validate_match_lobbies,
# )
# from src.feature_engineering import FEATURE_COLS, build_pre_match_features, feature_snapshot_for_team
# from src.prediction import get_feature_importance, predict_single_team
# from src.player_finish_model import PLAYER_FEATURES, build_player_prediction_frame
# from src.bmsd_source import (
#     BMSD_MATCHES_URL,
#     BMSD_FORMAT_URL,
#     BMSD_PROGRESSION_URL,
#     fetch_bmsd_format,
#     fetch_bmsd_progression,
#     fetch_bmsd_player_stats,
#     fetch_bmsd_source,
# )


# BASE_DIR = Path(__file__).resolve().parent
# DATA_DIR = BASE_DIR / "data"
# MODEL_DIR = BASE_DIR / "models"
# REPORT_DIR = BASE_DIR / "reports" / "model_results"
# MODEL_PATH = MODEL_DIR / "bgmi_winning_probability_model.joblib"
# PREPROCESSOR_PATH = MODEL_DIR / "preprocessing_pipeline.joblib"
# FEATURES_PATH = MODEL_DIR / "feature_names.joblib"
# MVP_MODEL_PATH = MODEL_DIR / "bmsd_player_finishes_model.joblib"
# MVP_FEATURES_PATH = MODEL_DIR / "bmsd_player_finish_feature_names.joblib"
# MVP_METADATA_PATH = REPORT_DIR / "player_finish_model_metadata.json"
# EAM_BMSD_URL = "https://esportsamaze.com/bgmi/tournaments/battlegrounds-mobile-india-showdown-2026/standings"
# LIQUIPEDIA_BMSD_URL = "https://liquipedia.net/pubgmobile/Battlegrounds_Mobile_India_Showdown/2026"
# FIXTURES_PATH = DATA_DIR / "raw" / "bmsd_fixtures.csv"
# PROGRESSION_PATH = DATA_DIR / "raw" / "bmsd_progression.csv"
# STAGE_FORMAT_PATH = DATA_DIR / "raw" / "bmsd_stage_format.csv"
# PLAYER_STATS_PATH = DATA_DIR / "raw" / "bmsd_player_stats.csv"
# PLAYER_MATCH_STATS_PATH = DATA_DIR / "raw" / "bmsd_player_match_stats.csv"

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("bgmi_app")

# # ---------------------------------------------------------------------------
# # Simple in-process TTL cache – avoids recomputing heavy route data on every
# # navbar click. Data only changes on manual refresh or CSV upload.
# # ---------------------------------------------------------------------------
# _CACHE_TTL = 86400  # seconds (24 hours)
# _route_cache: dict[str, tuple[float, object]] = {}
# _CSV_FILE_CACHE: dict[tuple[str, float], pd.DataFrame] = {}


# def _cache_get(key: str) -> object:
#     if app.config.get("TESTING"):
#         return None
#     entry = _route_cache.get(key)
#     if entry and (time.monotonic() - entry[0]) < _CACHE_TTL:
#         return entry[1]
#     return None


# def _cache_set(key: str, value: object) -> None:
#     _route_cache[key] = (time.monotonic(), value)


# def _cache_clear() -> None:
#     _route_cache.clear()
#     _HISTORY_SLICE_CACHE.clear()
#     _CSV_FILE_CACHE.clear()
#     LIVE_PREDICTION_CACHE.clear()
#     try:
#         from src.feature_engineering import _ELO_CACHE
#         _ELO_CACHE.clear()
#     except Exception:
#         pass


# app = Flask(__name__)
# app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# def _load_model_artifacts():
#     try:
#         if not all(path.exists() for path in (MODEL_PATH, PREPROCESSOR_PATH, FEATURES_PATH)):
#             return None, None, FEATURE_COLS
#         return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH), list(joblib.load(FEATURES_PATH))
#     except Exception as exc:  # Do not prevent dashboard access if artifacts need retraining.
#         logger.error("Could not load saved model artifacts: %s", exc)
#         return None, None, FEATURE_COLS


# MODEL, PREPROCESSOR, MODEL_FEATURES = _load_model_artifacts()
# try:
#     MVP_MODEL = joblib.load(MVP_MODEL_PATH) if MVP_MODEL_PATH.exists() else None
#     MVP_MODEL_FEATURES = list(joblib.load(MVP_FEATURES_PATH)) if MVP_FEATURES_PATH.exists() else []
#     MVP_METADATA = json.loads(MVP_METADATA_PATH.read_text(encoding="utf-8")) if MVP_METADATA_PATH.exists() else {}
# except Exception as exc:
#     logger.error("Could not load saved player finish artifacts: %s", exc)
#     MVP_MODEL, MVP_MODEL_FEATURES, MVP_METADATA = None, [], {}


# def _load_history() -> pd.DataFrame:
#     try:
#         return load_all_historical_data(DATA_DIR)
#     except Exception as exc:
#         logger.error("Could not load tournament data: %s", exc)
#         return pd.DataFrame(columns=CANONICAL_COLUMNS)


# HISTORY = _load_history()
# LIVE_PREDICTION_CACHE: dict[tuple[int, float, str], dict[str, object]] = {}


# _WARMUP_STARTED = False
# _WARMUP_LOCK = threading.Lock()


# def _warmup_cache() -> None:
#     """Pre-compute expensive /live data in a background thread at startup."""
#     global _WARMUP_STARTED
#     with _WARMUP_LOCK:
#         if _WARMUP_STARTED:
#             return
#         _WARMUP_STARTED = True

#     def _run() -> None:
#         try:
#             from src.data_preprocessing import load_all_historical_data
#             fixtures = _read_source_csv(FIXTURES_PATH)
#             progression = _read_source_csv(PROGRESSION_PATH)
#             stage_formats = _read_source_csv(STAGE_FORMAT_PATH)
#             _cache_set("live_fixtures", fixtures)
#             _cache_set("live_progression", progression)
#             _cache_set("live_stage_formats", stage_formats)
#             bmsd = HISTORY[HISTORY["tournament"].astype(str).str.casefold().eq("bmsd")] if not HISTORY.empty else HISTORY.__class__(columns=HISTORY.columns)
#             board = _team_summary(bmsd)
#             _cache_set("live_board", board)
#             stage_sections = _bmsd_stage_sections(stage_formats)
#             _cache_set("live_stage_sections", stage_sections)
#             section_map = {_stage_token(str(s["stage"])): s for s in stage_sections}
#             upcoming = _live_stage_lobbies(fixtures, stage_formats, stage_sections)
#             for lobby in upcoming:
#                 section = section_map.get(_stage_token(lobby.get("stage")))
#                 if section:
#                     lobby.setdefault("stage_total", section.get("total_matches"))
#                     lobby.setdefault("completed_in_stage", section.get("completed_matches"))
#             for lobby in upcoming:
#                 lobby["predictions"] = _lobby_predictions(lobby, cache=LIVE_PREDICTION_CACHE)
#             _cache_set("live_upcoming", upcoming)
#             if upcoming:
#                 first_lobby = upcoming[0]
#                 _cache_set(f"mvp_{first_lobby.get('id')}", _next_match_mvp_candidates(first_lobby))
#             logger.info("Cache warmup complete.")
#         except Exception as exc:
#             logger.warning("Cache warmup failed (non-fatal): %s", exc)

#     threading.Thread(target=_run, daemon=True).start()



# def _allowed_tournaments() -> list[str]:
#     cached = _cache_get("allowed_tournaments")
#     if cached is not None:
#         return cached  # type: ignore[return-value]
#     present = sorted(HISTORY["tournament"].dropna().astype(str).unique().tolist()) if not HISTORY.empty else []
#     result = ["All Tournaments", *present]
#     _cache_set("allowed_tournaments", result)
#     return result


# def _filter_history(tournament: str = "All Tournaments") -> pd.DataFrame:
#     if HISTORY.empty or tournament in ("", "All Tournaments", "All"):
#         return HISTORY.copy()
#     return HISTORY[HISTORY["tournament"].astype(str).str.casefold().eq(str(tournament).casefold())].copy()


# def _safe_number(value: object, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
#     try:
#         number = float(value)
#     except (TypeError, ValueError):
#         raise ValueError(f"{name} must be a number.") from None
#     if not np.isfinite(number):
#         raise ValueError(f"{name} must be a finite number.")
#     if minimum is not None and number < minimum:
#         raise ValueError(f"{name} must be at least {minimum:g}.")
#     if maximum is not None and number > maximum:
#         raise ValueError(f"{name} must be no more than {maximum:g}.")
#     return number


# def _prediction_input(
#     team: str,
#     supplied: dict[str, object] | None = None,
#     history_frame: pd.DataFrame | None = None,
# ) -> tuple[dict[str, float], bool]:
#     if not team or not team.strip():
#         raise ValueError("Enter a team name.")
#     source_history = HISTORY if history_frame is None else history_frame
#     fields = feature_snapshot_for_team(source_history, team)
#     has_history = bool(not source_history.empty and source_history["team_key"].eq(canonical_team_key(team)).any())
#     aliases = {
#         "avg_points": "prev_avg_points",
#         "avg_kills": "prev_avg_kills",
#         "win_rate": "prev_win_rate",
#         "recent_points": "recent3_avg_points",
#         "recent_kills": "recent3_avg_kills",
#     }
#     supplied = supplied or {}
#     for key, value in supplied.items():
#         destination = aliases.get(key, key)
#         if destination not in MODEL_FEATURES:
#             continue
#         if value in (None, ""):
#             continue
#         minimum = 0.0
#         maximum = 1.0 if destination in ("prev_win_rate", "recent3_win_rate") else None
#         fields[destination] = _safe_number(value, destination, minimum=minimum, maximum=maximum)
#     return {feature: float(fields.get(feature, 0.0)) for feature in MODEL_FEATURES}, has_history


# def _predict(team: str, supplied: dict[str, object] | None = None) -> dict[str, object]:
#     if MODEL is None or PREPROCESSOR is None:
#         raise RuntimeError("The trained model is not available. Run the training script first.")
#     features, has_history = _prediction_input(team, supplied)
#     probability = predict_single_team(MODEL, features, MODEL_FEATURES, PREPROCESSOR)
#     return {
#         "team": team,
#         "winning_probability": probability,
#         "winning_probability_percentage": round(probability * 100, 2),
#         "features": features,
#         "has_history": has_history,
#         "feature_importance": get_feature_importance(MODEL, MODEL_FEATURES)[:6],
#     }


# def _team_summary(frame: pd.DataFrame) -> list[dict[str, object]]:
#     if frame.empty:
#         return []
#     result = frame.groupby("team_key", as_index=False).agg(
#         team=("team", "last"),
#         matches=("match_key", "nunique"),
#         points=("points", "sum"),
#         kills=("kills", "sum"),
#         wins=("win", "sum"),
#         avg_points=("points", "mean"),
#         avg_kills=("kills", "mean"),
#         win_rate=("win", "mean"),
#     )
#     result["wins"] = result["wins"].astype(int)
#     return result.sort_values(["avg_points", "avg_kills"], ascending=False).to_dict("records")


# def _summary_stats(frame: pd.DataFrame) -> dict[str, int | float]:
#     if frame.empty:
#         return {"matches": 0, "teams": 0, "rows": 0, "wins": 0, "points": 0, "kills": 0}
#     return {
#         "matches": int(frame["match_key"].nunique()),
#         "teams": int(frame["team_key"].nunique()),
#         "rows": int(len(frame)),
#         "wins": int(frame["win"].sum()),
#         "points": int(frame["points"].sum()),
#         "kills": int(frame["kills"].sum()),
#     }


# def _read_source_csv(path: Path) -> pd.DataFrame:
#     try:
#         if not path.exists() or not path.stat().st_size:
#             return pd.DataFrame()
#         mtime = path.stat().st_mtime
#         key = (str(path.resolve()), mtime)
#         cached = _CSV_FILE_CACHE.get(key)
#         if cached is not None:
#             return cached.copy()
#         df = pd.read_csv(path)
#         _CSV_FILE_CACHE[key] = df
#         return df.copy()
#     except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
#         logger.warning("Unable to read BMSD source snapshot %s: %s", path.name, exc)
#         return pd.DataFrame()


# def _bmsd_group_rows(fixtures: pd.DataFrame) -> list[dict[str, object]]:
#     if fixtures.empty or "team" not in fixtures:
#         return []
#     frame = fixtures.copy()
#     frame["team"] = frame["team"].fillna("").astype(str).str.strip()
#     frame = frame[frame["team"].ne("")]
#     if frame.empty:
#         return []
#     rows = []
#     for (stage, group), subset in frame.groupby(["stage", "group"], dropna=False, sort=False):
#         teams = sorted(subset["team"].dropna().astype(str).unique().tolist(), key=str.casefold)
#         rows.append({
#             "stage": str(stage or "Unknown stage"),
#             "group": str(group) if pd.notna(group) and str(group).strip() else "Group not specified",
#             "teams": teams,
#             "matches": int(subset["source_match_id"].nunique()) if "source_match_id" in subset else int(subset["match"].nunique()),
#             "statuses": sorted(subset["status"].dropna().astype(str).str.upper().unique().tolist()) if "status" in subset else [],
#         })
#     return rows


# def _upcoming_lobbies(fixtures: pd.DataFrame) -> list[dict[str, object]]:
#     if fixtures.empty or "status" not in fixtures:
#         return []
#     scheduled_statuses = {"SCHEDULED", "UPCOMING", "NOT_STARTED", "PENDING"}
#     future = fixtures[fixtures["status"].fillna("").astype(str).str.upper().isin(scheduled_statuses)].copy()
#     if future.empty:
#         return []
#     identity = "source_match_id" if "source_match_id" in future and future["source_match_id"].fillna("").astype(str).str.strip().ne("").any() else "overall_match"
#     lobbies = []
#     for lobby_id, subset in future.groupby(identity, dropna=False, sort=False):
#         first = subset.iloc[0]
#         teams = sorted(set(subset["team"].fillna("").astype(str).str.strip()).difference({""}), key=str.casefold)
#         if len(teams) < 2:
#             continue
#         lobbies.append({
#             "id": str(lobby_id),
#             "overall_match": first.get("overall_match", ""),
#             "match": first.get("match", ""),
#             "stage": str(first.get("stage", "Unknown stage")),
#             "group": "" if str(first.get("stage", "")).strip().casefold() == "grand finals" else str(first.get("group", "")),
#             "map": str(first.get("map", "")),
#             "scheduled_at": str(first.get("scheduled_at", "")),
#             "teams": teams,
#         })
#     return sorted(lobbies, key=lambda row: (row["scheduled_at"] or "9999", str(row["overall_match"])))


# def _stage_token(value: object) -> str:
#     return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


# def _group_token(value: object) -> str:
#     text = re.sub(r"\s+", " ", str(value or "").strip()).casefold()
#     return re.sub(r"^group\s+", "", text)


# def _json_value(value: object, fallback: object) -> object:
#     try:
#         return json.loads(str(value))
#     except (TypeError, ValueError, json.JSONDecodeError):
#         return fallback


# def _rank_range(threshold: str, team_count: int) -> tuple[int, int] | None:
#     value = re.sub(r"\s+", " ", str(threshold or "").strip()).casefold()
#     top = re.search(r"top\s+(\d+)", value)
#     if top:
#         return 1, min(team_count, int(top.group(1)))
#     bottom = re.search(r"bottom\s+(\d+)", value)
#     if bottom:
#         size = min(team_count, int(bottom.group(1)))
#         return team_count - size + 1, team_count
#     ranks = [int(number) for number in re.findall(r"\d+", value)]
#     if len(ranks) >= 2:
#         return max(1, min(ranks)), min(team_count, max(ranks))
#     return None


# def _bmsd_standings(bmsd: pd.DataFrame, formats: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
#     """Rank completed stage groups using the published BMSD tiebreak order."""
#     empty = pd.DataFrame(columns=["stage", "group", "team", "team_key", "rank", "points", "wins"])
#     if bmsd.empty:
#         return empty, {}
#     frame = bmsd.copy()
#     for column in ("points", "kills", "win", "placement_points", "rank", "match_order"):
#         frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
#     frame["_stage_key"] = frame["stage"].map(_stage_token)
#     frame["_group_key"] = frame["group"].map(_group_token)
#     frame["_order"] = frame["match_order"].fillna(0)
#     complete: dict[str, bool] = {}
#     totals: dict[str, int] = {}
#     if not formats.empty:
#         for _, row in formats.iterrows():
#             total = pd.to_numeric(row.get("total_matches"), errors="coerce")
#             if pd.notna(total):
#                 totals[_stage_token(row.get("stage"))] = int(total)
#     for key, subset in frame.groupby("_stage_key", sort=False):
#         complete[key] = bool(totals.get(key, 0) > 0 and subset["match_order"].nunique() >= totals[key])

#     rows = []
#     grouped = frame.groupby(["_stage_key", "_group_key", "team_key"], sort=False, dropna=False)
#     for (stage_key, group_key, team_key), subset in grouped:
#         last = subset.sort_values(["_order", "date"], kind="stable", na_position="first").iloc[-1]
#         rows.append({
#             "stage_key": stage_key,
#             "group_key": group_key,
#             "stage": str(last["stage"]),
#             "group": str(last["group"]),
#             "team": str(last["team"]),
#             "team_key": str(team_key),
#             "points": float(subset["points"].sum()),
#             "wins": float(subset["win"].sum()),
#             "placement_points": float(subset["placement_points"].fillna(0).sum()),
#             "kills": float(subset["kills"].sum()),
#             "last_rank": float(last["rank"]) if pd.notna(last["rank"]) else float("inf"),
#         })
#     standings = pd.DataFrame(rows)
#     if standings.empty:
#         return empty, complete
#     standings = standings.sort_values(
#         ["stage_key", "group_key", "points", "wins", "placement_points", "kills", "last_rank", "team_key"],
#         ascending=[True, True, False, False, False, False, True, True],
#         kind="stable",
#     )
#     standings["rank"] = standings.groupby(["stage_key", "group_key"], sort=False).cumcount() + 1
#     return standings, complete


# def _bmsd_stage_sections(
#     stage_formats: pd.DataFrame,
#     history_frame: pd.DataFrame | None = None,
# ) -> list[dict[str, object]]:
#     """Resolve published bracket slots from completed match standings and rules."""
#     bmsd = _filter_history("BMSD") if history_frame is None else history_frame[
#         history_frame["tournament"].astype(str).str.casefold().eq("bmsd")
#     ].copy()
#     if stage_formats.empty:
#         return []
#     formats = stage_formats.copy()
#     formats["_stage_key"] = formats["stage"].map(_stage_token)
#     formats["sequence"] = pd.to_numeric(formats.get("sequence"), errors="coerce").fillna(0)
#     formats = formats.sort_values("sequence", kind="stable")
#     standings, stage_complete = _bmsd_standings(bmsd, formats)
#     stage_lookup = {row["_stage_key"]: row for _, row in formats.iterrows()}
#     result_by_stage: dict[str, pd.DataFrame] = {}
#     for key, subset in bmsd.assign(_stage_key=bmsd["stage"].map(_stage_token)).groupby("_stage_key", sort=False):
#         result_by_stage[key] = subset
#     names: dict[str, dict[str, str]] = {}
#     grouped_names: dict[str, dict[str, dict[str, str]]] = {}
#     pending_sources: dict[str, list[str]] = {}
#     for stage_key in stage_lookup:
#         names[stage_key] = {}
#         grouped_names[stage_key] = {}
#         pending_sources[stage_key] = []

#     display_by_key: dict[str, str] = {}
#     for _, row in bmsd.sort_values("match_order", kind="stable").iterrows():
#         display_by_key[str(row["team_key"])] = str(row["team"])

#     def add_team(stage_key: str, group_name: str, team_name: str) -> None:
#         team_key = canonical_team_key(team_name)
#         if not team_key:
#             return
#         display = display_by_key.get(team_key, team_name)
#         names.setdefault(stage_key, {})[team_key] = display
#         group_map = grouped_names.setdefault(stage_key, {}).setdefault(group_name or "Teams", {})
#         group_map[team_key] = display

#     # Resolve each stage's published team slots. Source ranks are only treated
#     # as confirmed once the source stage has completed all published matches.
#     for _, stage_row in formats.iterrows():
#         target_key = stage_row["_stage_key"]
#         if stage_complete.get(target_key, False):
#             continue
#         groups = _json_value(stage_row.get("groups_json", "{}"), {})
#         if isinstance(groups, dict):
#             for group_name, entries in groups.items():
#                 if not isinstance(entries, list):
#                     continue
#                 for entry in entries:
#                     if not isinstance(entry, dict):
#                         continue
#                     source = entry.get("source")
#                     if isinstance(source, dict):
#                         source_key = _stage_token(source.get("stage"))
#                         source_group = _group_token(source.get("group"))
#                         rank = pd.to_numeric(source.get("rank"), errors="coerce")
#                         source_table = standings[
#                             standings["stage_key"].eq(source_key)
#                             & standings["group_key"].eq(source_group)
#                         ] if not standings.empty else pd.DataFrame()
#                         match = source_table[source_table["rank"].eq(rank)] if pd.notna(rank) else pd.DataFrame()
#                         if stage_complete.get(source_key, False) and not match.empty:
#                             add_team(target_key, str(group_name), str(match.iloc[0]["team"]))
#                     else:
#                         team_name = str(entry.get("teamName", entry.get("name", "")) or "").strip()
#                         if team_name and not re.match(r"^Group\s+.+\s+#\d+$", team_name, flags=re.IGNORECASE):
#                             add_team(target_key, str(group_name), team_name)

#     # Completed scorecards are the strongest source for a stage's actual teams.
#     for stage_key, subset in result_by_stage.items():
#         if stage_key not in stage_lookup:
#             continue
#         groups = _json_value(stage_lookup[stage_key].get("groups_json", "{}"), {})
#         format_group_names = list(groups) if isinstance(groups, dict) else []
#         one_lobby = len(format_group_names) == 1
#         for _, row in subset.drop_duplicates(["team_key", "group"]).iterrows():
#             group_name = format_group_names[0] if one_lobby else (str(row.get("group", "")).strip() or "Teams")
#             add_team(stage_key, group_name, str(row["team"]))

#     # Apply every completed source-stage advancement rule, including teams
#     # that are not present in the static stage-group list (for example the
#     # top eight from the current Upper Bracket entering Semi Finals).
#     for _, source_row in formats.iterrows():
#         source_key = source_row["_stage_key"]
#         rules = _json_value(source_row.get("rules_json", "[]"), [])
#         if not isinstance(rules, list):
#             continue
#         source_completed = stage_complete.get(source_key, False)
#         source_name = str(source_row.get("stage", ""))
#         for rule in rules:
#             if not isinstance(rule, dict):
#                 continue
#             destination = str(rule.get("destination", "") or "").strip()
#             target_key = _stage_token(destination)
#             if target_key not in stage_lookup:
#                 continue
#             if stage_complete.get(target_key, False):
#                 continue
#             target_group = str(rule.get("groupName", "") or "").strip() or f"From {source_name}"
#             label = str(rule.get("thresholdRank", "") or "").strip()
#             if not source_completed:
#                 if label:
#                     pending_sources[target_key].append(f"{label} from {source_name}")
#                 continue
#             group_filter = _group_token(rule.get("groupName", ""))
#             source_table = standings[standings["stage_key"].eq(source_key)] if not standings.empty else pd.DataFrame()
#             if group_filter:
#                 source_table = source_table[source_table["group_key"].eq(group_filter)]
#             if source_table.empty:
#                 continue
#             rank_range = _rank_range(label, int(source_table.groupby("group_key")["team_key"].nunique().max()))
#             if rank_range is None:
#                 continue
#             qualified = source_table[source_table["rank"].between(*rank_range)]
#             for _, team_row in qualified.iterrows():
#                 if canonical_team_key(str(team_row["team"])) not in names.get(target_key, {}):
#                     add_team(target_key, target_group, str(team_row["team"]))

#     sections = []
#     for _, row in formats.iterrows():
#         stage_key = row["_stage_key"]
#         total_matches = pd.to_numeric(row.get("total_matches"), errors="coerce")
#         total_matches = int(total_matches) if pd.notna(total_matches) else 0
#         match_rows = result_by_stage.get(stage_key, pd.DataFrame())
#         completed_matches = int(match_rows["match_order"].nunique()) if not match_rows.empty else 0
#         expected_teams = pd.to_numeric(row.get("teams_count"), errors="coerce")
#         expected_teams = int(expected_teams) if pd.notna(expected_teams) else 0
#         teams = names.get(stage_key, {})
#         groups = []
#         for group_name, group_teams in grouped_names.get(stage_key, {}).items():
#             groups.append({"group": group_name, "teams": sorted(group_teams.values(), key=str.casefold)})
#         known_count = len(teams)
#         pending_count = max(0, expected_teams - known_count)
#         if pending_count:
#             groups.append({"group": "Qualification pending", "teams": [], "pending_count": pending_count})
#         pct = 100.0 * completed_matches / total_matches if total_matches else 0.0
#         status = "Complete" if total_matches and completed_matches >= total_matches else ("In progress" if completed_matches else "Not started")
#         sections.append({
#             "stage": str(row.get("stage", "Unknown stage")),
#             "sequence": int(row.get("sequence", 0)),
#             "groups": groups,
#             "teams": sorted(teams.values(), key=str.casefold),
#             "known_teams": known_count,
#             "expected_teams": expected_teams,
#             "pending_count": pending_count,
#             "pending_sources": list(dict.fromkeys(pending_sources.get(stage_key, []))),
#             "completed_matches": completed_matches,
#             "total_matches": total_matches,
#             "completion_pct": pct,
#             "status": status,
#             "division": str(row.get("groups_division", "")),
#             "_stage_key": stage_key,
#         })
#     return sections


# def _live_stage_lobbies(fixtures: pd.DataFrame, stage_formats: pd.DataFrame, sections: list[dict[str, object]]) -> list[dict[str, object]]:
#     """Build predictions for remaining games in a fully known 16-team lobby stage."""
#     lobbies = _upcoming_lobbies(fixtures)
#     if stage_formats.empty or not sections:
#         return lobbies
#     formats = stage_formats.copy()
#     formats["_stage_key"] = formats["stage"].map(_stage_token)
#     formats["sequence"] = pd.to_numeric(formats.get("sequence"), errors="coerce").fillna(0)
#     section_map = {str(section["_stage_key"]): section for section in sections}
#     bmsd = _filter_history("BMSD")
#     lobby_formats = {"single lobby", "smash rules", "round robin"}
#     active = []
#     for _, row in formats.iterrows():
#         section = section_map.get(str(row["_stage_key"]), {})
#         division = str(row.get("groups_division", "")).strip().casefold()
#         if (
#             division in lobby_formats
#             and int(section.get("known_teams", 0)) == int(section.get("expected_teams", 0))
#             and int(section.get("completed_matches", 0)) > 0
#             and int(section.get("completed_matches", 0)) < int(section.get("total_matches", 0))
#         ):
#             active.append((int(row["sequence"]), row, section))
#     candidate = max(active, key=lambda item: item[0]) if active else None
#     if candidate is None:
#         completed = [int(section.get("sequence", 0)) for section in sections if int(section.get("completed_matches", 0)) > 0]
#         last_sequence = max(completed, default=0)
#         future = [
#             (int(row["sequence"]), row, section_map.get(str(row["_stage_key"]), {}))
#             for _, row in formats.iterrows()
#             if int(row["sequence"]) > last_sequence
#             and str(row.get("groups_division", "")).strip().casefold() in lobby_formats
#             and int(section_map.get(str(row["_stage_key"]), {}).get("known_teams", 0)) == int(section_map.get(str(row["_stage_key"]), {}).get("expected_teams", 0))
#         ]
#         candidate = min(future, key=lambda item: item[0]) if future else None
#     if candidate is None:
#         return lobbies

#     _, stage_row, section = candidate
#     division = str(stage_row.get("groups_division", "")).strip().casefold()
#     if division not in lobby_formats:
#         return lobbies

#     stage_key = str(stage_row["_stage_key"])
#     stage_completed = bmsd[bmsd["stage"].map(_stage_token).eq(stage_key)] if not bmsd.empty else pd.DataFrame()
#     local_match = pd.to_numeric(stage_completed.get("match_id"), errors="coerce") if not stage_completed.empty and "match_id" in stage_completed else pd.Series(dtype=float)
#     last_local = int(local_match.max()) if not local_match.empty and local_match.notna().any() else 0
#     if last_local == 0:
#         last_local = 0
#     parsed_total = pd.to_numeric(stage_row.get("total_matches"), errors="coerce")
#     total_matches = int(parsed_total) if pd.notna(parsed_total) else 0
#     current_global = int(pd.to_numeric(bmsd.get("match_order"), errors="coerce").max()) if not bmsd.empty else 0
#     existing_ids = {(str(lobby.get("stage", "")).casefold(), str(lobby.get("match", ""))) for lobby in lobbies}

#     if division == "round robin":
#         groups = [g for g in section.get("groups", []) if g.get("teams")]
#         if len(groups) >= 2:
#             pairings = []
#             for i in range(len(groups)):
#                 for j in range(i + 1, len(groups)):
#                     pairings.append((groups[i], groups[j]))
#             if pairings:
#                 for number in range(last_local + 1, total_matches + 1):
#                     if (str(stage_row.get("stage", "")).casefold(), str(number)) in existing_ids:
#                         continue
#                     p_idx = (number - 1) % len(pairings)
#                     g1, g2 = pairings[p_idx]
#                     match_teams = sorted(set(g1["teams"] + g2["teams"]), key=str.casefold)
#                     lobbies.append({
#                         "id": f"inferred-{stage_key.replace(' ', '-')}-{number}",
#                         "overall_match": current_global + (number - last_local),
#                         "match": number,
#                         "stage": str(stage_row.get("stage", "")),
#                         "group": f"{g1['group']} vs {g2['group']}",
#                         "map": "",
#                         "scheduled_at": "",
#                         "teams": match_teams,
#                         "inferred": True,
#                         "stage_total": total_matches,
#                         "completed_in_stage": last_local,
#                         "as_of_match": current_global,
#                     })
#     else:
#         teams = list(section.get("teams", []))
#         expected = int(section.get("expected_teams", 0))
#         if expected < 2 or len(teams) != expected:
#             return lobbies
#         for number in range(last_local + 1, total_matches + 1):
#             if (str(stage_row.get("stage", "")).casefold(), str(number)) in existing_ids:
#                 continue
#             lobbies.append({
#                 "id": f"inferred-{stage_key.replace(' ', '-')}-{number}",
#                 "overall_match": current_global + (number - last_local),
#                 "match": number,
#                 "stage": str(stage_row.get("stage", "")),
#                 "group": "" if division == "smash rules" else "Single Lobby",
#                 "map": "",
#                 "scheduled_at": "",
#                 "teams": teams,
#                 "inferred": True,
#                 "stage_total": total_matches,
#                 "completed_in_stage": last_local,
#                 "as_of_match": current_global,
#             })
#     lobbies.sort(key=lambda item: (pd.to_numeric(item.get("overall_match"), errors="coerce") if pd.notna(pd.to_numeric(item.get("overall_match"), errors="coerce")) else float("inf"), str(item.get("stage", ""))))
#     return lobbies


# def _grand_finals_smash_state(next_match: object) -> dict[str, object]:
#     """Describe the Grand Finals Match Point target and eligibility before a match."""
#     try:
#         match_number = int(next_match)
#     except (TypeError, ValueError):
#         return {}

#     bmsd = _filter_history("BMSD")
#     finals = bmsd[bmsd["stage"].map(_stage_token).eq("grand finals")].copy()
#     if finals.empty:
#         return {
#             "match": match_number,
#             "target_score": None,
#             "leader_after_match_12": None,
#             "eligible_teams": [],
#             "champion": None,
#         }
#     finals["match_id"] = pd.to_numeric(finals["match_id"], errors="coerce")
#     finals["points"] = pd.to_numeric(finals["points"], errors="coerce").fillna(0)
#     finals["win"] = pd.to_numeric(finals["win"], errors="coerce").fillna(0)
#     completed = set(finals["match_id"].dropna().astype(int).tolist())
#     first_twelve = finals[finals["match_id"].between(1, 12)]
#     if not set(range(1, 13)).issubset(completed):
#         return {
#             "match": match_number,
#             "target_score": None,
#             "leader_after_match_12": None,
#             "eligible_teams": [],
#             "champion": None,
#         }

#     totals_after_12 = first_twelve.groupby("team")["points"].sum()
#     leader_score = float(totals_after_12.max())
#     target_score = leader_score + 10.0

#     champion = None
#     championship_match = None
#     for game in range(13, match_number):
#         before_game = finals[finals["match_id"].lt(game)].groupby("team")["points"].sum()
#         eligible_before_game = set(before_game[before_game.ge(target_score)].index.astype(str))
#         winners = set(finals.loc[finals["match_id"].eq(game) & finals["win"].eq(1), "team"].dropna().astype(str))
#         decided = sorted(eligible_before_game & winners, key=str.casefold)
#         if decided:
#             champion = decided[0]
#             championship_match = game
#             break

#     prior_totals = finals[finals["match_id"].lt(match_number)].groupby("team")["points"].sum()
#     eligible_teams = sorted(
#         prior_totals[prior_totals.ge(target_score)].index.astype(str).tolist(),
#         key=str.casefold,
#     )
#     return {
#         "match": match_number,
#         "smash_phase": match_number >= 13,
#         "target_score": target_score,
#         "leader_after_match_12": leader_score,
#         "eligible_teams": eligible_teams,
#         "champion": champion,
#         "championship_match": championship_match,
#     }


# _HISTORY_SLICE_CACHE: dict[float, pd.DataFrame] = {}


# def _history_before_bmsd_match(overall_match: object) -> pd.DataFrame:
#     try:
#         next_order = float(overall_match)
#     except (TypeError, ValueError):
#         return HISTORY.copy()
#     cached = _HISTORY_SLICE_CACHE.get(next_order)
#     if cached is not None:
#         return cached
#     other_events = HISTORY[~HISTORY["tournament"].astype(str).str.casefold().eq("bmsd")]
#     earlier_bmsd = HISTORY[
#         HISTORY["tournament"].astype(str).str.casefold().eq("bmsd")
#         & pd.to_numeric(HISTORY["match_order"], errors="coerce").lt(next_order)
#     ]
#     result = pd.concat([other_events, earlier_bmsd], ignore_index=True)
#     _HISTORY_SLICE_CACHE[next_order] = result
#     return result


# def _lobby_predictions(lobby: dict[str, object], cache: dict | None = None) -> list[dict[str, object]]:
#     if MODEL is None or PREPROCESSOR is None:
#         return []
#     history_before = _history_before_bmsd_match(lobby.get("overall_match"))
#     history_order = pd.to_numeric(history_before.get("match_order"), errors="coerce")
#     snapshot_order = float(history_order.max()) if not history_order.empty and history_order.notna().any() else -1.0
#     cache = LIVE_PREDICTION_CACHE if cache is None else cache

#     raw_teams = lobby.get("teams", [])
#     teams = [str(t).strip() for t in raw_teams if str(t).strip()]
#     if not teams:
#         return []

#     uncached_teams: list[str] = []
#     uncached_keys: list[tuple[int, float, str]] = []
#     for team in teams:
#         cache_key = (id(HISTORY), snapshot_order, canonical_team_key(team))
#         if cache_key not in cache:
#             uncached_teams.append(team)
#             uncached_keys.append(cache_key)

#     if uncached_teams:
#         from src.feature_engineering import batch_feature_snapshots
#         snapshots = batch_feature_snapshots(history_before, uncached_teams)
#         feature_rows: list[tuple[str, tuple[int, float, str], dict[str, float], bool]] = []
#         for team, ck in zip(uncached_teams, uncached_keys):
#             fields = snapshots.get(team, {})
#             has_history = bool(
#                 not history_before.empty
#                 and history_before["team_key"].eq(canonical_team_key(team)).any()
#             )
#             feature_rows.append((team, ck, fields, has_history))

#         input_data = [
#             {f: float(fields.get(f, 0.0)) for f in MODEL_FEATURES}
#             for _, _, fields, _ in feature_rows
#         ]
#         if input_data:
#             input_df = pd.DataFrame(input_data)[MODEL_FEATURES]
#             matrix = PREPROCESSOR.transform(input_df) if PREPROCESSOR is not None else input_df
#             probs = MODEL.predict_proba(matrix)
#             for (_, ck, fields, has_history), p_val in zip(feature_rows, probs):
#                 raw_p = float(np.clip(p_val[1], 0.0, 1.0))
#                 cache[ck] = {
#                     "raw_probability": raw_p,
#                     "matches": fields.get("matches_before", 0.0),
#                     "has_history": has_history,
#                 }

#     rows = []
#     for team in teams:
#         cache_key = (id(HISTORY), snapshot_order, canonical_team_key(team))
#         cached = cache.get(cache_key)
#         if cached:
#             rows.append({
#                 "team": team,
#                 "raw_probability": float(cached["raw_probability"]),
#                 "matches": float(cached["matches"]),
#                 "has_history": bool(cached["has_history"]),
#             })
#     total = sum(row["raw_probability"] for row in rows)
#     for row in rows:
#         row["probability"] = 100.0 * row["raw_probability"] / total if total > 0 else 100.0 / max(1, len(rows))
#     return sorted(rows, key=lambda row: row["probability"], reverse=True)


# def _next_match_mvp_candidates(lobby: dict[str, object] | None) -> list[dict[str, object]]:
#     """Rank known lobby players by leakage-safe estimated match finishes."""
#     if not lobby or not lobby.get("teams"):
#         return []
#     players = _read_source_csv(PLAYER_MATCH_STATS_PATH)
#     fixtures = _read_source_csv(DATA_DIR / "raw" / "bmsd_fixtures.csv")
#     if players.empty or "source_match_id" not in players or "elims" not in players:
#         return []
#     match_order = pd.to_numeric(lobby.get("overall_match"), errors="coerce")
#     snapshot = build_player_prediction_frame(
#         players,
#         fixtures,
#         [str(team) for team in lobby["teams"]],
#         float(match_order) if pd.notna(match_order) else None,
#     )
#     if snapshot.empty:
#         return []
#     if MVP_MODEL is not None and MVP_METADATA.get("selected_method") == "Random Forest regressor":
#         columns = MVP_MODEL_FEATURES or PLAYER_FEATURES
#         snapshot["predicted_finishes"] = np.clip(MVP_MODEL.predict(snapshot[columns].astype(float)), 0.0, None)
#     else:
#         snapshot["predicted_finishes"] = 0.5 * snapshot["prev_avg_finishes"] + 0.5 * snapshot["recent3_avg_finishes"]
#     rows = []
#     for _, row in snapshot.iterrows():
#         rows.append({
#             "player": str(row["player"]),
#             "team": str(row["team"]),
#             "matches": int(row["matches_before"]),
#             "avg_finishes": float(row["prev_avg_finishes"]),
#             "recent3_finishes": float(row["recent3_avg_finishes"]),
#             "predicted_finishes": float(row["predicted_finishes"]),
#         })
#     return sorted(rows, key=lambda row: (row["predicted_finishes"], row["recent3_finishes"], row["avg_finishes"]), reverse=True)[:12]


# def _grand_finals_scoreboard(teams: list[str]) -> list[dict[str, object]]:
#     """Build a score table for the 16 confirmed finalists, including zero-score rows before play."""
#     roster = sorted({str(team).strip() for team in teams if str(team).strip()}, key=str.casefold)
#     if len(roster) != 16:
#         return []

#     bmsd = _filter_history("BMSD")
#     finals = bmsd[bmsd["stage"].map(_stage_token).eq("grand finals")].copy()
#     stats_by_team: dict[str, dict[str, object]] = {}
#     if not finals.empty:
#         for column in ("points", "kills", "win", "placement_points", "rank"):
#             finals[column] = pd.to_numeric(finals.get(column), errors="coerce")
#         finals["team_key"] = finals["team"].map(canonical_team_key)
#         totals = finals.groupby("team_key", sort=False).agg(
#             matches=("match_id", "nunique"),
#             points=("points", "sum"),
#             finishes=("kills", "sum"),
#             wwcds=("win", "sum"),
#             placement_points=("placement_points", "sum"),
#             last_rank=("rank", "last"),
#         )
#         stats_by_team = totals.to_dict("index")

#     rows = []
#     for team in roster:
#         values = stats_by_team.get(canonical_team_key(team), {})
#         rows.append({
#             "team": team,
#             "matches": int(values.get("matches", 0) or 0),
#             "points": float(values.get("points", 0) or 0),
#             "finishes": float(values.get("finishes", 0) or 0),
#             "wwcds": int(values.get("wwcds", 0) or 0),
#             "placement_points": float(values.get("placement_points", 0) or 0),
#             "last_rank": float(values.get("last_rank", float("inf")) or float("inf")),
#         })
#     return sorted(
#         rows,
#         key=lambda row: (row["points"], row["wwcds"], row["placement_points"], row["finishes"], -row["last_rank"]),
#         reverse=True,
#     )


# def _mvp_watchlist(stage: str = "All Stages") -> tuple[list[dict[str, object]], list[str]]:
#     players = _read_source_csv(PLAYER_STATS_PATH)
#     match_stats = _read_source_csv(PLAYER_MATCH_STATS_PATH)
#     if stage not in ("", "All Stages") and not match_stats.empty:
#         selected = match_stats[match_stats["stage"].fillna("").astype(str).eq(stage)].copy()
#         if selected.empty:
#             return [], sorted(match_stats["stage"].dropna().astype(str).unique().tolist())
#         selected["elims"] = pd.to_numeric(selected["elims"], errors="coerce").fillna(0.0)
#         board = selected.groupby(["player_id", "player", "team"], as_index=False).agg(
#             matches_played=("source_match_id", "nunique"),
#             total_elims=("elims", "sum"),
#             avg_elims=("elims", "mean"),
#             max_elims=("elims", "max"),
#             five_plus_elims_matches=("elims", lambda values: int(values.ge(5).sum())),
#         )
#     elif not players.empty:
#         board = players.rename(columns={
#             "player": "player", "team": "team", "matches_played": "matches_played",
#             "total_elims": "total_elims", "avg_elims": "avg_elims", "max_elims": "max_elims",
#             "five_plus_elims_matches": "five_plus_elims_matches", "reported_mvp_awards": "reported_mvp_awards",
#         }).copy()
#     else:
#         return [], sorted(match_stats["stage"].dropna().astype(str).unique().tolist()) if not match_stats.empty else []

#     for column in ("matches_played", "total_elims", "avg_elims", "max_elims", "five_plus_elims_matches"):
#         if column in board:
#             board[column] = pd.to_numeric(board[column], errors="coerce").fillna(0.0)
#     board = board.sort_values(["total_elims", "avg_elims", "matches_played"], ascending=False, kind="stable")
#     stages = sorted(match_stats["stage"].dropna().astype(str).unique().tolist()) if not match_stats.empty else []
#     return board.head(30).to_dict("records"), stages


# @app.route("/")
# def home():
#     per_tournament = []
#     for name in _allowed_tournaments()[1:]:
#         subset = _filter_history(name)
#         stats = _summary_stats(subset)
#         per_tournament.append({"name": name, **stats})
#     return render_template(
#         "index.html",
#         stats=_summary_stats(HISTORY),
#         per_tournament=per_tournament,
#         model_available=MODEL is not None,
#         bmsd_matches=_summary_stats(_filter_history("BMSD"))["matches"],
#     )


# @app.route("/dashboard")
# def dashboard():
#     tournament = request.args.get("tournament", "All Tournaments")
#     frame = _filter_history(tournament)
#     cache_key = f"dashboard_{tournament}"
#     cached = _cache_get(cache_key)
#     if cached is not None:
#         board, latest_matches = cached
#     else:
#         board = _team_summary(frame)
#         latest_matches = []
#         if not frame.empty:
#             matches = frame.drop_duplicates("match_key").sort_values(["year", "date", "match_order"], na_position="first")
#             latest_keys = set(matches.tail(6)["match_key"])
#             latest_matches = frame[frame["match_key"].isin(latest_keys)].groupby("match_key", as_index=False).agg(
#                 tournament=("tournament", "first"),
#                 match_id=("match_id", "first"),
#                 stage=("stage", "first"),
#                 date=("date", "first"),
#                 teams=("team", "nunique"),
#                 winner=("team", lambda values: values[frame.loc[values.index, "win"].eq(1)].iloc[0] if frame.loc[values.index, "win"].eq(1).any() else "Unknown"),
#                 total_points=("points", "sum"),
#             ).to_dict("records")
#         _cache_set(cache_key, (board, latest_matches))
#     return render_template(
#         "dashboard.html",
#         tournament=tournament,
#         tournaments=_allowed_tournaments(),
#         stats=_summary_stats(frame),
#         leaderboard=board[:12],
#         latest_matches=latest_matches,
#         chart_teams=board[:10],
#     )


# @app.route("/leaderboard")
# def leaderboard():
#     tournament = request.args.get("tournament", "All Tournaments")
#     return render_template(
#         "leaderboard.html",
#         tournament=tournament,
#         tournaments=_allowed_tournaments(),
#         leaderboard=_team_summary(_filter_history(tournament)),
#     )


# @app.route("/history")
# def history():
#     tournament = request.args.get("tournament", "All Tournaments")
#     frame = _filter_history(tournament)
#     keys = []
#     match_data: list[dict[str, object]] = []
#     selected = request.args.get("match", "")
#     if not frame.empty:
#         metadata = frame.drop_duplicates("match_key").sort_values(["year", "date", "match_order"], na_position="first")
#         keys = metadata[["match_key", "tournament", "stage", "match_id"]].to_dict("records")
#         if selected:
#             match_data = frame[frame["match_key"].astype(str).eq(selected)].sort_values("points", ascending=False).to_dict("records")
#     return render_template("history.html", tournament=tournament, tournaments=_allowed_tournaments(), matches=keys, selected=selected, match_data=match_data)


# @app.route("/team-analytics")
# def team_analytics():
#     tournament = request.args.get("tournament", "All Tournaments")
#     board = _team_summary(_filter_history(tournament))
#     selected_team = request.args.get("team", board[0]["team"] if board else "")
#     team_rows = []
#     if selected_team:
#         selected_frame = _filter_history(tournament)
#         team_rows = selected_frame[selected_frame["team_key"].eq(canonical_team_key(selected_team))].sort_values(["year", "date", "match_order"], na_position="first").to_dict("records")
#     return render_template("team_analytics.html", tournament=tournament, tournaments=_allowed_tournaments(), teams=[row["team"] for row in board], selected_team=selected_team, team_rows=team_rows)


# @app.route("/compare", methods=["GET", "POST"])
# def compare():
#     tournaments = _allowed_tournaments()
#     tournament_a = request.values.get("tournament_a") or request.values.get("tournament") or "All Tournaments"
#     tournament_b = request.values.get("tournament_b") or request.values.get("tournament") or "All Tournaments"
#     tournament_teams = {
#         t: [row["team"] for row in _team_summary(_filter_history(t))]
#         for t in tournaments
#     }
#     teams_a = tournament_teams.get(tournament_a, [])
#     teams_b = tournament_teams.get(tournament_b, [])
#     team_a = request.values.get("team_a") or (teams_a[0] if teams_a else "")
#     team_b = request.values.get("team_b") or (teams_b[1] if len(teams_b) > 1 else (teams_b[0] if teams_b else ""))
#     summary_a = {row["team"]: row for row in _team_summary(_filter_history(tournament_a))}
#     summary_b = {row["team"]: row for row in _team_summary(_filter_history(tournament_b))}
#     a = summary_a.get(team_a)
#     b = summary_b.get(team_b)
#     comparison = None
#     if a and b:
#         a_copy = dict(a)
#         b_copy = dict(b)
#         a_copy["tournament"] = tournament_a
#         b_copy["tournament"] = tournament_b
#         try:
#             a_copy["model_probability"] = _predict(team_a)["winning_probability_percentage"] if MODEL is not None else None
#         except (ValueError, RuntimeError):
#             a_copy["model_probability"] = None
#         try:
#             b_copy["model_probability"] = _predict(team_b)["winning_probability_percentage"] if MODEL is not None else None
#         except (ValueError, RuntimeError):
#             b_copy["model_probability"] = None
#         comparison = {"team_a": a_copy, "team_b": b_copy}
#     return render_template(
#         "compare.html",
#         tournament=tournament_a,
#         tournament_a=tournament_a,
#         tournament_b=tournament_b,
#         tournaments=tournaments,
#         teams=teams_a,
#         teams_a=teams_a,
#         teams_b=teams_b,
#         team_a=team_a,
#         team_b=team_b,
#         tournament_teams=tournament_teams,
#         comparison=comparison,
#     )


# @app.route("/predict", methods=["GET", "POST"])
# @app.route("/predictor", methods=["GET", "POST"])
# def predict_page():
#     # Public-facing predictions are selected from real BMSD stage lobbies.
#     # Team statistics come from the saved scorecards, so users never have to
#     # type team names or pre-match feature values into the web form.
#     params = {}
#     for name in ("stage", "forecast"):
#         value = request.values.get(name, "").strip()
#         if value:
#             params[name] = value
#     return redirect(url_for("live", _anchor="team-prediction", **params))


# @app.route("/live", methods=["GET"])
# def live():
#     bmsd = _filter_history("BMSD")
#     board = _cache_get("live_board")
#     if board is None:
#         board = _team_summary(bmsd)
#         _cache_set("live_board", board)
#     fixtures = _cache_get("live_fixtures")
#     if fixtures is None:
#         fixtures = _read_source_csv(FIXTURES_PATH)
#         _cache_set("live_fixtures", fixtures)
#     progression = _cache_get("live_progression")
#     if progression is None:
#         progression = _read_source_csv(PROGRESSION_PATH)
#         _cache_set("live_progression", progression)
#     stage_formats = _cache_get("live_stage_formats")
#     if stage_formats is None:
#         stage_formats = _read_source_csv(STAGE_FORMAT_PATH)
#         _cache_set("live_stage_formats", stage_formats)
#     stage_sections = _cache_get("live_stage_sections")
#     if stage_sections is None:
#         stage_sections = _bmsd_stage_sections(stage_formats)
#         _cache_set("live_stage_sections", stage_sections)
#     all_groups = [
#         {"stage": section["stage"], **group, "known_teams": len(group.get("teams", [])), "expected_teams": section["expected_teams"]}
#         for section in stage_sections
#         for group in section["groups"]
#     ] or _bmsd_group_rows(fixtures)
#     fixture_stages = list(dict.fromkeys([row["stage"] for row in all_groups]))
#     format_stages = stage_formats.sort_values("sequence", kind="stable")["stage"].dropna().astype(str).tolist() if not stage_formats.empty and "sequence" in stage_formats else []
#     progression_stages = []
#     if not progression.empty and "source_stages" in progression:
#         for value in progression["source_stages"].dropna().astype(str):
#             progression_stages.extend(stage.strip() for stage in value.split("|") if stage.strip())
#     all_stages = list(dict.fromkeys(format_stages + fixture_stages + progression_stages))
#     stage_tracker = []
#     section_map = {_stage_token(str(section["stage"])): section for section in stage_sections}
#     for stage in all_stages:
#         section = section_map.get(_stage_token(stage))
#         if section:
#             status = f"{section['completed_matches']} of {section['total_matches']} matches · {section['known_teams']} of {section['expected_teams']} teams"
#         else:
#             status = "Team assignments not published"
#         stage_tracker.append({"stage": stage, "status": status})
#     selected_stage = request.args.get("stage", "All Stages")
#     group_rows = all_groups if selected_stage in ("", "All Stages") else [row for row in all_groups if row["stage"] == selected_stage]
#     upcoming = _cache_get("live_upcoming")
#     if upcoming is None:
#         upcoming = _live_stage_lobbies(fixtures, stage_formats, stage_sections)
#         for lobby in upcoming:
#             section = section_map.get(_stage_token(lobby.get("stage")))
#             if section:
#                 lobby.setdefault("stage_total", section.get("total_matches"))
#                 lobby.setdefault("completed_in_stage", section.get("completed_matches"))
#         for lobby in upcoming:
#             lobby["predictions"] = _lobby_predictions(lobby)
#         _cache_set("live_upcoming", upcoming)
#     else:
#         for lobby in upcoming:
#             section = section_map.get(_stage_token(lobby.get("stage")))
#             if section:
#                 lobby.setdefault("stage_total", section.get("total_matches"))
#                 lobby.setdefault("completed_in_stage", section.get("completed_matches"))
#     selected_forecast = request.args.get("forecast", "")
#     next_lobby = next((lobby for lobby in upcoming if str(lobby.get("id")) == selected_forecast), None)
#     if next_lobby is None and upcoming:
#         next_lobby = upcoming[0]
#     next_lobby_id = str(next_lobby.get("id")) if next_lobby else ""
#     gf_smash_key = f"gf_smash_{next_lobby.get('match') if next_lobby else ''}"
#     grand_finals_smash = _cache_get(gf_smash_key)
#     if grand_finals_smash is None and next_lobby and _stage_token(next_lobby.get("stage")) == "grand finals":
#         grand_finals_smash = _grand_finals_smash_state(next_lobby.get("match"))
#         _cache_set(gf_smash_key, grand_finals_smash)
#     grand_finals_section = next((section for section in stage_sections if _stage_token(section.get("stage")) == "grand finals"), None)
#     grand_finals_teams = list(grand_finals_section.get("teams", [])) if grand_finals_section else []
#     grand_finals_ready = bool(
#         grand_finals_section
#         and int(grand_finals_section.get("expected_teams", 0)) == 16
#         and len(set(grand_finals_teams)) == 16
#     )
#     gf_scoreboard_key = f"gf_scoreboard_{','.join(sorted(grand_finals_teams))}"
#     grand_finals_scoreboard = _cache_get(gf_scoreboard_key) if grand_finals_ready else []
#     if grand_finals_ready and grand_finals_scoreboard is None:
#         grand_finals_scoreboard = _grand_finals_scoreboard(grand_finals_teams)
#         _cache_set(gf_scoreboard_key, grand_finals_scoreboard)
#     grand_finals_mvp_candidates = []
#     if grand_finals_ready:
#         final_lobby = next((lobby for lobby in upcoming if _stage_token(lobby.get("stage")) == "grand finals"), None)
#         final_match_order = pd.to_numeric(final_lobby.get("overall_match"), errors="coerce") if final_lobby else float("nan")
#         if pd.isna(final_match_order):
#             completed_order = pd.to_numeric(bmsd.get("match_order"), errors="coerce")
#             final_match_order = float(completed_order.max() + 1) if not completed_order.empty and completed_order.notna().any() else 1.0
#         gf_mvp_key = f"gf_mvp_{final_match_order}"
#         grand_finals_mvp_candidates = _cache_get(gf_mvp_key)
#         if grand_finals_mvp_candidates is None:
#             grand_finals_mvp_candidates = _next_match_mvp_candidates({
#                 "teams": grand_finals_teams,
#                 "overall_match": float(final_match_order),
#                 "stage": "Grand Finals",
#                 "match": int(final_lobby.get("match", 1)) if final_lobby else 1,
#             })
#             _cache_set(gf_mvp_key, grand_finals_mvp_candidates)
#     mvp_key = f"mvp_{next_lobby.get('id') if next_lobby else ''}"
#     mvp_predictions = _cache_get(mvp_key)
#     if mvp_predictions is None:
#         mvp_predictions = _next_match_mvp_candidates(next_lobby)
#         _cache_set(mvp_key, mvp_predictions)
#     for section in stage_sections:
#         section["forecasts"] = [
#             {
#                 "match": lobby.get("match"),
#                 "id": lobby.get("id"),
#                 "stage_total": lobby.get("stage_total", section.get("total_matches")),
#                 "favorite": lobby["predictions"][0]["team"] if lobby.get("predictions") else "",
#                 "probability": lobby["predictions"][0]["probability"] if lobby.get("predictions") else 0.0,
#                 "teams": len(lobby.get("teams", [])),
#                 "id": lobby.get("id"),
#                 "inferred": bool(lobby.get("inferred")),
#             }
#             for lobby in upcoming
#             if _stage_token(lobby.get("stage")) == _stage_token(section.get("stage"))
#         ]
#     remaining_forecasts = [
#         {
#             "stage": lobby.get("stage"),
#             "match": lobby.get("match"),
#             "stage_total": lobby.get("stage_total"),
#             "favorite": lobby["predictions"][0]["team"] if lobby.get("predictions") else "",
#             "probability": lobby["predictions"][0]["probability"] if lobby.get("predictions") else 0.0,
#             "teams": len(lobby.get("teams", [])),
#             "inferred": bool(lobby.get("inferred")),
#             "id": lobby.get("id"),
#         }
#         for lobby in upcoming
#         if str(lobby.get("id")) != next_lobby_id
#     ]
#     source_checked = ""
#     if not fixtures.empty and "source_checked" in fixtures:
#         dates = fixtures["source_checked"].dropna().astype(str)
#         source_checked = dates.iloc[-1] if not dates.empty else ""
#     return render_template(
#         "live.html",
#         bmsd_matches=_summary_stats(bmsd)["matches"],
#         bmsd_teams=_summary_stats(bmsd)["teams"],
#         leaderboard=board,
#         groups=group_rows,
#         stage_sections=stage_sections,
#         all_stages=all_stages,
#         stage_tracker=stage_tracker,
#         selected_stage=selected_stage,
#         upcoming=upcoming,
#         next_lobby=next_lobby,
#         grand_finals_smash=grand_finals_smash,
#         grand_finals_ready=grand_finals_ready,
#         grand_finals_scoreboard=grand_finals_scoreboard,
#         grand_finals_mvp_candidates=grand_finals_mvp_candidates,
#         mvp_predictions=mvp_predictions,
#         mvp_model_method=str(MVP_METADATA.get("selected_method", "Rolling average baseline")),
#         next_lobby_id=next_lobby_id,
#         other_upcoming=[lobby for lobby in upcoming if str(lobby.get("id")) != next_lobby_id],
#         forecast_options=upcoming,
#         remaining_forecasts=remaining_forecasts,
#         finalists=progression.to_dict("records") if not progression.empty else [],
#         source_checked=source_checked,
#         model_available=MODEL is not None,
#         eam_url=EAM_BMSD_URL,
#         matches_url=BMSD_MATCHES_URL,
#         progression_url=BMSD_PROGRESSION_URL,
#         format_url=BMSD_FORMAT_URL,
#         liquipedia_url=LIQUIPEDIA_BMSD_URL,
#         uploaded=request.args.get("uploaded", "") == "1",
#         refreshed=request.args.get("refreshed", "") == "1",
#         error=request.args.get("error", ""),
#         source_warning=request.args.get("warning", ""),
#     )


# @app.route("/live/refresh", methods=["POST"])
# def refresh_bmsd_source():
#     global HISTORY
#     try:
#         raw_matches, fixtures = fetch_bmsd_source()
#         # Stage lengths and qualification rules are kept separately from match
#         # scorecards; they let the live page show remaining games and bracket
#         # teams without making up a schedule or team assignment.
#         stage_format = None
#         stage_format_warning = None
#         try:
#             stage_format = fetch_bmsd_format()
#         except Exception as exc:
#             logger.warning("BMSD scorecards refreshed but stage format could not be read: %s", exc)
#             stage_format_warning = "The format page was unavailable, so saved bracket rules are shown."
#         imported = normalize_match_data(raw_matches, "BMSD", default_year=2026)
#         if imported.empty:
#             raise ValueError("No completed scorecards were returned by the source.")
#         validate_match_lobbies(imported)
#         # This public endpoint returns the complete BMSD stageGroups collection.
#         # Replace the BMSD snapshot so an older manual seed with a differently
#         # formatted match ID cannot double-count a scorecard already on source.
#         merged_bmsd = imported.copy()
#         validate_match_lobbies(merged_bmsd)

#         DATA_DIR.mkdir(parents=True, exist_ok=True)
#         FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
#         raw_matches.to_csv(DATA_DIR / "raw" / "bmsd_matches.csv", index=False)
#         fixtures.to_csv(FIXTURES_PATH, index=False)
#         merged_bmsd.to_csv(DATA_DIR / "bmsd.csv", index=False)
#         if stage_format is not None and not stage_format.empty:
#             stage_format.to_csv(STAGE_FORMAT_PATH, index=False)

#         warnings = [stage_format_warning] if stage_format_warning else []
#         try:
#             routes = fetch_bmsd_progression()
#             route_rows = []
#             for route in routes:
#                 for team in route["teams"]:
#                     route_rows.append({
#                         "route_id": route["route_id"],
#                         "route": route["name"],
#                         "advancement": route["advancement"],
#                         "source_stages": "|".join(route.get("stages", [])),
#                         "team": team,
#                         "source_url": BMSD_PROGRESSION_URL,
#                         "source_checked": pd.Timestamp.now(tz="UTC").isoformat(),
#                     })
#             pd.DataFrame(route_rows).to_csv(PROGRESSION_PATH, index=False)
#         except Exception as exc:
#             logger.warning("BMSD scorecards refreshed but progression could not be read: %s", exc)
#             warnings.append("The progression page was unavailable, so the last saved finalist list is shown.")
#         try:
#             player_stats, player_match_stats = fetch_bmsd_player_stats()
#             player_stats.to_csv(PLAYER_STATS_PATH, index=False)
#             player_match_stats.to_csv(PLAYER_MATCH_STATS_PATH, index=False)
#         except Exception as exc:
#             logger.warning("BMSD scorecards refreshed but player statistics could not be read: %s", exc)
#             warnings.append("The player statistics page was unavailable, so the last saved MVP watchlist is shown.")
#         HISTORY = _load_history()
#         LIVE_PREDICTION_CACHE.clear()
#         _cache_clear()
#         return redirect(url_for("live", refreshed="1", warning=" ".join(warnings)))
#     except Exception as exc:
#         logger.warning("BMSD source refresh failed: %s", exc)
#         return redirect(url_for("live", error=f"Could not refresh BMSD data: {exc}"))


# @app.route("/live/import", methods=["POST"])
# def import_bmsd_scorecards():
#     global HISTORY
#     uploaded = request.files.get("scorecards")
#     if uploaded is None or not uploaded.filename:
#         return redirect(url_for("live", error="Choose a completed-match CSV scorecard first."))
#     if Path(uploaded.filename).suffix.casefold() != ".csv":
#         return redirect(url_for("live", error="Upload a CSV file."))
#     try:
#         raw = pd.read_csv(uploaded.stream)
#         imported = normalize_match_data(raw, "BMSD", default_year=2026)
#         if imported.empty:
#             raise ValueError("The CSV contains no match records.")
#         if not imported["tournament"].astype(str).str.casefold().eq("bmsd").all():
#             raise ValueError("The uploaded scorecards must be BMSD match data.")
#         validate_match_lobbies(imported)

#         # Uploads may contain just the latest completed match. Keep earlier
#         # imported BMSD scorecards and replace whole matches in this upload,
#         # so corrections do not leave duplicate team rows behind.
#         existing_bmsd = HISTORY[HISTORY["tournament"].astype(str).str.casefold().eq("bmsd")].copy()
#         replaced_matches = set(imported["match_key"].astype(str))
#         retained = existing_bmsd[~existing_bmsd["match_key"].astype(str).isin(replaced_matches)]
#         merged_bmsd = pd.concat([retained, imported], ignore_index=True)
#         validate_match_lobbies(merged_bmsd)

#         raw_path = DATA_DIR / "raw" / "bmsd_matches.csv"
#         raw_path.parent.mkdir(parents=True, exist_ok=True)
#         if raw_path.exists() and raw_path.stat().st_size:
#             previous_raw = pd.read_csv(raw_path)
#             raw = pd.concat([previous_raw, raw], ignore_index=True).drop_duplicates()
#         raw.to_csv(raw_path, index=False)
#         merged_bmsd.to_csv(DATA_DIR / "bmsd.csv", index=False)
#         HISTORY = _load_history()
#         LIVE_PREDICTION_CACHE.clear()
#         _cache_clear()
#         return redirect(url_for("live", uploaded="1"))
#     except Exception as exc:
#         logger.info("BMSD scorecard import rejected: %s", exc)
#         return redirect(url_for("live", error=str(exc)))


# @app.route("/live/template.csv")
# def download_bmsd_template():
#     template = pd.DataFrame(columns=[
#         "date", "overall_match", "match", "stage", "group", "map", "team", "rank",
#         "wwcd", "elim_points", "place_points", "total_points",
#     ])
#     output = io.BytesIO(template.to_csv(index=False).encode("utf-8"))
#     output.seek(0)
#     return send_file(output, mimetype="text/csv", as_attachment=True, download_name="bmsd_match_scorecard_template.csv")


# @app.route("/model-analytics")
# def model_analytics():
#     metrics = []
#     metric_path = REPORT_DIR / "model_comparison.csv"
#     if metric_path.exists():
#         try:
#             metrics = pd.read_csv(metric_path).to_dict("records")
#         except Exception as exc:
#             logger.warning("Unable to read model report: %s", exc)
#     importance = get_feature_importance(MODEL, MODEL_FEATURES) if MODEL is not None else []
#     metadata = {}
#     meta_path = REPORT_DIR / "model_metadata.json"
#     if meta_path.exists():
#         try:
#             import json
#             metadata = json.loads(meta_path.read_text(encoding="utf-8"))
#         except Exception:
#             pass
#     return render_template("model_analytics.html", metrics=metrics, importance=importance, model_available=MODEL is not None, metadata=metadata)


# @app.route("/model-report/<path:filename>")
# def model_report(filename: str):
#     allowed = {"model_comparison.png", "roc_curve.png", "calibration_curve.png", "feature_importance.png"}
#     if filename not in allowed or not (REPORT_DIR / filename).exists():
#         return render_template("error.html", code=404, message="That model report is not available yet."), 404
#     return send_file(REPORT_DIR / filename, mimetype="image/png", conditional=True)


# @app.route("/about")
# def about():
#     stats = [
#         {"name": name, **_summary_stats(_filter_history(name))}
#         for name in _allowed_tournaments()[1:]
#     ]
#     return render_template("about.html", stats=stats, model_available=MODEL is not None, eam_url=EAM_BMSD_URL, liquipedia_url=LIQUIPEDIA_BMSD_URL)


# @app.route("/mvp")
# def mvp():
#     selected_stage = request.args.get("stage", "All Stages")
#     players, stages = _mvp_watchlist(selected_stage)
#     source = _read_source_csv(PLAYER_STATS_PATH)
#     source_checked = ""
#     if not source.empty and "source_checked" in source:
#         values = source["source_checked"].dropna().astype(str)
#         source_checked = values.iloc[0] if not values.empty else ""
#     return render_template(
#         "mvp.html",
#         players=players,
#         stages=stages,
#         selected_stage=selected_stage,
#         has_data=not source.empty,
#         source_checked=source_checked,
#         stats_url="https://esportsamaze.com/bgmi/tournaments/battlegrounds-mobile-india-showdown-2026/statistics",
#     )


# @app.route("/api/predict", methods=["POST"])
# def api_predict():
#     data = request.get_json(silent=True)
#     if not isinstance(data, dict) or not data:
#         return jsonify({"success": False, "error": "Send a non-empty JSON object."}), 400
#     team = str(data.get("team", "")).strip()
#     if not team:
#         return jsonify({"success": False, "error": "The team field is required."}), 400
#     try:
#         prediction = _predict(team, data)
#     except ValueError as exc:
#         return jsonify({"success": False, "error": str(exc)}), 400
#     except RuntimeError as exc:
#         return jsonify({"success": False, "error": str(exc)}), 503
#     return jsonify({
#         "success": True,
#         "team": team,
#         "winning_probability": round(float(prediction["winning_probability"]), 6),
#         "winning_probability_percentage": prediction["winning_probability_percentage"],
#         "features_used": MODEL_FEATURES,
#         "historical_team_data_available": prediction["has_history"],
#     })


# @app.route("/api/predict-match", methods=["POST"])
# def api_predict_match():
#     data = request.get_json(silent=True)
#     if not isinstance(data, dict) or not isinstance(data.get("teams"), list):
#         return jsonify({"success": False, "error": "Send JSON with a teams array."}), 400
#     team_names = [str(team).strip() for team in data["teams"] if str(team).strip()]
#     if len(team_names) < 2 or len({team.casefold() for team in team_names}) != len(team_names):
#         return jsonify({"success": False, "error": "Provide at least two distinct team names."}), 400
#     if MODEL is None or PREPROCESSOR is None:
#         return jsonify({"success": False, "error": "The trained model is not available."}), 503
#     try:
#         from src.feature_engineering import batch_feature_snapshots
#         snapshots = batch_feature_snapshots(HISTORY, team_names)
#         input_data = []
#         for team in team_names:
#             features = snapshots.get(team, {})
#             input_data.append({f: float(features.get(f, 0.0)) for f in MODEL_FEATURES})
#         input_df = pd.DataFrame(input_data)[MODEL_FEATURES]
#         matrix = PREPROCESSOR.transform(input_df) if PREPROCESSOR is not None else input_df
#         probs = MODEL.predict_proba(matrix)
#         rows = []
#         for team, input_row, p_val in zip(team_names, input_data, probs):
#             has_history = bool(
#                 not HISTORY.empty
#                 and HISTORY["team_key"].eq(canonical_team_key(team)).any()
#             )
#             rows.append({
#                 "team": team,
#                 "raw_probability": float(np.clip(p_val[1], 0.0, 1.0)),
#                 "features": input_row,
#                 "history_available": has_history,
#             })
#         total = sum(row["raw_probability"] for row in rows)
#         for row in rows:
#             row["winning_probability"] = row["raw_probability"] / total if total > 0 else 1 / len(rows)
#             row["winning_probability_percentage"] = round(row["winning_probability"] * 100, 2)
#         rows.sort(key=lambda row: row["winning_probability"], reverse=True)
#         for rank, row in enumerate(rows, 1):
#             row["rank"] = rank
#         return jsonify({"success": True, "tournament": data.get("tournament", "BMSD"), "match": data.get("match"), "results": rows})
#     except (ValueError, NotFittedError) as exc:
#         return jsonify({"success": False, "error": str(exc)}), 400


# @app.route("/api/stats")
# def api_stats():
#     tournament = request.args.get("tournament", "All Tournaments")
#     frame = _filter_history(tournament)
#     return jsonify({"success": True, "tournament": tournament, "stats": _summary_stats(frame), "leaderboard": _team_summary(frame)[:10]})


# @app.route("/health")
# def health():
#     return jsonify({"status": "ok", "model_loaded": MODEL is not None, "historical_rows": int(len(HISTORY))})


# @app.errorhandler(404)
# def not_found(_error):
#     return render_template("error.html", code=404, message="That page could not be found."), 404


# @app.errorhandler(413)
# def upload_too_large(_error):
#     return render_template("error.html", code=413, message="The CSV is larger than the 5 MB upload limit."), 413


# @app.errorhandler(500)
# def server_error(_error):
#     return render_template("error.html", code=500, message="The request could not be completed. Check the dataset and model files."), 500


# if __name__ == "__main__":
#     _warmup_cache()
#     app.run(host="127.0.0.1", port=5055, debug=False)

# # Also warm cache when loaded by a WSGI server (e.g. gunicorn / waitress)
# _warmup_cache()



"""Flask application for the BGMI match win probability project."""

from __future__ import annotations

from pathlib import Path
import io
import json
import logging
import re
import time
import threading

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from sklearn.exceptions import NotFittedError

from src.data_preprocessing import (
    CANONICAL_COLUMNS,
    canonical_team_key,
    load_all_historical_data,
    normalize_match_data,
    validate_match_lobbies,
)
from src.feature_engineering import FEATURE_COLS, build_pre_match_features, feature_snapshot_for_team
from src.prediction import get_feature_importance, predict_single_team
from src.player_finish_model import PLAYER_FEATURES, build_player_prediction_frame
from src.bmsd_source import (
    BMSD_MATCHES_URL,
    BMSD_FORMAT_URL,
    BMSD_PROGRESSION_URL,
    fetch_bmsd_format,
    fetch_bmsd_progression,
    fetch_bmsd_player_stats,
    fetch_bmsd_source,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "model_results"
MODEL_PATH = MODEL_DIR / "bgmi_winning_probability_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessing_pipeline.joblib"
FEATURES_PATH = MODEL_DIR / "feature_names.joblib"
MVP_MODEL_PATH = MODEL_DIR / "bmsd_player_finishes_model.joblib"
MVP_FEATURES_PATH = MODEL_DIR / "bmsd_player_finish_feature_names.joblib"
MVP_METADATA_PATH = REPORT_DIR / "player_finish_model_metadata.json"
EAM_BMSD_URL = "https://esportsamaze.com/bgmi/tournaments/battlegrounds-mobile-india-showdown-2026/standings"
LIQUIPEDIA_BMSD_URL = "https://liquipedia.net/pubgmobile/Battlegrounds_Mobile_India_Showdown/2026"
FIXTURES_PATH = DATA_DIR / "raw" / "bmsd_fixtures.csv"
PROGRESSION_PATH = DATA_DIR / "raw" / "bmsd_progression.csv"
STAGE_FORMAT_PATH = DATA_DIR / "raw" / "bmsd_stage_format.csv"
PLAYER_STATS_PATH = DATA_DIR / "raw" / "bmsd_player_stats.csv"
PLAYER_MATCH_STATS_PATH = DATA_DIR / "raw" / "bmsd_player_match_stats.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgmi_app")

# ---------------------------------------------------------------------------
# Simple in-process TTL cache – avoids recomputing heavy route data on every
# navbar click. Data only changes on manual refresh or CSV upload.
# ---------------------------------------------------------------------------
_CACHE_TTL = 86400  # seconds (24 hours)
_route_cache: dict[str, tuple[float, object]] = {}
_CSV_FILE_CACHE: dict[tuple[str, float], pd.DataFrame] = {}


def _cache_get(key: str) -> object:
    if app.config.get("TESTING"):
        return None
    entry = _route_cache.get(key)
    if entry and (time.monotonic() - entry[0]) < _CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, value: object) -> None:
    _route_cache[key] = (time.monotonic(), value)


def _cache_clear() -> None:
    _route_cache.clear()
    _HISTORY_SLICE_CACHE.clear()
    _CSV_FILE_CACHE.clear()
    LIVE_PREDICTION_CACHE.clear()
    try:
        from src.feature_engineering import _ELO_CACHE
        _ELO_CACHE.clear()
    except Exception:
        pass


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


def _load_model_artifacts():
    try:
        if not all(path.exists() for path in (MODEL_PATH, PREPROCESSOR_PATH, FEATURES_PATH)):
            return None, None, FEATURE_COLS
        return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH), list(joblib.load(FEATURES_PATH))
    except Exception as exc:  # Do not prevent dashboard access if artifacts need retraining.
        logger.error("Could not load saved model artifacts: %s", exc)
        return None, None, FEATURE_COLS


MODEL, PREPROCESSOR, MODEL_FEATURES = _load_model_artifacts()
try:
    MVP_MODEL = joblib.load(MVP_MODEL_PATH) if MVP_MODEL_PATH.exists() else None
    MVP_MODEL_FEATURES = list(joblib.load(MVP_FEATURES_PATH)) if MVP_FEATURES_PATH.exists() else []
    MVP_METADATA = json.loads(MVP_METADATA_PATH.read_text(encoding="utf-8")) if MVP_METADATA_PATH.exists() else {}
except Exception as exc:
    logger.error("Could not load saved player finish artifacts: %s", exc)
    MVP_MODEL, MVP_MODEL_FEATURES, MVP_METADATA = None, [], {}


def _load_history() -> pd.DataFrame:
    try:
        return load_all_historical_data(DATA_DIR)
    except Exception as exc:
        logger.error("Could not load tournament data: %s", exc)
        return pd.DataFrame(columns=CANONICAL_COLUMNS)


HISTORY = _load_history()
LIVE_PREDICTION_CACHE: dict[tuple[int, float, str], dict[str, object]] = {}


_WARMUP_STARTED = False
_WARMUP_LOCK = threading.Lock()


def _warmup_cache() -> None:
    """Pre-compute expensive /live data in a background thread at startup."""
    global _WARMUP_STARTED
    with _WARMUP_LOCK:
        if _WARMUP_STARTED:
            return
        _WARMUP_STARTED = True

    def _run() -> None:
        try:
            from src.data_preprocessing import load_all_historical_data
            fixtures = _read_source_csv(FIXTURES_PATH)
            progression = _read_source_csv(PROGRESSION_PATH)
            stage_formats = _read_source_csv(STAGE_FORMAT_PATH)
            _cache_set("live_fixtures", fixtures)
            _cache_set("live_progression", progression)
            _cache_set("live_stage_formats", stage_formats)
            bmsd = HISTORY[HISTORY["tournament"].astype(str).str.casefold().eq("bmsd")] if not HISTORY.empty else HISTORY.__class__(columns=HISTORY.columns)
            board = _team_summary(bmsd)
            _cache_set("live_board", board)
            stage_sections = _bmsd_stage_sections(stage_formats)
            _cache_set("live_stage_sections", stage_sections)
            section_map = {_stage_token(str(s["stage"])): s for s in stage_sections}
            upcoming = _live_stage_lobbies(fixtures, stage_formats, stage_sections)
            for lobby in upcoming:
                section = section_map.get(_stage_token(lobby.get("stage")))
                if section:
                    lobby.setdefault("stage_total", section.get("total_matches"))
                    lobby.setdefault("completed_in_stage", section.get("completed_matches"))
            for lobby in upcoming:
                lobby["predictions"] = _lobby_predictions(lobby, cache=LIVE_PREDICTION_CACHE)
            _cache_set("live_upcoming", upcoming)
            if upcoming:
                first_lobby = upcoming[0]
                _cache_set(f"mvp_{first_lobby.get('id')}", _next_match_mvp_candidates(first_lobby))
            logger.info("Cache warmup complete.")
        except Exception as exc:
            logger.warning("Cache warmup failed (non-fatal): %s", exc)

    threading.Thread(target=_run, daemon=True).start()



def _allowed_tournaments() -> list[str]:
    cached = _cache_get("allowed_tournaments")
    if cached is not None:
        return cached  # type: ignore[return-value]
    present = sorted(HISTORY["tournament"].dropna().astype(str).unique().tolist()) if not HISTORY.empty else []
    result = ["All Tournaments", *present]
    _cache_set("allowed_tournaments", result)
    return result


def _filter_history(tournament: str = "All Tournaments") -> pd.DataFrame:
    if HISTORY.empty or tournament in ("", "All Tournaments", "All"):
        return HISTORY.copy()
    return HISTORY[HISTORY["tournament"].astype(str).str.casefold().eq(str(tournament).casefold())].copy()


def _safe_number(value: object, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.") from None
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be no more than {maximum:g}.")
    return number


def _prediction_input(
    team: str,
    supplied: dict[str, object] | None = None,
    history_frame: pd.DataFrame | None = None,
) -> tuple[dict[str, float], bool]:
    if not team or not team.strip():
        raise ValueError("Enter a team name.")
    source_history = HISTORY if history_frame is None else history_frame
    fields = feature_snapshot_for_team(source_history, team)
    has_history = bool(not source_history.empty and source_history["team_key"].eq(canonical_team_key(team)).any())
    aliases = {
        "avg_points": "prev_avg_points",
        "avg_kills": "prev_avg_kills",
        "win_rate": "prev_win_rate",
        "recent_points": "recent3_avg_points",
        "recent_kills": "recent3_avg_kills",
    }
    supplied = supplied or {}
    for key, value in supplied.items():
        destination = aliases.get(key, key)
        if destination not in MODEL_FEATURES:
            continue
        if value in (None, ""):
            continue
        minimum = 0.0
        maximum = 1.0 if destination in ("prev_win_rate", "recent3_win_rate") else None
        fields[destination] = _safe_number(value, destination, minimum=minimum, maximum=maximum)
    return {feature: float(fields.get(feature, 0.0)) for feature in MODEL_FEATURES}, has_history


def _predict(team: str, supplied: dict[str, object] | None = None) -> dict[str, object]:
    if MODEL is None or PREPROCESSOR is None:
        raise RuntimeError("The trained model is not available. Run the training script first.")
    features, has_history = _prediction_input(team, supplied)
    probability = predict_single_team(MODEL, features, MODEL_FEATURES, PREPROCESSOR)
    return {
        "team": team,
        "winning_probability": probability,
        "winning_probability_percentage": round(probability * 100, 2),
        "features": features,
        "has_history": has_history,
        "feature_importance": get_feature_importance(MODEL, MODEL_FEATURES)[:6],
    }


def _team_summary(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    result = frame.groupby("team_key", as_index=False).agg(
        team=("team", "last"),
        matches=("match_key", "nunique"),
        points=("points", "sum"),
        kills=("kills", "sum"),
        wins=("win", "sum"),
        avg_points=("points", "mean"),
        avg_kills=("kills", "mean"),
        win_rate=("win", "mean"),
    )
    result["wins"] = result["wins"].astype(int)
    return result.sort_values(["avg_points", "avg_kills"], ascending=False).to_dict("records")


def _summary_stats(frame: pd.DataFrame) -> dict[str, int | float]:
    if frame.empty:
        return {"matches": 0, "teams": 0, "rows": 0, "wins": 0, "points": 0, "kills": 0}
    return {
        "matches": int(frame["match_key"].nunique()),
        "teams": int(frame["team_key"].nunique()),
        "rows": int(len(frame)),
        "wins": int(frame["win"].sum()),
        "points": int(frame["points"].sum()),
        "kills": int(frame["kills"].sum()),
    }


def _read_source_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists() or not path.stat().st_size:
            return pd.DataFrame()
        mtime = path.stat().st_mtime
        key = (str(path.resolve()), mtime)
        cached = _CSV_FILE_CACHE.get(key)
        if cached is not None:
            return cached.copy()
        df = pd.read_csv(path)
        _CSV_FILE_CACHE[key] = df
        return df.copy()
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.warning("Unable to read BMSD source snapshot %s: %s", path.name, exc)
        return pd.DataFrame()


def _bmsd_group_rows(fixtures: pd.DataFrame) -> list[dict[str, object]]:
    if fixtures.empty or "team" not in fixtures:
        return []
    frame = fixtures.copy()
    frame["team"] = frame["team"].fillna("").astype(str).str.strip()
    frame = frame[frame["team"].ne("")]
    if frame.empty:
        return []
    rows = []
    for (stage, group), subset in frame.groupby(["stage", "group"], dropna=False, sort=False):
        teams = sorted(subset["team"].dropna().astype(str).unique().tolist(), key=str.casefold)
        rows.append({
            "stage": str(stage or "Unknown stage"),
            "group": str(group) if pd.notna(group) and str(group).strip() else "Group not specified",
            "teams": teams,
            "matches": int(subset["source_match_id"].nunique()) if "source_match_id" in subset else int(subset["match"].nunique()),
            "statuses": sorted(subset["status"].dropna().astype(str).str.upper().unique().tolist()) if "status" in subset else [],
        })
    return rows


def _upcoming_lobbies(fixtures: pd.DataFrame) -> list[dict[str, object]]:
    if fixtures.empty or "status" not in fixtures:
        return []
    scheduled_statuses = {"SCHEDULED", "UPCOMING", "NOT_STARTED", "PENDING"}
    future = fixtures[fixtures["status"].fillna("").astype(str).str.upper().isin(scheduled_statuses)].copy()
    if future.empty:
        return []
    identity = "source_match_id" if "source_match_id" in future and future["source_match_id"].fillna("").astype(str).str.strip().ne("").any() else "overall_match"
    lobbies = []
    for lobby_id, subset in future.groupby(identity, dropna=False, sort=False):
        first = subset.iloc[0]
        teams = sorted(set(subset["team"].fillna("").astype(str).str.strip()).difference({""}), key=str.casefold)
        if len(teams) < 2:
            continue
        lobbies.append({
            "id": str(lobby_id),
            "overall_match": first.get("overall_match", ""),
            "match": first.get("match", ""),
            "stage": str(first.get("stage", "Unknown stage")),
            "group": "" if str(first.get("stage", "")).strip().casefold() == "grand finals" else str(first.get("group", "")),
            "map": str(first.get("map", "")),
            "scheduled_at": str(first.get("scheduled_at", "")),
            "teams": teams,
        })
    return sorted(lobbies, key=lambda row: (row["scheduled_at"] or "9999", str(row["overall_match"])))


def _stage_token(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _group_token(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip()).casefold()
    return re.sub(r"^group\s+", "", text)


def _json_value(value: object, fallback: object) -> object:
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def _rank_range(threshold: str, team_count: int) -> tuple[int, int] | None:
    value = re.sub(r"\s+", " ", str(threshold or "").strip()).casefold()
    top = re.search(r"top\s+(\d+)", value)
    if top:
        return 1, min(team_count, int(top.group(1)))
    bottom = re.search(r"bottom\s+(\d+)", value)
    if bottom:
        size = min(team_count, int(bottom.group(1)))
        return team_count - size + 1, team_count
    ranks = [int(number) for number in re.findall(r"\d+", value)]
    if len(ranks) >= 2:
        return max(1, min(ranks)), min(team_count, max(ranks))
    return None


def _bmsd_standings(bmsd: pd.DataFrame, formats: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Rank completed stage groups using the published BMSD tiebreak order."""
    empty = pd.DataFrame(columns=["stage", "group", "team", "team_key", "rank", "points", "wins"])
    if bmsd.empty:
        return empty, {}
    frame = bmsd.copy()
    for column in ("points", "kills", "win", "placement_points", "rank", "match_order"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["_stage_key"] = frame["stage"].map(_stage_token)
    frame["_group_key"] = frame["group"].map(_group_token)
    frame["_order"] = frame["match_order"].fillna(0)
    complete: dict[str, bool] = {}
    totals: dict[str, int] = {}
    if not formats.empty:
        for _, row in formats.iterrows():
            total = pd.to_numeric(row.get("total_matches"), errors="coerce")
            if pd.notna(total):
                totals[_stage_token(row.get("stage"))] = int(total)
    for key, subset in frame.groupby("_stage_key", sort=False):
        complete[key] = bool(totals.get(key, 0) > 0 and subset["match_order"].nunique() >= totals[key])

    rows = []
    grouped = frame.groupby(["_stage_key", "_group_key", "team_key"], sort=False, dropna=False)
    for (stage_key, group_key, team_key), subset in grouped:
        last = subset.sort_values(["_order", "date"], kind="stable", na_position="first").iloc[-1]
        rows.append({
            "stage_key": stage_key,
            "group_key": group_key,
            "stage": str(last["stage"]),
            "group": str(last["group"]),
            "team": str(last["team"]),
            "team_key": str(team_key),
            "points": float(subset["points"].sum()),
            "wins": float(subset["win"].sum()),
            "placement_points": float(subset["placement_points"].fillna(0).sum()),
            "kills": float(subset["kills"].sum()),
            "last_rank": float(last["rank"]) if pd.notna(last["rank"]) else float("inf"),
        })
    standings = pd.DataFrame(rows)
    if standings.empty:
        return empty, complete
    standings = standings.sort_values(
        ["stage_key", "group_key", "points", "wins", "placement_points", "kills", "last_rank", "team_key"],
        ascending=[True, True, False, False, False, False, True, True],
        kind="stable",
    )
    standings["rank"] = standings.groupby(["stage_key", "group_key"], sort=False).cumcount() + 1
    return standings, complete


def _bmsd_stage_sections(
    stage_formats: pd.DataFrame,
    history_frame: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    """Resolve published bracket slots from completed match standings and rules."""
    bmsd = _filter_history("BMSD") if history_frame is None else history_frame[
        history_frame["tournament"].astype(str).str.casefold().eq("bmsd")
    ].copy()
    if stage_formats.empty:
        return []
    formats = stage_formats.copy()
    formats["_stage_key"] = formats["stage"].map(_stage_token)
    formats["sequence"] = pd.to_numeric(formats.get("sequence"), errors="coerce").fillna(0)
    formats = formats.sort_values("sequence", kind="stable")
    standings, stage_complete = _bmsd_standings(bmsd, formats)
    stage_lookup = {row["_stage_key"]: row for _, row in formats.iterrows()}
    result_by_stage: dict[str, pd.DataFrame] = {}
    for key, subset in bmsd.assign(_stage_key=bmsd["stage"].map(_stage_token)).groupby("_stage_key", sort=False):
        result_by_stage[key] = subset
    names: dict[str, dict[str, str]] = {}
    grouped_names: dict[str, dict[str, dict[str, str]]] = {}
    pending_sources: dict[str, list[str]] = {}
    for stage_key in stage_lookup:
        names[stage_key] = {}
        grouped_names[stage_key] = {}
        pending_sources[stage_key] = []

    display_by_key: dict[str, str] = {}
    for _, row in bmsd.sort_values("match_order", kind="stable").iterrows():
        display_by_key[str(row["team_key"])] = str(row["team"])

    def add_team(stage_key: str, group_name: str, team_name: str) -> None:
        team_key = canonical_team_key(team_name)
        if not team_key:
            return
        display = display_by_key.get(team_key, team_name)
        names.setdefault(stage_key, {})[team_key] = display
        group_map = grouped_names.setdefault(stage_key, {}).setdefault(group_name or "Teams", {})
        group_map[team_key] = display

    # Resolve each stage's published team slots. Source ranks are only treated
    # as confirmed once the source stage has completed all published matches.
    for _, stage_row in formats.iterrows():
        target_key = stage_row["_stage_key"]
        if stage_complete.get(target_key, False):
            continue
        groups = _json_value(stage_row.get("groups_json", "{}"), {})
        if isinstance(groups, dict):
            for group_name, entries in groups.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    source = entry.get("source")
                    if isinstance(source, dict):
                        source_key = _stage_token(source.get("stage"))
                        source_group = _group_token(source.get("group"))
                        rank = pd.to_numeric(source.get("rank"), errors="coerce")
                        source_table = standings[
                            standings["stage_key"].eq(source_key)
                            & standings["group_key"].eq(source_group)
                        ] if not standings.empty else pd.DataFrame()
                        match = source_table[source_table["rank"].eq(rank)] if pd.notna(rank) else pd.DataFrame()
                        if stage_complete.get(source_key, False) and not match.empty:
                            add_team(target_key, str(group_name), str(match.iloc[0]["team"]))
                    else:
                        team_name = str(entry.get("teamName", entry.get("name", "")) or "").strip()
                        if team_name and not re.match(r"^Group\s+.+\s+#\d+$", team_name, flags=re.IGNORECASE):
                            add_team(target_key, str(group_name), team_name)

    # Completed scorecards are the strongest source for a stage's actual teams.
    for stage_key, subset in result_by_stage.items():
        if stage_key not in stage_lookup:
            continue
        groups = _json_value(stage_lookup[stage_key].get("groups_json", "{}"), {})
        format_group_names = list(groups) if isinstance(groups, dict) else []
        one_lobby = len(format_group_names) == 1
        for _, row in subset.drop_duplicates(["team_key", "group"]).iterrows():
            group_name = format_group_names[0] if one_lobby else (str(row.get("group", "")).strip() or "Teams")
            add_team(stage_key, group_name, str(row["team"]))

    # Apply every completed source-stage advancement rule, including teams
    # that are not present in the static stage-group list (for example the
    # top eight from the current Upper Bracket entering Semi Finals).
    for _, source_row in formats.iterrows():
        source_key = source_row["_stage_key"]
        rules = _json_value(source_row.get("rules_json", "[]"), [])
        if not isinstance(rules, list):
            continue
        source_completed = stage_complete.get(source_key, False)
        source_name = str(source_row.get("stage", ""))
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            destination = str(rule.get("destination", "") or "").strip()
            target_key = _stage_token(destination)
            if target_key not in stage_lookup:
                continue
            if stage_complete.get(target_key, False):
                continue
            target_group = str(rule.get("groupName", "") or "").strip() or f"From {source_name}"
            label = str(rule.get("thresholdRank", "") or "").strip()
            if not source_completed:
                if label:
                    pending_sources[target_key].append(f"{label} from {source_name}")
                continue
            group_filter = _group_token(rule.get("groupName", ""))
            source_table = standings[standings["stage_key"].eq(source_key)] if not standings.empty else pd.DataFrame()
            if group_filter:
                source_table = source_table[source_table["group_key"].eq(group_filter)]
            if source_table.empty:
                continue
            rank_range = _rank_range(label, int(source_table.groupby("group_key")["team_key"].nunique().max()))
            if rank_range is None:
                continue
            qualified = source_table[source_table["rank"].between(*rank_range)]
            for _, team_row in qualified.iterrows():
                if canonical_team_key(str(team_row["team"])) not in names.get(target_key, {}):
                    add_team(target_key, target_group, str(team_row["team"]))

    sections = []
    for _, row in formats.iterrows():
        stage_key = row["_stage_key"]
        total_matches = pd.to_numeric(row.get("total_matches"), errors="coerce")
        total_matches = int(total_matches) if pd.notna(total_matches) else 0
        match_rows = result_by_stage.get(stage_key, pd.DataFrame())
        completed_matches = int(match_rows["match_order"].nunique()) if not match_rows.empty else 0
        expected_teams = pd.to_numeric(row.get("teams_count"), errors="coerce")
        expected_teams = int(expected_teams) if pd.notna(expected_teams) else 0
        teams = names.get(stage_key, {})
        groups = []
        for group_name, group_teams in grouped_names.get(stage_key, {}).items():
            groups.append({"group": group_name, "teams": sorted(group_teams.values(), key=str.casefold)})
        known_count = len(teams)
        pending_count = max(0, expected_teams - known_count)
        if pending_count:
            groups.append({"group": "Qualification pending", "teams": [], "pending_count": pending_count})
        pct = 100.0 * completed_matches / total_matches if total_matches else 0.0
        status = "Complete" if total_matches and completed_matches >= total_matches else ("In progress" if completed_matches else "Not started")
        sections.append({
            "stage": str(row.get("stage", "Unknown stage")),
            "sequence": int(row.get("sequence", 0)),
            "groups": groups,
            "teams": sorted(teams.values(), key=str.casefold),
            "known_teams": known_count,
            "expected_teams": expected_teams,
            "pending_count": pending_count,
            "pending_sources": list(dict.fromkeys(pending_sources.get(stage_key, []))),
            "completed_matches": completed_matches,
            "total_matches": total_matches,
            "completion_pct": pct,
            "status": status,
            "division": str(row.get("groups_division", "")),
            "_stage_key": stage_key,
        })
    return sections


def _live_stage_lobbies(fixtures: pd.DataFrame, stage_formats: pd.DataFrame, sections: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build predictions for remaining games in a fully known 16-team lobby stage."""
    lobbies = _upcoming_lobbies(fixtures)
    if stage_formats.empty or not sections:
        return lobbies
    formats = stage_formats.copy()
    formats["_stage_key"] = formats["stage"].map(_stage_token)
    formats["sequence"] = pd.to_numeric(formats.get("sequence"), errors="coerce").fillna(0)
    section_map = {str(section["_stage_key"]): section for section in sections}
    bmsd = _filter_history("BMSD")
    lobby_formats = {"single lobby", "smash rules", "round robin"}
    active = []
    for _, row in formats.iterrows():
        section = section_map.get(str(row["_stage_key"]), {})
        division = str(row.get("groups_division", "")).strip().casefold()
        if (
            division in lobby_formats
            and int(section.get("known_teams", 0)) == int(section.get("expected_teams", 0))
            and int(section.get("completed_matches", 0)) > 0
            and int(section.get("completed_matches", 0)) < int(section.get("total_matches", 0))
        ):
            active.append((int(row["sequence"]), row, section))
    candidate = max(active, key=lambda item: item[0]) if active else None
    if candidate is None:
        completed = [int(section.get("sequence", 0)) for section in sections if int(section.get("completed_matches", 0)) > 0]
        last_sequence = max(completed, default=0)
        future = [
            (int(row["sequence"]), row, section_map.get(str(row["_stage_key"]), {}))
            for _, row in formats.iterrows()
            if int(row["sequence"]) > last_sequence
            and str(row.get("groups_division", "")).strip().casefold() in lobby_formats
            and int(section_map.get(str(row["_stage_key"]), {}).get("known_teams", 0)) == int(section_map.get(str(row["_stage_key"]), {}).get("expected_teams", 0))
        ]
        candidate = min(future, key=lambda item: item[0]) if future else None
    if candidate is None:
        return lobbies

    _, stage_row, section = candidate
    division = str(stage_row.get("groups_division", "")).strip().casefold()
    if division not in lobby_formats:
        return lobbies

    stage_key = str(stage_row["_stage_key"])
    stage_completed = bmsd[bmsd["stage"].map(_stage_token).eq(stage_key)] if not bmsd.empty else pd.DataFrame()
    local_match = pd.to_numeric(stage_completed.get("match_id"), errors="coerce") if not stage_completed.empty and "match_id" in stage_completed else pd.Series(dtype=float)
    last_local = int(local_match.max()) if not local_match.empty and local_match.notna().any() else 0
    if last_local == 0:
        last_local = 0
    parsed_total = pd.to_numeric(stage_row.get("total_matches"), errors="coerce")
    total_matches = int(parsed_total) if pd.notna(parsed_total) else 0
    current_global = int(pd.to_numeric(bmsd.get("match_order"), errors="coerce").max()) if not bmsd.empty else 0
    existing_ids = {(str(lobby.get("stage", "")).casefold(), str(lobby.get("match", ""))) for lobby in lobbies}

    if division == "round robin":
        def clean_group_label(value: object) -> str:
            """Normalize group labels, including legacy combined labels."""
            label = re.sub(r"\s+", " ", str(value or "").strip())
            if not label:
                return ""

            explicit_groups = re.findall(
                r"\bGroup\s+([A-Za-z0-9]+)\b",
                label,
                flags=re.IGNORECASE,
            )
            if explicit_groups:
                return f"Group {explicit_groups[-1].upper()}"

            parts = re.split(r"\s+vs\s+", label, flags=re.IGNORECASE)
            token = parts[-1].strip() if parts else label
            token = re.sub(r"^group\s+", "", token, flags=re.IGNORECASE).strip()
            return f"Group {token.upper()}" if token else ""

        # Only use the groups declared for this round-robin stage. Advancement
        # rules can add helper buckets such as "From Week 3"; those are not
        # actual match groups and must not create Group C vs Group C pairings.
        declared = _json_value(stage_row.get("groups_json", "{}"), {})
        declared_labels = set()
        if isinstance(declared, dict):
            declared_labels = {
                clean_group_label(label)
                for label in declared.keys()
                if clean_group_label(label)
            }

        # Merge equivalent labels (for example "C" and "Group C") into one
        # real group, and deduplicate teams so a group cannot pair with itself.
        merged_groups: dict[str, dict[str, object]] = {}
        for source_group in section.get("groups", []):
            if not source_group.get("teams"):
                continue
            label = clean_group_label(source_group.get("group"))
            if not label or (declared_labels and label not in declared_labels):
                continue
            bucket = merged_groups.setdefault(label, {"group": label, "teams": []})
            bucket["teams"].extend(source_group.get("teams", []))

        groups = []
        for label, bucket in sorted(merged_groups.items(), key=lambda item: item[0].casefold()):
            unique_teams = sorted(
                {str(team).strip() for team in bucket["teams"] if str(team).strip()},
                key=str.casefold,
            )
            if unique_teams:
                groups.append({"group": label, "teams": unique_teams})

        if len(groups) >= 2:
            pairings = [
                (groups[i], groups[j])
                for i in range(len(groups))
                for j in range(i + 1, len(groups))
                if groups[i]["group"] != groups[j]["group"]
            ]
            for number in range(last_local + 1, total_matches + 1):
                if (str(stage_row.get("stage", "")).casefold(), str(number)) in existing_ids:
                    continue
                if not pairings:
                    break

                g1, g2 = pairings[(number - 1) % len(pairings)]
                match_teams = sorted(
                    set(g1["teams"] + g2["teams"]),
                    key=str.casefold,
                )
                lobbies.append({
                    "id": f"inferred-{stage_key.replace(' ', '-')}-{number}",
                    "overall_match": current_global + (number - last_local),
                    "match": number,
                    "stage": str(stage_row.get("stage", "")),
                    "group": f"{g1['group']} vs {g2['group']}",
                    "map": "",
                    "scheduled_at": "",
                    "teams": match_teams,
                    "inferred": True,
                    "stage_total": total_matches,
                    "completed_in_stage": last_local,
                    "as_of_match": current_global,
                })
    else:
        teams = list(section.get("teams", []))
        expected = int(section.get("expected_teams", 0))
        if expected < 2 or len(teams) != expected:
            return lobbies
        for number in range(last_local + 1, total_matches + 1):
            if (str(stage_row.get("stage", "")).casefold(), str(number)) in existing_ids:
                continue
            lobbies.append({
                "id": f"inferred-{stage_key.replace(' ', '-')}-{number}",
                "overall_match": current_global + (number - last_local),
                "match": number,
                "stage": str(stage_row.get("stage", "")),
                "group": "" if division == "smash rules" else "Single Lobby",
                "map": "",
                "scheduled_at": "",
                "teams": teams,
                "inferred": True,
                "stage_total": total_matches,
                "completed_in_stage": last_local,
                "as_of_match": current_global,
            })
    lobbies.sort(key=lambda item: (pd.to_numeric(item.get("overall_match"), errors="coerce") if pd.notna(pd.to_numeric(item.get("overall_match"), errors="coerce")) else float("inf"), str(item.get("stage", ""))))
    return lobbies


def _grand_finals_smash_state(next_match: object) -> dict[str, object]:
    """Describe the Grand Finals Match Point target and eligibility before a match."""
    try:
        match_number = int(next_match)
    except (TypeError, ValueError):
        return {}

    bmsd = _filter_history("BMSD")
    finals = bmsd[bmsd["stage"].map(_stage_token).eq("grand finals")].copy()
    if finals.empty:
        return {
            "match": match_number,
            "target_score": None,
            "leader_after_match_12": None,
            "eligible_teams": [],
            "champion": None,
        }
    finals["match_id"] = pd.to_numeric(finals["match_id"], errors="coerce")
    finals["points"] = pd.to_numeric(finals["points"], errors="coerce").fillna(0)
    finals["win"] = pd.to_numeric(finals["win"], errors="coerce").fillna(0)
    completed = set(finals["match_id"].dropna().astype(int).tolist())
    first_twelve = finals[finals["match_id"].between(1, 12)]
    if not set(range(1, 13)).issubset(completed):
        return {
            "match": match_number,
            "target_score": None,
            "leader_after_match_12": None,
            "eligible_teams": [],
            "champion": None,
        }

    totals_after_12 = first_twelve.groupby("team")["points"].sum()
    leader_score = float(totals_after_12.max())
    target_score = leader_score + 10.0

    champion = None
    championship_match = None
    for game in range(13, match_number):
        before_game = finals[finals["match_id"].lt(game)].groupby("team")["points"].sum()
        eligible_before_game = set(before_game[before_game.ge(target_score)].index.astype(str))
        winners = set(finals.loc[finals["match_id"].eq(game) & finals["win"].eq(1), "team"].dropna().astype(str))
        decided = sorted(eligible_before_game & winners, key=str.casefold)
        if decided:
            champion = decided[0]
            championship_match = game
            break

    prior_totals = finals[finals["match_id"].lt(match_number)].groupby("team")["points"].sum()
    eligible_teams = sorted(
        prior_totals[prior_totals.ge(target_score)].index.astype(str).tolist(),
        key=str.casefold,
    )
    return {
        "match": match_number,
        "smash_phase": match_number >= 13,
        "target_score": target_score,
        "leader_after_match_12": leader_score,
        "eligible_teams": eligible_teams,
        "champion": champion,
        "championship_match": championship_match,
    }


_HISTORY_SLICE_CACHE: dict[float, pd.DataFrame] = {}


def _history_before_bmsd_match(overall_match: object) -> pd.DataFrame:
    try:
        next_order = float(overall_match)
    except (TypeError, ValueError):
        return HISTORY.copy()
    cached = _HISTORY_SLICE_CACHE.get(next_order)
    if cached is not None:
        return cached
    other_events = HISTORY[~HISTORY["tournament"].astype(str).str.casefold().eq("bmsd")]
    earlier_bmsd = HISTORY[
        HISTORY["tournament"].astype(str).str.casefold().eq("bmsd")
        & pd.to_numeric(HISTORY["match_order"], errors="coerce").lt(next_order)
    ]
    result = pd.concat([other_events, earlier_bmsd], ignore_index=True)
    _HISTORY_SLICE_CACHE[next_order] = result
    return result


def _lobby_predictions(lobby: dict[str, object], cache: dict | None = None) -> list[dict[str, object]]:
    if MODEL is None or PREPROCESSOR is None:
        return []
    history_before = _history_before_bmsd_match(lobby.get("overall_match"))
    history_order = pd.to_numeric(history_before.get("match_order"), errors="coerce")
    snapshot_order = float(history_order.max()) if not history_order.empty and history_order.notna().any() else -1.0
    cache = LIVE_PREDICTION_CACHE if cache is None else cache

    raw_teams = lobby.get("teams", [])
    teams = [str(t).strip() for t in raw_teams if str(t).strip()]
    if not teams:
        return []

    uncached_teams: list[str] = []
    uncached_keys: list[tuple[int, float, str]] = []
    for team in teams:
        cache_key = (id(HISTORY), snapshot_order, canonical_team_key(team))
        if cache_key not in cache:
            uncached_teams.append(team)
            uncached_keys.append(cache_key)

    if uncached_teams:
        from src.feature_engineering import batch_feature_snapshots
        snapshots = batch_feature_snapshots(history_before, uncached_teams)
        feature_rows: list[tuple[str, tuple[int, float, str], dict[str, float], bool]] = []
        for team, ck in zip(uncached_teams, uncached_keys):
            fields = snapshots.get(team, {})
            has_history = bool(
                not history_before.empty
                and history_before["team_key"].eq(canonical_team_key(team)).any()
            )
            feature_rows.append((team, ck, fields, has_history))

        input_data = [
            {f: float(fields.get(f, 0.0)) for f in MODEL_FEATURES}
            for _, _, fields, _ in feature_rows
        ]
        if input_data:
            input_df = pd.DataFrame(input_data)[MODEL_FEATURES]
            matrix = PREPROCESSOR.transform(input_df) if PREPROCESSOR is not None else input_df
            probs = MODEL.predict_proba(matrix)
            for (_, ck, fields, has_history), p_val in zip(feature_rows, probs):
                raw_p = float(np.clip(p_val[1], 0.0, 1.0))
                cache[ck] = {
                    "raw_probability": raw_p,
                    "matches": fields.get("matches_before", 0.0),
                    "has_history": has_history,
                }

    rows = []
    for team in teams:
        cache_key = (id(HISTORY), snapshot_order, canonical_team_key(team))
        cached = cache.get(cache_key)
        if cached:
            rows.append({
                "team": team,
                "raw_probability": float(cached["raw_probability"]),
                "matches": float(cached["matches"]),
                "has_history": bool(cached["has_history"]),
            })
    total = sum(row["raw_probability"] for row in rows)
    for row in rows:
        row["probability"] = 100.0 * row["raw_probability"] / total if total > 0 else 100.0 / max(1, len(rows))
    return sorted(rows, key=lambda row: row["probability"], reverse=True)


def _next_match_mvp_candidates(lobby: dict[str, object] | None) -> list[dict[str, object]]:
    """Rank known lobby players by leakage-safe estimated match finishes."""
    if not lobby or not lobby.get("teams"):
        return []
    players = _read_source_csv(PLAYER_MATCH_STATS_PATH)
    fixtures = _read_source_csv(DATA_DIR / "raw" / "bmsd_fixtures.csv")
    if players.empty or "source_match_id" not in players or "elims" not in players:
        return []
    match_order = pd.to_numeric(lobby.get("overall_match"), errors="coerce")
    snapshot = build_player_prediction_frame(
        players,
        fixtures,
        [str(team) for team in lobby["teams"]],
        float(match_order) if pd.notna(match_order) else None,
    )
    if snapshot.empty:
        return []
    if MVP_MODEL is not None and MVP_METADATA.get("selected_method") == "Random Forest regressor":
        columns = MVP_MODEL_FEATURES or PLAYER_FEATURES
        snapshot["predicted_finishes"] = np.clip(MVP_MODEL.predict(snapshot[columns].astype(float)), 0.0, None)
    else:
        snapshot["predicted_finishes"] = 0.5 * snapshot["prev_avg_finishes"] + 0.5 * snapshot["recent3_avg_finishes"]
    rows = []
    for _, row in snapshot.iterrows():
        rows.append({
            "player": str(row["player"]),
            "team": str(row["team"]),
            "matches": int(row["matches_before"]),
            "avg_finishes": float(row["prev_avg_finishes"]),
            "recent3_finishes": float(row["recent3_avg_finishes"]),
            "predicted_finishes": float(row["predicted_finishes"]),
        })
    return sorted(rows, key=lambda row: (row["predicted_finishes"], row["recent3_finishes"], row["avg_finishes"]), reverse=True)[:12]


def _grand_finals_scoreboard(teams: list[str]) -> list[dict[str, object]]:
    """Build a score table for the 16 confirmed finalists, including zero-score rows before play."""
    roster = sorted({str(team).strip() for team in teams if str(team).strip()}, key=str.casefold)
    if len(roster) != 16:
        return []

    bmsd = _filter_history("BMSD")
    finals = bmsd[bmsd["stage"].map(_stage_token).eq("grand finals")].copy()
    stats_by_team: dict[str, dict[str, object]] = {}
    if not finals.empty:
        for column in ("points", "kills", "win", "placement_points", "rank"):
            finals[column] = pd.to_numeric(finals.get(column), errors="coerce")
        finals["team_key"] = finals["team"].map(canonical_team_key)
        totals = finals.groupby("team_key", sort=False).agg(
            matches=("match_id", "nunique"),
            points=("points", "sum"),
            finishes=("kills", "sum"),
            wwcds=("win", "sum"),
            placement_points=("placement_points", "sum"),
            last_rank=("rank", "last"),
        )
        stats_by_team = totals.to_dict("index")

    rows = []
    for team in roster:
        values = stats_by_team.get(canonical_team_key(team), {})
        rows.append({
            "team": team,
            "matches": int(values.get("matches", 0) or 0),
            "points": float(values.get("points", 0) or 0),
            "finishes": float(values.get("finishes", 0) or 0),
            "wwcds": int(values.get("wwcds", 0) or 0),
            "placement_points": float(values.get("placement_points", 0) or 0),
            "last_rank": float(values.get("last_rank", float("inf")) or float("inf")),
        })
    return sorted(
        rows,
        key=lambda row: (row["points"], row["wwcds"], row["placement_points"], row["finishes"], -row["last_rank"]),
        reverse=True,
    )


def _mvp_watchlist(stage: str = "All Stages") -> tuple[list[dict[str, object]], list[str]]:
    players = _read_source_csv(PLAYER_STATS_PATH)
    match_stats = _read_source_csv(PLAYER_MATCH_STATS_PATH)
    if stage not in ("", "All Stages") and not match_stats.empty:
        selected = match_stats[match_stats["stage"].fillna("").astype(str).eq(stage)].copy()
        if selected.empty:
            return [], sorted(match_stats["stage"].dropna().astype(str).unique().tolist())
        selected["elims"] = pd.to_numeric(selected["elims"], errors="coerce").fillna(0.0)
        board = selected.groupby(["player_id", "player", "team"], as_index=False).agg(
            matches_played=("source_match_id", "nunique"),
            total_elims=("elims", "sum"),
            avg_elims=("elims", "mean"),
            max_elims=("elims", "max"),
            five_plus_elims_matches=("elims", lambda values: int(values.ge(5).sum())),
        )
    elif not players.empty:
        board = players.rename(columns={
            "player": "player", "team": "team", "matches_played": "matches_played",
            "total_elims": "total_elims", "avg_elims": "avg_elims", "max_elims": "max_elims",
            "five_plus_elims_matches": "five_plus_elims_matches", "reported_mvp_awards": "reported_mvp_awards",
        }).copy()
    else:
        return [], sorted(match_stats["stage"].dropna().astype(str).unique().tolist()) if not match_stats.empty else []

    for column in ("matches_played", "total_elims", "avg_elims", "max_elims", "five_plus_elims_matches"):
        if column in board:
            board[column] = pd.to_numeric(board[column], errors="coerce").fillna(0.0)
    board = board.sort_values(["total_elims", "avg_elims", "matches_played"], ascending=False, kind="stable")
    stages = sorted(match_stats["stage"].dropna().astype(str).unique().tolist()) if not match_stats.empty else []
    return board.head(30).to_dict("records"), stages


@app.route("/")
def home():
    per_tournament = []
    for name in _allowed_tournaments()[1:]:
        subset = _filter_history(name)
        stats = _summary_stats(subset)
        per_tournament.append({"name": name, **stats})
    return render_template(
        "index.html",
        stats=_summary_stats(HISTORY),
        per_tournament=per_tournament,
        model_available=MODEL is not None,
        bmsd_matches=_summary_stats(_filter_history("BMSD"))["matches"],
    )


@app.route("/dashboard")
def dashboard():
    tournament = request.args.get("tournament", "All Tournaments")
    frame = _filter_history(tournament)
    cache_key = f"dashboard_{tournament}"
    cached = _cache_get(cache_key)
    if cached is not None:
        board, latest_matches = cached
    else:
        board = _team_summary(frame)
        latest_matches = []
        if not frame.empty:
            matches = frame.drop_duplicates("match_key").sort_values(["year", "date", "match_order"], na_position="first")
            latest_keys = set(matches.tail(6)["match_key"])
            latest_matches = frame[frame["match_key"].isin(latest_keys)].groupby("match_key", as_index=False).agg(
                tournament=("tournament", "first"),
                match_id=("match_id", "first"),
                stage=("stage", "first"),
                date=("date", "first"),
                teams=("team", "nunique"),
                winner=("team", lambda values: values[frame.loc[values.index, "win"].eq(1)].iloc[0] if frame.loc[values.index, "win"].eq(1).any() else "Unknown"),
                total_points=("points", "sum"),
            ).to_dict("records")
        _cache_set(cache_key, (board, latest_matches))
    return render_template(
        "dashboard.html",
        tournament=tournament,
        tournaments=_allowed_tournaments(),
        stats=_summary_stats(frame),
        leaderboard=board[:12],
        latest_matches=latest_matches,
        chart_teams=board[:10],
    )


@app.route("/leaderboard")
def leaderboard():
    tournament = request.args.get("tournament", "All Tournaments")
    return render_template(
        "leaderboard.html",
        tournament=tournament,
        tournaments=_allowed_tournaments(),
        leaderboard=_team_summary(_filter_history(tournament)),
    )


@app.route("/history")
def history():
    tournament = request.args.get("tournament", "All Tournaments")
    frame = _filter_history(tournament)
    keys = []
    match_data: list[dict[str, object]] = []
    selected = request.args.get("match", "")
    if not frame.empty:
        metadata = frame.drop_duplicates("match_key").sort_values(["year", "date", "match_order"], na_position="first")
        keys = metadata[["match_key", "tournament", "stage", "match_id"]].to_dict("records")
        if selected:
            match_data = frame[frame["match_key"].astype(str).eq(selected)].sort_values("points", ascending=False).to_dict("records")
    return render_template("history.html", tournament=tournament, tournaments=_allowed_tournaments(), matches=keys, selected=selected, match_data=match_data)


@app.route("/team-analytics")
def team_analytics():
    tournament = request.args.get("tournament", "All Tournaments")
    board = _team_summary(_filter_history(tournament))
    selected_team = request.args.get("team", board[0]["team"] if board else "")
    team_rows = []
    if selected_team:
        selected_frame = _filter_history(tournament)
        team_rows = selected_frame[selected_frame["team_key"].eq(canonical_team_key(selected_team))].sort_values(["year", "date", "match_order"], na_position="first").to_dict("records")
    return render_template("team_analytics.html", tournament=tournament, tournaments=_allowed_tournaments(), teams=[row["team"] for row in board], selected_team=selected_team, team_rows=team_rows)


@app.route("/compare", methods=["GET", "POST"])
def compare():
    tournaments = _allowed_tournaments()
    tournament_a = request.values.get("tournament_a") or request.values.get("tournament") or "All Tournaments"
    tournament_b = request.values.get("tournament_b") or request.values.get("tournament") or "All Tournaments"
    tournament_teams = {
        t: [row["team"] for row in _team_summary(_filter_history(t))]
        for t in tournaments
    }
    teams_a = tournament_teams.get(tournament_a, [])
    teams_b = tournament_teams.get(tournament_b, [])
    team_a = request.values.get("team_a") or (teams_a[0] if teams_a else "")
    team_b = request.values.get("team_b") or (teams_b[1] if len(teams_b) > 1 else (teams_b[0] if teams_b else ""))
    summary_a = {row["team"]: row for row in _team_summary(_filter_history(tournament_a))}
    summary_b = {row["team"]: row for row in _team_summary(_filter_history(tournament_b))}
    a = summary_a.get(team_a)
    b = summary_b.get(team_b)
    comparison = None
    if a and b:
        a_copy = dict(a)
        b_copy = dict(b)
        a_copy["tournament"] = tournament_a
        b_copy["tournament"] = tournament_b
        try:
            a_copy["model_probability"] = _predict(team_a)["winning_probability_percentage"] if MODEL is not None else None
        except (ValueError, RuntimeError):
            a_copy["model_probability"] = None
        try:
            b_copy["model_probability"] = _predict(team_b)["winning_probability_percentage"] if MODEL is not None else None
        except (ValueError, RuntimeError):
            b_copy["model_probability"] = None
        comparison = {"team_a": a_copy, "team_b": b_copy}
    return render_template(
        "compare.html",
        tournament=tournament_a,
        tournament_a=tournament_a,
        tournament_b=tournament_b,
        tournaments=tournaments,
        teams=teams_a,
        teams_a=teams_a,
        teams_b=teams_b,
        team_a=team_a,
        team_b=team_b,
        tournament_teams=tournament_teams,
        comparison=comparison,
    )


@app.route("/predict", methods=["GET", "POST"])
@app.route("/predictor", methods=["GET", "POST"])
def predict_page():
    # Public-facing predictions are selected from real BMSD stage lobbies.
    # Team statistics come from the saved scorecards, so users never have to
    # type team names or pre-match feature values into the web form.
    params = {}
    for name in ("stage", "forecast"):
        value = request.values.get(name, "").strip()
        if value:
            params[name] = value
    return redirect(url_for("live", _anchor="team-prediction", **params))


@app.route("/live", methods=["GET"])
def live():
    bmsd = _filter_history("BMSD")
    board = _cache_get("live_board")
    if board is None:
        board = _team_summary(bmsd)
        _cache_set("live_board", board)
    fixtures = _cache_get("live_fixtures")
    if fixtures is None:
        fixtures = _read_source_csv(FIXTURES_PATH)
        _cache_set("live_fixtures", fixtures)
    progression = _cache_get("live_progression")
    if progression is None:
        progression = _read_source_csv(PROGRESSION_PATH)
        _cache_set("live_progression", progression)
    stage_formats = _cache_get("live_stage_formats")
    if stage_formats is None:
        stage_formats = _read_source_csv(STAGE_FORMAT_PATH)
        _cache_set("live_stage_formats", stage_formats)
    stage_sections = _cache_get("live_stage_sections")
    if stage_sections is None:
        stage_sections = _bmsd_stage_sections(stage_formats)
        _cache_set("live_stage_sections", stage_sections)
    all_groups = [
        {"stage": section["stage"], **group, "known_teams": len(group.get("teams", [])), "expected_teams": section["expected_teams"]}
        for section in stage_sections
        for group in section["groups"]
    ] or _bmsd_group_rows(fixtures)
    fixture_stages = list(dict.fromkeys([row["stage"] for row in all_groups]))
    format_stages = stage_formats.sort_values("sequence", kind="stable")["stage"].dropna().astype(str).tolist() if not stage_formats.empty and "sequence" in stage_formats else []
    progression_stages = []
    if not progression.empty and "source_stages" in progression:
        for value in progression["source_stages"].dropna().astype(str):
            progression_stages.extend(stage.strip() for stage in value.split("|") if stage.strip())
    all_stages = list(dict.fromkeys(format_stages + fixture_stages + progression_stages))
    stage_tracker = []
    section_map = {_stage_token(str(section["stage"])): section for section in stage_sections}
    for stage in all_stages:
        section = section_map.get(_stage_token(stage))
        if section:
            status = f"{section['completed_matches']} of {section['total_matches']} matches · {section['known_teams']} of {section['expected_teams']} teams"
        else:
            status = "Team assignments not published"
        stage_tracker.append({"stage": stage, "status": status})
    selected_stage = request.args.get("stage", "All Stages")
    group_rows = all_groups if selected_stage in ("", "All Stages") else [row for row in all_groups if row["stage"] == selected_stage]
    upcoming = _cache_get("live_upcoming")
    if upcoming is None:
        upcoming = _live_stage_lobbies(fixtures, stage_formats, stage_sections)
        for lobby in upcoming:
            section = section_map.get(_stage_token(lobby.get("stage")))
            if section:
                lobby.setdefault("stage_total", section.get("total_matches"))
                lobby.setdefault("completed_in_stage", section.get("completed_matches"))
        for lobby in upcoming:
            lobby["predictions"] = _lobby_predictions(lobby)
        _cache_set("live_upcoming", upcoming)
    else:
        for lobby in upcoming:
            section = section_map.get(_stage_token(lobby.get("stage")))
            if section:
                lobby.setdefault("stage_total", section.get("total_matches"))
                lobby.setdefault("completed_in_stage", section.get("completed_matches"))
    selected_forecast = request.args.get("forecast", "")
    next_lobby = next((lobby for lobby in upcoming if str(lobby.get("id")) == selected_forecast), None)
    if next_lobby is None and upcoming:
        next_lobby = upcoming[0]
    next_lobby_id = str(next_lobby.get("id")) if next_lobby else ""
    gf_smash_key = f"gf_smash_{next_lobby.get('match') if next_lobby else ''}"
    grand_finals_smash = _cache_get(gf_smash_key)
    if grand_finals_smash is None and next_lobby and _stage_token(next_lobby.get("stage")) == "grand finals":
        grand_finals_smash = _grand_finals_smash_state(next_lobby.get("match"))
        _cache_set(gf_smash_key, grand_finals_smash)
    grand_finals_section = next((section for section in stage_sections if _stage_token(section.get("stage")) == "grand finals"), None)
    grand_finals_teams = list(grand_finals_section.get("teams", [])) if grand_finals_section else []
    grand_finals_ready = bool(
        grand_finals_section
        and int(grand_finals_section.get("expected_teams", 0)) == 16
        and len(set(grand_finals_teams)) == 16
    )
    gf_scoreboard_key = f"gf_scoreboard_{','.join(sorted(grand_finals_teams))}"
    grand_finals_scoreboard = _cache_get(gf_scoreboard_key) if grand_finals_ready else []
    if grand_finals_ready and grand_finals_scoreboard is None:
        grand_finals_scoreboard = _grand_finals_scoreboard(grand_finals_teams)
        _cache_set(gf_scoreboard_key, grand_finals_scoreboard)
    grand_finals_mvp_candidates = []
    if grand_finals_ready:
        final_lobby = next((lobby for lobby in upcoming if _stage_token(lobby.get("stage")) == "grand finals"), None)
        final_match_order = pd.to_numeric(final_lobby.get("overall_match"), errors="coerce") if final_lobby else float("nan")
        if pd.isna(final_match_order):
            completed_order = pd.to_numeric(bmsd.get("match_order"), errors="coerce")
            final_match_order = float(completed_order.max() + 1) if not completed_order.empty and completed_order.notna().any() else 1.0
        gf_mvp_key = f"gf_mvp_{final_match_order}"
        grand_finals_mvp_candidates = _cache_get(gf_mvp_key)
        if grand_finals_mvp_candidates is None:
            grand_finals_mvp_candidates = _next_match_mvp_candidates({
                "teams": grand_finals_teams,
                "overall_match": float(final_match_order),
                "stage": "Grand Finals",
                "match": int(final_lobby.get("match", 1)) if final_lobby else 1,
            })
            _cache_set(gf_mvp_key, grand_finals_mvp_candidates)
    mvp_key = f"mvp_{next_lobby.get('id') if next_lobby else ''}"
    mvp_predictions = _cache_get(mvp_key)
    if mvp_predictions is None:
        mvp_predictions = _next_match_mvp_candidates(next_lobby)
        _cache_set(mvp_key, mvp_predictions)
    for section in stage_sections:
        section["forecasts"] = [
            {
                "match": lobby.get("match"),
                "id": lobby.get("id"),
                "stage_total": lobby.get("stage_total", section.get("total_matches")),
                "favorite": lobby["predictions"][0]["team"] if lobby.get("predictions") else "",
                "probability": lobby["predictions"][0]["probability"] if lobby.get("predictions") else 0.0,
                "teams": len(lobby.get("teams", [])),
                "id": lobby.get("id"),
                "inferred": bool(lobby.get("inferred")),
            }
            for lobby in upcoming
            if _stage_token(lobby.get("stage")) == _stage_token(section.get("stage"))
        ]
    remaining_forecasts = [
        {
            "stage": lobby.get("stage"),
            "match": lobby.get("match"),
            "stage_total": lobby.get("stage_total"),
            "favorite": lobby["predictions"][0]["team"] if lobby.get("predictions") else "",
            "probability": lobby["predictions"][0]["probability"] if lobby.get("predictions") else 0.0,
            "teams": len(lobby.get("teams", [])),
            "inferred": bool(lobby.get("inferred")),
            "id": lobby.get("id"),
        }
        for lobby in upcoming
        if str(lobby.get("id")) != next_lobby_id
    ]
    source_checked = ""
    if not fixtures.empty and "source_checked" in fixtures:
        dates = fixtures["source_checked"].dropna().astype(str)
        source_checked = dates.iloc[-1] if not dates.empty else ""
    return render_template(
        "live.html",
        bmsd_matches=_summary_stats(bmsd)["matches"],
        bmsd_teams=_summary_stats(bmsd)["teams"],
        leaderboard=board,
        groups=group_rows,
        stage_sections=stage_sections,
        all_stages=all_stages,
        stage_tracker=stage_tracker,
        selected_stage=selected_stage,
        upcoming=upcoming,
        next_lobby=next_lobby,
        grand_finals_smash=grand_finals_smash,
        grand_finals_ready=grand_finals_ready,
        grand_finals_scoreboard=grand_finals_scoreboard,
        grand_finals_mvp_candidates=grand_finals_mvp_candidates,
        mvp_predictions=mvp_predictions,
        mvp_model_method=str(MVP_METADATA.get("selected_method", "Rolling average baseline")),
        next_lobby_id=next_lobby_id,
        other_upcoming=[lobby for lobby in upcoming if str(lobby.get("id")) != next_lobby_id],
        forecast_options=upcoming,
        remaining_forecasts=remaining_forecasts,
        finalists=progression.to_dict("records") if not progression.empty else [],
        source_checked=source_checked,
        model_available=MODEL is not None,
        eam_url=EAM_BMSD_URL,
        matches_url=BMSD_MATCHES_URL,
        progression_url=BMSD_PROGRESSION_URL,
        format_url=BMSD_FORMAT_URL,
        liquipedia_url=LIQUIPEDIA_BMSD_URL,
        uploaded=request.args.get("uploaded", "") == "1",
        refreshed=request.args.get("refreshed", "") == "1",
        error=request.args.get("error", ""),
        source_warning=request.args.get("warning", ""),
    )


@app.route("/live/refresh", methods=["POST"])
def refresh_bmsd_source():
    global HISTORY
    try:
        raw_matches, fixtures = fetch_bmsd_source()
        # Stage lengths and qualification rules are kept separately from match
        # scorecards; they let the live page show remaining games and bracket
        # teams without making up a schedule or team assignment.
        stage_format = None
        stage_format_warning = None
        try:
            stage_format = fetch_bmsd_format()
        except Exception as exc:
            logger.warning("BMSD scorecards refreshed but stage format could not be read: %s", exc)
            stage_format_warning = "The format page was unavailable, so saved bracket rules are shown."
        imported = normalize_match_data(raw_matches, "BMSD", default_year=2026)
        if imported.empty:
            raise ValueError("No completed scorecards were returned by the source.")
        validate_match_lobbies(imported)
        # This public endpoint returns the complete BMSD stageGroups collection.
        # Replace the BMSD snapshot so an older manual seed with a differently
        # formatted match ID cannot double-count a scorecard already on source.
        merged_bmsd = imported.copy()
        validate_match_lobbies(merged_bmsd)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw_matches.to_csv(DATA_DIR / "raw" / "bmsd_matches.csv", index=False)
        fixtures.to_csv(FIXTURES_PATH, index=False)
        merged_bmsd.to_csv(DATA_DIR / "bmsd.csv", index=False)
        if stage_format is not None and not stage_format.empty:
            stage_format.to_csv(STAGE_FORMAT_PATH, index=False)

        warnings = [stage_format_warning] if stage_format_warning else []
        try:
            routes = fetch_bmsd_progression()
            route_rows = []
            for route in routes:
                for team in route["teams"]:
                    route_rows.append({
                        "route_id": route["route_id"],
                        "route": route["name"],
                        "advancement": route["advancement"],
                        "source_stages": "|".join(route.get("stages", [])),
                        "team": team,
                        "source_url": BMSD_PROGRESSION_URL,
                        "source_checked": pd.Timestamp.now(tz="UTC").isoformat(),
                    })
            pd.DataFrame(route_rows).to_csv(PROGRESSION_PATH, index=False)
        except Exception as exc:
            logger.warning("BMSD scorecards refreshed but progression could not be read: %s", exc)
            warnings.append("The progression page was unavailable, so the last saved finalist list is shown.")
        try:
            player_stats, player_match_stats = fetch_bmsd_player_stats()
            player_stats.to_csv(PLAYER_STATS_PATH, index=False)
            player_match_stats.to_csv(PLAYER_MATCH_STATS_PATH, index=False)
        except Exception as exc:
            logger.warning("BMSD scorecards refreshed but player statistics could not be read: %s", exc)
            warnings.append("The player statistics page was unavailable, so the last saved MVP watchlist is shown.")
        HISTORY = _load_history()
        LIVE_PREDICTION_CACHE.clear()
        _cache_clear()
        return redirect(url_for("live", refreshed="1", warning=" ".join(warnings)))
    except Exception as exc:
        logger.warning("BMSD source refresh failed: %s", exc)
        return redirect(url_for("live", error=f"Could not refresh BMSD data: {exc}"))


@app.route("/live/import", methods=["POST"])
def import_bmsd_scorecards():
    global HISTORY
    uploaded = request.files.get("scorecards")
    if uploaded is None or not uploaded.filename:
        return redirect(url_for("live", error="Choose a completed-match CSV scorecard first."))
    if Path(uploaded.filename).suffix.casefold() != ".csv":
        return redirect(url_for("live", error="Upload a CSV file."))
    try:
        raw = pd.read_csv(uploaded.stream)
        imported = normalize_match_data(raw, "BMSD", default_year=2026)
        if imported.empty:
            raise ValueError("The CSV contains no match records.")
        if not imported["tournament"].astype(str).str.casefold().eq("bmsd").all():
            raise ValueError("The uploaded scorecards must be BMSD match data.")
        validate_match_lobbies(imported)

        # Uploads may contain just the latest completed match. Keep earlier
        # imported BMSD scorecards and replace whole matches in this upload,
        # so corrections do not leave duplicate team rows behind.
        existing_bmsd = HISTORY[HISTORY["tournament"].astype(str).str.casefold().eq("bmsd")].copy()
        replaced_matches = set(imported["match_key"].astype(str))
        retained = existing_bmsd[~existing_bmsd["match_key"].astype(str).isin(replaced_matches)]
        merged_bmsd = pd.concat([retained, imported], ignore_index=True)
        validate_match_lobbies(merged_bmsd)

        raw_path = DATA_DIR / "raw" / "bmsd_matches.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_path.exists() and raw_path.stat().st_size:
            previous_raw = pd.read_csv(raw_path)
            raw = pd.concat([previous_raw, raw], ignore_index=True).drop_duplicates()
        raw.to_csv(raw_path, index=False)
        merged_bmsd.to_csv(DATA_DIR / "bmsd.csv", index=False)
        HISTORY = _load_history()
        LIVE_PREDICTION_CACHE.clear()
        _cache_clear()
        return redirect(url_for("live", uploaded="1"))
    except Exception as exc:
        logger.info("BMSD scorecard import rejected: %s", exc)
        return redirect(url_for("live", error=str(exc)))


@app.route("/live/template.csv")
def download_bmsd_template():
    template = pd.DataFrame(columns=[
        "date", "overall_match", "match", "stage", "group", "map", "team", "rank",
        "wwcd", "elim_points", "place_points", "total_points",
    ])
    output = io.BytesIO(template.to_csv(index=False).encode("utf-8"))
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="bmsd_match_scorecard_template.csv")


@app.route("/model-analytics")
def model_analytics():
    metrics = []
    metric_path = REPORT_DIR / "model_comparison.csv"
    if metric_path.exists():
        try:
            metrics = pd.read_csv(metric_path).to_dict("records")
        except Exception as exc:
            logger.warning("Unable to read model report: %s", exc)
    importance = get_feature_importance(MODEL, MODEL_FEATURES) if MODEL is not None else []
    metadata = {}
    meta_path = REPORT_DIR / "model_metadata.json"
    if meta_path.exists():
        try:
            import json
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return render_template("model_analytics.html", metrics=metrics, importance=importance, model_available=MODEL is not None, metadata=metadata)


@app.route("/model-report/<path:filename>")
def model_report(filename: str):
    allowed = {"model_comparison.png", "roc_curve.png", "calibration_curve.png", "feature_importance.png"}
    if filename not in allowed or not (REPORT_DIR / filename).exists():
        return render_template("error.html", code=404, message="That model report is not available yet."), 404
    return send_file(REPORT_DIR / filename, mimetype="image/png", conditional=True)


@app.route("/about")
def about():
    stats = [
        {"name": name, **_summary_stats(_filter_history(name))}
        for name in _allowed_tournaments()[1:]
    ]
    return render_template("about.html", stats=stats, model_available=MODEL is not None, eam_url=EAM_BMSD_URL, liquipedia_url=LIQUIPEDIA_BMSD_URL)


@app.route("/mvp")
def mvp():
    selected_stage = request.args.get("stage", "All Stages")
    players, stages = _mvp_watchlist(selected_stage)
    source = _read_source_csv(PLAYER_STATS_PATH)
    source_checked = ""
    if not source.empty and "source_checked" in source:
        values = source["source_checked"].dropna().astype(str)
        source_checked = values.iloc[0] if not values.empty else ""
    return render_template(
        "mvp.html",
        players=players,
        stages=stages,
        selected_stage=selected_stage,
        has_data=not source.empty,
        source_checked=source_checked,
        stats_url="https://esportsamaze.com/bgmi/tournaments/battlegrounds-mobile-india-showdown-2026/statistics",
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data:
        return jsonify({"success": False, "error": "Send a non-empty JSON object."}), 400
    team = str(data.get("team", "")).strip()
    if not team:
        return jsonify({"success": False, "error": "The team field is required."}), 400
    try:
        prediction = _predict(team, data)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    return jsonify({
        "success": True,
        "team": team,
        "winning_probability": round(float(prediction["winning_probability"]), 6),
        "winning_probability_percentage": prediction["winning_probability_percentage"],
        "features_used": MODEL_FEATURES,
        "historical_team_data_available": prediction["has_history"],
    })


@app.route("/api/predict-match", methods=["POST"])
def api_predict_match():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("teams"), list):
        return jsonify({"success": False, "error": "Send JSON with a teams array."}), 400
    team_names = [str(team).strip() for team in data["teams"] if str(team).strip()]
    if len(team_names) < 2 or len({team.casefold() for team in team_names}) != len(team_names):
        return jsonify({"success": False, "error": "Provide at least two distinct team names."}), 400
    if MODEL is None or PREPROCESSOR is None:
        return jsonify({"success": False, "error": "The trained model is not available."}), 503
    try:
        from src.feature_engineering import batch_feature_snapshots
        snapshots = batch_feature_snapshots(HISTORY, team_names)
        input_data = []
        for team in team_names:
            features = snapshots.get(team, {})
            input_data.append({f: float(features.get(f, 0.0)) for f in MODEL_FEATURES})
        input_df = pd.DataFrame(input_data)[MODEL_FEATURES]
        matrix = PREPROCESSOR.transform(input_df) if PREPROCESSOR is not None else input_df
        probs = MODEL.predict_proba(matrix)
        rows = []
        for team, input_row, p_val in zip(team_names, input_data, probs):
            has_history = bool(
                not HISTORY.empty
                and HISTORY["team_key"].eq(canonical_team_key(team)).any()
            )
            rows.append({
                "team": team,
                "raw_probability": float(np.clip(p_val[1], 0.0, 1.0)),
                "features": input_row,
                "history_available": has_history,
            })
        total = sum(row["raw_probability"] for row in rows)
        for row in rows:
            row["winning_probability"] = row["raw_probability"] / total if total > 0 else 1 / len(rows)
            row["winning_probability_percentage"] = round(row["winning_probability"] * 100, 2)
        rows.sort(key=lambda row: row["winning_probability"], reverse=True)
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        return jsonify({"success": True, "tournament": data.get("tournament", "BMSD"), "match": data.get("match"), "results": rows})
    except (ValueError, NotFittedError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/stats")
def api_stats():
    tournament = request.args.get("tournament", "All Tournaments")
    frame = _filter_history(tournament)
    return jsonify({"success": True, "tournament": tournament, "stats": _summary_stats(frame), "leaderboard": _team_summary(frame)[:10]})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None, "historical_rows": int(len(HISTORY))})


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", code=404, message="That page could not be found."), 404


@app.errorhandler(413)
def upload_too_large(_error):
    return render_template("error.html", code=413, message="The CSV is larger than the 5 MB upload limit."), 413


@app.errorhandler(500)
def server_error(_error):
    return render_template("error.html", code=500, message="The request could not be completed. Check the dataset and model files."), 500


if __name__ == "__main__":
    _warmup_cache()
    app.run(host="127.0.0.1", port=5055, debug=False)

# Also warm cache when loaded by a WSGI server (e.g. gunicorn / waitress)
_warmup_cache()
