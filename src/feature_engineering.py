"""Leakage-safe, pre-match features for BGMI team match records."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .data_preprocessing import canonical_team_key
except ImportError:
    from data_preprocessing import canonical_team_key


FEATURE_COLS = [
    "matches_before",
    "prev_avg_points",
    "prev_avg_kills",
    "prev_win_rate",
    "recent3_avg_points",
    "recent3_avg_kills",
    "recent3_win_rate",
    "prev_std_points",
    "prev_std_kills",
    "pre_match_elo",
]

def _ordered_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Create one chronological row per match and map it back to records."""
    frame = df.copy()
    if "match_key" not in frame.columns:
        if "match_id" not in frame.columns:
            raise ValueError("Feature engineering needs match_key or match_id.")
        tournament = frame.get("tournament", pd.Series("Tournament", index=frame.index)).astype(str)
        frame["match_key"] = tournament + "|" + frame["match_id"].astype(str)
    if "team_key" not in frame.columns:
        frame["team_key"] = frame["team"].astype(str).str.strip().str.casefold()
    for column, default in (("tournament", "BGIS"), ("year", 0), ("match_order", 0), ("stage", "Tournament"), ("group", "")):
        if column not in frame:
            frame[column] = default
    if "date" in frame:
        frame["_event_date"] = pd.to_datetime(frame["date"], errors="coerce")
    else:
        frame["_event_date"] = pd.NaT
    frame["_year"] = pd.to_numeric(frame["year"], errors="coerce").fillna(0)
    frame["_match_order"] = pd.to_numeric(frame["match_order"], errors="coerce").fillna(0)

    match_meta = frame.drop_duplicates("match_key").copy()
    # Order by observed year and date. Match order and tournament label are
    # deterministic fallbacks when source dates are absent or tied.
    match_meta = match_meta.sort_values(
        ["_year", "_event_date", "_match_order", "tournament", "stage", "group", "match_id" if "match_id" in frame else "match_key"],
        kind="stable",
        na_position="first",
    )
    order = {key: i for i, key in enumerate(match_meta["match_key"].tolist())}
    frame["_chronology_order"] = frame["match_key"].map(order).astype(int)
    return frame.sort_values(["_chronology_order", "team_key"], kind="stable").reset_index(drop=True)


_ELO_CACHE: dict[tuple[int, int], tuple[pd.DataFrame, dict[str, float]]] = {}


def _elo_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute each team's rating before its match, updating only afterward."""
    if frame.empty:
        result = frame.copy()
        result["pre_match_elo"] = 1500.0
        return result, {}

    cache_key = (id(frame), len(frame))
    cached = _ELO_CACHE.get(cache_key)
    if cached is not None:
        return cached[0], cached[1]

    ratings: dict[str, float] = {}
    pre_match: dict[int, float] = {}
    indices = frame.index.to_numpy()
    team_keys = frame["team_key"].astype(str).to_numpy()
    wins = pd.to_numeric(frame["win"], errors="coerce").fillna(0.0).to_numpy()

    for _, match in frame.groupby("_chronology_order", sort=True):
        m_indices = match.index.to_numpy()
        m_teams = match["team_key"].astype(str).to_numpy()
        m_wins = pd.to_numeric(match["win"], errors="coerce").fillna(0.0).to_numpy()

        snapshot = {t: ratings.get(t, 1500.0) for t in m_teams}
        total_elo = sum(snapshot.values())
        n_teams = len(snapshot)
        mean_elo = total_elo / n_teams if n_teams else 1500.0
        updates: dict[str, float] = {}

        for idx, team, win_val in zip(m_indices, m_teams, m_wins):
            rating = snapshot[team]
            pre_match[idx] = rating
            n_opp = n_teams - 1
            mean_opp = (total_elo - rating) / n_opp if n_opp > 0 else mean_elo
            expected = 1.0 / (1.0 + 10.0 ** ((mean_opp - rating) / 400.0))
            updates[team] = rating + 20.0 * (float(win_val) - expected)
        ratings.update(updates)

    result = frame.copy()
    result["pre_match_elo"] = pd.Series(pre_match).reindex(result.index).fillna(1500.0)
    _ELO_CACHE[cache_key] = (result, ratings)
    return result, ratings


def build_pre_match_features(match_data: pd.DataFrame) -> pd.DataFrame:
    """Add historical team-form features using completed matches only.

    The rows are ordered by event year and match order.  Within a match all
    teams share the same pre-match state; target, points, and kills from that
    match are only added to the history after the features are calculated.
    """
    if match_data is None or match_data.empty:
        return pd.DataFrame()
    frame = _ordered_matches(match_data)
    required = {"team_key", "points", "kills", "win", "match_key"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for pre-match features: {sorted(missing)}")

    for column in ("points", "kills", "win"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[["points", "kills", "win"]] = frame[["points", "kills", "win"]].fillna(0.0)

    team_groups = frame.groupby("team_key", sort=False)
    frame["matches_before"] = team_groups.cumcount().astype(float)
    for source, target in (("points", "prev_avg_points"), ("kills", "prev_avg_kills"), ("win", "prev_win_rate")):
        frame[target] = team_groups[source].transform(lambda values: values.shift(1).expanding(min_periods=1).mean()).fillna(0.0)
    for source, target in (("points", "recent3_avg_points"), ("kills", "recent3_avg_kills"), ("win", "recent3_win_rate")):
        frame[target] = team_groups[source].transform(lambda values: values.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0)
    for source, target in (("points", "prev_std_points"), ("kills", "prev_std_kills")):
        frame[target] = team_groups[source].transform(lambda values: values.shift(1).expanding(min_periods=2).std(ddof=0)).fillna(0.0)

    frame, _ = _elo_features(frame)
    for column in FEATURE_COLS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame.sort_values(["_chronology_order", "team_key"], kind="stable").reset_index(drop=True)


def feature_snapshot_for_team(match_data: pd.DataFrame, team: str, ratings: dict[str, float] | None = None) -> dict[str, float]:
    """Build next-match features from a team's full completed-match history."""
    if not team or not str(team).strip():
        raise ValueError("A team name is required.")
    frame = _ordered_matches(match_data) if match_data is not None and not match_data.empty else pd.DataFrame()
    key = canonical_team_key(team)
    if not frame.empty:
        # Canonical aliases are normally applied by data_preprocessing. Keep
        # raw names usable for API callers that pass an exact historical name.
        matches = frame[frame["team_key"].astype(str).str.casefold().eq(key)].copy()
    else:
        matches = pd.DataFrame()
    if matches.empty:
        return {feature: (1500.0 if feature == "pre_match_elo" else 0.0) for feature in FEATURE_COLS}

    matches = matches.sort_values("_chronology_order", kind="stable")
    points = pd.to_numeric(matches["points"], errors="coerce").fillna(0.0)
    kills = pd.to_numeric(matches["kills"], errors="coerce").fillna(0.0)
    wins = pd.to_numeric(matches["win"], errors="coerce").fillna(0.0)
    recent_points, recent_kills, recent_wins = points.tail(3), kills.tail(3), wins.tail(3)
    if ratings is None:
        _, ratings = _elo_features(frame)
    return {
        "matches_before": float(len(matches)),
        "prev_avg_points": float(points.mean()),
        "prev_avg_kills": float(kills.mean()),
        "prev_win_rate": float(wins.mean()),
        "recent3_avg_points": float(recent_points.mean()),
        "recent3_avg_kills": float(recent_kills.mean()),
        "recent3_win_rate": float(recent_wins.mean()),
        "prev_std_points": float(points.std(ddof=0)) if len(points) > 1 else 0.0,
        "prev_std_kills": float(kills.std(ddof=0)) if len(kills) > 1 else 0.0,
        "pre_match_elo": float(ratings.get(key, 1500.0)),
    }


def batch_feature_snapshots(match_data: pd.DataFrame, teams: list[str]) -> dict[str, dict[str, float]]:
    """Build next-match features for multiple teams simultaneously in one fast grouped pass."""
    empty_result = {team: {feature: (1500.0 if feature == "pre_match_elo" else 0.0) for feature in FEATURE_COLS} for team in teams}
    if match_data is None or match_data.empty or not teams:
        return empty_result
    frame = _ordered_matches(match_data)
    _, ratings = _elo_features(frame)
    key_to_teams: dict[str, list[str]] = {}
    for team in teams:
        k = canonical_team_key(team)
        key_to_teams.setdefault(k, []).append(team)

    matched_frame = frame[frame["team_key"].isin(key_to_teams.keys())]
    if matched_frame.empty:
        return empty_result

    results = dict(empty_result)
    for key, matches in matched_frame.groupby("team_key", sort=False):
        matches = matches.sort_values("_chronology_order", kind="stable")
        p = pd.to_numeric(matches["points"], errors="coerce").fillna(0.0).to_numpy()
        k = pd.to_numeric(matches["kills"], errors="coerce").fillna(0.0).to_numpy()
        w = pd.to_numeric(matches["win"], errors="coerce").fillna(0.0).to_numpy()
        n = len(p)
        snap = {
            "matches_before": float(n),
            "prev_avg_points": float(p.mean()) if n else 0.0,
            "prev_avg_kills": float(k.mean()) if n else 0.0,
            "prev_win_rate": float(w.mean()) if n else 0.0,
            "recent3_avg_points": float(p[-3:].mean()) if n else 0.0,
            "recent3_avg_kills": float(k[-3:].mean()) if n else 0.0,
            "recent3_win_rate": float(w[-3:].mean()) if n else 0.0,
            "prev_std_points": float(p.std(ddof=0)) if n > 1 else 0.0,
            "prev_std_kills": float(k.std(ddof=0)) if n > 1 else 0.0,
            "pre_match_elo": float(ratings.get(key, 1500.0)),
        }
        for orig_team in key_to_teams.get(key, []):
            results[orig_team] = snap
    return results


# Backward-compatible entry points used by the original notebooks and scripts.
def compute_historical_features(
    df: pd.DataFrame,
    points_col: str = "points",
    kills_col: str = "kills",
    win_col: str = "win",
    match_col: str = "match_num",
    team_col: str = "team",
) -> pd.DataFrame:
    """Compatibility wrapper around the no-leakage feature builder."""
    source = df.copy()
    if points_col not in source.columns and "placement" in source.columns:
        source[points_col] = pd.to_numeric(source["placement"], errors="coerce")
    if kills_col not in source.columns:
        source[kills_col] = 0.0
    if win_col not in source.columns:
        source[win_col] = 0.0
    if team_col not in source.columns:
        raise ValueError(f"Missing team column {team_col!r}.")
    match_values = source[match_col] if match_col in source.columns else source.get("match_no")
    if match_values is None:
        raise ValueError("Missing match number column.")
    source["match_id"] = match_values.astype(str)
    source["tournament"] = source.get("tournament", "BGIS")
    source["year"] = pd.to_numeric(source.get("year", 2025), errors="coerce").fillna(2025) if isinstance(source.get("year", 2025), pd.Series) else 2025
    source["match_order"] = pd.to_numeric(match_values.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce").fillna(0)
    source["match_key"] = source["tournament"].astype(str) + "|" + source["match_id"].astype(str)
    source["team_key"] = source[team_col].astype(str).str.strip().str.casefold()
    source["points"] = pd.to_numeric(source[points_col], errors="coerce").fillna(0)
    source["kills"] = pd.to_numeric(source[kills_col], errors="coerce").fillna(0)
    source["win"] = pd.to_numeric(source[win_col], errors="coerce").fillna(0)
    return build_pre_match_features(source)


def compute_elo_features(
    df: pd.DataFrame,
    match_col: str = "match_num",
    team_col: str = "team",
    win_col: str = "win",
    K: float = 20.0,
    initial_elo: float = 1500.0,
) -> pd.DataFrame:
    """Compatibility wrapper that adds leakage-safe pre-match ELO columns."""
    result = compute_historical_features(df, win_col=win_col, match_col=match_col, team_col=team_col)
    if initial_elo != 1500.0 or K != 20.0:
        result.attrs["note"] = "Compatibility wrapper uses the standardized 1500 rating and K=20 scale."
    return result


def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Return a finite numeric feature matrix and its ordered feature names."""
    columns = list(feature_cols or FEATURE_COLS)
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from data: {missing}")
    matrix = df[columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return matrix, columns

