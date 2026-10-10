"""Load, normalize, and validate match-level BGMI scorecards.

The canonical row is one team in one completed match.  The target ``win``
means a match win (WWCD) when the source provides that signal.  For the BGIS
2025 file the source already provides a single binary ``win`` flag per match;
its original WWCD field is not present in the supplied CSV.
"""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


TEAM_ALIASES = {
    "vaslsta esports": "vasista esports",
    "team tamillas": "team tamilas",
    "iqoo team tamilas iq-tt": "team tamilas",
    "iqoo soul iq-soul": "team soul",
    "team soul": "team soul",
    "iqoo orangutan iq-og": "orangutan",
    "team orangutan": "orangutan",
    "hero xtreme godlike hx-godl": "godlike esports",
    "godlike": "godlike esports",
    "iqoo revenant xspark iq-rntx": "revenant xspark",
    "iqoo reckoning esports iq-rge": "reckoning esports",
    "vasista esports vasista": "vasista esports",
}


def _key(value: object) -> str:
    """Normalize an identifier for matching without changing display text."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    return re.sub(r"\s+", " ", text)


def canonical_team_key(value: object) -> str:
    key = _key(value)
    return TEAM_ALIASES.get(key, key)


def _column_token(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


COLUMN_ALIASES = {
    "team": ("team", "teamname", "squad", "squadname", "teamshortname"),
    "match_id": ("matchid", "match", "matchno", "matchnumber", "game", "gameid"),
    "overall_match": ("overallmatch", "overallmatchnumber", "globalmatch"),
    "date": ("date", "matchdate"),
    "stage": ("stage", "round", "week", "phase"),
    "group": ("group", "lobby", "matchgroup"),
    "map": ("map", "mapname"),
    "rank": ("rank", "placement", "position", "place"),
    "wwcd": ("wwcd", "winnerwinnerchickendinner", "chickendinner", "w"),
    "kills": ("kills", "kill", "elimpoints", "elims", "eliminations", "e"),
    "points": ("points", "totalpoints", "totalscore", "totals", "score", "t"),
    "placement_points": ("placementpoints", "placepoints", "p"),
    "win": ("win", "winner", "iswinner", "matchwin", "matchwinnerflag"),
    "match_winner": ("matchwinner", "winningteam", "winnerteam"),
    "tournament": ("tournament", "event", "eventname", "series"),
    "season": ("season", "edition"),
    "year": ("year", "eventyear", "seasonyear"),
}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common source headers to stable names, preserving other columns."""
    result = frame.copy()
    rename: dict[object, str] = {}
    tokens = {_column_token(column): column for column in result.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for token in aliases:
            if token in tokens:
                rename[tokens[token]] = canonical
                break
    return result.rename(columns=rename)


def _binary(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "y", "win", "won", "wwcd", "winner", "🍗"}:
        return 1.0
    if text in {"0", "false", "no", "n", "loss", "lost", "-", "—", ""}:
        return 0.0
    return np.nan


def _match_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    direct = pd.to_numeric(value, errors="coerce")
    if not pd.isna(direct):
        return float(direct)
    match = re.search(r"(\d+)", str(value))
    return float(match.group(1)) if match else np.nan


def normalize_match_data(
    frame: pd.DataFrame,
    tournament: str,
    default_year: int | None = None,
    target_source: str | None = None,
    require_target: bool = True,
) -> pd.DataFrame:
    """Return validated, canonical match-level records.

    ``target_source`` may be ``"wwcd"`` or ``"win"``.  If omitted, the
    source WWCD field is preferred over a rank-derived or general win field.
    ``rank == 1`` is only used as a last-resort proxy when the file has no
    explicit win/winner information.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = normalize_columns(frame)
    required = ["team"]
    if "team" not in df.columns:
        raise ValueError("Scorecard needs a team or squad column.")

    if "match_id" not in df.columns:
        if "overall_match" in df.columns:
            df["match_id"] = df["overall_match"].astype(str)
        else:
            raise ValueError("Scorecard needs a match, match_id, or overall_match column.")
    required.append("match_id")

    df["team"] = df["team"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["team"].notna() & df["team"].ne("")].copy()
    df["team_key"] = df["team"].map(canonical_team_key)
    df["tournament"] = (
        df["tournament"].astype("string").str.strip()
        if "tournament" in df.columns
        else tournament
    )
    df["tournament"] = df["tournament"].fillna(tournament).replace("", tournament)
    df["stage"] = df["stage"].astype("string").str.strip() if "stage" in df.columns else "Tournament"
    df["stage"] = df["stage"].fillna("Tournament").replace("", "Tournament")
    df["group"] = df["group"].astype("string").str.strip() if "group" in df.columns else ""
    df["group"] = df["group"].fillna("")
    df["map"] = df["map"].astype("string").str.strip() if "map" in df.columns else ""
    df["map"] = df["map"].fillna("")
    if "date" in df.columns:
        # Parse unambiguous ISO dates first. Pandas' mixed day-first parser can
        # misread ISO `YYYY-MM-DD` strings (for example 2026-10-02 as Feb 10).
        raw_dates = df["date"].astype("string").str.strip()
        parsed_dates = pd.to_datetime(raw_dates, errors="coerce", format="ISO8601")
        needs_fallback = parsed_dates.isna() & raw_dates.notna() & raw_dates.ne("")
        if needs_fallback.any():
            parsed_dates.loc[needs_fallback] = pd.to_datetime(
                raw_dates.loc[needs_fallback], errors="coerce", dayfirst=True, format="mixed"
            )
        df["date"] = parsed_dates
    else:
        df["date"] = pd.NaT

    if "overall_match" in df.columns:
        order_source = df["overall_match"]
    elif "match_order" in df.columns:
        # Canonical snapshots may already carry source-wide ordering even
        # when the source's local match number restarts in each stage.
        order_source = df["match_order"]
    else:
        order_source = df["match_id"]
    df["match_order"] = order_source.map(_match_number)
    if df["match_order"].isna().any():
        # Stable fallback for source IDs such as an opaque match key.
        opaque = df["match_id"].astype(str)
        fallback = pd.factorize(opaque, sort=True)[0] + 1
        df["match_order"] = df["match_order"].fillna(pd.Series(fallback, index=df.index))

    if "year" in df.columns:
        years = pd.to_numeric(df["year"], errors="coerce")
    else:
        years = pd.Series(np.nan, index=df.index)
    if "season" in df.columns:
        season_year = df["season"].astype(str).str.extract(r"(20\d{2})", expand=False)
        years = years.fillna(pd.to_numeric(season_year, errors="coerce"))
    if "date" in df.columns:
        years = years.fillna(df["date"].dt.year)
    if default_year is not None:
        years = years.fillna(default_year)
    df["year"] = years.fillna(0).astype(int)

    for column in ("points", "kills", "rank", "placement_points"):
        df[column] = pd.to_numeric(df[column], errors="coerce") if column in df.columns else np.nan
    df["points"] = df["points"].fillna(0.0)
    df["kills"] = df["kills"].fillna(0.0)

    if target_source == "wwcd" and "wwcd" in df.columns:
        target = df["wwcd"].map(_binary)
    elif target_source == "win" and "win" in df.columns:
        target = df["win"].map(_binary)
    elif "wwcd" in df.columns:
        target = df["wwcd"].map(_binary)
    elif "match_winner" in df.columns:
        winner_keys = df["match_winner"].map(canonical_team_key)
        target = winner_keys.eq(df["team_key"]).astype(float)
    elif "win" in df.columns:
        target = df["win"].map(_binary)
    elif "rank" in df.columns:
        target = df["rank"].eq(1).astype(float)
    else:
        target = pd.Series(np.nan, index=df.index)

    if target.isna().any() and require_target:
        raise ValueError("A completed match scorecard has missing or unrecognized winner flags.")
    df["win"] = target

    # Preserve source IDs, but build a unique key when a source restarts M1
    # across stages, weeks, or groups.
    df["match_id"] = df["match_id"].astype("string").str.strip()
    date_part = df["date"].dt.strftime("%Y%m%d").fillna("")
    df["match_key"] = (
        df["tournament"].astype(str).str.strip()
        + "|" + df["year"].astype(str)
        + "|" + df["stage"].astype(str).str.strip()
        + "|" + df["group"].astype(str).str.strip()
        + "|" + date_part
        + "|" + df["match_id"].astype(str)
    )
    df["source"] = tournament

    invalid_numeric = (df[["points", "kills"]] < 0).any(axis=1)
    if invalid_numeric.any():
        raise ValueError("Points and kills cannot be negative.")
    invalid_target = ~df["win"].isin([0.0, 1.0])
    if invalid_target.any() and require_target:
        raise ValueError("Winner flags must be binary 0/1 values.")

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[CANONICAL_COLUMNS].reset_index(drop=True)


CANONICAL_COLUMNS = [
    "tournament", "year", "stage", "group", "date", "match_id", "match_order",
    "match_key", "team", "team_key", "points", "kills", "rank", "placement_points",
    "map", "win", "source",
]


def validate_match_lobbies(df: pd.DataFrame, require_complete: bool = True) -> None:
    """Check for one row per team and exactly one winner in every match."""
    if df.empty:
        return
    key = "match_key" if "match_key" in df.columns else "match_id"
    duplicated = df.duplicated([key, "team_key"], keep=False)
    if duplicated.any():
        rows = df.loc[duplicated, [key, "team"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate team rows in a match: {rows}")
    summary = df.groupby(key, dropna=False).agg(teams=("team_key", "nunique"), winners=("win", "sum"))
    if summary["winners"].isna().any() or not summary["winners"].eq(1).all():
        bad = summary.loc[~summary["winners"].eq(1)].head(5).to_dict("index")
        raise ValueError(f"Each completed match must have exactly one winner. Problem matches: {bad}")
    if require_complete and summary["teams"].lt(2).any():
        raise ValueError("Each match must contain at least two teams.")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_bgis_data(data_dir: str | Path) -> pd.DataFrame:
    """Load BGIS match-level records, excluding the incomplete 2026 fragment."""
    data_dir = Path(data_dir)
    source = _read_csv(data_dir / "bgis_combined.csv")
    if not source.empty and "tournament" in source.columns:
        source = source[source["tournament"].astype(str).str.casefold().eq("bgis")].copy()
    if source.empty:
        source = _read_csv(data_dir / "bgis.csv")
    if source.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    result = normalize_match_data(source, "BGIS", default_year=2025, target_source="win")
    validate_match_lobbies(result)
    return result


def load_bmps_data(data_dir: str | Path) -> pd.DataFrame:
    """Load BMPS final match scorecards and derive wins from the WWCD field."""
    data_dir = Path(data_dir)
    source = _read_csv(data_dir / "bmps_clean.csv")
    if source.empty:
        source = _read_csv(data_dir / "bmps_2026.csv")
    if source.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    result = normalize_match_data(source, "BMPS", default_year=2026, target_source="wwcd")
    validate_match_lobbies(result)
    return result


def load_bmsd_data(data_dir: str | Path) -> pd.DataFrame:
    """Load imported completed BMSD match scorecards, if any are available."""
    data_dir = Path(data_dir)
    source = _read_csv(data_dir / "bmsd.csv")
    if source.empty:
        source = _read_csv(data_dir / "raw" / "bmsd_matches.csv")
    if source.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    result = normalize_match_data(source, "BMSD", default_year=2026)
    validate_match_lobbies(result)
    return result


def load_all_historical_data(data_dir: str | Path) -> pd.DataFrame:
    """Return complete BGIS/BMPS history plus BMSD results when imported."""
    parts = [
        load_bgis_data(data_dir),
        load_bmps_data(data_dir),
        load_bmsd_data(data_dir),
    ]
    nonempty = [part for part in parts if not part.empty]
    if not nonempty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    combined = pd.concat(nonempty, ignore_index=True)
    validate_match_lobbies(combined)
    return combined


def clean_team_names(df: pd.DataFrame, team_col: str = "team") -> pd.DataFrame:
    """Backward-compatible helper: trim and normalize repeated whitespace."""
    result = df.copy()
    if team_col in result.columns:
        result[team_col] = result[team_col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    return result


def validate_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Backward-compatible structural validation function."""
    errors: list[str] = []
    if df is None or df.empty:
        return False, ["DataFrame is empty."]
    normalized = normalize_columns(df)
    for column in ("team", "match_id", "points", "kills"):
        if column not in normalized.columns:
            errors.append(f"Missing required column: {column}")
    if not ("win" in normalized.columns or "wwcd" in normalized.columns or "rank" in normalized.columns):
        errors.append("Missing winner signal (win, wwcd, or rank).")
    return not errors, errors

