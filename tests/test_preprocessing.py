from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_preprocessing import load_all_historical_data, normalize_match_data, validate_match_lobbies
from src.feature_engineering import FEATURE_COLS, build_pre_match_features


def test_historical_sources_have_complete_single_winner_lobbies():
    data = load_all_historical_data(ROOT / "data")
    assert not data.empty
    counts = data["tournament"].value_counts().to_dict()
    assert counts["BGIS"] == 288
    assert counts["BMPS"] == 288
    assert counts["BMSD"] % 16 == 0
    matches = data.drop_duplicates("match_key").groupby("tournament").size().to_dict()
    assert matches["BGIS"] == 18
    assert matches["BMPS"] == 18
    assert matches["BMSD"] == counts["BMSD"] // 16
    validate_match_lobbies(data)
    assert data.groupby("match_key")["win"].sum().eq(1).all()


def test_invalid_duplicate_or_multiwinner_match_is_rejected():
    rows = pd.DataFrame({
        "team": ["A", "A", "B"], "match": [1, 1, 1], "rank": [1, 1, 2],
        "wwcd": [1, 1, 0], "total_points": [10, 10, 5], "elims": [3, 3, 1],
    })
    normalized = normalize_match_data(rows, "BMSD", default_year=2026)
    with pytest.raises(ValueError):
        validate_match_lobbies(normalized)


def test_pre_match_features_do_not_use_the_current_match_result():
    rows = []
    for match in range(1, 5):
        rows.extend([
            {"tournament": "BGIS", "year": 2025, "match_id": str(match), "match_order": match,
             "match_key": f"BGIS|{match}", "team": "A", "team_key": "a", "points": 12 + match,
             "kills": match, "win": int(match == 1), "rank": 1 if match == 1 else 2},
            {"tournament": "BGIS", "year": 2025, "match_id": str(match), "match_order": match,
             "match_key": f"BGIS|{match}", "team": "B", "team_key": "b", "points": 8 + match,
             "kills": 2, "win": int(match == 2), "rank": 2 if match == 1 else 1},
        ])
    original = pd.DataFrame(rows)
    changed = original.copy()
    changed.loc[changed["match_key"].eq("BGIS|3"), ["points", "kills", "win"]] = [999, 999, 1]
    base_features = build_pre_match_features(original)
    changed_features = build_pre_match_features(changed)
    row_original = base_features[(base_features["match_key"] == "BGIS|3") & (base_features["team_key"] == "a")].iloc[0]
    row_changed = changed_features[(changed_features["match_key"] == "BGIS|3") & (changed_features["team_key"] == "a")].iloc[0]
    assert row_original[FEATURE_COLS].to_numpy(dtype=float) == pytest.approx(row_changed[FEATURE_COLS].to_numpy(dtype=float))
    later = changed_features[(changed_features["match_key"] == "BGIS|4") & (changed_features["team_key"] == "a")].iloc[0]
    assert later["prev_avg_points"] > row_original["prev_avg_points"]
    assert np.isfinite(base_features[FEATURE_COLS].to_numpy()).all()


def test_bmsd_rows_use_explicit_source_wwcd():
    data = load_all_historical_data(ROOT / "data")
    bmsd = data[data["tournament"].eq("BMSD")]
    source = pd.read_csv(ROOT / "data" / "raw" / "bmsd_matches.csv")
    expected = normalize_match_data(source, "BMSD", default_year=2026)
    expected_winners = set(zip(expected.loc[expected["win"].eq(1), "match_key"], expected.loc[expected["win"].eq(1), "team"]))
    actual_winners = set(zip(bmsd.loc[bmsd["win"].eq(1), "match_key"], bmsd.loc[bmsd["win"].eq(1), "team"]))
    assert len(actual_winners) == bmsd["match_key"].nunique()
    assert actual_winners == expected_winners
