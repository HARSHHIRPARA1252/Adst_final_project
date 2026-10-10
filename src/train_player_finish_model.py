"""Train and evaluate the BMSD match-level player finish estimator."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.player_finish_model import train_player_finish_model


def main() -> None:
    data_dir = ROOT / "data" / "raw"
    player_path = data_dir / "bmsd_player_match_stats.csv"
    fixture_path = data_dir / "bmsd_fixtures.csv"
    if not player_path.exists() or not fixture_path.exists():
        raise SystemExit("Refresh BMSD player and fixture data before training the player finish model.")
    result = train_player_finish_model(pd.read_csv(player_path), pd.read_csv(fixture_path))
    print("Chronological player finish model comparison:")
    print(result["comparison"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Selected: {result['selected_method']}")


if __name__ == "__main__":
    main()
