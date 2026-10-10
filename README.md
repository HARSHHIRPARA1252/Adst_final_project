# BGMI Tournament Winning Probability Prediction System

A reproducible Flask and machine-learning project that estimates a team's probability of winning **one upcoming match** from completed BGMI match history. It reuses the supplied BGIS and BMPS data, adds leakage-safe historical form features, compares calibrated classifiers, and refreshes BMSD match, stage, progression, and player data from eSportsAmaze.

> A match probability is a statistical estimate, not a guarantee. The current data supports match-level WWCD estimates; it does not support a validated probability of winning the entire BMSD tournament.

## What is included

- Flask home, historical dashboard, match history, leaderboard, team comparison, prediction, methodology, and model report pages.
- BMSD live tournament center with all bracket teams, match completion counts, qualification spots, and match-by-match forecasts for source-backed lobbies.
- A link from the Live BMSD page to the official KRAFTON India Esports YouTube live page. The stream opens on YouTube; the project does not embed or depend on video playback.
- Automatic prediction flow: choose a bracket and match; the app loads teams and pre-match statistics from saved scorecards, so no team statistics need to be entered.
- A two-part selected-match view: calibrated team WWCD estimates and a player finish-leader/MVP-style estimate from the latest player match history.
- A Grand Finals-only section that appears once all 16 finalists are confirmed, with the final score table and finish-based MVP candidates from those finalist teams.
- A separately trained player finishes regressor, selected against a rolling-average baseline on a chronological holdout; it predicts finishes, not an MVP-award probability.
- An editable 3-slide model report card at `reports/model_results/BGMI_Model_Report_Card.pptx`, covering both model evaluations, limitations, and improvement priorities. The report card is a PowerPoint file, not a portal page.
- Tournament-wide BMSD MVP watchlist using player finishes, with all-stage and per-stage views.
- JSON endpoints at `/api/predict`, `/api/predict-match`, and `/api/stats`.
- A manual CSV fallback and a “Refresh BMSD data” action at `/live`.
- A Jupyter analysis notebook at `notebooks/BGMI_Winning_Probability_Analysis.ipynb`.
- Training and prediction code under `src/`.
- Saved model and preprocessing artifacts under `models/`.
- Evaluation metrics, predictions, and diagnostic charts under `reports/model_results/`.

## Existing project inspection

The original project already had a Flask application and useful BGIS/BMPS match loaders, rolling features, ELO logic, comparison views, and dark esports styling. Those ideas and routes were retained and reorganized. The existing Streamlit pages remain in `src/` as legacy references; Flask is the primary application.

The inspection identified these issues and the fixes used here:

- The BMSD source snapshot includes 66 completed scorecards (1,056 team-match rows) through Upper Bracket Survival Match 12, all eight stage definitions and advancement rules, four currently listed Grand Finals qualifiers, and player-level finish statistics. The app refreshes these public pages on request and retains the latest local snapshots.
- The root file `BGIS_Grand_Finals_Matches_1_18_Combined.csv` is labelled BGIS, but its June 19–21, 2026 results correspond to the separate BMPS finals data and its winner column does not consistently match WWCD. It is not used as a training source.
- `data/bmps_features.csv` uses points rank as its `win` label in some rows even where the WWCD team differs. Training now derives BMPS wins from the source `wwcd` field.
- The 2026 BGIS fragment has only two teams in each of two matches, so it is incomplete and excluded from whole-lobby training.
- Previous training scripts calibrated against their test set, which leaks test labels into displayed results. The new training script holds out later complete matches and uses expanding chronological calibration folds.
- The former Streamlit live page calculated percentages with a fixed points/kills formula. The Flask app uses the saved trained estimator instead.
- The five files under the old `Notebook/` folder are zero-byte files. The new notebook contains the full analysis workflow.

See [PROJECT_AUDIT.md](PROJECT_AUDIT.md) for the concise audit record.

## Dataset

The project adapts to the columns that are actually available:

| Source | Current records | Use |
|---|---:|---|
| `data/bgis_combined.csv` | 288 complete team-match rows across 18 matches, plus four incomplete rows with missing event metadata | BGIS 2025 history; incomplete rows are filtered out |
| `data/bmps_clean.csv` | 288 team-match rows across 18 matches | BMPS 2026 finals; WWCD is the target |
| `data/bmsd.csv` | 1,056 team-match rows across 66 matches | BMSD 2026 scorecards; source rows are in `data/raw/bmsd_matches.csv` |

The complete historical set currently contains 102 matches and 1,632 team-match rows. The BGIS source has a binary `win` column but no original WWCD column, so the pipeline preserves that supplied flag and records the semantic limitation. BMPS and BMSD use explicit WWCD fields. Exactly one winner flag is required per completed match.

The team win classifier uses common match-level measurements: points, finishes (stored as kills in the existing CSV schema), and match wins. Player finish data is stored separately for the MVP-style estimate. Damage, survival time, and assists are not used where the source does not populate them. Team aliases are normalized only where the spelling/branding match is known; the displayed source name remains available in the data.

## Data processing and features

`src/data_preprocessing.py` standardizes common column names, converts numeric fields, normalizes team names, identifies the match winner, removes no valid match rows silently, and validates match groups. A BMSD import must contain one row per team in each completed match, non-negative points/kills, and exactly one winner flag.

The validated, canonical 1,632-row dataset is also saved at `data/processed/tournament_history.csv` for inspection and reuse.

`src/feature_engineering.py` computes these features using only completed matches before the row's match:

- Matches played before the match.
- Expanding average points, kills, and win rate.
- Average points, kills, and wins over the last three completed matches.
- Expanding standard deviation of points and kills as consistency measures.
- Pre-match ELO rating.

Features are generated match by match in event order. Rows from the same match stay together in every validation fold. Current-match points, kills, and winner labels become history only for later matches.

## Model training and evaluation

The primary target is match win / WWCD. BGMI match wins are imbalanced: one winner per 16-team match is about 6.25% of team-match rows in the provided finals. SMOTE and class weighting are not used because changing the natural class ratio would distort probability estimates.

The pipeline compares Logistic Regression, Random Forest, and Gradient Boosting. The latest six complete matches are held out in chronological order. Calibration uses sigmoid calibration on expanding match-group folds from earlier matches. The selected estimator minimizes conditional match log loss, with Brier score as the tie-breaker. It also reports accuracy, precision, recall, F1, ROC-AUC, log loss, Brier score, confusion matrix, ROC curve, reliability curve, and feature importance.

There are 102 complete historical matches in the current snapshot; the final six are held out chronologically. Random Forest was selected by the lowest conditional match log loss (2.7640; Brier score 0.0585; ROC-AUC 0.5444). It selected the actual winner in one of the six held-out matches (top-one accuracy 16.7%). A uniform 16-team lobby has match log loss about 2.77, so the measured improvement is small. Treat this as a limited backtest, not established accuracy.

The saved classifier is calibrated on historical probabilities using forward-only folds. Feature importance describes how the selected model uses inputs; it does not show causation.

The separate player finishes model uses 4,224 BMSD player-match rows and holds out the last six matches. Random Forest was selected by MAE (1.0919 finishes) over the rolling-average baseline (1.1070); its RMSE is 1.4357 and R² is -0.0047. This is weak player-level evidence, so the UI calls its top estimate an MVP-style candidate and does not show an award probability.

## BMSD live workflow

The live page includes an explicit source refresh. The current BMSD snapshot has 66 completed matches: Week 1, Week 2, and Week 3 are complete, and Upper Bracket Survival has 12 of its 18 scheduled matches complete. The stage format data includes all eight stages, their match totals, groups, and advancement rules. The page resolves completed qualifiers from stage standings and shows pending slots for teams that have not qualified yet.

1. Press **Refresh BMSD data** on `/live`. The app reads the public scorecards, progression, and player performance pages; Flask does not fetch continuously in the background.
2. Completed scorecards are saved to `data/raw/bmsd_matches.csv` and normalized to `data/bmsd.csv`. Stage format, fixture, progression, and player data are saved separately under `data/raw/`.
3. Choose a bracket and match from the prediction selector. For the current single-lobby Upper Bracket, the official format says 18 matches and Match 12 is complete, so the page offers Matches 13–18. It reuses the 16 teams from the official completed lobby; no team or performance-stat entry is required. Map and start time remain blank until published.
4. Each team forecast uses only completed results before that match. Remaining future match rows are “as of now” estimates; they refresh as new scorecards are imported. They do not assume that a predicted result has happened.
5. Grand Finals uses 18 match slots. Once all 16 finalists are confirmed, the app can show predictions starting at Match 1 and advance to the next match after a scorecard refresh. After Matches 1–12 are complete, it calculates the Match Point target as the Match 12 leaderboard leader’s total plus 10. From Match 13 onward, a team that reaches the target during a match becomes eligible in the following match; an eligible team must then win a WWCD to take the title. If no eligible team wins by Match 18, the highest points total wins. The model estimate remains a next-match WWCD estimate, not a calibrated tournament-champion probability.
6. Once the final roster has 16 confirmed teams, the page shows a Grand Finals score table for those teams and finish-based MVP candidates limited to those finalists. Scores start at zero before final scorecards arrive and update from published results. This is not an overall tournament MVP prediction or a calibrated official-award probability.
7. The selected-match player panel estimates the player most likely to lead the lobby in finishes from historical and recent finishes. The source does not provide validated per-match MVP award labels, so this is an MVP-style finish-leader estimate, not an award probability. `/mvp` shows the corresponding tournament/stage finish watchlist.
8. To make the classifier learn from newly added team match scores, run `python src/model_training.py` again, then restart Flask. Refreshing data updates prediction features but does not retrain the saved model.

An aggregate standings page without individual completed match scorecards cannot supply recent match-by-match form. The model's current objective is next-match win chance; it does not estimate tournament champion probability.

Sources:

- [eSportsAmaze BMSD standings](https://esportsamaze.com/bgmi/tournaments/battlegrounds-mobile-india-showdown-2026/standings)
- [eSportsAmaze BMSD match scorecards](https://esportsamaze.com/bgmi/tournaments/battlegrounds-mobile-india-showdown-2026/matches)
- [eSportsAmaze BMSD progression](https://esportsamaze.com/bgmi/tournaments/battlegrounds-mobile-india-showdown-2026/progression)
- [eSportsAmaze BMSD stage format](https://esportsamaze.com/bgmi/tournaments/battlegrounds-mobile-india-showdown-2026/format)
- [eSportsAmaze BMSD player statistics](https://esportsamaze.com/bgmi/tournaments/battlegrounds-mobile-india-showdown-2026/statistics)
- [BMSD Grand Finals Smash Rule details](https://liquipedia.net/pubgmobile/Battlegrounds_Mobile_India_Showdown/2026/Grand_Finals)
- [Liquipedia BMSD 2026](https://liquipedia.net/pubgmobile/Battlegrounds_Mobile_India_Showdown/2026)
- [KRAFTON India Esports official channel listing](https://krafton.in/esports/)
- [KRAFTON India Esports YouTube live page](https://www.youtube.com/@kraftonindiaesports/live)

## Project structure

```text
BGI_Win_Probability/
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_AUDIT.md
├── data/
│   ├── raw/
│   ├── bgis_combined.csv
│   ├── bmps_clean.csv
│   ├── bmsd.csv
│   ├── raw/bmsd_matches.csv
│   ├── raw/bmsd_fixtures.csv
│   ├── raw/bmsd_progression.csv
│   ├── raw/bmsd_stage_format.csv
│   ├── raw/bmsd_player_stats.csv
│   ├── raw/bmsd_player_match_stats.csv
│   └── processed/tournament_history.csv
├── notebooks/
│   └── BGMI_Winning_Probability_Analysis.ipynb
├── models/
│   ├── bgmi_winning_probability_model.joblib
│   ├── bmsd_player_finishes_model.joblib
│   ├── preprocessing_pipeline.joblib
│   └── feature_names.joblib
├── src/
│   ├── bmsd_source.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── player_finish_model.py
│   ├── train_player_finish_model.py
│   └── prediction.py
├── templates/
├── static/
├── tests/
└── reports/model_results/
    ├── model_comparison.csv
    ├── confusion_matrix.csv
    ├── test_predictions.csv
    └── diagnostic charts
```

Other existing data files, models, notebooks, and legacy Streamlit scripts have been kept in the workspace. The active Flask routes are `/`, `/dashboard`, `/live`, `/mvp`, `/predict`, `/compare`, `/leaderboard`, `/history`, `/team-analytics`, `/model-analytics`, and `/about`.

## Installation

Use Python 3.12. From the project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and review

Train the models, create evaluation reports, and save artifacts:

```powershell
python src/model_training.py
python src/train_player_finish_model.py
```

Open the analysis notebook:

```powershell
jupyter notebook notebooks/BGMI_Winning_Probability_Analysis.ipynb
```

## Run Flask

```powershell
python app.py
```

Open [http://127.0.0.1:5055](http://127.0.0.1:5055). The app uses port 5055 so it does not collide with another local Flask app already using the common port 5000. Flask loads the existing model artifacts; it does not retrain on startup. Retrain after adding completed tournament results when you intend the model itself to learn from them.

## JSON prediction API

`POST /api/predict` accepts a team and optional scorecard-derived feature values. Missing feature values are computed from completed history when that team is known.

```json
{
  "team": "Team SouL",
  "avg_points": 14.2,
  "avg_kills": 7.1,
  "win_rate": 0.07
}
```

The response includes `winning_probability` as a 0–1 value and `winning_probability_percentage` as a percentage. `POST /api/predict-match` accepts a `teams` array and returns probabilities normalized over that supplied match lobby.

## Tests

```powershell
pytest
```

The tests cover data validation, leakage-safe feature construction, saved model loading, Flask pages, prediction validation, and the JSON endpoints.

