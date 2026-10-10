# Existing project audit

## What was present

- The workspace already contained a large Flask application, a smaller Streamlit entry point, data loaders, historical feature code, several training scripts, and dark esports templates.
- The data folder contained 18 complete BGIS matches and 18 complete BMPS matches, each represented as 16 team rows per match. `data/bmsd.csv` was empty at first inspection.
- Five notebook files in the original `Notebook/` folder were zero bytes.
- Available match statistics are points, kills, placement/rank, event identifiers, and winner flags. There are no player-level damage, assists, or survival measurements in the supplied historical CSVs.

## Problems found and corrections

- The root file named `BGIS_Grand_Finals_Matches_1_18_Combined.csv` contains BMPS-dated records that overlap `data/bmps_clean.csv`; it also has winner labels inconsistent with WWCD. The training loader excludes it.
- `data/bmps_features.csv` uses a rank-derived winner in some rows. BMPS labels now come from its explicit WWCD column.
- The supplied BGIS 2026 fragment has four records across two incomplete matches and is excluded from complete-lobby training.
- The former Streamlit live calculation used a fixed points/kills formula. The Flask app loads the calibrated classifier and has an explicit eSportsAmaze refresh route plus a manual CSV fallback.
- Earlier training code calibrated on its evaluation set. The replacement keeps the latest six complete matches as a chronological holdout and calibrates candidates on expanding, whole-match folds.
- Training rows are one team in one match. Every feature uses only earlier completed matches; rows from the same match remain together during evaluation.

## BMSD running-event data and live forecasts

The first local seed was one completed Upper Bracket Survival scorecard. The workspace now includes 66 completed BMSD match scorecards (1,056 team-match rows) through Upper Bracket Survival Match 12, player match-level finishes, progression routes, and the source's eight-stage format definitions. Source snapshots are stored under `data/raw/`; BMSD canonical rows are in `data/bmsd.csv`. Refresh is user-triggered rather than a continuous feed.

The published format defines Upper Bracket Survival as a 16-team single lobby with 18 matches; 12 are complete. The live page therefore offers Match 13 through Match 18 using the source-backed 16-team roster from Match 12, while explicitly leaving map and start time unspecified. Forecasts for later games use the latest completed history available at page refresh and do not feed predicted results back into later match features. All eight bracket sections show completed/total match progress, resolved teams, and unfilled qualification slots. Team prediction and player prediction are shown as separate sections for the selected bracket match.

Grand Finals is a distinct 16-team, 18-match Smash Rules stage. Once all 16 qualifiers are officially resolved, the live predictor can infer its unplayed match slots and move to the next match after each refreshed scorecard. After Match 12, the page calculates the Match Point target as the leader's points plus 10, lists teams already eligible before the selected match, and distinguishes the per-match WWCD forecast from tournament champion status. A team crossing the target during a match is eligible starting in the following match and must then win a WWCD. If no eligible team wins by Match 18, the highest points total is champion.

The live page also has a Grand Finals-only score table and a finish-based MVP candidate list, gated on all 16 finalists being known. The score table starts at zero before the first final scorecard and updates from actual final match rows; the player list is limited to the finalists and is clearly not an official MVP-award probability. Grand Finals is displayed without a group label.

The model was retrained after adding the 66-match BMSD snapshot. Its last six matches were held out chronologically. Random Forest was selected on conditional match log loss (2.7640), with Brier score 0.0585 and ROC-AUC 0.5444; it ranked the winner first in one of six matches. This is close to a uniform 16-team baseline and remains an important limitation.

A player-level Random Forest regressor was trained on 4,224 BMSD player-match rows to estimate individual match finishes. On the last-six-match holdout it achieved MAE 1.0919 and RMSE 1.4357, slightly improving MAE over the rolling-average baseline (1.1070) while R² remained negative. The source has no validated per-match MVP award label, so the estimate identifies likely finish leaders rather than claiming an MVP award probability.

## Prediction meaning and limits

The trained target is a team's chance to win one upcoming match (WWCD). It is not the chance to win the whole BMSD tournament. The historical sample is small, the BGIS winner column has no accompanying raw WWCD field, and some BMSD short tags cannot safely be mapped to older team spellings. Predictions are model estimates and should be read with those limitations in mind.
