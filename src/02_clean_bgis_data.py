import pandas as pd
import re


# Load raw BGIS data
df = pd.read_csv("data/bgis_raw.csv")

# Remove repeated header row
df = df[df["team"] != "TEAM"].copy()
match_data = []


# Process every team
for _, row in df.iterrows():

    team = row["team"]

    # Process M1 to M18
    for match_num in range(1, 19):

        match_id = f"M{match_num}"

        value = str(row[match_id]).strip()

        # Check if team won the match
        win = 1 if "🏆" in value else 0

        # Remove trophy
        value = value.replace("🏆", "").strip()

        # Extract numbers
        numbers = re.findall(r"\d+", value)

        if len(numbers) == 0:
            continue

        # First number = match points
        points = int(numbers[0])

        # Second number = kills
        if len(numbers) >= 2:
            kills = int(numbers[1])
        else:
            kills = 0

        match_data.append({
            "team": team,
            "match_id": match_id,
            "points": points,
            "kills": kills,
            "win": win
        })


# Create clean DataFrame
df_matches = pd.DataFrame(match_data)


# Validation
print("Rows:", len(df_matches))
print("Teams:", df_matches["team"].nunique())
print("Matches:", df_matches["match_id"].nunique())

print("\nColumns:")
print(df_matches.columns.tolist())

print("\nFirst 20 rows:")
print(df_matches.head(20))

print("\nMissing values:")
print(df_matches.isnull().sum())

print("\nDuplicate team + match:")
print(
    df_matches.duplicated(
        subset=["team", "match_id"]
    ).sum()
)

print("\nPoints range:")
print(
    df_matches["points"].min(),
    "to",
    df_matches["points"].max()
)

print("\nKills range:")
print(
    df_matches["kills"].min(),
    "to",
    df_matches["kills"].max()
)

print("\nWin distribution:")
print(df_matches["win"].value_counts())


# Save clean dataset
df_matches.to_csv(
    "data/bgis.csv",
    index=False
)

print("\nClean BGIS dataset saved successfully!")
print("File: data/bgis.csv")