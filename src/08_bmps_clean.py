import pandas as pd

df = pd.read_csv("data/bmps_2026.csv")

print("Original Shape:", df.shape)
print(df.head())

df = df.drop_duplicates()

numeric_columns = [
    "match",
    "overall_match",
    "rank",
    "wwcd",
    "elim_points",
    "place_points",
    "total_score"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["team"] = df["team"].astype(str).str.strip()
df["map"] = df["map"].astype(str).str.strip()
df["match_winner"] = df["match_winner"].astype(str).str.strip()

df = df.dropna(
    subset=[
        "match",
        "overall_match",
        "rank",
        "team",
        "total_score",
        "match_winner"
    ]
)

df = df.sort_values(["match", "rank"])

df.to_csv("data/bmps_clean.csv", index=False)

print("Cleaned Shape:", df.shape)
print("BMPS cleaning completed.")
print("Saved: data/bmps_clean.csv")