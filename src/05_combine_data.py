import pandas as pd

# Load old BGIS data
old_df = pd.read_csv("data/bgis.csv")

# Load BGIS 2026 data
new_df = pd.read_csv("data/bgis_2026_updated.csv")

print("Old BGIS rows:", len(old_df))
print("BGIS 2026 rows:", len(new_df))

# Find common columns
common_columns = [
    "team",
    "match_id",
    "points",
    "kills",
    "win"
]

# Keep the main columns
old_data = old_df[common_columns]
new_data = new_df[common_columns]

# Add tournament information
old_data = old_data.copy()
old_data["tournament"] = "BGIS"
old_data["season"] = "Previous"
old_data["year"] = 2025
old_data["stage"] = "Tournament"

# Combine both datasets
combined_df = pd.concat(
    [old_data, new_data],
    ignore_index=True
)

# Save combined data
output_file = "data/bgis_combined.csv"

combined_df.to_csv(output_file, index=False)

print("\nData Combined Successfully!")
print("Total Rows:", len(combined_df))
print("\nSaved to:", output_file)

print("\nFirst 5 rows:")
print(combined_df.head())