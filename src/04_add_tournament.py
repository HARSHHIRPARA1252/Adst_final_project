import pandas as pd

# Load BGIS 2026 data
file_path = "data/bgis_2026.csv"

df = pd.read_csv(file_path)

# Add tournament information
df["tournament"] = "BGIS"
df["season"] = "2026"
df["year"] = 2026
df["stage"] = "Tournament"

# Save updated data
output_file = "data/bgis_2026_updated.csv"

df.to_csv(output_file, index=False)

print("BGIS 2026 Tournament Data Added Successfully!")
print()
print(df.head())
print()
print("Total Rows:", len(df))
print("Saved to:", output_file)