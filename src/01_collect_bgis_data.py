import requests
from bs4 import BeautifulSoup
import pandas as pd
import re


URL = "https://esportsamaze.in/BGMI/Tournaments/Battlegrounds_Mobile_India_Series_2026/Grand_Finals"


def collect_bgis_data():

    print("Downloading BGIS data...")

    response = requests.get(URL)

    if response.status_code != 200:
        print("Failed to download data.")
        print("Status code:", response.status_code)
        return

    soup = BeautifulSoup(response.text, "html.parser")

    tables = soup.find_all("table")

    print("Tables found:", len(tables))

    if len(tables) == 0:
        print("No table found.")
        return

    # Read the first table
    table = tables[0]

    rows = table.find_all("tr")

    data = []

    for row in rows:

        cells = row.find_all(["th", "td"])

        values = []

        for cell in cells:
            text = cell.get_text(" ", strip=True)
            values.append(text)

        if len(values) >= 29:
            data.append(values[:29])

    print("Rows extracted:", len(data))

    if len(data) == 0:
        print("No BGIS match data extracted.")
        return

    columns = [
        "rank",
        "team",
        "matches",
        "wwcd",
        "place",
        "elims",
        "matches2",
        "wins",
        "placement_points",
        "elim_points",
        "total_points",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "M8",
        "M9",
        "M10",
        "M11",
        "M12",
        "M13",
        "M14",
        "M15",
        "M16",
        "M17",
        "M18"
    ]

    df = pd.DataFrame(data, columns=columns)

    # Clean team names
    df["team"] = (
        df["team"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Save raw extracted data
    output_file = "data/bgis_raw.csv"

    df.to_csv(output_file, index=False)

    print()
    print("BGIS data collected successfully!")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print()
    print(df[["rank", "team", "M1", "M2", "M3", "M18"]].head(10))


if __name__ == "__main__":
    collect_bgis_data()