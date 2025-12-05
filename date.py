from pathlib import Path
import pandas as pd

# Folder where this script (date.py) lives
BASE_DIR = Path(__file__).resolve().parent

# Point to the CSV inside DS410_final_project_datasets
input_file = BASE_DIR / "DS410_final_project_datasets" / "SPX_full_5min.csv"
output_file = BASE_DIR / "DS410_final_project_datasets" / "SPX_full_5min_with_datetime_parts.csv"

df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"])

df["year"]   = df["Date"].dt.year
df["month"]  = df["Date"].dt.month
df["day"]    = df["Date"].dt.day
df["hour"]   = df["Date"].dt.hour
df["minute"] = df["Date"].dt.minute
df["second"] = df["Date"].dt.second

df.to_csv(output_file, index=False)

print("Done! Saved to:", output_file)