import pandas as pd
from pathlib import Path

# Project root = folder containing ml, data, backend, frontend, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset location
DATA_DIR = PROJECT_ROOT / "data" / "fakeddit"

files = [
    "multimodal_train.tsv",
    "multimodal_validate.tsv",
    "multimodal_test_public.tsv"
]

for filename in files:
    file_path = DATA_DIR / filename

    print("\n" + "=" * 60)
    print(filename)
    print("=" * 60)

    df = pd.read_csv(file_path, sep="\t")

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\n2-way label distribution:")
    print(df["2_way_label"].value_counts())

    print("\nMissing clean_title:", df["clean_title"].isna().sum())