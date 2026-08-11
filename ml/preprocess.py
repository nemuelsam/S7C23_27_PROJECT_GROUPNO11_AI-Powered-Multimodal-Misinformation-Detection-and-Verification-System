from pathlib import Path
import pandas as pd


# ============================================================
# Paths
# ============================================================

# Project root = one level above the ml folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "fakeddit"

TRAIN_FILE = DATA_DIR / "multimodal_train.tsv"
VALIDATE_FILE = DATA_DIR / "multimodal_validate.tsv"
TEST_FILE = DATA_DIR / "multimodal_test_public.tsv"


# ============================================================
# Load dataset
# ============================================================

def load_dataset(file_path):
    """Load a Fakeddit TSV file."""
    return pd.read_csv(file_path, sep="\t")


# ============================================================
# Basic text preprocessing
# ============================================================

def clean_text(text):
    """Basic cleaning of text."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.strip()
    text = " ".join(text.split())

    return text


def preprocess_dataframe(df):
    """Prepare a Fakeddit dataframe for further processing."""

    # Use clean_title as the main text feature
    df["clean_title"] = df["clean_title"].apply(clean_text)

    # Remove rows where the text is empty
    df = df[df["clean_title"].str.len() > 0].copy()

    # Make sure the binary label is numeric
    df["2_way_label"] = pd.to_numeric(
        df["2_way_label"],
        errors="coerce"
    )

    # Remove rows with missing labels
    df = df.dropna(subset=["2_way_label"])

    # Convert label to integer
    df["2_way_label"] = df["2_way_label"].astype(int)

    return df


# ============================================================
# Main preprocessing
# ============================================================

def main():

    print("Loading Fakeddit datasets...")

    train_df = load_dataset(TRAIN_FILE)
    validate_df = load_dataset(VALIDATE_FILE)
    test_df = load_dataset(TEST_FILE)

    print(f"Train:    {train_df.shape}")
    print(f"Validate: {validate_df.shape}")
    print(f"Test:     {test_df.shape}")

    print("\nPreprocessing...")

    train_df = preprocess_dataframe(train_df)
    validate_df = preprocess_dataframe(validate_df)
    test_df = preprocess_dataframe(test_df)

    print("\nAfter preprocessing:")
    print(f"Train:    {train_df.shape}")
    print(f"Validate: {validate_df.shape}")
    print(f"Test:     {test_df.shape}")

    print("\nLabel distribution:")

    print("\nTrain:")
    print(train_df["2_way_label"].value_counts())

    print("\nValidate:")
    print(validate_df["2_way_label"].value_counts())

    print("\nTest:")
    print(test_df["2_way_label"].value_counts())

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()