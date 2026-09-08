import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Fakeddit training TSV
TRAIN_FILE = PROJECT_ROOT / "data" / "fakeddit" / "multimodal_train.tsv"

# Save selected sample information here
SAMPLE_DIR = PROJECT_ROOT / "data" / "fakeddit"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_SAMPLE_FILE = SAMPLE_DIR / "original_10k_image_training_set.csv"
NEW_SAMPLE_FILE = SAMPLE_DIR / "new_10k_image_training_set.csv"


# ============================================================
# FAKEDDIT COLUMNS
# ============================================================

columns = [
    "author",
    "clean_title",
    "created_utc",
    "domain",
    "hasImage",
    "id",
    "image_url",
    "linked_submission_id",
    "num_comments",
    "score",
    "subreddit",
    "title",
    "upvote_ratio",
    "2_way_label",
    "3_way_label",
    "6_way_label"
]


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading Fakeddit training dataset...")

train_df = pd.read_csv(
    TRAIN_FILE,
    sep="\t",
    header=None,
    names=columns,
    dtype=str,
    engine="python",
    on_bad_lines="skip"
)


# ============================================================
# CLEAN DATA
# ============================================================

# Remove accidental header rows
train_df = train_df[
    train_df["2_way_label"]
    .astype(str)
    .str.strip()
    .isin(["0", "1"])
].copy()


# Keep only rows with image URLs
train_df = train_df[
    train_df["image_url"].notna() &
    (train_df["image_url"].astype(str).str.strip() != "")
].copy()


# Convert labels to integers
train_df["2_way_label"] = train_df["2_way_label"].astype(int)


print("Usable rows:", len(train_df))

print("\nLabel distribution:")
print(train_df["2_way_label"].value_counts())


# ============================================================
# RECREATE FRIEND'S ORIGINAL 10K SAMPLE
# ============================================================

print("\nRecreating friend's original 10K sample...")

original_fake = train_df[
    train_df["2_way_label"] == 0
].sample(
    n=5000,
    random_state=42
)

original_true = train_df[
    train_df["2_way_label"] == 1
].sample(
    n=5000,
    random_state=42
)


original_10k = pd.concat(
    [original_fake, original_true],
    ignore_index=True
).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Save the original sample IDs
original_10k.to_csv(
    ORIGINAL_SAMPLE_FILE,
    index=False
)


print("Original 10K sample recreated.")
print("Original Fake:", (original_10k["2_way_label"] == 0).sum())
print("Original True:", (original_10k["2_way_label"] == 1).sum())


# ============================================================
# REMOVE ORIGINAL 10K
# ============================================================

print("\nRemoving original 10K from available data...")

original_ids = set(original_10k["id"].astype(str))

remaining_df = train_df[
    ~train_df["id"].astype(str).isin(original_ids)
].copy()


print("Remaining usable rows:", len(remaining_df))


# ============================================================
# SELECT NEW 10K SAMPLE
# ============================================================

print("\nSelecting NEW 10K sample...")
print("5,000 Fake + 5,000 True")
print("These images are different from the original 10K.")


new_fake = remaining_df[
    remaining_df["2_way_label"] == 0
].sample(
    n=5000,
    random_state=123
)

new_true = remaining_df[
    remaining_df["2_way_label"] == 1
].sample(
    n=5000,
    random_state=123
)


# Combine and shuffle
train_subset = pd.concat(
    [new_fake, new_true],
    ignore_index=True
).sample(
    frac=1,
    random_state=123
).reset_index(drop=True)


# ============================================================
# SAVE NEW SAMPLE
# ============================================================

train_subset.to_csv(
    NEW_SAMPLE_FILE,
    index=False
)


# ============================================================
# VERIFY THERE IS NO OVERLAP
# ============================================================

new_ids = set(train_subset["id"].astype(str))

overlap = original_ids.intersection(new_ids)


print("\n========================================")
print("NEW TRAINING SAMPLE CREATED")
print("========================================")

print("Total images:", len(train_subset))
print("Fake images:", (train_subset["2_way_label"] == 0).sum())
print("True images:", (train_subset["2_way_label"] == 1).sum())
print("Overlap with friend's original 10K:", len(overlap))

print("\nSaved:")
print("Original sample:", ORIGINAL_SAMPLE_FILE)
print("New sample:", NEW_SAMPLE_FILE)

print("\nFake = 0")
print("True = 1")

print("\nDone!")