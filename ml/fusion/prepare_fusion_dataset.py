import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

VALIDATE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "fakeddit",
    "multimodal_validate.tsv"
)

# Your validation images are stored in "validate"
IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "fakeddit",
    "images",
    "validate"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "fakeddit",
    "fusion"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("Loading Fakeddit validation data...")

df = pd.read_csv(
    VALIDATE_FILE,
    sep="\t"
)

print("Total validation rows:", len(df))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

print("\nColumns available:")
print(df.columns.tolist())


# ============================================================
# FILTER VALID TEXT AND LABEL
# ============================================================

df = df[
    df["clean_title"].notna()
].copy()

df = df[
    df["clean_title"].astype(str).str.strip() != ""
]

df = df[
    df["2_way_label"].isin([0, 1])
].copy()

print(
    "\nRows with valid text and label:",
    len(df)
)


# ============================================================
# CHECK LOCAL IMAGES
# ============================================================

print("\nChecking local validation images...")
print("Image directory:")
print(IMAGE_DIR)


def find_image(image_id):
    """
    Find the locally downloaded image for a Fakeddit ID.
    Supports common image extensions.
    """

    image_id = str(image_id)

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]

    for ext in extensions:

        image_path = os.path.join(
            IMAGE_DIR,
            image_id + ext
        )

        if os.path.exists(image_path):
            return image_path

    return None


df["image_path"] = df["id"].apply(find_image)


# Count available images
available_images = df["image_path"].notna().sum()
missing_images = df["image_path"].isna().sum()

print("Images found:", available_images)
print("Images missing:", missing_images)


# ============================================================
# KEEP ONLY COMPLETE MULTIMODAL SAMPLES
# ============================================================

fusion_df = df[
    df["image_path"].notna()
].copy()

print(
    "\nUsable multimodal samples:",
    len(fusion_df)
)


# ============================================================
# KEEP REQUIRED COLUMNS
# ============================================================

fusion_df = fusion_df[
    [
        "id",
        "clean_title",
        "image_path",
        "2_way_label"
    ]
].copy()


# ============================================================
# CHECK IF DATASET IS EMPTY
# ============================================================

if len(fusion_df) == 0:

    print("\nERROR: No usable multimodal samples found.")

    print("\nPlease check that images exist in:")
    print(IMAGE_DIR)

    raise SystemExit(1)


# ============================================================
# TRAIN / VALIDATION / HOLDOUT SPLIT
# ============================================================

print("\nSplitting fusion dataset...")

# 85% development data
# 15% holdout data

train_val, holdout = train_test_split(
    fusion_df,
    test_size=0.15,
    random_state=42,
    stratify=fusion_df["2_way_label"]
)


# Split remaining 85% into:
# approximately 70% total train
# approximately 15% total validation

train, val = train_test_split(
    train_val,
    test_size=0.17647,
    random_state=42,
    stratify=train_val["2_way_label"]
)


# ============================================================
# SAVE DATASETS
# ============================================================

train_file = os.path.join(
    OUTPUT_DIR,
    "fusion_train.csv"
)

val_file = os.path.join(
    OUTPUT_DIR,
    "fusion_val.csv"
)

holdout_file = os.path.join(
    OUTPUT_DIR,
    "fusion_holdout.csv"
)


train.to_csv(
    train_file,
    index=False
)

val.to_csv(
    val_file,
    index=False
)

holdout.to_csv(
    holdout_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("FUSION DATASET CREATED")
print("========================================")

print(
    "\nTotal usable multimodal samples:",
    len(fusion_df)
)

print(
    "Training samples:",
    len(train)
)

print(
    "Validation samples:",
    len(val)
)

print(
    "Holdout samples:",
    len(holdout)
)


print("\nTrain label distribution:")
print(
    train["2_way_label"].value_counts()
    .sort_index()
)

print("\nValidation label distribution:")
print(
    val["2_way_label"].value_counts()
    .sort_index()
)

print("\nHoldout label distribution:")
print(
    holdout["2_way_label"].value_counts()
    .sort_index()
)


# ============================================================
# VERIFY NO DUPLICATE IDs
# ============================================================

print("\nChecking split overlap...")

train_ids = set(train["id"].astype(str))
val_ids = set(val["id"].astype(str))
holdout_ids = set(holdout["id"].astype(str))

print(
    "Train ↔ Validation:",
    len(train_ids & val_ids)
)

print(
    "Train ↔ Holdout:",
    len(train_ids & holdout_ids)
)

print(
    "Validation ↔ Holdout:",
    len(val_ids & holdout_ids)
)


# ============================================================
# FILE LOCATIONS
# ============================================================

print("\nSaved files:")

print(
    "Train:",
    train_file
)

print(
    "Validation:",
    val_file
)

print(
    "Holdout:",
    holdout_file
)

print("\nDone.")