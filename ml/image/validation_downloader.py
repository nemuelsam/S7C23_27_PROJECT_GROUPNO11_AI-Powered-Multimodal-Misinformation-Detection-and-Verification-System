import os
import time
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

VAL_TSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "fakeddit",
    "multimodal_validate.tsv"
)

VAL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "fakeddit",
    "images",
    "validate"
)

FAILED_VAL_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "fakeddit",
    "validate_image_download_failed.csv"
)

SAMPLES_PER_CLASS = 5000

os.makedirs(VAL_DIR, exist_ok=True)


# ============================================================
# READ VALIDATION DATASET
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

print("=" * 60)
print("Fakeddit Validation Image Downloader")
print("=" * 60)

print("\nReading validation dataset...")

val_df = pd.read_csv(
    VAL_TSV,
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

# Remove accidental/header rows
val_df = val_df[
    val_df["2_way_label"]
    .astype(str)
    .str.strip()
    .isin(["0", "1"])
].copy()

# Keep only rows with image URLs
val_df = val_df[
    val_df["image_url"].notna()
    &
    (val_df["image_url"].astype(str).str.strip() != "")
].copy()

# Convert labels to integers
val_df["2_way_label"] = val_df["2_way_label"].astype(int)

print("Usable validation rows:", len(val_df))

print("\nLabel distribution:")
print(val_df["2_way_label"].value_counts())


# ============================================================
# SELECT 10,000 VALIDATION IMAGES
# ============================================================

fake_df = val_df[
    val_df["2_way_label"] == 0
].sample(
    n=SAMPLES_PER_CLASS,
    random_state=42
)

true_df = val_df[
    val_df["2_way_label"] == 1
].sample(
    n=SAMPLES_PER_CLASS,
    random_state=42
)

val_subset = pd.concat(
    [fake_df, true_df],
    ignore_index=True
).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print("\n" + "=" * 60)
print("VALIDATION SUBSET")
print("=" * 60)

print("Selected validation images:", len(val_subset))

print("\nLabel distribution:")
print(val_subset["2_way_label"].value_counts())

print("\nFake = 0")
print("True = 1")


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_val_image(url, image_id, max_retries=3):

    save_path = os.path.join(
        VAL_DIR,
        f"{image_id}.jpg"
    )

    # Skip images already downloaded
    if os.path.exists(save_path):
        return True

    for attempt in range(max_retries):

        try:
            response = requests.get(
                url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            response.raise_for_status()

            image = Image.open(
                BytesIO(response.content)
            )

            # Convert to RGB so PNG/transparency/etc.
            # can safely be saved as JPEG
            if image.mode != "RGB":
                image = image.convert("RGB")

            image.save(
                save_path,
                format="JPEG",
                quality=95
            )

            return True

        except Exception:

            if attempt < max_retries - 1:
                time.sleep(1)

    return False


# ============================================================
# DOWNLOAD VALIDATION IMAGES
# ============================================================

failed_downloads = []

print("\n" + "=" * 60)
print("DOWNLOADING VALIDATION IMAGES")
print("=" * 60)

print("Image cache:")
print(VAL_DIR)

print("\nImages to process:", len(val_subset))


for _, row in tqdm(
    val_subset.iterrows(),
    total=len(val_subset),
    desc="Downloading validation images"
):

    success = download_val_image(
        row["image_url"],
        row["id"]
    )

    if not success:

        failed_downloads.append({
            "id": row["id"],
            "image_url": row["image_url"],
            "2_way_label": row["2_way_label"]
        })


# ============================================================
# SAVE FAILED DOWNLOADS
# ============================================================

failed_df = pd.DataFrame(
    failed_downloads
)

if len(failed_df) > 0:

    failed_df.to_csv(
        FAILED_VAL_FILE,
        index=False
    )


# ============================================================
# FINAL REPORT
# ============================================================

cached_images = 0

for image_id in val_subset["id"]:

    image_path = os.path.join(
        VAL_DIR,
        f"{image_id}.jpg"
    )

    if os.path.exists(image_path):
        cached_images += 1


print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print(f"\nCached images: {cached_images}")
print(f"Expected samples: {len(val_subset)}")
print(f"Failed downloads: {len(failed_downloads)}")

if len(failed_downloads) > 0:

    print("\nFailure log:")
    print(FAILED_VAL_FILE)

print("\nImages are stored locally.")
print("Existing images will be skipped on future runs.")