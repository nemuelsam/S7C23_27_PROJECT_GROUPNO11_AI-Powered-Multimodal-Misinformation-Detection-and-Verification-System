from pathlib import Path
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_FILE = (
    PROJECT_ROOT
    / "data"
    / "fakeddit"
    / "new_10k_image_training_set.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "fakeddit"
    / "images"
    / "train_new"
)

FAILED_FILE = (
    PROJECT_ROOT
    / "data"
    / "fakeddit"
    / "image_download_failed_new.csv"
)

IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Load NEW 10K dataset
# ============================================================

print("=" * 60)
print("Fakeddit NEW 10K Image Downloader")
print("=" * 60)

print("\nLoading:")
print(SAMPLE_FILE)

train_subset = pd.read_csv(SAMPLE_FILE)

print(f"\nImages to process: {len(train_subset)}")
print(f"Fake images: {(train_subset['2_way_label'] == 0).sum()}")
print(f"True images: {(train_subset['2_way_label'] == 1).sum()}")

print("\nImage cache:")
print(IMAGE_DIR)


# ============================================================
# 3. Download settings
# ============================================================

headers = {
    "User-Agent": "Mozilla/5.0"
}

timeout = 15

failed = []


# ============================================================
# 4. Download images
# ============================================================

for _, row in tqdm(
    train_subset.iterrows(),
    total=len(train_subset),
    desc="Downloading new images"
):

    image_id = str(row["id"]).strip()
    image_url = str(row["image_url"]).strip()

    output_path = IMAGE_DIR / f"{image_id}.jpg"

    # Skip already downloaded images
    if output_path.exists():
        continue

    try:

        response = requests.get(
            image_url,
            headers=headers,
            timeout=timeout
        )

        response.raise_for_status()

        # Verify that the response is actually an image
        image = Image.open(
            BytesIO(response.content)
        )

        image = image.convert("RGB")

        # Save as JPEG
        image.save(
            output_path,
            format="JPEG"
        )

    except Exception as e:

        failed.append({
            "id": image_id,
            "image_url": image_url,
            "error": str(e)
        })


# ============================================================
# 5. Save failed downloads
# ============================================================

if failed:

    failed_df = pd.DataFrame(failed)

    failed_df.to_csv(
        FAILED_FILE,
        index=False
    )

    print(
        f"\nFailed downloads: {len(failed)}"
    )

    print(
        f"Failure log: {FAILED_FILE}"
    )

else:

    print("\n✓ No download failures")


# ============================================================
# 6. Final statistics
# ============================================================

downloaded = len(
    list(IMAGE_DIR.glob("*.jpg"))
)

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print(f"\nCached images: {downloaded}")
print(f"Expected samples: {len(train_subset)}")

print("\nImages are stored in:")
print(IMAGE_DIR)

print("\nExisting images will be skipped on future runs.")

print("=" * 60)