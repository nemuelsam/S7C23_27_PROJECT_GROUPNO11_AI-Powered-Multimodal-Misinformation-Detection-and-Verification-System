import os
import time
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm

# Drive folder
TRAIN_DIR = "/content/drive/MyDrive/fakeddit_images/train"
os.makedirs(TRAIN_DIR, exist_ok=True)

# Files for tracking
FAILED_FILE = "/content/drive/MyDrive/fakeddit_images/train_failed.csv"

failed = []

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

def download_image(url, save_path):
    try:
        response = session.get(
            url,
            timeout=10,
            allow_redirects=True
        )

        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"

        # Check that the downloaded content is actually an image
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Save as JPEG
        image.save(save_path, "JPEG", quality=90)

        return True, None

    except Exception as e:
        return False, str(e)


print("Starting download...")
print("Total images:", len(train_subset))
print("Saving to:", TRAIN_DIR)

downloaded = 0
already_exists = 0
failed_count = 0

for _, row in tqdm(
    train_subset.iterrows(),
    total=len(train_subset),
    desc="Downloading training images"
):

    post_id = str(row["id"]).strip()
    url = str(row["image_url"]).strip()

    save_path = os.path.join(
        TRAIN_DIR,
        post_id + ".jpg"
    )

    # Skip if already downloaded
    if os.path.exists(save_path):
        already_exists += 1
        continue

    success, error = download_image(
        url,
        save_path
    )

    if success:
        downloaded += 1
    else:
        failed_count += 1

        failed.append({
            "id": post_id,
            "image_url": url,
            "label": int(row["2_way_label"]),
            "error": error
        })

    # Small delay to avoid hammering servers
    time.sleep(0.03)

# Save failure log
if failed:
    pd.DataFrame(failed).to_csv(
        FAILED_FILE,
        index=False
    )

print("\n==============================")
print("DOWNLOAD COMPLETE")
print("==============================")
print("Downloaded:", downloaded)
print("Already existed:", already_exists)
print("Failed:", failed_count)

print("\nImages currently in Drive:")

image_files = [
    f for f in os.listdir(TRAIN_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

print("Cached images:", len(image_files))

if failed:
    print("\nFailure log:")
    print(FAILED_FILE)