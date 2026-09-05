import os
import time
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm

VAL_DIR = "/content/drive/MyDrive/fakeddit_images/validate"
os.makedirs(VAL_DIR, exist_ok=True)

FAILED_VAL_FILE = "/content/drive/MyDrive/fakeddit_images/validate_failed.csv"

failed_val = []

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

def download_val_image(url, save_path):
    try:
        response = session.get(
            url,
            timeout=10,
            allow_redirects=True
        )

        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        image.save(
            save_path,
            "JPEG",
            quality=90
        )

        return True, None

    except Exception as e:
        return False, str(e)


print("Starting validation download...")
print("Total images:", len(val_subset))
print("Saving to:", VAL_DIR)

downloaded = 0
already_exists = 0
failed_count = 0

for _, row in tqdm(
    val_subset.iterrows(),
    total=len(val_subset),
    desc="Downloading validation images"
):

    post_id = str(row["id"]).strip()
    url = str(row["image_url"]).strip()

    save_path = os.path.join(
        VAL_DIR,
        post_id + ".jpg"
    )

    if os.path.exists(save_path):
        already_exists += 1
        continue

    success, error = download_val_image(
        url,
        save_path
    )

    if success:
        downloaded += 1
    else:
        failed_count += 1

        failed_val.append({
            "id": post_id,
            "image_url": url,
            "label": int(row["2_way_label"]),
            "error": error
        })

    time.sleep(0.03)


# Save failure log
if failed_val:
    pd.DataFrame(failed_val).to_csv(
        FAILED_VAL_FILE,
        index=False
    )


image_files = [
    f for f in os.listdir(VAL_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    )
]

print("\n==============================")
print("VALIDATION DOWNLOAD COMPLETE")
print("==============================")
print("Downloaded:", downloaded)
print("Already existed:", already_exists)
print("Failed:", failed_count)
print("Cached validation images:", len(image_files))

if failed_val:
    print("\nFailure log:")
    print(FAILED_VAL_FILE)