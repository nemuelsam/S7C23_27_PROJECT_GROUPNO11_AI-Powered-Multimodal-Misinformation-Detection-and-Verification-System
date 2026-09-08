from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "fakeddit"
    / "images"
    / "train_new"
)


# ============================================================
# 2. Import the exact 10K dataset
# ============================================================

from image_dataset_preprocess import train_subset


# ============================================================
# 3. Image transformations
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. Dataset class
# ============================================================

class FakedditImageDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_id = str(row["id"]).strip()
        label = int(row["2_way_label"])

        image_path = self.image_dir / f"{image_id}.jpg"

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, image_id


# ============================================================
# 5. Check which images actually exist locally
# ============================================================

print("\n" + "=" * 60)
print("Checking local image cache")
print("=" * 60)

available_rows = []

missing_count = 0

for _, row in train_subset.iterrows():

    image_id = str(row["id"]).strip()

    image_path = IMAGE_DIR / f"{image_id}.jpg"

    if image_path.exists():

        available_rows.append(row)

    else:

        missing_count += 1


train_subset_available = pd.DataFrame(
    available_rows
).reset_index(drop=True)


print("\nExpected images:", len(train_subset))
print("Images available:", len(train_subset_available))
print("Images missing:", missing_count)


# ============================================================
# 6. Safety check
# ============================================================

if len(train_subset_available) == 0:

    raise RuntimeError(
        "\nNo cached images were found.\n"
        "Run image_downloader.py first."
    )


# ============================================================
# 7. Create dataset
# ============================================================

image_dataset = FakedditImageDataset(
    dataframe=train_subset_available,
    image_dir=IMAGE_DIR,
    transform=train_transform
)


# ============================================================
# 8. Create DataLoader
# ============================================================

BATCH_SIZE = 16

NUM_WORKERS = 0

train_loader = DataLoader(
    image_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 9. Print information
# ============================================================

print("\n" + "=" * 60)
print("IMAGE DATASET READY")
print("=" * 60)

print("\nDataset samples:", len(image_dataset))
print("Batch size:", BATCH_SIZE)
print("Number of workers:", NUM_WORKERS)

print("\nLabel distribution:")

print(
    train_subset_available["2_way_label"]
    .value_counts()
    .sort_index()
)

print("\nFake (0):",
      (train_subset_available["2_way_label"] == 0).sum())

print("True (1):",
      (train_subset_available["2_way_label"] == 1).sum())

print("\n✓ Local image dataset ready")
print("=" * 60)