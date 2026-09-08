from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_TSV = (
    PROJECT_ROOT
    / "data"
    / "fakeddit"
    / "multimodal_validate.tsv"
)

VAL_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "fakeddit"
    / "images"
    / "validate"
)

VAL_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Fakeddit columns
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
# 3. Load validation TSV
# ============================================================

if not VAL_TSV.exists():

    raise FileNotFoundError(
        f"\nValidation file not found:\n{VAL_TSV}"
    )


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
# 4. Filter valid rows
# ============================================================

val_df = val_df[
    val_df["2_way_label"]
    .astype(str)
    .str.strip()
    .isin(["0", "1"])
].copy()


val_df = val_df[
    val_df["image_url"].notna()
    &
    (
        val_df["image_url"]
        .astype(str)
        .str.strip()
        != ""
    )
].copy()


val_df["2_way_label"] = (
    val_df["2_way_label"].astype(int)
)


val_df = val_df.drop_duplicates(
    subset=["id"]
).reset_index(drop=True)


# ============================================================
# 5. Keep locally available images
# ============================================================

available_rows = []

for _, row in val_df.iterrows():

    image_id = str(row["id"]).strip()

    image_path = (
        VAL_IMAGE_DIR
        / f"{image_id}.jpg"
    )

    if image_path.exists():

        available_rows.append(row)


val_subset = pd.DataFrame(
    available_rows
).reset_index(drop=True)


print("\nValidation samples available:",
      len(val_subset))


# ============================================================
# 6. Transform
# ============================================================

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 7. Dataset
# ============================================================

class FakedditValidationDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        transform=None
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.image_dir = Path(image_dir)

        self.transform = transform


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_id = str(row["id"]).strip()

        label = int(row["2_way_label"])

        image_path = (
            self.image_dir
            / f"{image_id}.jpg"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, label, image_id


# ============================================================
# 8. DataLoader
# ============================================================

val_dataset = FakedditValidationDataset(
    dataframe=val_subset,
    image_dir=VAL_IMAGE_DIR,
    transform=val_transform
)


val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


print("✓ Validation DataLoader ready")

