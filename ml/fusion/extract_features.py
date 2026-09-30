import os
import torch
import torch.nn as nn
import pandas as pd

from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MULTIMODAL FEATURE EXTRACTION")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# MODEL PATHS
# ============================================================

TEXT_MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "ml",
    "text",
    "distilbert_model_50k"
)

IMAGE_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ml",
    "image",
    "models",
    "efficientnet_b0_new10k_best.pth"
)


# ============================================================
# FUSION DATASET PATHS
# ============================================================

FUSION_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "fakeddit",
    "fusion"
)

TRAIN_FILE = os.path.join(
    FUSION_DIR,
    "fusion_train.csv"
)

VAL_FILE = os.path.join(
    FUSION_DIR,
    "fusion_val.csv"
)

HOLDOUT_FILE = os.path.join(
    FUSION_DIR,
    "fusion_holdout.csv"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "ml",
    "fusion",
    "features"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# BATCH SIZES
# ============================================================

TEXT_BATCH_SIZE = 8
IMAGE_BATCH_SIZE = 16
MAX_LENGTH = 128


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

required_files = [
    TEXT_MODEL_DIR,
    IMAGE_MODEL_PATH,
    TRAIN_FILE,
    VAL_FILE,
    HOLDOUT_FILE
]

for path in required_files:

    if os.path.exists(path):
        print("[OK]", path)

    else:
        print("[MISSING]", path)
        raise FileNotFoundError(
            f"\nRequired file/folder not found:\n{path}"
        )


# ============================================================
# LOAD DISTILBERT
# ============================================================

print("\n" + "=" * 70)
print("LOADING DISTILBERT 50K MODEL")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    TEXT_MODEL_DIR
)

text_model = AutoModelForSequenceClassification.from_pretrained(
    TEXT_MODEL_DIR
)

text_model = text_model.to(DEVICE)

text_model.eval()

# Freeze DistilBERT
for parameter in text_model.parameters():
    parameter.requires_grad = False

print("DistilBERT loaded successfully.")
print("Text feature dimension: 768")


# ============================================================
# LOAD EFFICIENTNET-B0
# ============================================================

print("\n" + "=" * 70)
print("LOADING EFFICIENTNET-B0")
print("=" * 70)

# Create the same architecture used during training
image_model = models.efficientnet_b0(
    weights=None
)

# Original classifier architecture
image_model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1280, 2)
)


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(
    IMAGE_MODEL_PATH,
    map_location=DEVICE
)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

        print(
            "Checkpoint type: training checkpoint"
        )

    else:

        state_dict = checkpoint

        print(
            "Checkpoint type: model state dictionary"
        )

else:

    state_dict = checkpoint

    print(
        "Checkpoint type: direct state dictionary"
    )


# Handle possible DataParallel prefix
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):

        key = key[7:]

    clean_state_dict[key] = value


image_model.load_state_dict(
    clean_state_dict,
    strict=True
)


# ============================================================
# REMOVE CLASSIFIER FOR FEATURE EXTRACTION
# ============================================================

# We do NOT want the final Fake/Real classifier output.
#
# We want the 1280-dimensional feature representation
# produced immediately before the classifier.

image_model.classifier = nn.Identity()

image_model = image_model.to(DEVICE)

image_model.eval()


# Freeze EfficientNet
for parameter in image_model.parameters():
    parameter.requires_grad = False


print("EfficientNet-B0 loaded successfully.")
print("Image feature dimension: 1280")


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

image_transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ]
)


# ============================================================
# TEXT FEATURE EXTRACTION
# ============================================================

def extract_text_features(df):

    print(
        "\nExtracting DistilBERT features..."
    )

    all_features = []

    texts = (
        df["clean_title"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    total = len(texts)

    with torch.no_grad():

        for start in range(
            0,
            total,
            TEXT_BATCH_SIZE
        ):

            end = min(
                start + TEXT_BATCH_SIZE,
                total
            )

            batch_texts = texts[start:end]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(DEVICE)
                for key, value in encoded.items()
            }

            # Get DistilBERT encoder output
            outputs = text_model.distilbert(
                **encoded
            )

            # First token representation
            features = outputs.last_hidden_state[:, 0, :]

            features = features.cpu()

            all_features.append(
                features
            )

            if (
                (end % 100 == 0)
                or end == total
            ):

                print(
                    f"Text: {end}/{total}"
                )

    return torch.cat(
        all_features,
        dim=0
    )


# ============================================================
# IMAGE FEATURE EXTRACTION
# ============================================================

def extract_image_features(df):

    print(
        "\nExtracting EfficientNet-B0 features..."
    )

    all_features = []

    image_paths = (
        df["image_path"]
        .astype(str)
        .tolist()
    )

    total = len(image_paths)

    with torch.no_grad():

        for start in range(
            0,
            total,
            IMAGE_BATCH_SIZE
        ):

            end = min(
                start + IMAGE_BATCH_SIZE,
                total
            )

            batch_paths = image_paths[
                start:end
            ]

            images = []

            valid_indices = []

            for index, image_path in enumerate(
                batch_paths
            ):

                try:

                    image = Image.open(
                        image_path
                    ).convert("RGB")

                    image = image_transform(
                        image
                    )

                    images.append(
                        image
                    )

                    valid_indices.append(
                        index
                    )

                except Exception as error:

                    print(
                        "\nWarning: Could not read image:"
                    )

                    print(
                        image_path
                    )

                    print(
                        "Error:",
                        error
                    )

            # This should normally never happen
            # because prepare_fusion_dataset.py
            # already checked the images.

            if len(images) == 0:

                raise RuntimeError(
                    "No valid images found in batch."
                )

            image_batch = torch.stack(
                images
            ).to(DEVICE)

            features = image_model(
                image_batch
            )

            features = features.cpu()

            # Normally all images should be valid.
            # If an image failed, the dataset alignment
            # would be affected, so stop instead of silently
            # producing incorrect text/image pairs.

            if len(valid_indices) != len(batch_paths):

                raise RuntimeError(
                    "\nAn image could not be loaded.\n"
                    "The text and image features must remain "
                    "perfectly aligned.\n"
                    f"Batch starting at: {start}"
                )

            all_features.append(
                features
            )

            if (
                (end % 100 == 0)
                or end == total
            ):

                print(
                    f"Image: {end}/{total}"
                )

    return torch.cat(
        all_features,
        dim=0
    )


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(
    csv_file,
    output_file,
    dataset_name
):

    print("\n" + "=" * 70)

    print(
        f"PROCESSING {dataset_name.upper()}"
    )

    print("=" * 70)

    df = pd.read_csv(
        csv_file
    )

    print(
        "Samples:",
        len(df)
    )

    # --------------------------------------------------------
    # Extract text features
    # --------------------------------------------------------

    text_features = extract_text_features(
        df
    )

    # --------------------------------------------------------
    # Extract image features
    # --------------------------------------------------------

    image_features = extract_image_features(
        df
    )

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    print(
        "\nChecking feature dimensions..."
    )

    print(
        "Text features:",
        text_features.shape
    )

    print(
        "Image features:",
        image_features.shape
    )

    if len(text_features) != len(df):

        raise RuntimeError(
            "Text feature count does not match dataset."
        )

    if len(image_features) != len(df):

        raise RuntimeError(
            "Image feature count does not match dataset."
        )

    # --------------------------------------------------------
    # Concatenate features
    # --------------------------------------------------------

    fusion_features = torch.cat(
        [
            text_features,
            image_features
        ],
        dim=1
    )

    print(
        "Combined features:",
        fusion_features.shape
    )

    # Expected:
    #
    # 768 + 1280 = 2048
    #

    if fusion_features.shape[1] != 2048:

        raise RuntimeError(
            f"Unexpected fusion feature dimension: "
            f"{fusion_features.shape[1]}. "
            f"Expected 2048."
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = torch.tensor(
        df["2_way_label"].values,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    data = {
        "ids": df["id"].astype(str).tolist(),

        "text_features": text_features,

        "image_features": image_features,

        "features": fusion_features,

        "labels": labels
    }

    torch.save(
        data,
        output_file
    )

    print(
        "\nSaved:",
        output_file
    )

    print(
        "Feature tensor:",
        fusion_features.shape
    )

    print(
        "Labels:",
        labels.shape
    )


# ============================================================
# PROCESS TRAINING DATA
# ============================================================

process_dataset(
    TRAIN_FILE,

    os.path.join(
        OUTPUT_DIR,
        "fusion_train_features.pt"
    ),

    "TRAIN"
)


# ============================================================
# PROCESS VALIDATION DATA
# ============================================================

process_dataset(
    VAL_FILE,

    os.path.join(
        OUTPUT_DIR,
        "fusion_val_features.pt"
    ),

    "VALIDATION"
)


# ============================================================
# PROCESS HOLDOUT DATA
# ============================================================

process_dataset(
    HOLDOUT_FILE,

    os.path.join(
        OUTPUT_DIR,
        "fusion_holdout_features.pt"
    ),

    "HOLDOUT"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print("\nFeature files created:")

print(
    os.path.join(
        OUTPUT_DIR,
        "fusion_train_features.pt"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "fusion_val_features.pt"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "fusion_holdout_features.pt"
    )
)

print("\nFeature dimensions:")
print("DistilBERT:     768")
print("EfficientNet:  1280")
print("Combined:       2048")

print(
    "\nBoth base models were frozen."
)

print(
    "Ready for MLP fusion training."
)

print("=" * 70)