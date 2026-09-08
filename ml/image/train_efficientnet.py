from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "image"
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Checkpoints
# ============================================================

# Friend's original trained checkpoint
RESUME_CHECKPOINT = (
    MODEL_DIR
    / "efficientnet_b0_best.pth"
)

# New checkpoints for this NEW 10K training experiment
NEW_BEST_CHECKPOINT = (
    MODEL_DIR
    / "efficientnet_b0_new10k_best.pth"
)

NEW_FINAL_CHECKPOINT = (
    MODEL_DIR
    / "efficientnet_b0_new10k_final.pth"
)


# ============================================================
# 3. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("EfficientNet-B0 Image Model Training")
print("=" * 60)

print("\nDevice:", device)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory / (1024 ** 3),
            2
        ),
        "GB"
    )


# ============================================================
# 4. Load datasets
# ============================================================

from image_dataset import train_subset_available
from image_dataset import train_loader

from validation_dataset import val_loader


# ============================================================
# 5. Class weights
# ============================================================

train_labels = (
    train_subset_available["2_way_label"]
    .astype(int)
    .values
)

class_counts = np.bincount(
    train_labels
)

class_weights = (
    len(train_labels)
    /
    (2 * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

print("\nClass counts:")

print(
    "Fake (0):",
    class_counts[0]
)

print(
    "True (1):",
    class_counts[1]
)


# ============================================================
# 6. Create EfficientNet-B0
# ============================================================

model = efficientnet_b0(
    weights=EfficientNet_B0_Weights.DEFAULT
)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1280, 2)
)

model = model.to(device)


# ============================================================
# 7. Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)


# ============================================================
# 8. Optimizer
# ============================================================

# IMPORTANT:
# Start a fresh optimizer for the NEW 10K dataset.
# We load the old model weights, but NOT the old optimizer state.

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=1e-4
)


# ============================================================
# 9. Scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)


# ============================================================
# 10. Mixed precision
# ============================================================

if device.type == "cuda":

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# 11. Load friend's trained model weights
# ============================================================

start_epoch = 0
best_f1 = 0.0

if RESUME_CHECKPOINT.exists():

    print("\n" + "=" * 60)
    print("Loading friend's trained model weights")
    print("=" * 60)

    checkpoint = torch.load(
        RESUME_CHECKPOINT,
        map_location=device
    )

    # Full checkpoint
    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print("\nModel weights loaded successfully.")

        print(
            "Previous validation F1:",
            checkpoint.get("val_f1", "N/A")
        )

        print(
            "\nStarting a NEW training phase."
        )

        print(
            "Old optimizer state will NOT be restored."
        )

    # Plain model state_dict
    else:

        model.load_state_dict(
            checkpoint
        )

        print(
            "\nModel weights loaded successfully."
        )

        print(
            "Optimizer state was not available."
        )

else:

    raise FileNotFoundError(
        "\nFriend's trained checkpoint was not found:\n"
        f"{RESUME_CHECKPOINT}\n\n"
        "Copy the .pth checkpoint into:\n"
        "ml/image/models/"
    )


# ============================================================
# 12. Training settings
# ============================================================

# Train the loaded model on the NEW 10K dataset.
EPOCHS = 8

PATIENCE = 3

epochs_without_improvement = 0

best_model_state = copy.deepcopy(
    model.state_dict()
)


# ============================================================
# 13. Training loop
# ============================================================

for epoch in range(EPOCHS):

    print("\n" + "=" * 60)

    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels, _ in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()


        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


    train_loss = (
        running_loss / total
    )

    train_acc = (
        correct / total
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    all_labels = []

    all_predictions = []

    val_loss_total = 0.0

    val_total = 0


    with torch.no_grad():

        for images, labels, _ in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            if device.type == "cuda":

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )

            else:

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            val_loss_total += (
                loss.item()
                * images.size(0)
            )

            val_total += labels.size(0)


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )


    val_loss = (
        val_loss_total / val_total
    )


    val_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    val_precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    val_recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    val_f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        val_f1
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nTrain Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_acc:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    print(
        f"Val Precision: {val_precision:.4f}"
    )

    print(
        f"Val Recall: {val_recall:.4f}"
    )

    print(
        f"Val F1: {val_f1:.4f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_f1 > best_f1:

        best_f1 = val_f1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch + 1,

                "val_f1":
                    val_f1,

                "val_accuracy":
                    val_accuracy
            },
            NEW_BEST_CHECKPOINT
        )

        epochs_without_improvement = 0

        print(
            "\n✓ New best model saved."
        )

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping."
        )

        break


# ============================================================
# 14. Restore best model
# ============================================================

model.load_state_dict(
    best_model_state
)


# ============================================================
# 15. Save final model
# ============================================================

torch.save(
    model.state_dict(),
    NEW_FINAL_CHECKPOINT
)


# ============================================================
# 16. Final output
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "\nBest validation F1:",
    round(best_f1, 4)
)

print(
    "\nFinal model:",
    NEW_FINAL_CHECKPOINT
)

print(
    "\nBest model:",
    NEW_BEST_CHECKPOINT
)

print("=" * 60)