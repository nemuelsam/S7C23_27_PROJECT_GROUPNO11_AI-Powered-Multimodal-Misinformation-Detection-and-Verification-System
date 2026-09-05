# ============================================================
# IMPROVED EFFICIENTNET-B0 TRAINING
# ============================================================

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. Calculate class weights
# ============================================================

train_labels = train_subset["2_way_label"].astype(int).values

class_counts = np.bincount(train_labels)

# Weight = total samples / (number of classes * class count)
class_weights = len(train_labels) / (
    2 * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

print("\nClass counts:")
print("Fake (0):", class_counts[0])
print("True (1):", class_counts[1])

print("\nClass weights:")
print(class_weights)


# ============================================================
# 3. Load pretrained EfficientNet-B0
# ============================================================

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(
    weights=weights
)

# Replace classifier
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1280, 2)
)

model = model.to(device)

print("\nEfficientNet-B0 loaded.")


# ============================================================
# 4. Loss function
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)


# ============================================================
# 5. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=1e-4
)


# ============================================================
# 6. Learning-rate scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)


# ============================================================
# 7. Mixed precision
# ============================================================

if device.type == "cuda":
    scaler = torch.amp.GradScaler("cuda")
else:
    scaler = None


# ============================================================
# 8. Training settings
# ============================================================

NUM_EPOCHS = 8
PATIENCE = 3

best_f1 = 0.0
best_epoch = 0
epochs_without_improvement = 0

BEST_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "efficientnet_b0_best.pth"
)

FINAL_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "efficientnet_b0_final.pth"
)


# ============================================================
# 9. Training loop
# ============================================================

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)
    print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
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

            scaler.scale(loss).backward()

            scaler.step(optimizer)

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
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_total = 0

    all_labels = []
    all_predictions = []

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

            val_loss += (
                loss.item() * images.size(0)
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


    val_loss = val_loss / val_total

    val_acc = accuracy_score(
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

    scheduler.step(val_f1)

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc * 100:.2f}%"
    )

    print(
        f"Val Precision: {val_precision * 100:.2f}%"
    )

    print(
        f"Val Recall: {val_recall * 100:.2f}%"
    )

    print(
        f"Val F1: {val_f1 * 100:.2f}%"
    )

    print(
        f"Learning Rate: {current_lr:.2e}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_f1 > best_f1:

        best_f1 = val_f1
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_f1": val_f1,
                "val_accuracy": val_acc
            },
            BEST_MODEL_PATH
        )

        print(
            f"✓ NEW BEST MODEL "
            f"(F1 = {val_f1 * 100:.2f}%)"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")

        break


# ============================================================
# 10. Save final model
# ============================================================

torch.save(
    model.state_dict(),
    FINAL_MODEL_PATH
)


# ============================================================
# 11. Final evaluation
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation F1: "
    f"{best_f1 * 100:.2f}%"
)

print(
    "\nBest model saved to:"
)

print(BEST_MODEL_PATH)

print(
    "\nFinal model saved to:"
)

print(FINAL_MODEL_PATH)


# ============================================================
# 12. Confusion matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)