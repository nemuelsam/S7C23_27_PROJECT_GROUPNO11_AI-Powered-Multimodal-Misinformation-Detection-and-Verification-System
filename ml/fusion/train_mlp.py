import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 5

INPUT_DIM = 2048
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 128
NUM_CLASSES = 2

DROPOUT = 0.30

FEATURE_DIR = os.path.join(
    os.path.dirname(__file__),
    "features"
)

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "models"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


TRAIN_FEATURES = os.path.join(
    FEATURE_DIR,
    "fusion_train_features.pt"
)

VAL_FEATURES = os.path.join(
    FEATURE_DIR,
    "fusion_val_features.pt"
)

HOLDOUT_FEATURES = os.path.join(
    FEATURE_DIR,
    "fusion_holdout_features.pt"
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mlp_fusion_best.pth"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mlp_fusion_final.pth"
)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "mlp_fusion_results.txt"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MLP MULTIMODAL FUSION TRAINING")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking feature files...")

required_files = [
    TRAIN_FEATURES,
    VAL_FEATURES,
    HOLDOUT_FEATURES
]

for file_path in required_files:

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    print(f"[OK] {file_path}")


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading feature files...")

train_data = torch.load(
    TRAIN_FEATURES,
    map_location="cpu",
    weights_only=False
)

val_data = torch.load(
    VAL_FEATURES,
    map_location="cpu",
    weights_only=False
)

holdout_data = torch.load(
    HOLDOUT_FEATURES,
    map_location="cpu",
    weights_only=False
)


# ============================================================
# EXTRACT FEATURES AND LABELS
# ============================================================

X_train = train_data["features"].float()
y_train = train_data["labels"].long()

X_val = val_data["features"].float()
y_val = val_data["labels"].long()

X_holdout = holdout_data["features"].float()
y_holdout = holdout_data["labels"].long()


# ============================================================
# CHECK DIMENSIONS
# ============================================================

print("\nFeature dimensions:")

print(f"Train:   {X_train.shape}")
print(f"Val:     {X_val.shape}")
print(f"Holdout: {X_holdout.shape}")

if X_train.shape[1] != INPUT_DIM:
    raise ValueError(
        f"Expected {INPUT_DIM} features, "
        f"but received {X_train.shape[1]}"
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nDataset sizes:")

print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Holdout samples:    {len(X_holdout)}")

print("\nTraining label distribution:")
print(
    f"Fake (0): {int((y_train == 0).sum())}"
)
print(
    f"True (1): {int((y_train == 1).sum())}"
)

print("\nValidation label distribution:")
print(
    f"Fake (0): {int((y_val == 0).sum())}"
)
print(
    f"True (1): {int((y_val == 1).sum())}"
)

print("\nHoldout label distribution:")
print(
    f"Fake (0): {int((y_holdout == 0).sum())}"
)
print(
    f"True (1): {int((y_holdout == 1).sum())}"
)


# ============================================================
# FEATURE STANDARDIZATION
# ============================================================
# IMPORTANT:
# Mean and standard deviation are calculated ONLY from
# the training data.
#
# Validation and holdout data use the same training statistics.
# This prevents data leakage.
# ============================================================

print("\nStandardizing features...")

train_mean = X_train.mean(dim=0)
train_std = X_train.std(dim=0)

# Avoid division by zero
train_std[train_std < 1e-6] = 1.0

X_train = (X_train - train_mean) / train_std
X_val = (X_val - train_mean) / train_std
X_holdout = (X_holdout - train_mean) / train_std

print("Feature standardization complete.")


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)

holdout_dataset = TensorDataset(
    X_holdout,
    y_holdout
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

holdout_loader = DataLoader(
    holdout_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MLP FUSION MODEL
# ============================================================

class MLPFusion(nn.Module):

    def __init__(
        self,
        input_dim=2048,
        hidden_dim_1=512,
        hidden_dim_2=128,
        num_classes=2,
        dropout=0.30
    ):

        super().__init__()

        self.network = nn.Sequential(

            # 2048 -> 512
            nn.Linear(
                input_dim,
                hidden_dim_1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                hidden_dim_1
            ),

            nn.Dropout(
                dropout
            ),

            # 512 -> 128
            nn.Linear(
                hidden_dim_1,
                hidden_dim_2
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                hidden_dim_2
            ),

            nn.Dropout(
                dropout
            ),

            # 128 -> 2
            nn.Linear(
                hidden_dim_2,
                num_classes
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# CREATE MODEL
# ============================================================

model = MLPFusion(
    input_dim=INPUT_DIM,
    hidden_dim_1=HIDDEN_DIM_1,
    hidden_dim_2=HIDDEN_DIM_2,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)

model = model.to(device)


print("\n" + "=" * 70)
print("MLP ARCHITECTURE")
print("=" * 70)

print(model)


# ============================================================
# CLASS WEIGHTS
# ============================================================
# Handles the small class imbalance in the fusion training set.
# ============================================================

class_counts = torch.bincount(
    y_train,
    minlength=NUM_CLASSES
).float()

class_weights = (
    len(y_train)
    /
    (
        NUM_CLASSES * class_counts
    )
)

class_weights = class_weights.to(device)

print("\nClass counts:")
print(
    f"Fake (0): {int(class_counts[0])}"
)
print(
    f"True (1): {int(class_counts[1])}"
)

print("\nClass weights:")
print(class_weights)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(
                outputs,
                labels
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = (
        total_loss / total_samples
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    return (
        average_loss,
        accuracy,
        precision,
        recall,
        f1,
        all_labels,
        all_predictions
    )


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING MLP TRAINING")
print("=" * 70)

best_val_f1 = 0.0
best_epoch = 0
epochs_without_improvement = 0

history = []


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            features
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * labels.size(0)
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

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
        _,
        _
    ) = evaluate(
        model,
        val_loader
    )

    # --------------------------------------------------------
    # LEARNING RATE
    # --------------------------------------------------------

    scheduler.step(
        val_f1
    )

    current_lr = optimizer.param_groups[0]["lr"]

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print("=" * 70)

    print(
        f"Train Loss:       {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:   {train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss:         {val_loss:.4f}"
    )

    print(
        f"Val Accuracy:     {val_accuracy * 100:.2f}%"
    )

    print(
        f"Val Precision:    {val_precision * 100:.2f}%"
    )

    print(
        f"Val Recall:       {val_recall * 100:.2f}%"
    )

    print(
        f"Val F1:           {val_f1 * 100:.2f}%"
    )

    print(
        f"Learning Rate:    {current_lr:.2e}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1
    })

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "input_dim":
                    INPUT_DIM,

                "hidden_dim_1":
                    HIDDEN_DIM_1,

                "hidden_dim_2":
                    HIDDEN_DIM_2,

                "num_classes":
                    NUM_CLASSES,

                "dropout":
                    DROPOUT,

                "train_mean":
                    train_mean,

                "train_std":
                    train_std,

                "best_val_f1":
                    best_val_f1,

                "best_epoch":
                    best_epoch
            },
            BEST_MODEL_PATH
        )

        print(
            f"✓ NEW BEST MODEL "
            f"(F1 = {best_val_f1 * 100:.2f}%)"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# SAVE FINAL MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "input_dim":
            INPUT_DIM,

        "hidden_dim_1":
            HIDDEN_DIM_1,

        "hidden_dim_2":
            HIDDEN_DIM_2,

        "num_classes":
            NUM_CLASSES,

        "dropout":
            DROPOUT,

        "train_mean":
            train_mean,

        "train_std":
            train_std
    },
    FINAL_MODEL_PATH
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MLP MODEL")
print("=" * 70)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

print(
    f"Best epoch: {checkpoint['best_epoch']}"
)

print(
    f"Best validation F1: "
    f"{checkpoint['best_val_f1'] * 100:.2f}%"
)


# ============================================================
# FINAL VALIDATION EVALUATION
# ============================================================

(
    val_loss,
    val_accuracy,
    val_precision,
    val_recall,
    val_f1,
    val_labels,
    val_predictions
) = evaluate(
    model,
    val_loader
)


# ============================================================
# HOLDOUT EVALUATION
# ============================================================
# This is the first time the MLP sees the holdout set.
# ============================================================

print("\n" + "=" * 70)
print("FINAL HOLDOUT EVALUATION")
print("=" * 70)

(
    holdout_loss,
    holdout_accuracy,
    holdout_precision,
    holdout_recall,
    holdout_f1,
    holdout_labels,
    holdout_predictions
) = evaluate(
    model,
    holdout_loader
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

holdout_cm = confusion_matrix(
    holdout_labels,
    holdout_predictions
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MLP FUSION RESULTS")
print("=" * 70)

print(
    f"Best epoch:          {best_epoch}"
)

print(
    f"Best validation F1:  "
    f"{best_val_f1 * 100:.2f}%"
)

print("\nValidation:")
print(
    f"Accuracy:  {val_accuracy * 100:.2f}%"
)
print(
    f"Precision: {val_precision * 100:.2f}%"
)
print(
    f"Recall:    {val_recall * 100:.2f}%"
)
print(
    f"F1-score:  {val_f1 * 100:.2f}%"
)

print("\nHoldout:")
print(
    f"Accuracy:  {holdout_accuracy * 100:.2f}%"
)
print(
    f"Precision: {holdout_precision * 100:.2f}%"
)
print(
    f"Recall:    {holdout_recall * 100:.2f}%"
)
print(
    f"F1-score:  {holdout_f1 * 100:.2f}%"
)

print("\nHoldout Confusion Matrix:")
print(holdout_cm)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MLP MULTIMODAL FUSION RESULTS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Project:\n"
        "AI-Powered Multimodal "
        "Misinformation Detection "
        "and Verification System\n\n"
    )

    f.write(
        "Fusion method:\n"
        "Feature-level early fusion\n\n"
    )

    f.write(
        "Base models:\n"
        "DistilBERT 50K\n"
        "EfficientNet-B0 new10k best\n\n"
    )

    f.write(
        "Base model feature dimensions:\n"
        "DistilBERT: 768\n"
        "EfficientNet-B0: 1280\n"
        "Combined: 2048\n\n"
    )

    f.write(
        "MLP architecture:\n"
        "2048 -> 512 -> 128 -> 2\n"
        "ReLU activation\n"
        "Batch Normalization\n"
        "Dropout: 0.30\n\n"
    )

    f.write(
        "Base models were frozen during MLP training.\n\n"
    )

    f.write(
        "Dataset:\n"
        f"Training samples: {len(X_train)}\n"
        f"Validation samples: {len(X_val)}\n"
        f"Holdout samples: {len(X_holdout)}\n\n"
    )

    f.write(
        "Training configuration:\n"
        f"Batch size: {BATCH_SIZE}\n"
        f"Learning rate: {LEARNING_RATE}\n"
        f"Weight decay: {WEIGHT_DECAY}\n"
        f"Maximum epochs: {EPOCHS}\n"
        f"Early stopping patience: {PATIENCE}\n\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    f.write(
        "EPOCH RESULTS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    for result in history:

        f.write(
            f"Epoch {result['epoch']}:\n"
        )

        f.write(
            f"Train Loss: "
            f"{result['train_loss']:.4f}\n"
        )

        f.write(
            f"Train Accuracy: "
            f"{result['train_accuracy'] * 100:.2f}%\n"
        )

        f.write(
            f"Val Loss: "
            f"{result['val_loss']:.4f}\n"
        )

        f.write(
            f"Val Accuracy: "
            f"{result['val_accuracy'] * 100:.2f}%\n"
        )

        f.write(
            f"Val Precision: "
            f"{result['val_precision'] * 100:.2f}%\n"
        )

        f.write(
            f"Val Recall: "
            f"{result['val_recall'] * 100:.2f}%\n"
        )

        f.write(
            f"Val F1: "
            f"{result['val_f1'] * 100:.2f}%\n\n"
        )

    f.write(
        "=" * 70 + "\n"
    )

    f.write(
        "FINAL RESULTS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Best epoch: {best_epoch}\n"
    )

    f.write(
        f"Best validation F1: "
        f"{best_val_f1 * 100:.2f}%\n\n"
    )

    f.write(
        "Validation Results:\n"
    )

    f.write(
        f"Accuracy: "
        f"{val_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Precision: "
        f"{val_precision * 100:.2f}%\n"
    )

    f.write(
        f"Recall: "
        f"{val_recall * 100:.2f}%\n"
    )

    f.write(
        f"F1-score: "
        f"{val_f1 * 100:.2f}%\n\n"
    )

    f.write(
        "Holdout Results:\n"
    )

    f.write(
        f"Accuracy: "
        f"{holdout_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Precision: "
        f"{holdout_precision * 100:.2f}%\n"
    )

    f.write(
        f"Recall: "
        f"{holdout_recall * 100:.2f}%\n"
    )

    f.write(
        f"F1-score: "
        f"{holdout_f1 * 100:.2f}%\n\n"
    )

    f.write(
        "Holdout Confusion Matrix:\n"
    )

    f.write(
        str(holdout_cm)
        + "\n\n"
    )

    f.write(
        "Model files:\n"
    )

    f.write(
        f"Best model: {BEST_MODEL_PATH}\n"
    )

    f.write(
        f"Final model: {FINAL_MODEL_PATH}\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("MLP TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest model saved to:\n"
    f"{BEST_MODEL_PATH}"
)

print(
    f"\nFinal model saved to:\n"
    f"{FINAL_MODEL_PATH}"
)

print(
    f"\nResults saved to:\n"
    f"{RESULT_FILE}"
)

print("\nReady for multimodal prediction.")