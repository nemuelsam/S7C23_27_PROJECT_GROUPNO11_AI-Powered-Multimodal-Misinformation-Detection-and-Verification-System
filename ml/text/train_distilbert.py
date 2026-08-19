import os
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128

OUTPUT_DIR = "ml/text/distilbert_model"
RESULTS_DIR = "results"


# ============================================================
# Find project root
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)


# ============================================================
# Dataset paths
# ============================================================

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "fakeddit",
    "multimodal_train.tsv"
)

VALIDATE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "fakeddit",
    "multimodal_validate.tsv"
)


# ============================================================
# Load datasets
# ============================================================
print("PROJECT_ROOT:",PROJECT_ROOT)
print("TRAIN_FILE:",TRAIN_FILE)
print("VALIDATE_FILE:",VALIDATE_FILE)
print("Train exists:",os.path.exists(TRAIN_FILE))
print("Validate exists:",os.path.exists(VALIDATE_FILE))

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE,sep="\t")
validate_df = pd.read_csv(VALIDATE_FILE,sep="\t")

# Use a smaller subset for the first training run
train_df = train_df.sample(
    n=10000,
    random_state=42
).reset_index(drop=True)

print("Train:", train_df.shape)
print("Validate:", validate_df.shape)


# ============================================================
# Select required columns
# ============================================================

TEXT_COLUMN = "clean_title"
LABEL_COLUMN = "2_way_label"

train_df = train_df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
validate_df = validate_df[[TEXT_COLUMN, LABEL_COLUMN]].copy()


# ============================================================
# Remove missing values
# ============================================================

train_df = train_df.dropna(
    subset=[TEXT_COLUMN, LABEL_COLUMN]
)

validate_df = validate_df.dropna(
    subset=[TEXT_COLUMN, LABEL_COLUMN]
)


# ============================================================
# Make sure labels are integers
# ============================================================

train_df[LABEL_COLUMN] = train_df[LABEL_COLUMN].astype(int)
validate_df[LABEL_COLUMN] = validate_df[LABEL_COLUMN].astype(int)


# ============================================================
# Rename columns for Hugging Face Dataset
# ============================================================

train_df = train_df.rename(
    columns={
        TEXT_COLUMN: "text",
        LABEL_COLUMN: "label"
    }
)

validate_df = validate_df.rename(
    columns={
        TEXT_COLUMN: "text",
        LABEL_COLUMN: "label"
    }
)


# ============================================================
# Convert Pandas → Hugging Face Dataset
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

validate_dataset = Dataset.from_pandas(
    validate_df,
    preserve_index=False
)


# ============================================================
# Load tokenizer
# ============================================================

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# Tokenization
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )


print("Tokenizing training data...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

print("Tokenizing validation data...")

validate_dataset = validate_dataset.map(
    tokenize_function,
    batched=True
)


# ============================================================
# Load DistilBERT model
# ============================================================

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# ============================================================
# Evaluation metrics
# ============================================================

def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    predictions = predictions.argmax(axis=-1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# Training configuration
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=500,

    learning_rate=2e-5,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    gradient_accumulation_steps=4,

    num_train_epochs=2,

    weight_decay=0.01,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    fp16=torch.cuda.is_available(),

    report_to="none"
)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validate_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics
)


# ============================================================
# Train
# ============================================================

print("\n========================================")
print("Starting DistilBERT training...")
print("========================================\n")

trainer.train()


# ============================================================
# Final evaluation
# ============================================================

print("\n========================================")
print("Evaluating DistilBERT...")
print("========================================\n")

results = trainer.evaluate()

print("DistilBERT Results:")

for key, value in results.items():

    print(f"{key}: {value}")


# ============================================================
# Save model
# ============================================================

print("\nSaving model...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


# ============================================================
# Save results
# ============================================================

os.makedirs(
    os.path.join(PROJECT_ROOT, RESULTS_DIR),
    exist_ok=True
)

results_file = os.path.join(
    PROJECT_ROOT,
    RESULTS_DIR,
    "distilbert_results.txt"
)

with open(results_file, "w") as f:

    f.write("DistilBERT Text Model Results\n")
    f.write("=============================\n\n")

    for key, value in results.items():

        f.write(f"{key}: {value}\n")


print("\nTraining completed successfully.")

print(
    f"Model saved to: {OUTPUT_DIR}"
)

print(
    f"Results saved to: {results_file}"
)