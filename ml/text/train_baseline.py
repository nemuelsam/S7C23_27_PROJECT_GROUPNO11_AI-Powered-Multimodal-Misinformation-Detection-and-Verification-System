import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Paths
TRAIN_FILE = "../../data/fakeddit/multimodal_train.tsv"
VALIDATE_FILE = "../../data/fakeddit/multimodal_validate.tsv"

# Load datasets
print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE, sep="\t")
validate_df = pd.read_csv(VALIDATE_FILE, sep="\t")

print("Train:", train_df.shape)
print("Validate:", validate_df.shape)

# Text and labels
X_train_text = train_df["clean_title"].fillna("")
y_train = train_df["2_way_label"]

X_validate_text = validate_df["clean_title"].fillna("")
y_validate = validate_df["2_way_label"]

# Convert text into TF-IDF features
print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2
)

X_train = vectorizer.fit_transform(X_train_text)
X_validate = vectorizer.transform(X_validate_text)

print("TF-IDF train shape:", X_train.shape)
print("TF-IDF validation shape:", X_validate.shape)

# Train Logistic Regression
print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

# Predict
print("\nEvaluating...")

y_pred = model.predict(X_validate)

# Results
accuracy = accuracy_score(y_validate, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_validate, y_pred))