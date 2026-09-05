import pandas as pd

columns = [
    "author", "clean_title", "created_utc", "domain", "hasImage", "id",
    "image_url", "linked_submission_id", "num_comments", "score",
    "subreddit", "title", "upvote_ratio", "2_way_label",
    "3_way_label", "6_way_label"
]

train_df = pd.read_csv(
    "/content/multimodal_train.tsv",
    sep="\t",
    header=None,
    names=columns,
    dtype=str,
    engine="python",
    on_bad_lines="skip"
)

# Remove accidental header rows
train_df = train_df[
    train_df["2_way_label"].astype(str).str.strip().isin(["0", "1"])
].copy()

# Keep only rows with image URLs
train_df = train_df[
    train_df["image_url"].notna() &
    (train_df["image_url"].astype(str).str.strip() != "")
].copy()

# Convert label to integer
train_df["2_way_label"] = train_df["2_way_label"].astype(int)

print("Usable rows:", len(train_df))
print("\nLabel distribution:")
print(train_df["2_way_label"].value_counts())

# Select 5,000 Fake + 5,000 True
fake_df = train_df[
    train_df["2_way_label"] == 0
].sample(n=5000, random_state=42)

true_df = train_df[
    train_df["2_way_label"] == 1
].sample(n=5000, random_state=42)

# Combine and shuffle
train_subset = pd.concat(
    [fake_df, true_df],
    ignore_index=True
).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nSelected training images:", len(train_subset))
print("\nSelected label distribution:")
print(train_subset["2_way_label"].value_counts())

print("\nFake = 0")
print("True = 1")