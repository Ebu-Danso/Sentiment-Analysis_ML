from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "annotations.csv"
IMAGE_ROOT = PROJECT_ROOT / "data" / "raw"

print("Project root:", PROJECT_ROOT)
print("CSV path:", CSV_PATH)
print("Image root:", IMAGE_ROOT)

# -----------------------------
# Load annotations
# -----------------------------
df = pd.read_csv(CSV_PATH)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

# -----------------------------
# Basic checks
# -----------------------------
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

# -----------------------------
# Build full image path
# -----------------------------
df["full_image_path"] = df["image_path"].apply(lambda x: IMAGE_ROOT / x)

print("\nSample full paths:")
print(df[["image_path", "full_image_path"]].head())

# -----------------------------
# Check whether image files exist
# -----------------------------
df["file_exists"] = df["full_image_path"].apply(lambda x: x.exists())

print("\nFile existence check:")
print(df["file_exists"].value_counts())

if not df["file_exists"].all():
    print("\nSome files are missing!")
else:
    print("\nAll listed image files exist.")

# -----------------------------
# Show sample images (3 seconds each)
# -----------------------------
sample_df = df[df["file_exists"] == True].head(5)

for _, row in sample_df.iterrows():
    img = Image.open(row["full_image_path"]).convert("RGB")

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f"Sentiment: {row['sentiment']}")
    plt.axis("off")

    plt.pause(3)   # show for 3 seconds
    plt.close()

# -----------------------------
# Distribution Visualization
# -----------------------------
counts = df['sentiment'].value_counts()
percentages = df['sentiment'].value_counts(normalize=True) * 100

# Color mapping
color_map = {
    'positive': 'green',
    'neutral': 'gray',
    'negative': 'red'
}

bar_colors = [color_map[label] for label in counts.index]

plt.figure(figsize=(8, 5))
bars = plt.bar(counts.index, counts.values, color=bar_colors)

# Add percentage labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    pct = percentages.iloc[i]

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{pct:.1f}%",
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.title("Sentiment Distribution (Count & Percentage)")
plt.xlabel("Sentiment")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label='Positive'),
    Patch(facecolor='gray', label='Neutral'),
    Patch(facecolor='red', label='Negative')
]
plt.legend(handles=legend_elements)

plt.show()

# -----------------------------
# Print summary
# -----------------------------
print("\nDistribution Summary:")
for label in counts.index:
    print(f"{label}: {counts[label]} ({percentages[label]:.2f}%)")