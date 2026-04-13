from pathlib import Path
import yaml
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}


def load_config(config_path: str = "configs/base.yaml") -> dict:
    project_root = Path(__file__).resolve().parents[1]
    full_config_path = project_root / config_path

    with open(full_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def load_annotations(config: dict) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    annotations_path = project_root / config["data"]["annotations"]

    df = pd.read_csv(annotations_path)

    # Keep only needed columns for image-only project
    df = df[["image_path", "sentiment"]].copy()

    # Map labels to integers
    df["label"] = df["sentiment"].map(LABEL_MAP)

    # Build full image path
    df["full_image_path"] = df["image_path"].apply(
        lambda x: project_root / "data" / "raw" / x
    )

    # Keep only rows where image exists and label is valid
    df = df[df["full_image_path"].apply(lambda x: x.exists())].copy()
    df = df[df["label"].notna()].copy()

    # Reset index after filtering
    df.reset_index(drop=True, inplace=True)

    return df


def split_data(df: pd.DataFrame, val_split: float = 0.2, test_split: float = 0.1, seed: int = 42):
    train_df, temp_df = train_test_split(
        df,
        test_size=val_split + test_split,
        random_state=seed,
        stratify=df["label"]
    )

    relative_test_size = test_split / (val_split + test_split)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def get_transforms(config: dict):
    mean = config["data"]["mean"]
    std = config["data"]["std"]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, eval_transform


class SentimentImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]

        image_path = row["full_image_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloaders(config_path: str = "configs/base.yaml"):
    config = load_config(config_path)
    df = load_annotations(config)

    train_df, val_df, test_df = split_data(
        df,
        val_split=config["training"]["val_split"],
        test_split=0.1,
        seed=config["training"]["seed"]
    )

    train_transform, eval_transform = get_transforms(config)

    train_dataset = SentimentImageDataset(train_df, transform=train_transform)
    val_dataset = SentimentImageDataset(val_df, transform=eval_transform)
    test_dataset = SentimentImageDataset(test_df, transform=eval_transform)

    batch_size = config["training"]["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    images, labels = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Sample labels:", labels[:10])