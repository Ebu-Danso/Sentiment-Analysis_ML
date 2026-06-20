"""Batch demo - show multiple predictions in a grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from .dataset import ImageSentimentDataset, INDEX_TO_SENTIMENT
    from .model import build_resnet18, get_device
    from .utils import load_config
except ImportError:
    from dataset import ImageSentimentDataset, INDEX_TO_SENTIMENT
    from model import build_resnet18, get_device
    from utils import load_config


def create_batch_demo(model, device, transform, dataset, data_root, num_samples=12, filter_type="all"):
    """Create a grid visualization of predictions."""
    
    model.eval()
    
    # Get indices based on filter
    indices = list(range(len(dataset)))
    
    if filter_type == "correct":
        filtered_indices = []
        for idx in indices:
            _, true_label = dataset[idx]
            image_tensor, _ = dataset[idx]
            image_tensor = image_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                pred_idx = model(image_tensor).argmax(dim=1).item()
            
            if INDEX_TO_SENTIMENT[pred_idx] == INDEX_TO_SENTIMENT[true_label]:
                filtered_indices.append(idx)
            if len(filtered_indices) >= num_samples:
                break
        indices = filtered_indices
    
    elif filter_type == "incorrect":
        filtered_indices = []
        for idx in indices:
            _, true_label = dataset[idx]
            image_tensor, _ = dataset[idx]
            image_tensor = image_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                pred_idx = model(image_tensor).argmax(dim=1).item()
            
            if INDEX_TO_SENTIMENT[pred_idx] != INDEX_TO_SENTIMENT[true_label]:
                filtered_indices.append(idx)
            if len(filtered_indices) >= num_samples:
                break
        indices = filtered_indices
    else:
        indices = indices[:num_samples]
    
    # Create grid
    grid_size = int(np.ceil(np.sqrt(len(indices))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    axes = axes.flatten()
    
    color_map = {"negative": "#d62728", "neutral": "#ff7f0e", "positive": "#2ca02c"}
    
    for plot_idx, data_idx in enumerate(indices):
        image_tensor, true_label = dataset[data_idx]
        image_tensor_input = image_tensor.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor_input)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            pred_idx = outputs.argmax(dim=1).item()
        
        pred_sentiment = INDEX_TO_SENTIMENT[pred_idx]
        true_sentiment = INDEX_TO_SENTIMENT[true_label]
        confidence = probs[pred_idx]
        is_correct = pred_sentiment == true_sentiment
        
        # Get image for display
        image_path = dataset.annotations.iloc[data_idx]["image_path"]
        full_path = dataset.root_dir / image_path
        image = Image.open(full_path).convert("RGB")
        
        # Plot
        ax = axes[plot_idx]
        ax.imshow(image)
        
        # Title with results
        title_color = "green" if is_correct else "red"
        title = f"True: {true_sentiment}\nPred: {pred_sentiment}\nConf: {confidence:.0%}"
        ax.set_title(title, fontsize=10, fontweight='bold', color=title_color)
        ax.axis("off")
        
        # Border
        border_color = color_map[pred_sentiment]
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)
    
    # Hide extra subplots
    for idx in range(len(indices), len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle(f"Model Predictions ({len(indices)} samples) - Green=Correct, Red=Wrong", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Batch demo - show multiple predictions")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/saved_models/best.pth"))
    parser.add_argument("--data_root", type=Path, default=Path("data/raw"))
    parser.add_argument("--annotations", type=Path, default=Path("data/processed/annotations_test.csv"))
    parser.add_argument("--num_samples", type=int, default=12, help="Number of samples to show")
    parser.add_argument("--filter", type=str, choices=["correct", "incorrect", "all"], default="all")
    parser.add_argument("--save", type=Path, default=None, help="Save figure to path")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Load model
    device = get_device()
    model = build_resnet18(num_classes=cfg.model.num_classes, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
    ])
    
    # Load dataset
    dataset = ImageSentimentDataset(
        annotations_file=args.annotations,
        root_dir=args.data_root,
        transform=transform,
    )
    
    # Create demo
    print(f"🎬 Creating batch demo ({args.num_samples} images, filter: {args.filter})...")
    fig = create_batch_demo(model, device, transform, dataset, args.data_root, 
                           num_samples=args.num_samples, filter_type=args.filter)
    
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to {args.save}")
    
    plt.show()


if __name__ == "__main__":
    main()
