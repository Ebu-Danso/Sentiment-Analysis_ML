"""Interactive demo - predict on multiple images with keyboard navigation."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

try:
    from .dataset import ImageSentimentDataset, INDEX_TO_SENTIMENT
    from .model import build_resnet18, get_device
    from .utils import load_config, get_inference_transform
except ImportError:
    from dataset import ImageSentimentDataset, INDEX_TO_SENTIMENT
    from model import build_resnet18, get_device
    from utils import load_config, get_inference_transform


class InteractiveDemo:
    def __init__(self, model, device, transform, cfg, dataset, data_root):
        self.model = model
        self.device = device
        self.transform = transform
        self.cfg = cfg
        self.dataset = dataset
        self.data_root = data_root
        self.current_idx = 0
        self.predictions = []
        self.fig = None
        
        # Pre-compute all predictions
        self._compute_predictions()
    
    def _compute_predictions(self):
        """Compute predictions for all images in dataset."""
        print("🔄 Computing predictions for all images (this may take a minute)...")
        self.model.eval()
        
        image_paths = self.dataset.annotations["image_path"].values
        sentiments = self.dataset.annotations["sentiment"].values
        
        with torch.no_grad():
            for idx in range(len(self.dataset)):
                if idx % 100 == 0:
                    print(f"  Processed {idx}/{len(self.dataset)} images...")
                
                image_tensor, true_label = self.dataset[idx]
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                outputs = self.model(image_tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                pred_idx = outputs.argmax(dim=1).item()
                
                self.predictions.append({
                    "image_idx": idx,
                    "image_path": image_paths[idx],
                    "true_sentiment": sentiments[idx],
                    "pred_sentiment": INDEX_TO_SENTIMENT[pred_idx],
                    "confidence": probs[pred_idx],
                    "probabilities": probs,
                    "correct": sentiments[idx] == INDEX_TO_SENTIMENT[pred_idx],
                })
        
        print(f"✅ Predictions computed!\n")
    
    def _load_image(self, image_path):
        """Load image from disk."""
        full_path = self.data_root / image_path
        image = Image.open(full_path).convert("RGB")
        return image
    
    def show_prediction(self, idx):
        """Display a single prediction."""
        if idx < 0 or idx >= len(self.predictions):
            print(f"Invalid index: {idx}")
            return
        
        pred = self.predictions[idx]
        image = self._load_image(pred["image_path"])
        
        # Create figure
        if self.fig is not None:
            plt.close(self.fig)
        
        self.fig = plt.figure(figsize=(14, 6))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[0.7, 0.3], width_ratios=[0.6, 0.4])
        
        # Image
        ax_img = self.fig.add_subplot(gs[0, :])
        ax_img.imshow(image)
        ax_img.axis("off")
        
        title_color = "green" if pred["correct"] else "red"
        title = f"Image {idx + 1}/{len(self.predictions)}: {Path(pred['image_path']).name}"
        ax_img.set_title(title, fontsize=12, fontweight='bold', color=title_color)
        
        # Prediction info
        ax_info = self.fig.add_subplot(gs[1, 0])
        ax_info.axis("off")
        
        status = "✅ CORRECT" if pred["correct"] else "❌ WRONG"
        info_text = f"""
{status}
True:       {pred['true_sentiment'].upper()}
Predicted:  {pred['pred_sentiment'].upper()}
Confidence: {pred['confidence']:.1%}
"""
        ax_info.text(0.05, 0.5, info_text, fontsize=11, fontweight='bold', 
                     verticalalignment='center', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Probability bars
        ax_probs = self.fig.add_subplot(gs[1, 1])
        ax_probs.axis("off")
        
        sentiments = list(INDEX_TO_SENTIMENT.values())
        probs = pred["probabilities"]
        colors = {"negative": "#d62728", "neutral": "#ff7f0e", "positive": "#2ca02c"}
        
        y_pos = [0.7, 0.4, 0.1]
        for i, sentiment in enumerate(sentiments):
            prob = probs[i]
            color = colors[sentiment]
            bar_length = prob * 0.8
            ax_probs.barh(y_pos[i], bar_length, height=0.15, color=color, alpha=0.7)
            ax_probs.text(bar_length + 0.02, y_pos[i], f"{prob:.1%}", 
                         va='center', fontsize=10, fontweight='bold')
            ax_probs.text(-0.05, y_pos[i], sentiment.upper(), ha='right', va='center', fontsize=10)
        
        ax_probs.set_xlim(-0.15, 1.0)
        ax_probs.set_ylim(-0.1, 0.9)
        
        plt.suptitle("← Press LEFT/LEFT ARROW for previous  |  Press RIGHT/RIGHT ARROW for next  |  Press Q/ESC to quit →", 
                    fontsize=10, style='italic', color='gray', y=0.98)
        
        self.current_idx = idx
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Keyboard handling
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        plt.show()
    
    def _on_key_press(self, event):
        """Handle keyboard events."""
        if event.key in ['left', 'Left', 'a', 'A']:
            self.show_prediction(self.current_idx - 1)
        elif event.key in ['right', 'Right', 'd', 'D']:
            self.show_prediction(self.current_idx + 1)
        elif event.key in ['q', 'Q', 'escape']:
            plt.close('all')
            self._print_summary()
    
    def _print_summary(self):
        """Print summary statistics."""
        correct = sum(1 for p in self.predictions if p["correct"])
        total = len(self.predictions)
        accuracy = correct / total if total > 0 else 0
        
        print("\n" + "="*60)
        print("DEMO SUMMARY")
        print("="*60)
        print(f"Total images reviewed: {total}")
        print(f"Correct predictions: {correct}/{total} ({accuracy:.1%})")
        
        # Per-class accuracy
        print("\nPer-class accuracy:")
        for sentiment in ["negative", "neutral", "positive"]:
            mask = np.array([p["true_sentiment"] == sentiment for p in self.predictions])
            if mask.sum() > 0:
                class_acc = np.array([p["correct"] for p in self.predictions])[mask].mean()
                class_total = mask.sum()
                print(f"  {sentiment.capitalize():10s}: {class_acc:.1%} ({mask.sum()} samples)")
        
        print("="*60)
    
    def run(self, start_idx=0):
        """Start interactive demo."""
        self.show_prediction(start_idx)


def main():
    parser = argparse.ArgumentParser(description="Interactive demo - browse predictions")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/saved_models/best.pth"))
    parser.add_argument("--data_root", type=Path, default=Path("data/raw"))
    parser.add_argument("--annotations", type=Path, default=Path("data/processed/annotations_test.csv"))
    parser.add_argument("--start_idx", type=int, default=0, help="Start from image index")
    parser.add_argument("--filter", type=str, choices=["correct", "incorrect", "all"], default="all", help="Filter which predictions to show")
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
    transform = get_inference_transform(cfg)
    
    # Load dataset
    dataset = ImageSentimentDataset(
        annotations_file=args.annotations,
        root_dir=args.data_root,
        transform=transform,
    )
    
    # Create and run demo
    print("\n🎬 Starting Interactive Demo...")
    print(f"📊 Dataset: {len(dataset)} images from {args.annotations}\n")
    
    demo = InteractiveDemo(model, device, transform, cfg, dataset, args.data_root)
    demo.run(start_idx=args.start_idx)


if __name__ == "__main__":
    main()
