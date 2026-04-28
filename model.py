"""Model definitions for image-based sentiment analysis."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


def build_resnet18(num_classes: int = 3, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """Create a ResNet-18 backbone with an improved classification head.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained ImageNet weights
        freeze_backbone: Whether to freeze early layers of backbone (recommended for fine-tuning)
    """

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    
    # Freeze early layers for fine-tuning (prevent catastrophic forgetting)
    if freeze_backbone:
        for name, param in model.named_parameters():
            # Freeze everything except layer3 and layer4
            if 'layer3' not in name and 'layer4' not in name:
                param.requires_grad = False
    
    in_features = model.fc.in_features
    
    # Improved classification head with stronger regularization
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.5),  # Increased from 0.3
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, num_classes),
    )
    return model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
