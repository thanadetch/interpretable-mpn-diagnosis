"""
Model factory for MPN Classification and Fibrosis Grading.
Supports ResNet18, EfficientNet-B0, and DenseNet121 with pretrained ImageNet weights.
"""
import torch
import torch.nn as nn
from torchvision import models


def get_model(
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """
    Factory function to create a model with pretrained weights.

    Args:
        model_name: Name of the model ('resnet18' or 'efficientnet_b0')
        num_classes: Number of output classes
        device: Device to move the model to

    Returns:
        PyTorch model with modified final layer

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name == "resnet18":
        # Load pretrained ResNet18
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify the final fully connected layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        # Load pretrained EfficientNet-B0
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Modify the classifier layer
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "densenet121":
        # Load pretrained DenseNet121 - better for texture/fiber detection
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # DenseNet uses a single Linear layer as classifier
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from: 'resnet18', 'efficientnet_b0', 'densenet121'"
        )

    # Move model to device
    model = model.to(device)

    return model


def get_target_layer(model: nn.Module, model_name: str):
    """
    Get the target layer for Grad-CAM visualization.

    Args:
        model: PyTorch model
        model_name: Name of the model architecture

    Returns:
        Target layer for Grad-CAM
    """
    if model_name == "resnet18":
        # Last convolutional block in ResNet
        return model.layer4[-1]
    elif model_name == "efficientnet_b0":
        # Last convolutional layer in EfficientNet
        return model.features[-1]
    elif model_name == "densenet121":
        # Last DenseBlock in DenseNet (features.denseblock4)
        return model.features.denseblock4
    else:
        raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, model_name: str) -> None:
    """
    Print a summary of the model architecture.

    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    total_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"{'='*60}\n")

