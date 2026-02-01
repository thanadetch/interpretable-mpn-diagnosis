"""
Model factory for MPN Classification and Fibrosis Grading.
Supports ResNet18, EfficientNet-B0, and DenseNet121 with pretrained ImageNet weights.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models


# =============================================================================
# CBAM (Convolutional Block Attention Module) Implementation
# =============================================================================


class ChannelAttention(nn.Module):
    """
    Channel Attention module for CBAM.
    Uses both average-pooled and max-pooled features with a shared MLP.
    """

    def __init__(self, in_planes: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention module for CBAM.
    Uses channel-wise average and max pooling followed by a convolution.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    """
    Combined CBAM (Convolutional Block Attention Module).
    Applies channel attention followed by spatial attention.
    """

    def __init__(self, in_planes: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class CBAMBasicBlockWrapper(nn.Module):
    """
    Wrapper that applies CBAM attention inside the residual path of a ResNet BasicBlock.

    CBAM is applied after bn2 but BEFORE the skip connection addition to preserve
    the identity mapping gradient flow ("highway" path).

    Logic: out = cbam(bn2(conv2(relu(bn1(conv1(x)))))); out += identity; return relu(out)
    """

    def __init__(self, block: nn.Module, planes: int, reduction: int = 16):
        super().__init__()
        # Extract layers from the original BasicBlock
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.relu = block.relu
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.downsample = block.downsample  # May be None

        # CBAM module applied on residual branch
        self.cbam = CBAM(planes, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save identity for skip connection
        identity = x

        # Residual path: conv1 -> bn1 -> relu -> conv2 -> bn2
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply CBAM on residual branch (before addition)
        out = self.cbam(out)

        # Handle downsample for identity path
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add skip connection, then ReLU
        out += identity
        out = self.relu(out)

        return out


# =============================================================================
# CBAM Application Helpers
# =============================================================================


def apply_cbam_to_resnet(model: nn.Module) -> nn.Module:
    """
    Apply CBAM wrappers to all BasicBlocks in a ResNet model.

    Args:
        model: A ResNet model (e.g., resnet18)

    Returns:
        The modified model with CBAM attention applied to each block
    """
    # ResNet18/34 layer channel sizes
    layer_planes = {
        "layer1": 64,
        "layer2": 128,
        "layer3": 256,
        "layer4": 512,
    }

    for layer_name, planes in layer_planes.items():
        layer = getattr(model, layer_name)
        wrapped_blocks = nn.Sequential(
            *[CBAMBasicBlockWrapper(block, planes) for block in layer]
        )
        setattr(model, layer_name, wrapped_blocks)

    return model


def apply_cbam_to_densenet(model: nn.Module) -> nn.Module:
    """
    Apply CBAM modules to a DenseNet model by inserting them after
    each _DenseBlock and _Transition layer.

    Args:
        model: A DenseNet model (e.g., densenet121)

    Returns:
        The modified model with CBAM attention inserted after dense blocks
    """
    from torchvision.models.densenet import _DenseBlock, _Transition

    # DenseNet121 channel counts after each block/transition
    # Initial: 64, after conv0/pool0: 64
    # DenseBlock1: 64 + 6*32 = 256, Transition1: 128
    # DenseBlock2: 128 + 12*32 = 512, Transition2: 256
    # DenseBlock3: 256 + 24*32 = 1024, Transition3: 512
    # DenseBlock4: 512 + 16*32 = 1024 (no transition after)
    block_output_channels = {
        "denseblock1": 256,
        "transition1": 128,
        "denseblock2": 512,
        "transition2": 256,
        "denseblock3": 1024,
        "transition3": 512,
        "denseblock4": 1024,
    }

    # Reconstruct the features Sequential with CBAM inserted
    new_features = []
    for name, module in model.features.named_children():
        new_features.append((name, module))

        # Insert CBAM after DenseBlocks and Transitions
        if name in block_output_channels:
            channels = block_output_channels[name]
            cbam_name = f"cbam_{name}"
            new_features.append((cbam_name, CBAM(channels)))

    # Replace model.features with the new Sequential
    model.features = nn.Sequential(OrderedDict(new_features))

    return model


# =============================================================================
# Model Factory Function
# =============================================================================


def get_model(
    model_name: str,
    num_classes: int,
    device: torch.device,
    use_cbam: bool = False,
) -> nn.Module:
    """
    Factory function to create a model with pretrained weights.

    Args:
        model_name: Name of the model ('resnet18', 'efficientnet_b0', or 'densenet121')
        num_classes: Number of output classes
        device: Device to move the model to
        use_cbam: Whether to apply CBAM attention (supported for resnet18 and densenet121)

    Returns:
        PyTorch model with modified final layer

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name == "resnet18":
        # Load pretrained ResNet18
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Apply CBAM attention if requested
        if use_cbam:
            print("[CBAM] Applying CBAM attention to ResNet18 blocks")
            model = apply_cbam_to_resnet(model)

        # Modify the final fully connected layer
        in_features = model.fc.in_features
        if use_cbam:
            # CBAM recipe: add Dropout for regularization
            model.fc = nn.Sequential(
                nn.Dropout(p=0.25),
                nn.Linear(in_features, num_classes),
            )
        else:
            # Baseline recipe: no Dropout
            model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        # Load pretrained EfficientNet-B0
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Modify the classifier layer
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "densenet121":
        # Load pretrained DenseNet121 - better for texture/fiber detection
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # Apply CBAM attention if requested
        if use_cbam:
            print("[CBAM] Applying CBAM attention to DenseNet121 blocks")
            model = apply_cbam_to_densenet(model)

        # Modify the classifier layer
        in_features = model.classifier.in_features
        if use_cbam:
            # CBAM recipe: add Dropout for regularization
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.25),
                nn.Linear(in_features, num_classes),
            )
        else:
            # Baseline recipe: no Dropout
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
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"{'=' * 60}\n")
