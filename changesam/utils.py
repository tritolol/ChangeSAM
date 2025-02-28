from typing import Iterable

from torch import nn
import torch.nn.functional as F

import torch

import torchvision.transforms.functional as TF


def set_requires_grad(parameters: Iterable[nn.Parameter], requires_grad: bool):
    """
    Sets the `requires_grad` flag for all given PyTorch parameters.

    Args:
        parameters (Iterable[nn.Parameter]): The parameters to update.
        requires_grad (bool): Whether to enable gradient computation.

    Example:
        >>> set_requires_grad(model.parameters(), False)
    """
    for param in parameters:
        param.requires_grad = requires_grad


def compute_f1_score(
    preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8
) -> float:
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return f1.item()


def set_finetune_modules_train_mode(model: nn.Module):
    for name, module in model.named_modules():
        if (
            "mask_decoder.fusion_layer" in name
            or "lora" in name
            or "sparse_prompt_embeddings" in name
        ):
            module.train()


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class GpuTripleAugmentation(nn.Module):
    def __init__(self, size: int, flip_prob: float = 0.3):
        """
        Args:
            size (int): Target spatial size (square) for resizing.
            flip_prob (float): Probability to apply each random flip.
        """
        super().__init__()
        self.size = size
        self.flip_prob = flip_prob
        # Normalization parameters (applied to images only)
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]

    def forward(
        self, image_a: torch.Tensor, image_b: torch.Tensor, label: torch.Tensor
    ):
        """
        Args:
            image_a: [B, C, H, W] tensor (float) for first images.
            image_b: [B, C, H, W] tensor (float) for second images.
            label: [B, H, W] tensor (int) for segmentation labels.
        Returns:
            Tuple of augmented (and normalized) image_a, image_b, and resized label.
        """
        B = image_a.shape[0]
        device = image_a.device

        # In training mode, apply random flips consistently to both images and label.
        if self.training:
            do_hflip = torch.rand(B, device=device) < self.flip_prob
            do_vflip = torch.rand(B, device=device) < self.flip_prob
            if do_hflip.any():
                idx = torch.nonzero(do_hflip).squeeze(1)
                image_a[idx] = torch.flip(image_a[idx], dims=[-1])
                image_b[idx] = torch.flip(image_b[idx], dims=[-1])
                label[idx] = torch.flip(label[idx], dims=[-1])
            if do_vflip.any():
                idx = torch.nonzero(do_vflip).squeeze(1)
                image_a[idx] = torch.flip(image_a[idx], dims=[-2])
                image_b[idx] = torch.flip(image_b[idx], dims=[-2])
                label[idx] = torch.flip(label[idx], dims=[-2])

                # Normalize images (applied only to images).
        image_a = TF.normalize(image_a.float(), mean=self.mean, std=self.std)
        image_b = TF.normalize(image_b.float(), mean=self.mean, std=self.std)

        # Resize images using bilinear interpolation.
        image_a = F.interpolate(
            image_a, size=(self.size, self.size), mode="bilinear", align_corners=False
        )
        image_b = F.interpolate(
            image_b, size=(self.size, self.size), mode="bilinear", align_corners=False
        )
        # Resize label using nearest neighbor (add channel dim, then remove).
        label = label.unsqueeze(1).float()
        label = F.interpolate(label, size=(self.size, self.size), mode="nearest")
        label = label.squeeze(1).long()

        return image_a, image_b, label
