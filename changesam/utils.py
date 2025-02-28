from typing import Iterable
from torch import nn
import torch.nn.functional as F
import torch
import torchvision.transforms.functional as TF


def set_requires_grad(parameters: Iterable[nn.Parameter], requires_grad: bool):
    """
    Sets the `requires_grad` attribute for all given PyTorch parameters.

    Args:
        parameters (Iterable[nn.Parameter]): An iterable of PyTorch parameters.
        requires_grad (bool): If True, enables gradient computation for the parameters; otherwise disables it.

    Example:
        >>> set_requires_grad(model.parameters(), False)
    """
    for param in parameters:
        param.requires_grad = requires_grad


def compute_f1_score(
    preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8
) -> float:
    """
    Computes the F1 score given predictions and targets.

    The F1 score is calculated as:
        F1 = 2 * TP / (2 * TP + FP + FN + eps)
    where:
        TP: True Positives,
        FP: False Positives,
        FN: False Negatives.

    Args:
        preds (torch.Tensor): Predicted binary values (0 or 1). Expected to be a tensor of floats.
        targets (torch.Tensor): Ground truth binary values (0 or 1). Expected to be a tensor of floats.
        eps (float): A small constant added to avoid division by zero. Default is 1e-8.

    Returns:
        float: The computed F1 score.
    """
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return f1.item()


def set_finetune_modules_train_mode(model: nn.Module):
    """
    Sets specific sub-modules of the model to training mode, allowing only the fine-tuned parts to update.

    This function iterates through all sub-modules of the model and sets those whose names contain any of:
      - "mask_decoder.fusion_layer"
      - "lora"
      - "sparse_prompt_embeddings"
    to training mode. Other parts of the model remain unaffected.

    Args:
        model (nn.Module): The PyTorch model containing the sub-modules.
    """
    for name, module in model.named_modules():
        if (
            "mask_decoder.fusion_layer" in name
            or "lora" in name
            or "sparse_prompt_embeddings" in name
        ):
            module.train()


class DiceBCELoss(nn.Module):
    """
    Combines Dice loss and Binary Cross-Entropy (BCE) loss for segmentation tasks.

    This loss function applies a sigmoid activation to the model outputs,
    computes the Dice loss (which measures the overlap between predictions and targets) and the BCE loss,
    and then returns their sum as the final loss value.
    """

    def __init__(self, weight=None, size_average=True):
        """
        Initializes the DiceBCELoss module.

        Args:
            weight: Optional weight tensor for BCE loss.
            size_average (bool): Whether to average the loss over observations.
        """
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass for the DiceBCELoss.

        Applies a sigmoid activation on the inputs, flattens both inputs and targets, computes the Dice loss
        and Binary Cross-Entropy loss, and returns the sum as the final loss.

        Args:
            inputs (torch.Tensor): The raw output from the model (logits).
            targets (torch.Tensor): The ground truth segmentation labels.
            smooth (float): A smoothing constant to avoid division by zero. Default is 1.

        Returns:
            torch.Tensor: The combined Dice and BCE loss.
        """
        # Comment out the sigmoid if the model already includes an activation layer.
        inputs = torch.sigmoid(inputs)

        # Flatten the label and prediction tensors.
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
    """
    Performs augmentation on a triplet of tensors: two images and one label.

    This module applies random horizontal and vertical flips (with a specified probability)
    and then normalizes and resizes the images and labels. The same augmentation is applied consistently
    across both images and the corresponding label.
    """

    def __init__(self, size: int, flip_prob: float = 0.3):
        """
        Initializes the GpuTripleAugmentation module.

        Args:
            size (int): The target spatial size (height and width) for resizing images and labels.
            flip_prob (float): The probability of applying a random flip along each spatial axis.
        """
        super().__init__()
        self.size = size
        self.flip_prob = flip_prob
        # Normalization parameters (applied to images only).
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]

    def forward(
        self, image_a: torch.Tensor, image_b: torch.Tensor, label: torch.Tensor
    ):
        """
        Applies augmentation to the input images and label.

        In training mode, random horizontal and vertical flips are applied consistently to both images and the label.
        After augmentation, the images are normalized using the predefined mean and standard deviation, and then
        resized to the target size using bilinear interpolation. The label is resized using nearest neighbor interpolation.

        Args:
            image_a (torch.Tensor): A tensor of shape [B, C, H, W] representing the first set of images.
            image_b (torch.Tensor): A tensor of shape [B, C, H, W] representing the second set of images.
            label (torch.Tensor): A tensor of shape [B, H, W] representing segmentation labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Augmented and normalized image_a tensor.
                - Augmented and normalized image_b tensor.
                - Resized label tensor.
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
        # Resize label using nearest neighbor interpolation.
        label = label.unsqueeze(1).float()
        label = F.interpolate(label, size=(self.size, self.size), mode="nearest")
        label = label.squeeze(1).long()

        return image_a, image_b, label
