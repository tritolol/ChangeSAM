#!/usr/bin/env python3
"""
Testing script for ChangeSam on the VL-CMU-CD dataset.

This script loads a previously trained ChangeSam model (from the best adapted checkpoint)
and evaluates its performance on a test split (here, the validation split is used as a test set).
It applies GPU-based augmentation (with no randomness) and computes the F1 score of the predictions.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from changesam.datasets.vl_cmu_cd import VlCmuCdDataset
from changesam.build_changesam import build_changesam
from changesam.utils import (
    compute_f1_score,
    GpuTripleAugmentation,
)


def get_args():
    """
    Parses command-line arguments for testing the ChangeSam model.

    Returns:
        argparse.Namespace: Parsed command-line arguments, which include:
            --dataset-root: Root directory of the VL-CMU-CD dataset.
            --batch-size: Batch size to use during testing.
            --device: Device to use for inference (e.g., "cuda" or "cpu").
            --sam-checkpoint: Path to the SAM checkpoint file.
            --encoder-type: Type of image encoder to use ("sam_vit_h" or "mobile_sam_vit_t").
            --decoder-type: Type of mask decoder to use ("predf" or "postdf").
            --num-tokens: Number of prompt tokens to tune.
            --lora-r: LoRA rank.
            --lora-layers: Indices of transformer layers to apply LoRA.
            --adapted-checkpoint-path: Path to the best adapted checkpoint file.
    """
    parser = argparse.ArgumentParser(description="Test ChangeSam on VL-CMU-CD dataset")
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="Dataset root directory",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for testing")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        help="Path to the SAM checkpoint file",
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="mobile_sam_vit_t",
        choices=["sam_vit_h", "mobile_sam_vit_t"],
        help="Type of image encoder to use",
    )
    parser.add_argument(
        "--decoder-type",
        type=str,
        default="predf",
        choices=["predf", "postdf"],
        help="Type of mask decoder to use",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=4,
        help="Number of prompt tokens to tune",
    )
    parser.add_argument("--lora-r", type=int, default=4, help="LoRA rank")
    parser.add_argument(
        "--lora-layers",
        type=int,
        nargs="+",
        default=[-1, -2],
        help="Indices of transformer layers to apply LoRA",
    )
    parser.add_argument(
        "--adapted-checkpoint-path",
        type=str,
        default="best_adapted_checkpoint.pth",
        help="Path to load the best adapted checkpoint",
    )
    return parser.parse_args()


def test(model: nn.Module, test_loader: DataLoader, device: str, gpu_aug: nn.Module) -> float:
    """
    Evaluates the ChangeSam model on the test set and computes the F1 score.

    The function performs the following steps:
      - Sets the model and GPU augmentation module to evaluation mode.
      - Iterates over the test dataset to apply augmentation, perform forward passes,
        and collect predictions and corresponding ground truth labels.
      - Computes the F1 score based on valid pixels (labels > 0).

    Args:
        model (nn.Module): The ChangeSam model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device used for evaluation.
        gpu_aug (nn.Module): GPU augmentation module (configured with no randomness).

    Returns:
        float: The computed F1 score on the test set.
    """
    model.eval()
    gpu_aug.eval()  # Disable randomness in augmentation during testing.
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            img_a, img_b, labels = batch
            img_a, img_b, labels = img_a.to(device), img_b.to(device), labels.to(device)
            # Apply GPU augmentation (with flip_prob=0, images remain unchanged).
            img_a, img_b, labels = gpu_aug(img_a, img_b, labels)
            batched_input = {"images_a": img_a, "images_b": img_b}
            outputs = model.forward_with_images(batched_input)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            labels_flat = labels.flatten()
            valid_idx = torch.where(labels_flat > 0)[0]
            targets = (labels_flat[valid_idx] - 1).float()
            preds_valid = preds.flatten()[valid_idx]
            all_preds.append(preds_valid.cpu())
            all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    f1 = compute_f1_score(all_preds, all_targets)
    return f1


def main():
    """
    Main function for testing the ChangeSam model.

    The function performs the following:
      1. Parses command-line arguments.
      2. Sets up the device and GPU augmentation module for testing.
      3. Loads the VL-CMU-CD test dataset (using the validation split as test data).
      4. Builds the ChangeSam model using the specified configuration.
      5. Loads the best adapted checkpoint into the model.
      6. Evaluates the model on the test dataset and prints the F1 score.
    """
    args = get_args()
    device = args.device

    # Instantiate GPU augmentation for testing (with no randomness).
    gpu_aug = GpuTripleAugmentation(size=1024, flip_prob=0).to(device)
    gpu_aug.eval()

    # Create the test dataset (using the validation split).
    _, test_dataset = VlCmuCdDataset.get_train_test_split(
        dataset_root=args.dataset_root, return_embeddings=False, return_images=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    # Build the ChangeSam model.
    model = build_changesam(
        sam_checkpoint=args.sam_checkpoint,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        lora_layers=args.lora_layers,
        lora_r=args.lora_r,
        lora_alpha=1,
        num_sparse_prompts=args.num_tokens,
        prompt_init_strategy="none",  # Set strategy to "none" for testing if desired.
    ).to(device)

    # Load the best adapted checkpoint.
    model.load_adapted_checkpoint(args.adapted_checkpoint_path)

    # Evaluate the model.
    f1 = test(model, test_loader, device, gpu_aug)
    print(f"Test F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
