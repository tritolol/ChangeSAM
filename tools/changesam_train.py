#!/usr/bin/env python3
"""
Training script for ChangeSam on the VL-CMU-CD dataset.

This script builds and trains a ChangeSam model for change detection using paired image embeddings.
It uses GPU-based augmentation, a composite Dice and BCE loss, and LoRA-adapted image encoder and mask decoder.
The training loop monitors performance via the F1 score on a validation split and saves the best model checkpoint.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from changesam.datasets.vl_cmu_cd import VlCmuCdDataset
from changesam.build_changesam import build_changesam
from changesam.utils import (
    DiceBCELoss,
    GpuTripleAugmentation,
    compute_f1_score,
    set_finetune_modules_train_mode,
)


def get_args():
    """
    Parses command-line arguments for training ChangeSam.

    Returns:
        argparse.Namespace: An object containing all command-line parameters.
            - dataset_root (str): Root directory for the VL-CMU-CD dataset.
            - batch_size (int): Batch size for training.
            - epochs (int): Number of training epochs.
            - lr (float): Learning rate for adaptable parameters.
            - prompt_lr (float): Learning rate for sparse prompt embeddings.
            - aug_flip_prob (float): Probability of image flipping during augmentation.
            - num_tokens (int): Number of prompt tokens to tune.
            - token_init_strategy (str): Strategy for initializing prompt embeddings.
            - lora_r (int): LoRA rank.
            - lora_layers (List[int]): Indices of transformer layers to apply LoRA.
            - device (str): Device to use for training.
            - sam_checkpoint (str): Path to the SAM checkpoint file.
            - encoder_type (str): Type of image encoder ("sam_vit_h" or "mobile_sam_vit_t").
            - decoder_type (str): Type of mask decoder ("predf" or "postdf").
            - checkpoint_path (str): Path to save the best model checkpoint.
    """
    parser = argparse.ArgumentParser(description="Train ChangeSam on VL-CMU-CD dataset")
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="Dataset root directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for adaptable parameters"
    )
    parser.add_argument(
        "--prompt-lr",
        type=float,
        default=1e-3,
        help="Learning rate for sparse_prompt_embeddings.weight",
    )
    parser.add_argument(
        "--aug-flip-prob",
        type=float,
        default=0.3,
        help="Probability of image flipping in augmentation",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=4,
        help="The number of prompt tokens to tune",
    )
    parser.add_argument(
        "--token-init-strategy",
        type=str,
        default="center",
        choices=["center", "grid", "random", "random_embedding"],
        help="Initialization strategy for prompt embeddings",
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        help="Path to the SAM/MobileSAM checkpoint file",
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
        "--checkpoint-path",
        type=str,
        default="best_adapted_checkpoint.pth",
        help="Path to save the best checkpoint",
    )

    return parser.parse_args()


def validate(
    model: nn.Module, val_loader: DataLoader, device: str, gpu_aug: nn.Module
) -> float:
    """
    Evaluates the model on a validation dataset and computes the F1 score.

    The validation function sets the model and GPU augmentation module to evaluation mode,
    processes the validation data, applies the model to obtain predictions, and computes the F1 score
    based on valid pixels (where label > 0).

    Args:
        model (nn.Module): The ChangeSam model to evaluate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (str): The device on which evaluation is performed.
        gpu_aug (nn.Module): GPU augmentation module (with no randomness in eval mode).

    Returns:
        float: The computed F1 score on the validation set.
    """
    model.eval()
    gpu_aug.eval()  # Ensure augmentation is in eval mode (no randomness)
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            img_a, img_b, labels = batch  # Loaded on CPU
            img_a, img_b, labels = img_a.to(device), img_b.to(device), labels.to(device)
            img_a, img_b, labels = gpu_aug(img_a, img_b, labels)
            batched_input = {"images_a": img_a, "images_b": img_b}
            outputs = model.forward_with_images(batched_input)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            labels_flat = labels.flatten()
            valid_idx = torch.where(labels_flat > 0)
            targets = (labels_flat[valid_idx] - 1).float()
            preds_valid = preds.flatten()[valid_idx]
            all_preds.append(preds_valid.cpu())
            all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    f1 = compute_f1_score(all_preds, all_targets)
    # Set back to training mode.
    model.train()
    gpu_aug.train()
    return f1


def main():
    """
    Main function to train the ChangeSam model.

    The function performs the following steps:
        1. Parses command-line arguments.
        2. Sets up the device and GPU augmentation module.
        3. Loads the VL-CMU-CD training and validation datasets.
        4. Builds the ChangeSam model based on the provided configuration and moves it to the device.
        5. Constructs optimizer parameter groups for the fine-tuned modules.
        6. Trains the model for a specified number of epochs, computing loss and updating parameters.
        7. Evaluates the model on the validation set after each epoch.
        8. Saves the best model checkpoint based on the highest validation F1 score.
    """
    args = get_args()
    device = args.device

    # Instantiate the GPU augmentation module.
    gpu_aug = GpuTripleAugmentation(size=1024, flip_prob=args.aug_flip_prob).to(device)

    # Create training and validation datasets.
    train_dataset, val_dataset = VlCmuCdDataset.get_train_test_split(
        dataset_root=args.dataset_root, return_embeddings=False, return_images=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    # Build the ChangeSam model.
    model = (
        build_changesam(
            sam_checkpoint=args.sam_checkpoint,
            encoder_type=args.encoder_type,
            decoder_type=args.decoder_type,
            lora_layers=args.lora_layers,
            lora_r=args.lora_r,
            lora_alpha=1,
            num_sparse_prompts=args.num_tokens,
            prompt_init_strategy=args.token_init_strategy,
        )
        .to(device)
        .eval()
    )

    # Build optimizer parameter groups.
    sparse_params = []
    other_params = []
    trainable_params = 0
    for name, param in model.named_parameters():
        if "sparse_prompt_embeddings.weight" in name:
            sparse_params.append(param)
            trainable_params += np.prod(param.size())
        if "mask_decoder.fusion_layer" in name or "lora" in name:
            other_params.append(param)
            trainable_params += np.prod(param.size())

    optimizer = optim.Adam(
        [
            {"params": sparse_params, "lr": args.prompt_lr},
            {"params": other_params, "lr": args.lr},
        ]
    )

    print(f"Trainable parameters: {trainable_params}")

    criterion = DiceBCELoss()
    best_f1 = 0.0

    # Training loop.
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        train_preds_list = []
        train_targets_list = []
        set_finetune_modules_train_mode(model)
        gpu_aug.train()  # Enable randomness in augmentation.
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            # Load batch from DataLoader (CPU tensors).
            img_a, img_b, labels = batch
            img_a, img_b, labels = img_a.to(device), img_b.to(device), labels.to(device)
            # Apply GPU augmentation.
            img_a, img_b, labels = gpu_aug(img_a, img_b, labels)
            batched_input = {"images_a": img_a, "images_b": img_b}
            outputs = model.forward_with_images(batched_input)
            # Compute loss on valid pixels (labels > 0).
            labels_flat = labels.flatten()
            valid_idx = torch.where(labels_flat > 0)[0]
            labels_valid = labels_flat[valid_idx] - 1  # Adjust label values.
            outputs_valid = outputs.flatten()[valid_idx]

            loss = 0.1 * criterion(outputs_valid, labels_valid.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float()
                preds_valid = preds.flatten()[valid_idx]
                train_preds_list.append(preds_valid.cpu())
                train_targets_list.append(labels_valid.float().cpu())
        avg_loss = epoch_loss / len(train_loader)
        train_preds = torch.cat(train_preds_list)
        train_targets = torch.cat(train_targets_list)
        train_f1 = compute_f1_score(train_preds, train_targets)

        print(
            f"Epoch {epoch+1}/{args.epochs}, Training Loss: {avg_loss:.4f}, Training F1: {train_f1:.4f}"
        )

        # Run validation.
        val_f1 = validate(model, val_loader, device, gpu_aug)
        print(f"Epoch {epoch+1}/{args.epochs}, Validation F1: {val_f1:.4f}")

        # Save checkpoint if validation improves.
        if val_f1 > best_f1:
            best_f1 = val_f1
            _ = model.save_adapted_checkpoint(filepath=args.checkpoint_path)
            print(
                f"New best Validation F1: {best_f1:.4f}. Checkpoint saved to {args.checkpoint_path}"
            )


if __name__ == "__main__":
    main()
