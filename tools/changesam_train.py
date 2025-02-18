import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import the VL-CMU-CD dataset.
from changesam.datasets.vl_cmu_cd import VlCmuCdDataset

# Utility function (if needed elsewhere).
from changesam.utils import set_requires_grad

# Import builder functions for ChangeSam.
from changesam.build_changesam import (
    build_changesam_predf_from_sam_vit_h_checkpoint,
    build_changesam_postdf_from_sam_vit_h_checkpoint,
)


def get_args():
    parser = argparse.ArgumentParser(description="Train ChangeSam on VL-CMU-CD dataset")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/path/to/VL-CMU-CD",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--lora-r", type=int, default=4, help="LoRA rank to use for selected layers"
    )
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
        default="sam_vit_h_4b8939.pth",
        help="Path to the SAM checkpoint file. Only the ViT-H model checkpoint (sam_vit_h_4b8939.pth) is supported.",
    )
    return parser.parse_args()


def freeze_except_lora(module: nn.Module) -> None:
    """
    Freezes all parameters of the given module except those whose names contain
    'lora_A' or 'lora_B'.
    """
    for name, param in module.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def main():
    args = get_args()

    # 1. Create the training dataset and dataloader.
    # We assume get_train_test_split returns a tuple (train_dataset, test_dataset) where
    # train_dataset returns raw images (instead of precomputed embeddings).
    train_dataset, _ = VlCmuCdDataset.get_train_test_split(
        dataset_root=args.dataset_root, return_embeddings=False, return_images=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # 2. Instantiate model components.
    # Build the ChangeSam model using the pre-DF mask decoder.
    model = build_changesam_predf_from_sam_vit_h_checkpoint(
        args.sam_checkpoint, lora_layers=args.lora_layers, lora_r=args.lora_r
    ).to(args.device)

    # 
    set_requires_grad(model.parameters(), False)

    # 3. Freeze image encoder parameters except for LoRA parameters.
    freeze_except_lora(model.image_encoder)

    # 4. Define loss function and optimizer.
    # For this example, we assume binary segmentation so we use BCEWithLogitsLoss.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            # Assume each batch is a tuple: (images_a, images_b, labels)
            img_a, img_b, labels = batch

            # Create the batched input dictionary with raw images.
            batched_input = {
                "images_a": img_a.to(args.device),
                "images_b": img_b.to(args.device),
            }

            optimizer.zero_grad()

            # Forward pass: compute logits using images.
            outputs = model.forward_with_images(batched_input)

            # Flatten the labels and outputs for computing loss.
            labels_device = labels.to(args.device).flatten()
            # Ignore background: assume labels > 0 indicate valid pixels,
            # and subtract 1 so that valid labels start at 0.
            valid_idx = torch.where(labels_device > 0)
            labels_valid = labels_device[valid_idx] - 1
            outputs_valid = outputs.flatten()[valid_idx]
            loss = criterion(outputs_valid, labels_valid.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(
            f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss / len(train_loader):.4f}"
        )


if __name__ == "__main__":
    main()
