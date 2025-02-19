import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the VL-CMU-CD dataset.
from changesam.datasets.vl_cmu_cd import VlCmuCdDataset
# Import the unified builder for ChangeSam.
from changesam.build_changesam import build_changesam


def get_args():
    parser = argparse.ArgumentParser(description="Train ChangeSam on VL-CMU-CD dataset")
    parser.add_argument("--dataset-root", type=str, default="/path/to/VL-CMU-CD",
                        help="Dataset root directory")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for adaptable parameters (except sparse_prompt_embeddings)")
    parser.add_argument("--sparse-lr", type=float, default=1e-3,
                        help="Learning rate for sparse_prompt_embeddings.weight")
    parser.add_argument("--lora-r", type=int, default=4,
                        help="LoRA rank to use for selected layers")
    parser.add_argument("--lora-layers", type=int, nargs="+", default=[-1, -2],
                        help="Indices of transformer layers to apply LoRA")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")
    parser.add_argument("--sam-checkpoint", type=str, default="sam_vit_h_4b8939.pth",
                        help="Path to the SAM checkpoint file. Only the ViT-H checkpoint is supported.")
    parser.add_argument("--encoder-type", type=str, default="sam_vit_h",
                        choices=["sam_vit_h", "mobile_sam_vit_t"],
                        help="Type of image encoder to use")
    parser.add_argument("--decoder-type", type=str, default="predf",
                        choices=["predf", "postdf"],
                        help="Type of mask decoder to use")
    parser.add_argument("--checkpoint-path", type=str, default="best_adapted_checkpoint.pth",
                        help="Path to save the best adapted checkpoint.")
    return parser.parse_args()


def compute_f1_score(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Computes the F1 score for binary segmentation.
    Args:
        preds (torch.Tensor): Binary predictions (0/1) as a flat tensor.
        targets (torch.Tensor): Ground truth labels (0/1) as a flat tensor.
    Returns:
        f1 (float): The computed F1 score.
    """
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return f1.item()


def validate(model: nn.Module, val_loader: DataLoader, device: str) -> float:
    """
    Runs validation on the provided validation loader and returns the F1 score.
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            # Assume each batch is a tuple: (images_a, images_b, labels)
            img_a, img_b, labels = batch

            batched_input = {
                "images_a": img_a.to(device),
                "images_b": img_b.to(device),
            }
            outputs = model.forward_with_images(batched_input)
            # Apply sigmoid to obtain probabilities and threshold at 0.5.
            preds = (torch.sigmoid(outputs) > 0.5).float()
            # Flatten and select valid pixels.
            labels_device = labels.to(device).flatten()
            valid_idx = torch.where(labels_device > 0)
            # Subtract 1 so that valid labels become 0 or 1.
            targets = (labels_device[valid_idx] - 1).float()
            preds_valid = preds.flatten()[valid_idx]
            all_preds.append(preds_valid)
            all_targets.append(targets)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    f1 = compute_f1_score(all_preds, all_targets)
    model.train()
    return f1


def main():
    args = get_args()

    # 1. Create training and validation datasets and dataloaders.
    train_dataset, val_dataset = VlCmuCdDataset.get_train_test_split(
        dataset_root=args.dataset_root, 
        return_embeddings=False, 
        return_images=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=12, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=12, persistent_workers=True
    )

    # 2. Instantiate the ChangeSam model using the unified builder.
    # The builder loads the SAM checkpoint and then calls freeze_except() so that only
    # the newly introduced adaptation parameters remain unfrozen.
    model = build_changesam(
        sam_checkpoint=args.sam_checkpoint,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        lora_layers=args.lora_layers,
        lora_r=args.lora_r,
        lora_alpha=1,
    ).to(args.device)

    # 3. Build optimizer parameter groups.
    # We assign one learning rate (args.sparse_lr) for sparse_prompt_embeddings.weight
    # and another (args.lr) for all other adaptable parameters.
    sparse_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "sparse_prompt_embeddings.weight" in name:
                sparse_params.append(param)
            else:
                other_params.append(param)
    optimizer = optim.Adam([
        {"params": sparse_params, "lr": args.sparse_lr},
        {"params": other_params, "lr": args.lr},
    ])

    # 4. Define the loss function (assuming binary segmentation).
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0.0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        train_preds_list = []
        train_targets_list = []
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            # Assume each batch is a tuple: (images_a, images_b, labels)
            img_a, img_b, labels = batch

            batched_input = {
                "images_a": img_a.to(args.device),
                "images_b": img_b.to(args.device),
            }
            optimizer.zero_grad()

            outputs = model.forward_with_images(batched_input)

            # Flatten outputs and labels for computing the loss.
            labels_device = labels.to(args.device).flatten()
            valid_idx = torch.where(labels_device > 0)[0]
            labels_valid = labels_device[valid_idx] - 1
            outputs_valid = outputs.flatten()[valid_idx]

            loss = criterion(outputs_valid, labels_valid.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Accumulate training predictions for F1 computation.
            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float()
                preds_valid = preds.flatten()[valid_idx]
                train_preds_list.append(preds_valid)
                train_targets_list.append(labels_valid.float())

        avg_loss = epoch_loss / len(train_loader)
        train_preds = torch.cat(train_preds_list)
        train_targets = torch.cat(train_targets_list)
        train_f1 = compute_f1_score(train_preds, train_targets)
        print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {avg_loss:.4f}, Training F1: {train_f1:.4f}")

        # 5. Validation loop.
        val_f1 = validate(model, val_loader, args.device)
        print(f"Epoch {epoch+1}/{args.epochs}, Validation F1: {val_f1:.4f}")

        # 6. If a new best F1 is reached, save an adapted checkpoint.
        if val_f1 > best_f1:
            best_f1 = val_f1
            _ = model.save_adapted_checkpoint(filepath=args.checkpoint_path)
            print(f"New best Validation F1: {best_f1:.4f}. Adapted checkpoint saved to {args.checkpoint_path}")

if __name__ == "__main__":
    main()
