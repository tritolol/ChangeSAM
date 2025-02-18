import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import the VL-CMU-CD dataset.
from changesam.datasets.vl_cmu_cd import VlCmuCdDataset

from changesam.build_changesam import build_changesam_predf_from_sam_vit_h_checkpoint, build_changesam_postdf_from_sam_vit_h_checkpoint


# For reproducibility.
torch.manual_seed(42)

def get_args():
    parser = argparse.ArgumentParser(description="Train ChangeSam on VL-CMU-CD dataset")
    parser.add_argument("--dataset-root", type=str, default="/path/to/VL-CMU-CD", help="Dataset root directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=4, help="LoRA rank to use for selected layers")
    parser.add_argument("--lora-layers", type=int, nargs="+", default=[2, 3, 4, 5], help="Indices of transformer layers to apply LoRA")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    return parser.parse_args()


def main():
    args = get_args()

    # 1. Create the training dataset and dataloader.
    train_dataset, _ = VlCmuCdDataset.get_train_test_split(dataset_root=args.dataset_root, return_embeddings=False, return_images=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # Collate returns three lists: list of emb_a, emb_b, label_tensor.
    
    # 2. Instantiate model components.
    model = build_changesam_predf_from_sam_vit_h_checkpoint("sam_vit_h_4b8939.pth").to(args.device)


    # 3. Define loss function and optimizer.
    # Assume output logits shape: (B, H, W, C) and target shape: (B, H, W)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            # batch is a tuple: (list_emb_a, list_emb_b, list_labels)
            img_a, img_b, labels = batch

            # Create the batched input dictionary.
            batched_input = {
                "images_a": img_a.to(args.device),
                "images_b": img_b.to(args.device),
            }

            optimizer.zero_grad()
            # Forward pass.
            outputs = model.forward_with_images(batched_input)  # outputs: (B, H, W, num_classes)
            # Rearrange to (B, num_classes, H, W) for cross entropy.
            outputs = outputs.permute(0, 3, 1, 2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss/len(train_loader):.4f}")


if __name__ == "__main__":
    main()
