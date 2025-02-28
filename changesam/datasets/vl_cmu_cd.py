import os
import glob
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

from tqdm import tqdm


class VlCmuCdDataset(Dataset):
    """
    A PyTorch Dataset for the VL-CMU-CD dataset that loads image embeddings,
    images, and ground-truth segmentation masks.

    Args:
        dataset_root (str): Root directory of the dataset.
        seq_ids (List[str]): List of sequence IDs to include.
        binary_gt (bool): Whether to map all change classes to a single class.
        return_embeddings (bool): Whether to load precomputed embeddings.
        return_images (bool): Whether to load the raw images.
    """

    def __init__(
        self,
        dataset_root: str,
        seq_ids: List[str],
        binary_gt: bool = True,
        return_embeddings: bool = True,
        return_images: bool = False,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.binary_gt = binary_gt
        self.return_embeddings = return_embeddings
        self.return_images = return_images

        # Retrieve all ground truth mask paths recursively.
        gt_dir = os.path.join(dataset_root, "GTclass")
        mask_pattern = os.path.join(gt_dir, "**", "*.png")
        all_masks = glob.glob(mask_pattern, recursive=True)

        # Each entry is a tuple: (sequence, image_id)
        self.seq_and_img_ids = [
            (
                os.path.basename(os.path.dirname(x)),
                os.path.basename(x).replace("gt", "")[:-4],
            )
            for x in all_masks
            if os.path.basename(os.path.dirname(x)) in seq_ids
        ]

        # Define the color mapping. The index in the list corresponds to the label.
        colors = [
            [0, 0, 0],  # 0: ignore / mask-out
            [255, 255, 255],  # 1: no-change
            [136, 0, 21],  # 2: barrier
            [237, 28, 36],  # 3: bin
            [255, 127, 39],  # 4: construction-maintenance
            [255, 242, 0],  # 5: misc
            [34, 177, 76],  # 6: other-objects
            [0, 162, 232],  # 7: person-cycle
            [63, 72, 204],  # 8: rubbish
            [163, 73, 164],  # 9: sign
            [255, 174, 201],  # 10: traffic-cone
            [181, 230, 29],  # 11: vehicle
        ]
        # Precompute a tensor for colors (int32) and its hash.
        self.colors_tensor = torch.tensor(
            colors, dtype=torch.int32
        )  # shape: [num_colors, 3]
        self.palette_hash = (
            (self.colors_tensor[:, 0] << 16)
            | (self.colors_tensor[:, 1] << 8)
            | self.colors_tensor[:, 2]
        )  # shape: [num_colors]

    def _convert_mask(self, gt_tensor: torch.Tensor) -> torch.Tensor:
        """
        Converts an RGB GT mask to a label mask using a precomputed hash-based lookup.

        Args:
            gt_tensor (torch.Tensor): Tensor of shape [H, W, 3] in int32.

        Returns:
            label_mask (torch.Tensor): Tensor of shape [H, W] with class indices.
        """
        # Compute a unique hash per pixel.
        hash_img = (
            (gt_tensor[..., 0] << 16) | (gt_tensor[..., 1] << 8) | gt_tensor[..., 2]
        )
        # Compare each pixel's hash to the precomputed palette hash.
        # hash_img: [H, W], palette_hash: [num_colors]
        matches = hash_img.unsqueeze(-1) == self.palette_hash.to(
            gt_tensor.device
        ).unsqueeze(0).unsqueeze(0)
        label_mask = matches.float().argmax(dim=-1)
        valid = matches.any(dim=-1)
        label_mask[~valid] = 0
        if self.binary_gt:
            label_mask[label_mask > 1] = 2
        return label_mask

    def consistency_check(self) -> None:
        """
        Checks that all image embeddings exist.
        """
        images_dir = os.path.join(self.dataset_root, "images")
        image_pattern = os.path.join(images_dir, "**", "*.png")
        all_images = glob.glob(image_pattern, recursive=True)

        embedding_pattern = os.path.join(images_dir, "**", "*.pt")
        all_embeddings = glob.glob(embedding_pattern, recursive=True)

        image_bases = {os.path.splitext(x)[0] for x in all_images}
        embedding_bases = {os.path.splitext(x)[0] for x in all_embeddings}

        missing = image_bases - embedding_bases
        if missing:
            raise ValueError(
                "Not all image embeddings have been found. "
                "Please run changesam_precompute_embeddings.py first."
            )

    def __len__(self) -> int:
        return len(self.seq_and_img_ids)

    def getitem(self, seq, img_id):
        # Load ground truth mask as tensor in RGB mode.
        gt_path = os.path.join(self.dataset_root, "GTclass", seq, f"gt{img_id}.png")
        gt_tensor = read_image(gt_path, mode=ImageReadMode.RGB)  # [3, H, W]
        gt_tensor = gt_tensor.permute(1, 2, 0).to(torch.int32)  # [H, W, 3]
        label_mask = self._convert_mask(gt_tensor)

        if self.return_embeddings:
            emb_path_a = os.path.join(
                self.dataset_root, "images", seq, "RGB", f"1_{img_id}.pt"
            )
            emb_path_b = os.path.join(
                self.dataset_root, "images", seq, "RGB", f"2_{img_id}.pt"
            )
            emb_a = torch.load(emb_path_a, map_location="cpu")
            emb_b = torch.load(emb_path_b, map_location="cpu")

        if self.return_images:
            img_path_a = os.path.join(
                self.dataset_root, "images", seq, "RGB", f"1_{img_id}.png"
            )
            img_path_b = os.path.join(
                self.dataset_root, "images", seq, "RGB", f"2_{img_id}.png"
            )
            img_a = read_image(img_path_a, mode=ImageReadMode.RGB)  # [3, H, W]
            img_b = read_image(img_path_b, mode=ImageReadMode.RGB)  # [3, H, W]

        if self.return_embeddings and self.return_images:
            return emb_a, emb_b, img_a, img_b, label_mask
        elif self.return_embeddings:
            return emb_a, emb_b, label_mask
        elif self.return_images:
            return img_a, img_b, label_mask
        else:
            return label_mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        seq, img_id = self.seq_and_img_ids[index]
        return self.getitem(seq, img_id)

    def get_class_counts(self) -> torch.Tensor:
        print("Calculating class counts...")
        num_classes = 3 if self.binary_gt else self.colors_tensor.shape[0]
        class_counts = torch.zeros(num_classes, dtype=torch.int64)
        for idx in tqdm(range(len(self.seq_and_img_ids)), desc="Counting classes"):
            item = self[idx]
            if self.return_embeddings and self.return_images:
                label_tensor = item[4]
            elif self.return_embeddings or self.return_images:
                label_tensor = item[2]
            else:
                label_tensor = item
            classes, counts = label_tensor.unique(return_counts=True)
            class_counts[classes] += counts
        return class_counts

    @staticmethod
    def _get_split(
        dataset_root: str,
        id_file: str,
        binary_gt: bool,
        return_embeddings: bool,
        return_images: bool,
    ) -> "VlCmuCdDataset":
        split_file = os.path.join(dataset_root, id_file)
        with open(split_file, encoding="utf-8") as f:
            content = f.read()
        seq_ids = [f"{int(s):03d}" for s in content.split()]
        return VlCmuCdDataset(
            dataset_root,
            seq_ids,
            binary_gt=binary_gt,
            return_embeddings=return_embeddings,
            return_images=return_images,
        )

    @staticmethod
    def get_train_test_split(
        dataset_root: str,
        return_class_counts: bool = False,
        binary_gt: bool = True,
        return_embeddings: bool = True,
        return_images: bool = False,
    ) -> Tuple["VlCmuCdDataset", "VlCmuCdDataset", Optional[torch.Tensor]]:
        train_ds = VlCmuCdDataset._get_split(
            dataset_root, "train_split.txt", binary_gt, return_embeddings, return_images
        )
        test_ds = VlCmuCdDataset._get_split(
            dataset_root, "test_split.txt", binary_gt, return_embeddings, return_images
        )
        if return_class_counts:
            counts = train_ds.get_class_counts()
            return train_ds, test_ds, counts
        else:
            return train_ds, test_ds
