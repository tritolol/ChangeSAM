import os
import glob
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from PIL import Image
import numpy as np

from tqdm import tqdm

from torchvision.transforms import Compose
from segment_anything.utils.transforms import ResizeLongestSide



class VlCmuCdDataset(Dataset):
    """
    A PyTorch Dataset for the VL-CMU-CD dataset that loads precomputed image embeddings
    and ground-truth segmentation masks, with optional binary ground-truth mapping.

    Args:
        dataset_root (str): Root directory of the dataset.
        seq_ids (List[str]): List of sequence IDs to include.
        binary_gt (bool): Whether to map all change classes to a single class.
    """

    def __init__(self, dataset_root: str, seq_ids: List[str], binary_gt: bool = True, return_embeddings: bool = True, return_images: bool = False):
        super().__init__()
        self.dataset_root = dataset_root
        self.binary_gt = binary_gt
        self.return_embeddings = return_embeddings
        self.return_images = return_images

        self.transform = ResizeLongestSide(1024)
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53])
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375])

        # Retrieve all ground truth mask paths recursively.
        gt_dir = os.path.join(dataset_root, "GTclass")
        mask_pattern = os.path.join(gt_dir, "**", "*.png")
        all_masks = glob.glob(mask_pattern, recursive=True)

        # Each entry is a tuple: (sequence, image_id)
        self.seq_and_img_ids = [
            (os.path.basename(os.path.dirname(x)), x.split(os.sep)[-1].replace("gt", "")[:-4])
            for x in all_masks
            if os.path.basename(os.path.dirname(x)) in seq_ids
        ]

        # Define the color mapping. The index in the list corresponds to the label.
        colors = [
            [0, 0, 0],         # 0: ignore / mask-out
            [255, 255, 255],   # 1: no-change
            [136, 0, 21],      # 2: barrier
            [237, 28, 36],     # 3: bin
            [255, 127, 39],    # 4: construction-maintenance
            [255, 242, 0],     # 5: misc
            [34, 177, 76],     # 6: other-objects
            [0, 162, 232],     # 7: person-cycle
            [63, 72, 204],     # 8: rubbish
            [163, 73, 164],    # 9: sign
            [255, 174, 201],   # 10: traffic-cone
            [181, 230, 29]     # 11: vehicle
        ]
        self.colors_arr = np.array(colors, dtype=np.int32)  # Shape: (num_classes, 3)

    def consistency_check(self) -> None:
        """
        Checks whether all image embeddings exist for the images in the dataset.
        Raises:
            ValueError: If there are image embeddings missing.
        """
        images_dir = os.path.join(self.dataset_root, "images")
        image_pattern = os.path.join(images_dir, "**", "*.png")
        all_images = glob.glob(image_pattern, recursive=True)

        embedding_pattern = os.path.join(images_dir, "**", "*.pt")
        all_embeddings = glob.glob(embedding_pattern, recursive=True)

        # Remove extension to compare base filenames.
        image_bases = {os.path.splitext(x)[0] for x in all_images}
        embedding_bases = {os.path.splitext(x)[0] for x in all_embeddings}

        missing = image_bases - embedding_bases
        if missing:
            raise ValueError(
                "Not all image embeddings have been found. Use changesam_precompute_embeddings.py first."
            )

    def __len__(self) -> int:
        return len(self.seq_and_img_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple containing:
            - emb_a (torch.Tensor): Embedding from view A.
            - emb_b (torch.Tensor): Embedding from view B.
            - label_tensor (torch.Tensor): Label mask as an integer tensor.
        """
        seq, img_id = self.seq_and_img_ids[index]

        # Construct the ground truth mask path.
        gt_path = os.path.join(
            self.dataset_root, "GTclass", seq, f"gt{img_id}.png"
        )

        with Image.open(gt_path) as im:
            # Ensure the image is in RGB mode.
            gt_img = im.convert("RGB")
            gt_np = np.array(gt_img)

        # Compute per-pixel differences to each class color.
        # diff has shape (H, W, num_classes)
        diff = np.abs(gt_np[:, :, None, :] - self.colors_arr[None, None, :, :]).sum(axis=-1)
        # The label for each pixel is the index with the minimal difference.
        label_mask = np.argmin(diff, axis=-1)
        # For pixels with no exact color match (min diff != 0), assign ignore label (0).
        min_diff = diff.min(axis=-1)
        label_mask[min_diff != 0] = 0

        label_tensor = torch.from_numpy(label_mask.astype(np.int64))

        if self.binary_gt:
            # Map all change classes (labels > 1) to a single class (2).
            label_tensor[label_tensor > 1] = 2

        if self.return_embeddings:
            # Construct paths for the embeddings.
            emb_path_a = os.path.join(
                self.dataset_root, "images", seq, "RGB", f"1_{img_id}.pt"
            )
            emb_path_b = os.path.join(
                self.dataset_root, "images", seq, "RGB", f"2_{img_id}.pt"
            )

            emb_a = torch.load(emb_path_a, map_location="cpu")
            emb_b = torch.load(emb_path_b, map_location="cpu")

        if self.return_images:
            # Construct paths for the images.
            img_path_a = os.path.join(
                self.dataset_root, "images", seq, "RGB", f"1_{img_id}.png"
            )
            img_path_b = os.path.join(
                self.dataset_root, "images", seq, "RGB", f"2_{img_id}.png"
            )

            with Image.open(img_path_a) as im:
                # Ensure the image is in RGB mode.
                a_img = im.convert("RGB")
                img_a = torch.tensor(self.transform.apply_image(np.array(a_img))).permute(2, 0, 1).contiguous()
                img_a = self.preprocess(img_a)

            with Image.open(img_path_b) as im:
                # Ensure the image is in RGB mode.
                b_img = im.convert("RGB")
                img_b = torch.tensor(self.transform.apply_image(np.array(b_img))).permute(2, 0, 1).contiguous()
                img_b = self.preprocess(img_b)

        if self.return_embeddings and self.return_images:
            return emb_a, emb_b, img_a, img_b, label_tensor
        else:
            if self.return_embeddings:
                return emb_a, emb_b, label_tensor
            if self.return_images:
                return img_a, img_a, label_tensor

        return label_tensor

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean[:, None, None]) / self.pixel_std[:, None, None]

        # Pad
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def get_class_counts(self) -> torch.Tensor:
        """
        Computes the total count of each class label across the dataset.

        Returns:
            A tensor with counts per class. If binary_gt is True, the tensor has 3 elements;
            otherwise, it has as many elements as there are classes in the color mapping.
        """
        print("Calculating class counts...")
        num_classes = 3 if self.binary_gt else self.colors_arr.shape[0]
        class_counts = torch.zeros(num_classes, dtype=torch.int64)

        for idx in tqdm(range(len(self.seq_and_img_ids)), desc="Counting classes"):
            _, _, label_tensor = self[idx]
            classes, counts = label_tensor.unique(return_counts=True)
            class_counts[classes] += counts

        return class_counts

    @staticmethod
    def _get_split(dataset_root: str, id_file: str, binary_gt: bool, return_embeddings: bool, return_images: bool) -> "VlCmuCdDataset":
        """
        Creates a dataset split based on an ID file.

        Args:
            dataset_root (str): The dataset root directory.
            id_file (str): The file containing the sequence IDs.
            binary_gt (bool): Whether to map all change classes to a single class.

        Returns:
            VlCmuCdDataset: The dataset corresponding to the given split.
        """
        split_file = os.path.join(dataset_root, id_file)
        with open(split_file, encoding="utf-8") as f:
            content = f.read()
        seq_ids = content.split()
        return VlCmuCdDataset(dataset_root, seq_ids, binary_gt=binary_gt, return_embeddings=return_embeddings, return_images=return_images)

    @staticmethod
    def get_train_test_split(
        dataset_root: str,
        return_class_counts: bool = False,
        binary_gt: bool = True,
        return_embeddings: bool = True, 
        return_images: bool = False
    ) -> Tuple["VlCmuCdDataset", "VlCmuCdDataset", Optional[torch.Tensor]]:
        """
        Splits the dataset into training and testing sets based on predefined split files.

        Args:
            dataset_root (str): The dataset root directory.
            return_class_counts (bool): Whether to also return class counts.
            binary_gt (bool): Whether to map all change classes to a single class.

        Returns:
            A tuple (train_dataset, test_dataset, class_counts) if return_class_counts is True,
            otherwise (train_dataset, test_dataset).
        """
        train_ds = VlCmuCdDataset._get_split(dataset_root, "train_split.txt", binary_gt, return_embeddings, return_images)
        test_ds = VlCmuCdDataset._get_split(dataset_root, "test_split.txt", binary_gt, return_embeddings, return_images)

        if return_class_counts:
            counts = train_ds.get_class_counts()
            return train_ds, test_ds, counts
        else:
            return train_ds, test_ds
