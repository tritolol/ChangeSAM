from typing import Tuple

import torch
import torch.nn.functional as F

from torchvision.transforms import functional as TF
import random
from PIL import Image
from torchvision.transforms import RandomCrop
import numpy as np


class TripleCompose:
    """Composes several triple transforms together.
    Args:
        transforms (list): List of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, items: Tuple[Image.Image, Image.Image, Image.Image]) -> Tuple:
        for t in self.transforms:
            items = t(items)
        return items

class TripleRandomHorizontalFlip:
    """Randomly horizontally flips the triple with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, items: Tuple[Image.Image, Image.Image, Image.Image]) -> Tuple:
        if random.random() < self.p:
            return (TF.hflip(items[0]), TF.hflip(items[1]), TF.hflip(items[2]))
        return items

class TripleRandomVerticalFlip:
    """Randomly vertically flips the triple with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, items: Tuple[Image.Image, Image.Image, Image.Image]) -> Tuple:
        if random.random() < self.p:
            return (TF.vflip(items[0]), TF.vflip(items[1]), TF.vflip(items[2]))
        return items

class TripleRandomCrop:
    """Randomly crops the triple to the given size using the same crop for all three."""
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, items: Tuple[Image.Image, Image.Image, Image.Image]) -> Tuple:
        i, j, h, w = RandomCrop.get_params(items[0], self.size)
        return (TF.crop(items[0], i, j, h, w),
                TF.crop(items[1], i, j, h, w),
                TF.crop(items[2], i, j, h, w))

class TripleResize:
    """
    Resizes the triple to the given size.
    For images uses bilinear interpolation; for the mask uses nearest neighbor.
    """
    def __init__(self, size, interpolation=Image.BILINEAR, mask_interpolation=Image.NEAREST):
        self.size = size
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    def __call__(self, items: Tuple[Image.Image, Image.Image, Image.Image]) -> Tuple:
        return (TF.resize(items[0], self.size, interpolation=self.interpolation),
                TF.resize(items[1], self.size, interpolation=self.interpolation),
                TF.resize(items[2], self.size, interpolation=self.mask_interpolation))

class TripleToTensor:
    """
    Converts a triple of PIL images to tensors.
    For the images, uses TF.to_tensor (scaling pixel values to [0,1]).
    For the mask, converts the image (assumed to be single-channel) to a LongTensor.
    """
    def __call__(self, items: Tuple[Image.Image, Image.Image, Image.Image]) -> Tuple:
        img_a = TF.to_tensor(items[0])
        img_b = TF.to_tensor(items[1])
        # For the mask, we convert the PIL image to a numpy array and then to a LongTensor.
        mask = torch.as_tensor(np.array(items[2]), dtype=torch.long)
        return (img_a, img_b, mask)

class TripleNormalize:
    """
    Normalizes the image tensors using the given mean and std.
    Leaves the mask unchanged.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, items: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple:
        return (TF.normalize(items[0], self.mean, self.std),
                TF.normalize(items[1], self.mean, self.std),
                items[2])

class TriplePad:
    """Pads each tensor in the triple to a square of the given size."""
    def __init__(self, size: int):
        self.size = size

    def __call__(self, items: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple:
        padded = []
        for tensor in items:
            if tensor.dim() == 3:
                _, h, w = tensor.shape
            else:
                h, w = tensor.shape
            pad_h = self.size - h if h < self.size else 0
            pad_w = self.size - w if w < self.size else 0
            padded.append(F.pad(tensor, (0, pad_w, 0, pad_h)))
        return tuple(padded)