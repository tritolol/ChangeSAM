#!/usr/bin/env python3
"""
This script processes images using the Segment Anything Model (SAM) on CUDA, MPS, or CPU.

It downloads the SAM checkpoint if not present, loads the model onto the specified devices,
and computes image embeddings for all images in a given dataset. Each embedding is saved
as a .pt file in the same folder as the corresponding image.
"""

import os
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor
import requests

from PIL import Image
import numpy as np
from tqdm import tqdm
import torch

# Import SAM modules
from segment_anything.build_sam import build_sam_vit_h
from segment_anything.predictor import SamPredictor


def parse_args():
    """
    Parses command-line arguments for processing images using SAM.

    Returns:
        argparse.Namespace: Parsed command-line arguments, which include:
            --cuda-devices (str): Comma-separated CUDA device IDs (or "all") to use.
            --dataset-root (str): Path to the dataset root directory (expects an 'images' folder).
            --checkpoint (str): Path to the SAM checkpoint file.
    """
    parser = argparse.ArgumentParser(
        description="Process images using SAM on CUDA, MPS, or CPU."
    )
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default="all",
        help='Comma-separated list of CUDA device IDs to use (e.g. "0,1") or "all" to use all available devices. (Only used if CUDA is available.)',
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="VL-CMU-CD",
        help="Path to the dataset root directory. Expects an 'images' folder within.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="sam_vit_h_4b8939.pth",
        help="Path to the SAM checkpoint file. Only the ViT-H model checkpoint (sam_vit_h_4b8939.pth) is supported.",
    )
    return parser.parse_args()


def download_checkpoint_if_needed(checkpoint):
    """
    Downloads the SAM checkpoint from a remote URL if it does not exist locally.

    Args:
        checkpoint (str): The local file path for the SAM checkpoint.
    
    Raises:
        requests.HTTPError: If the download request returns a bad status code.
    """
    if not os.path.exists(checkpoint):
        print(f"Downloading checkpoint {checkpoint}...")
        url = f"https://dl.fbaipublicfiles.com/segment_anything/{checkpoint}"
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses
        with open(checkpoint, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def process_images_for_device(image_list, device, checkpoint):
    """
    Loads the SAM model onto the specified device and processes all images in image_list.

    For each image, the SAM predictor is used to compute the image embedding which is then saved
    in the same folder as the image with the file extension replaced by .pt.

    Args:
        image_list (List[str]): List of image file paths to process.
        device (str): The device identifier (e.g., "cuda:0", "mps", or "cpu") on which to load the model.
        checkpoint (str): Path to the SAM checkpoint file.
    """
    print(f"[Device {device}] Loading SAM model from {checkpoint}...")
    sam = build_sam_vit_h(checkpoint).to(device=device)
    predictor = SamPredictor(sam)

    for image_path in tqdm(image_list, desc=f"Device {device}", leave=False):
        # Compute target embedding path by replacing the extension with .pt.
        target_path = os.path.splitext(image_path)[0] + ".pt"
        if os.path.exists(target_path):
            continue

        try:
            with Image.open(image_path) as im:
                im_array = np.array(im)
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        predictor.set_image(im_array)
        emb = predictor.get_image_embedding()
        torch.save(emb, target_path)


def main():
    """
    Main function to process images using the SAM model.

    The function performs the following steps:
      1. Parses command-line arguments.
      2. Downloads the SAM checkpoint if it does not exist.
      3. Configures the available device(s) based on CUDA, MPS, or CPU availability.
      4. Partitions the list of images across the selected devices.
      5. Processes the images in parallel across devices (or in the main thread if a single device is used).
    """
    args = parse_args()
    download_checkpoint_if_needed(args.checkpoint)

    torch.multiprocessing.set_start_method('spawn')

    # Gather all .png images recursively under dataset_root/images.
    images_dir = os.path.join(args.dataset_root, "images")
    image_pattern = os.path.join(images_dir, "**", "*.png")
    all_images = glob.glob(image_pattern, recursive=True)
    if not all_images:
        print(f"No images found with pattern: {image_pattern}")
        return
    print(f"Found {len(all_images)} images.")

    # Device selection logic:
    devices = []
    if torch.cuda.is_available():
        total_cuda = torch.cuda.device_count()
        print(f"Found {total_cuda} CUDA device(s).")
        if args.cuda_devices.lower() == "all":
            devices = [f"cuda:{i}" for i in range(total_cuda)]
        else:
            try:
                selected = [int(x.strip()) for x in args.cuda_devices.split(",")]
            except Exception as e:
                raise ValueError("Invalid CUDA device IDs provided.") from e
            for d in selected:
                if d < 0 or d >= total_cuda:
                    raise ValueError(f"CUDA device ID {d} is out of range.")
                devices.append(f"cuda:{d}")
    elif torch.backends.mps.is_available():
        print("CUDA not available. Using MPS device on macOS.")
        devices = ["mps"]
    else:
        print("Neither CUDA nor MPS devices are available. Falling back to CPU.")
        devices = ["cpu"]

    print("Using device(s):", devices)

    # Partition images across devices (round-robin if more than one).
    partitions = {device: [] for device in devices}
    for idx, image_path in enumerate(all_images):
        device = devices[idx % len(devices)]
        partitions[device].append(image_path)

    # Process images in parallel (if more than one device, use a process pool).
    if len(devices) > 1:
        with ProcessPoolExecutor(max_workers=len(devices)) as executor:
            futures = []
            for device, image_list in partitions.items():
                if image_list:
                    futures.append(
                        executor.submit(process_images_for_device, image_list, device, args.checkpoint)
                    )
            for future in futures:
                future.result()
    else:
        # If only one device, process in the main thread.
        only_device = devices[0]
        process_images_for_device(all_images, only_device, args.checkpoint)


if __name__ == "__main__":
    main()
