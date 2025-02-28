"""
Derived from segment_anything/build_sam.py

This module provides functionality to build and configure a ChangeSam model,
which leverages a modified SAM (Segment Anything Model) with LoRA adaptations
for change detection tasks. It includes functions to load and verify checkpoints,
construct image encoders with LoRA, and build ChangeSam with selectable mask decoders.
"""

from functools import partial
import hashlib
import io
from typing import Any, Callable, List, Optional, Union

import torch

from segment_anything.modeling import (
    PromptEncoder,
    TwoWayTransformer,
)
from changesam.modeling.changesam import ChangeSam
from changesam.modeling.image_encoder_vit_lora import ImageEncoderViTLoRA
from changesam.modeling.tiny_vit_lora import TinyViTLoRA
from changesam.modeling.change_decoders import ChangeDecoderPreDF, ChangeDecoderPostDF

# The actual expected SHA256 hash of the sam_vit_h_4b8939.pth file.
EXPECTED_SAM_VIT_H_CHECKPOINT_HASH = (
    "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
)

EXPECTED_MOBILESAM_VIT_TCHECKPOINT_HASH = (
    "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
)


def load_and_verify_checkpoint(
    filepath: str, expected_hash: str, hash_algo: str = "sha256"
) -> Any:
    """
    Loads a checkpoint file, verifies its integrity via hash comparison, and returns the state dict.

    This function reads the checkpoint file as bytes, computes its hash using the specified
    algorithm, and compares it against the expected hash. If the hashes match, it loads the
    checkpoint via torch.load; otherwise, it raises a ValueError.

    Args:
        filepath (str): Path to the checkpoint file.
        expected_hash (str): The expected hash string.
        hash_algo (str): The hashing algorithm to use (default is 'sha256').

    Returns:
        Any: The checkpoint loaded via torch.load (typically a state dict).

    Raises:
        ValueError: If the computed hash does not match the expected hash.
    """
    with open(filepath, "rb") as f:
        file_data = f.read()

    computed_hash = hashlib.new(hash_algo, file_data).hexdigest()
    if computed_hash != expected_hash:
        raise ValueError(
            f"Hash mismatch for {filepath}: computed {computed_hash}, expected {expected_hash}"
        )

    buffer = io.BytesIO(file_data)
    checkpoint = torch.load(buffer)
    return checkpoint


def _build_image_encoder_sam_vit_h_lora(
    lora_layers: List[int] | None = None,
    lora_r: int = 0,
    lora_alpha: float = 1,
) -> ImageEncoderViTLoRA:
    """
    Constructs an ImageEncoderViTLoRA model configured for SAM ViT-H.

    Args:
        lora_layers (List[int] | None): Optional list of transformer layer indices to apply LoRA adaptation.
        lora_r (int): LoRA rank.
        lora_alpha (float): LoRA scaling factor.

    Returns:
        ImageEncoderViTLoRA: An instance of ImageEncoderViTLoRA configured for SAM ViT-H.
    """
    return ImageEncoderViTLoRA(
        depth=32,
        embed_dim=1280,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=16,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[7, 15, 23, 31],
        window_size=14,
        out_chans=256,
        lora_layers=lora_layers,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )


def _build_image_encoder_mobile_sam_vit_t_lora(
    lora_layers: List[int] | None = None,
    lora_r: int = 0,
    lora_alpha: float = 1,
) -> TinyViTLoRA:
    """
    Constructs a TinyViTLoRA model configured for Mobile SAM ViT-T.

    Args:
        lora_layers (List[int] | None): Optional list of transformer layer indices to apply LoRA adaptation.
        lora_r (int): LoRA rank.
        lora_alpha (float): LoRA scaling factor.

    Returns:
        TinyViTLoRA: An instance of TinyViTLoRA configured for Mobile SAM ViT-T.
    """
    return TinyViTLoRA(
        img_size=1024,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
        lora_layers=lora_layers,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )


def _build_changesam_common(
    mask_decoder_builder: Callable[[], Union[ChangeDecoderPreDF, ChangeDecoderPostDF]],
    image_encoder: Union[ImageEncoderViTLoRA, TinyViTLoRA],
    image_embedding_size: int,
    image_size: int,
    prompt_embed_dim: int,
    num_sparse_prompts: int,
    prompt_init_strategy: str,
    sam_state_dict: Optional[Any] = None,
) -> ChangeSam:
    """
    Constructs a ChangeSam model with the specified image encoder, prompt encoder, and mask decoder.

    This function creates the prompt encoder, builds the mask decoder using the provided builder function,
    and instantiates a ChangeSam model. If a SAM state dict is provided, it loads the checkpoint into the
    model, verifies the keys, and freezes all parameters except those corresponding to the missing keys.

    Args:
        mask_decoder_builder (Callable[[], Union[ChangeDecoderPreDF, ChangeDecoderPostDF]]):
            A callable that returns an instance of the desired mask decoder.
        image_encoder (Union[ImageEncoderViTLoRA, TinyViTLoRA]): The image encoder model instance.
        image_embedding_size (int): The size of the image embedding.
        image_size (int): The spatial size of the input images.
        prompt_embed_dim (int): The dimension of the prompt embeddings.
        num_sparse_prompts (int): The number of sparse prompt tokens.
        prompt_init_strategy (str): The strategy for initializing prompt embeddings.
        sam_state_dict (Optional[Any]): Optional SAM state dict for loading pretrained weights.

    Returns:
        ChangeSam: The constructed ChangeSam model.
    
    Raises:
        ValueError: If the loaded SAM state dict does not contain the expected keys.
    """
    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    mask_decoder = mask_decoder_builder()

    changesam = ChangeSam(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        num_sparse_prompts=num_sparse_prompts,
        prompt_init_strategy=prompt_init_strategy
    )

    if sam_state_dict is not None:
        missing_keys, unexpected_keys = changesam.load_state_dict(
            sam_state_dict, strict=False
        )
        # Allow missing keys for fusion layer and sparse prompt embeddings.
        expected_missing = [
            x.startswith("mask_decoder.fusion_layer")
            or x.startswith("sparse_prompt_embeddings")
            or "lora_" in x
            for x in missing_keys
        ]
        if not all(expected_missing) or unexpected_keys:
            raise ValueError("SAM state dict doesn't contain the expected keys")
        
        changesam.freeze_except(missing_keys)

    return changesam


def build_changesam(
    sam_checkpoint: str,
    encoder_type: str = "sam_vit_h",  # Options: "sam_vit_h" or "mobile_sam_vit_t"
    decoder_type: str = "predf",       # Options: "predf" or "postdf"
    lora_layers: List[int] | None = None,
    lora_r: int = 0,
    lora_alpha: float = 1,
    num_sparse_prompts: int = 4,
    prompt_init_strategy: str = "center",
) -> ChangeSam:
    """
    Builds and returns a ChangeSam model with the specified configuration.

    This function selects an image encoder based on the provided encoder type (either SAM ViT-H or Mobile SAM ViT-T),
    verifies the corresponding SAM checkpoint, and builds a mask decoder based on the decoder type.
    It then constructs a ChangeSam model using common configuration parameters.

    Args:
        sam_checkpoint (str): Path to the SAM checkpoint file.
        encoder_type (str): Which image encoder to use. Options are "sam_vit_h" (for ImageEncoderViTLoRA)
            or "mobile_sam_vit_t" (for TinyViTLoRA). Default is "sam_vit_h".
        decoder_type (str): Which mask decoder to use. Options are "predf" (for ChangeDecoderPreDF)
            or "postdf" (for ChangeDecoderPostDF). Default is "predf".
        lora_layers (List[int] | None): List of transformer layer indices to apply LoRA adaptation. Default is None.
        lora_r (int): LoRA rank. Default is 0.
        lora_alpha (float): LoRA scaling factor. Default is 1.
        num_sparse_prompts (int): Number of sparse prompt tokens. Default is 4.
        prompt_init_strategy (str): Strategy for initializing prompt embeddings (e.g., "center", "grid", "random", "random_embedding").
            Default is "center".

    Returns:
        ChangeSam: The constructed ChangeSam model.
    
    Raises:
        ValueError: If an unsupported encoder_type or decoder_type is provided.
    """
    # Select the image encoder.
    if encoder_type == "sam_vit_h":
        # Load and verify the SAM checkpoint for SAM ViT-H.
        sam_state_dict = load_and_verify_checkpoint(
            sam_checkpoint, EXPECTED_SAM_VIT_H_CHECKPOINT_HASH
        )
        image_encoder = _build_image_encoder_sam_vit_h_lora(
            lora_layers=lora_layers, lora_r=lora_r, lora_alpha=lora_alpha
        )
    elif encoder_type == "mobile_sam_vit_t":
        # Load and verify the SAM checkpoint for Mobile SAM ViT-T.
        sam_state_dict = load_and_verify_checkpoint(
            sam_checkpoint, EXPECTED_MOBILESAM_VIT_TCHECKPOINT_HASH
        )
        image_encoder = _build_image_encoder_mobile_sam_vit_t_lora(
            lora_layers=lora_layers, lora_r=lora_r, lora_alpha=lora_alpha
        )
    else:
        raise ValueError(f"Unsupported encoder_type: {encoder_type}")

    # Common configuration parameters.
    image_size = 1024
    image_embedding_size = 64
    prompt_embed_dim = 256

    # Select the mask decoder builder.
    if decoder_type == "predf":
        mask_decoder_builder = lambda: ChangeDecoderPreDF(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
    elif decoder_type == "postdf":
        mask_decoder_builder = lambda: ChangeDecoderPostDF(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
    else:
        raise ValueError(f"Unsupported decoder_type: {decoder_type}")

    changesam = _build_changesam_common(
        mask_decoder_builder=mask_decoder_builder,
        image_encoder=image_encoder,
        prompt_embed_dim=prompt_embed_dim,
        image_size=image_size,
        image_embedding_size=image_embedding_size,
        num_sparse_prompts=num_sparse_prompts,
        prompt_init_strategy=prompt_init_strategy,
        sam_state_dict=sam_state_dict,
    )

    return changesam
