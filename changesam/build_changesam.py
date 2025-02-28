"""
Derived from segment_anything/build_sam.py
"""

from functools import partial
import hashlib
import io
from typing import Any, Callable, List, Optional,  Union

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
    Loads the checkpoint file into memory, computes its hash, verifies it,
    and then loads the checkpoint as a torch state dict.

    Args:
        filepath (str): Path to the checkpoint file.
        expected_hash (str): The expected hash string.
        hash_algo (str): The hashing algorithm to use (default is 'sha256').

    Returns:
        The checkpoint loaded via torch.load if the hash matches.

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
    Builds a ChangeSam model with selectable image encoder and mask decoder modules.

    Args:
        sam_checkpoint (str): Path to the SAM ViT-H checkpoint.
        encoder_type (str): Which image encoder to use. Either "sam_vit_h" (for ImageEncoderViTLoRA)
            or "mobile_sam_vit_t" (for TinyViTLoRA).
        decoder_type (str): Which mask decoder to use. Either "predf" (for ChangeDecoderPreDF)
            or "postdf" (for ChangeDecoderPostDF).
        lora_layers (List[int] | None): List of block indices to apply LoRA adaptation.
        lora_r (int): LoRA rank.
        lora_alpha (float): LoRA scaling factor.
        num_sparse_prompts (int): The number of prompt tokens.
        prompt_init_strategy (str): The prompt initialization strategy.

    Returns:
        ChangeSam: The constructed ChangeSam model.
    """

    # Select the image encoder
    if encoder_type == "sam_vit_h":
        # Load and verify the SAM checkpoint
        sam_state_dict = load_and_verify_checkpoint(
            sam_checkpoint, EXPECTED_SAM_VIT_H_CHECKPOINT_HASH
        )
        image_encoder = _build_image_encoder_sam_vit_h_lora(
            lora_layers=lora_layers, lora_r=lora_r, lora_alpha=lora_alpha
        )
    elif encoder_type == "mobile_sam_vit_t":
        # Load and verify the SAM checkpoint
        sam_state_dict = load_and_verify_checkpoint(
            sam_checkpoint, EXPECTED_MOBILESAM_VIT_TCHECKPOINT_HASH
        )
        image_encoder = _build_image_encoder_mobile_sam_vit_t_lora(
            lora_layers=lora_layers, lora_r=lora_r, lora_alpha=lora_alpha
        )
    else:
        raise ValueError(f"Unsupported encoder_type: {encoder_type}")

    # Common configuration parameters
    image_size = 1024
    image_embedding_size = 64
    prompt_embed_dim = 256

    # Select the mask decoder builder
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
