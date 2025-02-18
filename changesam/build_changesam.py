"""
Derived from segment_anything/build_sam.py
"""

from functools import partial
import hashlib
import io
from typing import Any, Callable, List, Optional, Tuple, Union

import torch

from segment_anything.modeling import (
    ImageEncoderViT,
    PromptEncoder,
    TwoWayTransformer,
)
from changesam.modeling.changesam import ChangeSam
from changesam.modeling.image_encoder_vit_lora import ImageEncoderViTLoRA
from changesam.modeling.change_decoder_pre_df import ChangeDecoderPreDF
from changesam.modeling.change_decoder_post_df import ChangeDecoderPostDF

# The actual expected SHA256 hash of the sam_vit_h_4b8939.pth file.
EXPECTED_SAM_VIT_H_CHECKPOINT_HASH = (
    "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
)


def load_and_verify_checkpoint(filepath: str, expected_hash: str, hash_algo: str = "sha256") -> Any:
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


def _build_changesam_common(
    mask_decoder_builder: Callable[[], Union[ChangeDecoderPreDF, ChangeDecoderPostDF]],
    encoder_embed_dim: int,
    encoder_depth: int,
    encoder_num_heads: int,
    encoder_global_attn_indexes: List[int],
    sam_state_dict: Optional[Any] = None,
    lora_layers: List[int] | None = None,
    lora_r: int = 0,
    lora_alpha: float = 1
) -> ChangeSam:
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    image_encoder = ImageEncoderViTLoRA(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
        lora_layers = lora_layers,
        lora_r = lora_r,
        lora_alpha = lora_alpha
    )

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
    )

    if sam_state_dict is not None:
        missing_keys, unexpected_keys = changesam.load_state_dict(sam_state_dict, strict=False)
        # Allow missing keys for fusion layer and sparse prompt embeddings.
        expected_missing = [
            x.startswith("mask_decoder.fusion_layer") or 
            x.startswith("sparse_prompt_embeddings") or
            "lora" in x
            for x in missing_keys
        ]
        if not all(expected_missing) or unexpected_keys:
            raise ValueError("SAM state dict doesn't contain the expected keys")
    return changesam


def build_changesam_predf_from_sam_vit_h_checkpoint(
    sam_checkpoint: str, 
    changesam_mask_decoder_checkpoint: Optional[str] = None,
    lora_layers: List[int] | None = None,
    lora_r: int = 0,
    lora_alpha: float = 1
) -> ChangeSam:
    """
    Builds a pre-DF ChangeSam model from a SAM ViT-H checkpoint.
    
    Args:
        sam_checkpoint (str): Path to the SAM ViT-H checkpoint.
        changesam_mask_decoder_checkpoint (str, optional): Path to a checkpoint for the mask decoder.
    
    Returns:
        ChangeSam: The constructed ChangeSam model.
    """
    sam_state_dict = load_and_verify_checkpoint(sam_checkpoint, EXPECTED_SAM_VIT_H_CHECKPOINT_HASH)
    changesam = _build_changesam_common(
        mask_decoder_builder=lambda: ChangeDecoderPreDF(
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
        ),
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        sam_state_dict=sam_state_dict,
        lora_layers=lora_layers,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )
    if changesam_mask_decoder_checkpoint is not None:
        state_dict = torch.load(changesam_mask_decoder_checkpoint)
        changesam.mask_decoder.load_state_dict(state_dict)
    else:
        print("Returning ChangeSAM with uninitialized fusion layer!")
    return changesam


def build_changesam_postdf_from_sam_vit_h_checkpoint(
    sam_checkpoint: str, 
    changesam_mask_decoder_checkpoint: Optional[str] = None,
    lora_layers: List[int] | None = None,
    lora_r: int = 0,
    lora_alpha: float = 1
) -> ChangeSam:
    """
    Builds a post-DF ChangeSam model from a SAM ViT-H checkpoint.
    
    Args:
        sam_checkpoint (str): Path to the SAM ViT-H checkpoint.
        changesam_mask_decoder_checkpoint (str, optional): Path to a checkpoint for the mask decoder.
    
    Returns:
        ChangeSam: The constructed ChangeSam model.
    """
    sam_state_dict = load_and_verify_checkpoint(sam_checkpoint, EXPECTED_SAM_VIT_H_CHECKPOINT_HASH)
    changesam = _build_changesam_common(
        mask_decoder_builder=lambda: ChangeDecoderPostDF(
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
        ),
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        sam_state_dict=sam_state_dict,
        lora_layers=lora_layers,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )
    if changesam_mask_decoder_checkpoint is not None:
        state_dict = torch.load(changesam_mask_decoder_checkpoint)
        changesam.mask_decoder.load_state_dict(state_dict)
    else:
        print("Returning ChangeSAM with uninitialized fusion layer!")
    return changesam
