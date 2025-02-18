"""
Derived from segment_anything/build_sam.py
"""

from functools import partial
import hashlib
import io

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


def load_and_verify_checkpoint(filepath, expected_hash, hash_algo="sha256"):
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
    # Load the file into memory once.
    with open(filepath, "rb") as f:
        file_data = f.read()

    # Compute the hash on the in-memory bytes.
    computed_hash = hashlib.new(hash_algo, file_data).hexdigest()
    if computed_hash != expected_hash:
        raise ValueError(
            f"Hash mismatch for {filepath}: computed {computed_hash}, expected {expected_hash}"
        )

    # Wrap the in-memory bytes with BytesIO and load the checkpoint.
    buffer = io.BytesIO(file_data)
    checkpoint = torch.load(buffer)
    return checkpoint


def build_changesam_predf_from_sam_vit_h_checkpoint(
    sam_checkpoint, changesam_mask_decoder_checkpoint=None
):
    # Compute and verify the hash for the sam_checkpoint file.
    sam_state_dict = load_and_verify_checkpoint(
        sam_checkpoint, EXPECTED_SAM_VIT_H_CHECKPOINT_HASH
    )

    # Build the RoofSAM model using the provided sam_checkpoint.
    changesam = _build_changesam_pre_df(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        sam_state_dict=sam_state_dict,
    )

    if changesam_mask_decoder_checkpoint is not None:
        state_dict = torch.load(changesam_mask_decoder_checkpoint)
        changesam.mask_decoder.load_state_dict(state_dict)
    else:
        print("Returning ChangeSAM with uninitialized fusion layer!")

    return changesam

def build_changesam_postdf_from_sam_vit_h_checkpoint(
    sam_checkpoint, changesam_mask_decoder_checkpoint=None
):
    # Compute and verify the hash for the sam_checkpoint file.
    sam_state_dict = load_and_verify_checkpoint(
        sam_checkpoint, EXPECTED_SAM_VIT_H_CHECKPOINT_HASH
    )

    # Build the RoofSAM model using the provided sam_checkpoint.
    changesam = _build_changesam_post_df(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        sam_state_dict=sam_state_dict,
    )

    if changesam_mask_decoder_checkpoint is not None:
        state_dict = torch.load(changesam_mask_decoder_checkpoint)
        changesam.mask_decoder.load_state_dict(state_dict)
    else:
        print("Returning ChangeSAM with uninitialized fusion layer!")

    return changesam

def _build_changesam_pre_df(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    sam_state_dict=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    changesam = ChangeSam(
        image_encoder=ImageEncoderViTLoRA(
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
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=ChangeDecoderPreDF(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
    )

    if sam_state_dict is not None:
        missing_keys, unexpected_keys = changesam.load_state_dict(
            sam_state_dict, strict=False
        )
        expected_missing = [
            x.startswith("mask_decoder.fusion_layer") or x.startswith("sparse_prompt_embeddings") for x in missing_keys
        ]
        if not all(expected_missing) or unexpected_keys:
            raise ValueError("SAM state dict doesn't contain the expected keys")

    return changesam

def _build_changesam_post_df(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    sam_state_dict=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    changesam = ChangeSam(
        image_encoder=ImageEncoderViTLoRA(
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
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=ChangeDecoderPostDF(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
    )

    if sam_state_dict is not None:
        missing_keys, unexpected_keys = changesam.load_state_dict(
            sam_state_dict, strict=False
        )
        expected_missing = [
            x.startswith("mask_decoder.fusion_layer") or x.startswith("sparse_prompt_embeddings") for x in missing_keys
        ]
        if not all(expected_missing) or unexpected_keys:
            raise ValueError("SAM state dict doesn't contain the expected keys")

    return changesam
