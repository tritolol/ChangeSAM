from typing import Any, Dict, Union, List
import torch
from torch import nn
from segment_anything.modeling import ImageEncoderViT, PromptEncoder
from torch.nn import functional as F

from changesam.modeling.image_encoder_vit_lora import ImageEncoderViTLoRA
from changesam.modeling.change_decoder_pre_df import ChangeDecoderPreDF
from changesam.modeling.change_decoder_post_df import ChangeDecoderPostDF


class ChangeSam(nn.Module):
    """
    ChangeSam is a change detection segmentation model that uses paired image embeddings along with
    prompt embeddings to predict change mask logits. This variant replaces the dynamic prompt inputs
    with fixed prompt embeddings. The dense prompt embedding is obtained from the given prompt encoder,
    while the sparse prompt embedding is replaced by a learnable parameter that is initialized to
    mimic a centered point prompt with the original padding behavior.
    """

    def __init__(
        self,
        image_encoder: Union[ImageEncoderViT, ImageEncoderViTLoRA],
        prompt_encoder: PromptEncoder,
        mask_decoder: Union[ChangeDecoderPreDF, ChangeDecoderPostDF],
        num_sparse_prompts: int = 4,
    ) -> None:
        """
        Initializes the ChangeSam model.

        Args:
            image_encoder (Union[ImageEncoderViT, ImageEncoderViTLoRA]): Backbone for encoding images into embeddings.
            prompt_encoder (PromptEncoder): Provides the fixed dense prompt embedding.
            mask_decoder (Union[ChangeDecoderPreDF, ChangeDecoderPostDF]): Decodes fused embeddings into change mask logits.
            num_sparse_prompts (int): Total number of sparse prompt embeddings. This number should match the original
                                      number of point embeddings (including the extra padding token) as produced by
                                      the prompt encoder.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.num_sparse_prompts = num_sparse_prompts

        # Initialize learnable sparse prompt embeddings.
        # To mimic the original prompt encoder behavior, we simulate a centered point prompt.
        # In the original implementation, if boxes are None, _embed_points pads the input by adding one extra
        # "not a point" embedding. Thus, to obtain `num_sparse_prompts` total embeddings, we provide
        # (num_sparse_prompts - 1) centered points.
        input_image_size = prompt_encoder.input_image_size  # (H, W)
        H, W = input_image_size
        # Compute center coordinates (in pixel space) adjusted so that when 0.5 is added, they are centered.
        center_x = (W / 2) - 0.5
        center_y = (H / 2) - 0.5

        # Create (1, num_sparse_prompts - 1, 2) tensor filled with the center coordinates.
        default_points = torch.full(
            (1, num_sparse_prompts - 1, 2),
            float(0),
            dtype=torch.float32
        )
        # Set x and y coordinates appropriately.
        default_points[..., 0] = center_x  # x coordinate
        default_points[..., 1] = center_y  # y coordinate

        # Create corresponding labels (using 1 for a positive prompt).
        default_labels = torch.ones((1, num_sparse_prompts - 1), dtype=torch.int64)

        # Call the prompt encoder's internal function with pad=True so that it appends the padding token.
        # The resulting embedding will have shape (1, num_sparse_prompts, embed_dim).
        with torch.no_grad():
            default_sparse = prompt_encoder._embed_points(default_points, default_labels, pad=True)
        # Remove the batch dimension.
        init_sparse = default_sparse.squeeze(0)  # Shape: (num_sparse_prompts, embed_dim)
        embed_dim = init_sparse.shape[-1]

        # Initialize the learnable sparse prompt embeddings.
        self.sparse_prompt_embeddings = nn.Embedding(num_sparse_prompts, embed_dim)
        with torch.no_grad():
            self.sparse_prompt_embeddings.weight.copy_(init_sparse)

    def forward(self, batched_input: Dict[str, Any]) -> torch.Tensor:
        batch_size = batched_input["embeddings_a"].shape[0]

        # Obtain the dense prompt embedding from the prompt encoder.
        dense_embeddings = self.prompt_encoder.get_dense_pe()  # Shape: (1, C, H, W)

        # Expand the learnable sparse prompt embeddings to shape (B, num_sparse_prompts, embed_dim).
        sparse_embeddings = self.sparse_prompt_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)

        mask = self.mask_decoder(
            image_embeddings_a=batched_input["embeddings_a"],
            image_embeddings_b=batched_input["embeddings_b"],
            image_pe=dense_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )
        return mask
    
    def forward_with_images(self, batched_input: Dict[str, Any]) -> torch.Tensor:
        """
        Processes raw images through the image encoder to obtain embeddings, and then
        produces change mask logits by calling forward().

        Args:
            batched_input (Dict[str, Any]): A dictionary containing:
                - "images_a" (torch.Tensor): Raw images for view A.
                - "images_b" (torch.Tensor): Raw images for view B.

        Returns:
            torch.Tensor: Predicted change mask logits.
        """
        # Obtain image embeddings from the image encoder.


        embeddings_a = self.image_encoder(batched_input["images_a"])
        embeddings_b = self.image_encoder(batched_input["images_b"])
        
        # Build a new batched input dict with the embeddings.
        new_batched_input = {
            "embeddings_a": embeddings_a,
            "embeddings_b": embeddings_b,
        }
        # Call the fixed-prompt forward method.
        return self(new_batched_input)
