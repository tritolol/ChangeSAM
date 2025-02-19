from typing import Any, Dict, Union, List, Tuple
import torch
from torch import nn
from segment_anything.modeling import ImageEncoderViT, PromptEncoder
from torch.nn import functional as F

from mobile_sam.modeling.tiny_vit_sam import TinyViT

from changesam.modeling.image_encoder_vit_lora import ImageEncoderViTLoRA
from changesam.modeling.tiny_vit_lora import TinyViTLoRA
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
        image_encoder: Union[ImageEncoderViT, ImageEncoderViTLoRA, TinyViT, TinyViTLoRA],
        prompt_encoder: PromptEncoder,
        mask_decoder: Union[ChangeDecoderPreDF, ChangeDecoderPostDF],
        num_sparse_prompts: int = 4,
        original_image_size: Tuple[int, int] = (768, 1024)
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
        self.original_image_size = original_image_size

        self.parameters_to_adapt = []

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
        ).unsqueeze(1)

        mask = self.postprocess_masks(mask).squeeze(1)

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

    def postprocess_masks(
        self,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : self.original_image_size[0], : self.original_image_size[1]]
        masks = F.interpolate(masks, self.original_image_size, mode="bilinear", align_corners=False)
        return masks

    def freeze_except(self, adapt_param_names: List[str]) -> None:
        """
        Freezes all parameters except those whose names contain any of the substrings
        in the provided list.

        Args:
            adapt_param_names (List[str]): List of parameter name strings that should
                remain trainable.
        """
        print("freezing all but these parameters: " + ', '.join(adapt_param_names))

        self.parameters_to_adapt = adapt_param_names

        for name, param in self.named_parameters():
            if any(adapt_str in name for adapt_str in adapt_param_names):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def save_adapted_checkpoint(self, filepath: Union[str, None] = None) -> Dict[str, Any]:
        """
        Saves a checkpoint containing only the adapted parameters.

        Args:
            filepath (str, optional): If provided, the checkpoint will be saved to this file.

        Returns:
            A dict mapping parameter names to parameter tensors (on CPU) for all adapted parameters.
        """
        # Clone each parameter tensor to ensure a deep copy is made.
        adapted_state = {
            name: param.detach().cpu().clone()
            for name, param in self.named_parameters()
            if any(adapt_str in name for adapt_str in self.parameters_to_adapt)
        }
        if filepath is not None:
            torch.save(adapted_state, filepath)
        return adapted_state

    def load_adapted_checkpoint(self, checkpoint: Union[str, Dict[str, Any]], strict: bool = True) -> None:
        """
        Loads a checkpoint containing only the adapted parameters into the model.

        Args:
            checkpoint (str or dict): Either the path to a checkpoint file or a state dict.
            strict (bool): If True, raises an error if the checkpoint contains unexpected keys or is missing
                        expected keys.

        Raises:
            RuntimeError: If strict is True and there is a mismatch between the adapted parameters in the model
                        and those in the checkpoint.
        """
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location="cpu")

        own_state = self.state_dict()
        missing_keys = []
        unexpected_keys = []

        for name, param in checkpoint.items():
            if name in own_state:
                try:
                    own_state[name].copy_(param)
                except Exception as e:
                    raise RuntimeError(f"Error copying parameter {name}: {e}")
            else:
                unexpected_keys.append(name)

        # Check that every adapted parameter in the model is present in the checkpoint.
        for name in own_state:
            if any(adapt_str in name for adapt_str in self.parameters_to_adapt) and name not in checkpoint:
                missing_keys.append(name)

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Adapted checkpoint load failed. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
            )

