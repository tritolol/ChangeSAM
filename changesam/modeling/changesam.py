from typing import Any, Dict, Union, List, Tuple
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np


class ChangeSam(nn.Module):
    """
    ChangeSam is a change detection segmentation model that uses paired image embeddings along with
    prompt embeddings to predict change mask logits. This variant replaces the dynamic prompt inputs
    with fixed prompt embeddings. The dense prompt embedding is obtained from the given prompt encoder,
    while the sparse prompt embedding is replaced by a learnable parameter.
    """

    def __init__(
        self,
        image_encoder: Union[Any, Any],  # Replace Any with your actual encoder types.
        prompt_encoder: Any,  # Replace Any with your PromptEncoder type.
        mask_decoder: Union[Any, Any],  # Replace Any with your mask decoder types.
        num_sparse_prompts: int = 4,
        original_image_size: Tuple[int, int] = (768, 1024),
        prompt_init_strategy: str = "center",  # Options: "center", "grid", "random", "random_embedding"
    ) -> None:
        """
        Initializes the ChangeSam model.

        Args:
            image_encoder: Backbone for encoding images into embeddings.
            prompt_encoder: Provides the fixed dense prompt embedding.
            mask_decoder: Decodes fused embeddings into change mask logits.
            num_sparse_prompts (int): Total number of sparse prompt embeddings.
            original_image_size (Tuple[int, int]): Size of the input image.
            prompt_init_strategy (str): Strategy for initializing the sparse prompt embeddings.
                                        "center" (default): center positive points.
                                        "grid": uniformly distributed positive points.
                                        "random": randomly sampled point coordinates.
                                        "random_embedding": directly initialize from torch.randn.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.num_sparse_prompts = num_sparse_prompts
        self.original_image_size = original_image_size
        self.prompt_init_strategy = prompt_init_strategy

        self.parameters_to_adapt = []
        self.initialize_prompt_embeddings()

    def forward(self, batched_input: Dict[str, Any]) -> torch.Tensor:
        batch_size = batched_input["embeddings_a"].shape[0]

        # Expand the learnable sparse prompt embeddings to shape (B, num_sparse_prompts, embed_dim).
        sparse_embeddings = self.sparse_prompt_embeddings.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        _, dense_embeddings = self.prompt_encoder(points=None, boxes=None, masks=None)

        mask = self.mask_decoder(
            image_embeddings_a=batched_input["embeddings_a"],
            image_embeddings_b=batched_input["embeddings_b"],
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )

        mask = self.postprocess_masks(mask).squeeze(1)

        return mask

    def initialize_prompt_embeddings(self):
        """
        Initialize learnable sparse prompt embeddings based on the chosen strategy.
        The current strategies include:
         - "center": Use centered positive points.
         - "grid": Generate a uniform grid of positive points.
         - "random": Randomly sample point coordinates within the image.
         - "random_embedding": Directly initialize prompt embeddings using torch.randn.
        """
        input_image_size = self.prompt_encoder.input_image_size  # (H, W)
        H, W = input_image_size

        # Create the embedding module. We add one extra token if needed by the prompt encoder.
        self.sparse_prompt_embeddings = nn.Embedding(
            self.num_sparse_prompts + 1, self.prompt_encoder.embed_dim
        )

        with torch.no_grad():
            if self.prompt_init_strategy == "none":
                return
            elif self.prompt_init_strategy == "center":
                # Center strategy: All prompts are set to the image center.
                center_x = W / 2
                center_y = H / 2
                default_points = torch.full(
                    (1, self.num_sparse_prompts, 2), 0.0, dtype=torch.float32
                )
                default_points[..., 0] = center_x  # x coordinate
                default_points[..., 1] = center_y  # y coordinate
                default_labels = torch.ones(
                    (1, self.num_sparse_prompts), dtype=torch.int64
                )
                default_sparse = self.prompt_encoder._embed_points(
                    default_points, default_labels, pad=True
                )

            elif self.prompt_init_strategy == "grid":
                # Grid strategy: Generate evenly spaced positive points.
                num_points = self.num_sparse_prompts
                num_cols = int(np.ceil(np.sqrt(num_points)))
                num_rows = int(np.ceil(num_points / num_cols))
                margin = 0.0  # Adjust margin if desired.
                xs = torch.linspace(margin, W - margin, steps=num_cols + 2)[1:-1]
                ys = torch.linspace(margin, H - margin, steps=num_rows + 2)[1:-1]
                mesh_x, mesh_y = torch.meshgrid(xs, ys, indexing="ij")
                grid_points = torch.stack([mesh_x.flatten(), mesh_y.flatten()], dim=1)[
                    :num_points
                ].unsqueeze(0)
                default_labels = torch.ones((1, num_points), dtype=torch.int64)
                default_sparse = self.prompt_encoder._embed_points(
                    grid_points, default_labels, pad=True
                )

            elif self.prompt_init_strategy == "random":
                # Random strategy: Sample random points uniformly over the image.
                num_points = self.num_sparse_prompts
                random_points = torch.zeros((1, num_points, 2), dtype=torch.float32)
                random_points[..., 0] = torch.rand(num_points) * W
                random_points[..., 1] = torch.rand(num_points) * H
                default_labels = torch.ones((1, num_points), dtype=torch.int64)
                default_sparse = self.prompt_encoder._embed_points(
                    random_points, default_labels, pad=True
                )

            elif self.prompt_init_strategy == "random_embedding":
                # Random embedding strategy: Directly initialize embeddings with values from torch.randn.
                init_values = torch.randn(
                    self.num_sparse_prompts + 1, self.prompt_encoder.embed_dim
                )
                self.sparse_prompt_embeddings.weight.copy_(init_values)
                return  # Early return since initialization is complete.

            else:
                raise ValueError(
                    f"Unknown token_init_strategy: {self.prompt_init_strategy}"
                )

            # For the first three strategies, copy the computed prompt embeddings.
            init_sparse = default_sparse.squeeze(0).detach().clone()
            self.sparse_prompt_embeddings.weight.copy_(init_sparse)

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

        return masks

    def freeze_except(self, adapt_param_names: List[str]) -> None:
        """
        Freezes all parameters except those whose names contain any of the substrings
        in the provided list.

        Args:
            adapt_param_names (List[str]): List of parameter name strings that should
                remain trainable.
        """
        print("freezing all but these parameters: " + ", ".join(adapt_param_names))

        self.parameters_to_adapt = adapt_param_names

        for name, param in self.named_parameters():
            if any(adapt_str in name for adapt_str in adapt_param_names):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def save_adapted_checkpoint(
        self, filepath: Union[str, None] = None
    ) -> Dict[str, Any]:
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

    def load_adapted_checkpoint(
        self, checkpoint: Union[str, Dict[str, Any]], strict: bool = True
    ) -> None:
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
            if (
                any(adapt_str in name for adapt_str in self.parameters_to_adapt)
                and name not in checkpoint
            ):
                missing_keys.append(name)

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Adapted checkpoint load failed. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
            )
