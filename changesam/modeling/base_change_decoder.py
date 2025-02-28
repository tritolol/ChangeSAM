from typing import List

import torch
from torch import nn

from segment_anything.modeling.mask_decoder import MaskDecoder


class BaseChangeDecoder(MaskDecoder):
    """
    BaseChangeDecoder extends the functionality of MaskDecoder by providing common methods
    for processing prompt tokens, running the transformer, and upscaling outputs.

    This base class is designed to be subclassed by specific decoder implementations that
    fuse or modify image and prompt embeddings in different ways.
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: nn.Module = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256
    ):
        """
        Initializes the BaseChangeDecoder with transformer parameters and output configuration.

        Args:
            transformer_dim (int): The channel dimension used in the transformer.
            transformer (nn.Module): The transformer module used for decoding.
            num_multimask_outputs (int, optional): Number of mask outputs when disambiguating masks.
                                                   Default is 3.
            activation (nn.Module, optional): Activation function used during upscaling.
                                              Default is nn.GELU.
            iou_head_depth (int, optional): The depth of the MLP for predicting mask quality.
                                            Default is 3.
            iou_head_hidden_dim (int, optional): Hidden dimension size for the IoU prediction MLP.
                                                 Default is 256.
        """
        super().__init__(
            transformer_dim=transformer_dim,
            transformer=transformer,
            num_multimask_outputs=num_multimask_outputs,
            activation=activation,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
        )

    def prepare_tokens(self, sparse_prompt_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Prepares the token embeddings for the transformer by concatenating the learnable IoU and mask tokens
        with the provided sparse prompt embeddings.

        This method performs the following:
            1. Concatenates the IoU token weights and mask token weights.
            2. Expands the concatenated tokens to match the batch size of the sparse prompt embeddings.
            3. Concatenates the expanded tokens with the sparse prompt embeddings along the token dimension.

        Args:
            sparse_prompt_embeddings (torch.Tensor): A tensor of sparse prompt embeddings with shape
                                                     [batch_size, num_sparse_prompts, embed_dim].

        Returns:
            torch.Tensor: The combined token tensor to be used as input to the transformer, with shape
                          [batch_size, num_total_tokens, embed_dim], where num_total_tokens is the sum
                          of the number of base tokens and sparse prompt tokens.
        """
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        return torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

    def run_transformer_and_upscale(
        self, src: torch.Tensor, pos_src: torch.Tensor, tokens: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Runs the transformer on the input source tensor and upscales the output embeddings.

        The process includes:
            1. Running the transformer with the given source, positional encodings, and tokens.
            2. Extracting the mask tokens from the transformer output.
            3. Reshaping and upscaling the source output using the output upscaling module.
            4. Generating per-token mask predictions by applying hypernetwork MLPs on each mask token.
            5. Producing final masks by performing a matrix multiplication between the hypernetwork outputs
               and the upscaled embeddings.

        Args:
            src (torch.Tensor): The fused or processed image embeddings with shape [B, C, H, W].
            pos_src (torch.Tensor): Positional encodings expanded to shape [B, C, H, W] or similar.
            tokens (torch.Tensor): The combined token embeddings for the transformer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - masks (torch.Tensor): The predicted masks with shape [B, num_mask_tokens, H_upscaled, W_upscaled].
                - hs (torch.Tensor): The hidden states output from the transformer, which may be used for further processing.
        """
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        b, c, h, w = upscaled_embedding.shape

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        return masks, hs

    def forward(
        self,
        image_embeddings_a: torch.Tensor,
        image_embeddings_b: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Abstract forward method to be implemented by subclasses.

        This method is intended to process image embeddings along with prompt embeddings,
        run the transformer and upscaling steps, and return the predicted change masks.
        Subclasses should override this method to define their own fusion strategies or modifications
        to the inputs before passing them through the transformer.

        Args:
            image_embeddings_a (torch.Tensor): The first set of image embeddings.
            image_embeddings_b (torch.Tensor): The second set of image embeddings.
            image_pe (torch.Tensor): The positional encodings corresponding to the image embeddings.
            sparse_prompt_embeddings (torch.Tensor): The sparse prompt embeddings.
            dense_prompt_embeddings (torch.Tensor): The dense prompt embeddings.

        Returns:
            torch.Tensor: The predicted change mask logits.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")
