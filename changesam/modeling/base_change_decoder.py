from typing import List

import torch
from torch import nn

from segment_anything.modeling.mask_decoder import MaskDecoder


class BaseChangeDecoder(MaskDecoder):
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
        super().__init__(
            transformer_dim=transformer_dim,
            transformer=transformer,
            num_multimask_outputs=num_multimask_outputs,
            activation=activation,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
        )

    def prepare_tokens(self, sparse_prompt_embeddings: torch.Tensor) -> torch.Tensor:
        # Common token concatenation logic.
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
        # Common transformer call and upscaling logic.
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
        return masks, hs  # or other outputs if needed

    def forward(
        self,
        image_embeddings_a: torch.Tensor,
        image_embeddings_b: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # Subclasses can decide how to fuse or modify the inputs
        raise NotImplementedError("Subclasses must implement the forward method.")
