from typing import List

import torch
import torch.nn as nn

from changesam.modeling.base_change_decoder import BaseChangeDecoder


class ChangeDecoderPreDF(BaseChangeDecoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Specific fusion layer for pre-decoder.
        self.fusion_layer = nn.Conv2d(2 * self.transformer_dim, self.transformer_dim, 1)

    def forward(
        self,
        image_embeddings_a: torch.Tensor,
        image_embeddings_b: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.prepare_tokens(sparse_prompt_embeddings)
        # Fuse image embeddings via concatenation and convolution.
        fusion_input = torch.cat((image_embeddings_a, image_embeddings_b), dim=1)
        src = self.fusion_layer(fusion_input) + dense_prompt_embeddings

        # Expand positional encoding to match tokens.
        pos_src = image_pe.expand((tokens.shape[0], -1, -1, -1))
        masks, _ = self.run_transformer_and_upscale(src, pos_src, tokens)
        # Return only the first mask as required.
        return masks[:, 0, :].unsqueeze(0)


class ChangeDecoderPostDF(BaseChangeDecoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Specific fusion layer for post-decoder.
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )

    def run_transformer(
        self, src: torch.Tensor, pos_src: torch.Tensor, tokens: torch.Tensor
    ):
        # This helper can be used to process each src separately.
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
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
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred

    def forward(
        self,
        image_embeddings_a: torch.Tensor,
        image_embeddings_b: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.prepare_tokens(sparse_prompt_embeddings)
        pos_src = image_pe.expand((tokens.shape[0], -1, -1, -1))

        # Process each image embedding separately.
        src_a = image_embeddings_a + dense_prompt_embeddings
        src_b = image_embeddings_b + dense_prompt_embeddings
        masks_a, _ = self.run_transformer(src_a, pos_src, tokens)
        masks_b, _ = self.run_transformer(src_b, pos_src, tokens)

        # Fuse the outputs via the fusion layer.
        input_masks = torch.cat((masks_a, masks_b), dim=1)
        fused_mask = self.fusion_layer(input_masks).squeeze(-1)
        return fused_mask
