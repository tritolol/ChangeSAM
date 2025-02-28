from typing import List

import torch
import torch.nn as nn

from changesam.modeling.base_change_decoder import BaseChangeDecoder


class ChangeDecoderPreDF(BaseChangeDecoder):
    """
    ChangeDecoderPreDF is a variant of BaseChangeDecoder designed for the "pre" configuration.

    This decoder fuses two image embeddings by concatenation, applies a convolutional fusion layer,
    and then runs the transformer and upscaling steps inherited from BaseChangeDecoder. It returns
    only the first mask from the set of predicted masks.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the ChangeDecoderPreDF module.

        Keyword Args:
            All keyword arguments required by BaseChangeDecoder, including:
              - transformer_dim (int): Dimension of the transformer embeddings.
              - transformer (nn.Module): The transformer module used in the decoder.
              - num_multimask_outputs (int): Number of multimask outputs.
              - activation (nn.Module): Activation function for upscaling.
              - iou_head_depth (int): Depth of the IoU prediction head.
              - iou_head_hidden_dim (int): Hidden dimension for the IoU prediction head.
              - etc.

        Additionally, it defines a fusion layer (a 1x1 convolution) that fuses concatenated image embeddings.
        """
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
        """
        Forward pass for the ChangeDecoderPreDF module.

        The forward method performs the following steps:
            1. Prepares tokens by concatenating the learnable prompt tokens with sparse prompt embeddings.
            2. Fuses image embeddings from two sources via concatenation followed by a convolution.
            3. Adds dense prompt embeddings to the fused result.
            4. Expands the positional encoding to match the batch size.
            5. Runs the transformer and upscaling process.
            6. Returns only the first mask from the predicted outputs.

        Args:
            image_embeddings_a (torch.Tensor): First set of image embeddings.
            image_embeddings_b (torch.Tensor): Second set of image embeddings.
            image_pe (torch.Tensor): Positional encodings for the images.
            sparse_prompt_embeddings (torch.Tensor): Sparse prompt embeddings.
            dense_prompt_embeddings (torch.Tensor): Dense prompt embeddings.

        Returns:
            torch.Tensor: The predicted change mask logits with shape adjusted to include only the first mask.
        """
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
    """
    ChangeDecoderPostDF is a variant of BaseChangeDecoder designed for the "post" configuration.

    In this decoder, each image embedding is processed separately through the transformer,
    and the resulting masks are fused using a sequential convolutional fusion layer.
    Additionally, an IoU prediction head is applied to the first token of the transformer output.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the ChangeDecoderPostDF module.

        Keyword Args:
            All keyword arguments required by BaseChangeDecoder.
        
        Additionally, it defines a fusion layer using a sequential model with multiple convolutions
        and ReLU activations to fuse mask outputs from separate processing streams.
        """
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
        """
        Runs the transformer on a given source tensor and processes the outputs to generate masks and IoU predictions.

        Steps:
            1. Passes the input `src` along with positional encodings `pos_src` and tokens through the transformer.
            2. Extracts the IoU token output and mask tokens from the transformer output.
            3. Upscales the transformer output using the output upscaling module.
            4. For each mask token, passes it through its corresponding hypernetwork MLP to generate a mask.
            5. Computes the final mask predictions and the IoU prediction.

        Args:
            src (torch.Tensor): The input tensor after fusion or addition of dense prompt embeddings.
            pos_src (torch.Tensor): The positional encodings expanded to match the batch size.
            tokens (torch.Tensor): The combined token embeddings for the transformer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - masks: The predicted masks.
                - iou_pred: The IoU prediction corresponding to the first token.
        """
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
        """
        Forward pass for the ChangeDecoderPostDF module.

        This forward method executes the following:
            1. Prepares tokens by combining the learnable prompt tokens with sparse prompt embeddings.
            2. Expands positional encodings to match the batch size.
            3. Processes each image embedding separately by adding dense prompt embeddings and running the transformer.
            4. Obtains two sets of masks from the separate transformer runs.
            5. Fuses the two sets of masks using a sequential fusion layer.
            6. Returns the fused mask output.

        Args:
            image_embeddings_a (torch.Tensor): First set of image embeddings.
            image_embeddings_b (torch.Tensor): Second set of image embeddings.
            image_pe (torch.Tensor): Positional encodings for the images.
            sparse_prompt_embeddings (torch.Tensor): Sparse prompt embeddings.
            dense_prompt_embeddings (torch.Tensor): Dense prompt embeddings.

        Returns:
            torch.Tensor: The final fused mask prediction.
        """
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
