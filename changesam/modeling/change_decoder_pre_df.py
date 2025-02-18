"""
Derived from segment_anything/modeling/mask_decoder.py
"""
from typing import List
from segment_anything.modeling.mask_decoder import MaskDecoder
import torch
from torch import nn


class ChangeDecoderPreDF(MaskDecoder):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: nn.Module = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts class probability logits given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_classes (int): the number of classes to predict
          mlp_hidden_dim (int): the hidden dimension size of the class prediction MLP
          mlp_layers (int): the number of layers of the class prediction MLP
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__(
            transformer_dim=transformer_dim,
            transformer=transformer,
            num_multimask_outputs=num_multimask_outputs,
            activation=activation,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
        )
        self.fusion_layer = nn.Linear(2 * transformer_dim, transformer_dim)

    def forward(
        self,
        image_embeddings_a: torch.Tensor,
        image_embeddings_b: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict class logits given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs

        Returns:
          torch.Tensor: batched predicted class logits
        """
        mask = self.predict_masks(
            image_embeddings_a=image_embeddings_a,
            image_embeddings_b=image_embeddings_b,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        return mask

    def predict_masks(
        self,
        image_embeddings_a: torch.Tensor,
        image_embeddings_b: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 
        src_a = image_embeddings_a + dense_prompt_embeddings
        src_b = image_embeddings_b + dense_prompt_embeddings
        fusion_input = torch.cat((src_a, src_b), dim=1).permute((0, 2, 3, 1))
        src = self.fusion_layer(fusion_input).permute((0, 3, 1, 2))

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        # iou_pred = self.iou_prediction_head(iou_token_out)

        # only return the fist mask
        return masks[:,0,:]

