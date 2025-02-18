"""
Derived from segment_anything/modeling/image_encoder.py
"""

import math
from typing import Optional, Tuple, Type, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling.common import LayerNorm2d, MLPBlock

###############################################################################
# LoRA Linear Module
###############################################################################
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Linear):
    """
    A Linear layer with Low-Rank Adaptation (LoRA) implemented by subclassing nn.Linear.
    The effective weight is computed as:
        W_eff = W + lora_alpha * (lora_B @ lora_A)
    where W is the original weight of the linear layer, and lora_A and lora_B are
    low-rank matrices learned during training.
    If r == 0, this layer behaves exactly like a standard nn.Linear.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 0, lora_alpha: float = 1.0, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.r = r
        self.lora_alpha = lora_alpha

        if self.r > 0:
            # Initialize the low-rank factors.
            self.lora_A = nn.Parameter(torch.Tensor(r, in_features))
            self.lora_B = nn.Parameter(torch.Tensor(out_features, r))
            self.reset_lora_parameters()
        else:
            self.lora_A = None
            self.lora_B = None

    def reset_lora_parameters(self) -> None:
        # Initialize lora_A with Kaiming uniform initialization and lora_B with zeros.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard linear output.
        result = super().forward(x)
        if self.r > 0:
            # Compute the low-rank update and add it to the result.
            lora_update = F.linear(x, self.lora_B @ self.lora_A)
            result += self.lora_alpha * lora_update
        return result


###############################################################################
# Attention Module with LoRA Support
###############################################################################
class Attention(nn.Module):
    """
    Multi-head Attention block with optional relative position embeddings and LoRA.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        """
        Args:
            dim (int): Input channel dimension.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, add a learnable bias to QKV projections.
            use_rel_pos (bool): If True, use relative positional embeddings.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple or None): Input spatial resolution, required if using relative positional encoding.
            lora_r (int): Rank of the LoRA decomposition. If 0, LoRA is disabled.
            lora_alpha (float): Scaling factor for the LoRA updates.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Use LoRA-adapted linear for QKV if requested.
        if lora_r > 0:
            self.qkv = LoRALinear(dim, dim * 3, r=lora_r * 3, lora_alpha=lora_alpha, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            if rel_pos_zero_init:
                nn.init.zeros_(self.rel_pos_h)
                nn.init.zeros_(self.rel_pos_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x

###############################################################################
# Block Module (Transformer Block)
###############################################################################
class Block(nn.Module):
    """
    Transformer block with support for window attention and residual propagation.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

###############################################################################
# Helper Functions for Window Partitioning and Relative Position Encoding
###############################################################################
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    return attn

###############################################################################
# Patch Embedding Module
###############################################################################
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # Rearrange from B x C x H x W to B x H x W x C.
        x = x.permute(0, 2, 3, 1)
        return x

###############################################################################
# LoRA-Adapted ViT Image Encoder with Specified LoRA Layers
###############################################################################
class ImageEncoderViTLoRA(nn.Module):
    """
    Vision Transformer image encoder with optional LoRA adaptation on self-attention QKV,
    with the ability to select specific transformer layers for LoRA.

    Args:
        img_size (int): Input image size.
        patch_size (int): Patch size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Patch embedding dimension.
        depth (int): Number of Transformer blocks.
        num_heads (int): Number of attention heads per block.
        mlp_ratio (float): MLP hidden dimension ratio.
        out_chans (int): Number of output channels after the neck.
        qkv_bias (bool): If True, use bias in QKV projection.
        norm_layer (nn.Module): Normalization layer.
        act_layer (nn.Module): Activation layer.
        use_abs_pos (bool): If True, use absolute positional embeddings.
        use_rel_pos (bool): If True, add relative positional embeddings.
        rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
        window_size (int): Window size for window attention; if 0, global attention is used.
        global_attn_indexes (tuple): Block indexes that use global attention.
        lora_layers (List[int]): Indices of transformer layers where LoRA should be applied.
        lora_r (int): LoRA rank (applied only on layers specified in lora_layers).
        lora_alpha (float): Scaling factor for LoRA updates.
    """
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        lora_layers: Optional[List[int]] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        # Normalize lora_layers: convert negative indices to positive ones.
        if lora_layers is not None:
            normalized_lora_layers = set(idx if idx >= 0 else idx + depth for idx in lora_layers)
        else:
            normalized_lora_layers = set()

        for i in range(depth):
            # Determine if LoRA should be applied for this block.
            if i in normalized_lora_layers:
                block_lora_r = lora_r
            else:
                block_lora_r = 0

            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                lora_r=block_lora_r,
                lora_alpha=lora_alpha,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x
