"""
Derived from mobile_sam/modeling/tiny_vit_sam.py
"""

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath as TimmDropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from typing import Tuple


class Conv2d_BN(torch.nn.Sequential):
    def __init__(
        self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1
    ):
        super().__init__()
        self.add_module(
            "c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None):
        super().__init__(drop_prob=drop_prob)
        self.drop_prob = drop_prob

    def __repr__(self):
        msg = super().__repr__()
        msg += f"(drop_prob={self.drop_prob})"
        return msg


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        img_size: Tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)


class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(
            self.hidden_chans,
            self.hidden_chans,
            ks=3,
            stride=1,
            pad=1,
            groups=self.hidden_chans,
        )
        self.act2 = activation()

        self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        x = self.act3(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c = 2
        if out_dim in (320, 448, 576):
            stride_c = 1
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        activation,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        out_dim=None,
        conv_expand_ratio=4.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                MBConv(
                    dim,
                    dim,
                    conv_expand_ratio,
                    activation,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation
            )
            if downsample is not None
            else None
        )

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=(14, 14),
        lora_r=0,
        lora_alpha=1.0,
        lora_dropout=0.0,
    ):
        """
        Standard attention with combined qkv layer.
        When lora_r > 0, legacy LoRA parameters are expected in the state dict.
        """
        super().__init__()
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        # d: per-head value dimension
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads
        self.attn_ratio = attn_ratio

        self.norm = nn.LayerNorm(dim)
        # Combined qkv layer
        self.qkv = nn.Linear(dim, 3 * (num_heads * key_dim))
        # Output projection remains standard.
        self.proj = nn.Linear(self.dh, dim)

        # Relative positional attention biases (unchanged)
        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))
        )
        self.register_buffer(
            "attention_bias_idxs", torch.LongTensor(idxs).view(N, N), persistent=False
        )

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.register_buffer(
                "ab",
                self.attention_biases[:, self.attention_bias_idxs],
                persistent=False,
            )

    def forward(self, x):  # x: (B, L, C)
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d], dim=3
        )
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs]
            if self.training
            else self.ab
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class LoRAAttention(Attention):
    def __init__(self, *args, rank, **kwargs):
        """
        Inherits from Attention.
        Args:
          rank: LoRA rank for q and v.
          Other args/kwargs are passed to Attention.
        """
        super().__init__(*args, **kwargs)
        self.rank = rank
        # LoRA layers for q and v.
        # These operate on the combined qkv output channels (which equals num_heads * key_dim).
        self.lora_w_a_q = nn.Linear(self.num_heads * self.key_dim, rank, bias=False)
        self.lora_w_b_q = nn.Linear(rank, self.num_heads * self.key_dim, bias=False)
        self.lora_w_a_v = nn.Linear(self.num_heads * self.key_dim, rank, bias=False)
        self.lora_w_b_v = nn.Linear(rank, self.num_heads * self.key_dim, bias=False)

        nn.init.zeros_(self.lora_w_b_q.weight)
        nn.init.zeros_(self.lora_w_b_v.weight)

    def forward(self, x):
        B, L, _ = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)  # shape: (B, L, 3 * (num_heads * key_dim))
        # Reshape to (B, L, 3, C) where C = num_heads * key_dim
        q, k, v = qkv.view(B, L, 3, -1).split(1, dim=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)  # each: (B, L, C)
        # Apply LoRA updates to query and value only.
        q_lora = q + self.lora_w_b_q(self.lora_w_a_q(q))
        v_lora = v + self.lora_w_b_v(self.lora_w_a_v(v))
        # Recombine with key unchanged.
        qkv_lora = torch.cat(
            [q_lora.unsqueeze(2), k.unsqueeze(2), v_lora.unsqueeze(2)], dim=2
        )
        # Flatten back to (B, L, 3 * C)
        qkv_lora = qkv_lora.view(B, L, -1)
        # Now split into q, k, and v for multi-head attention.
        q, k, v = qkv_lora.view(B, L, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d], dim=3
        )
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.training:
            attn = attn + self.attention_biases[:, self.attention_bias_idxs]
        else:
            attn = attn + self.ab
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, self.dh)
        out = self.proj(out)
        return out


class TinyViTBlock(nn.Module):
    r"""TinyViT Block with optional LoRA in the attention.
    Additional parameters:
      - lora_r: if > 0, applies LoRA via LoRAAttention.
      - lora_alpha, lora_dropout: passed to the attention layer.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        local_conv_size=3,
        activation=nn.GELU,
        lora_r=0,
        lora_alpha=1.0,
        lora_dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, "window_size must be greater than 0"
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        head_dim = dim // num_heads
        window_resolution = (window_size, window_size)
        # Use LoRAAttention if lora_r > 0, else standard Attention.
        if lora_r > 0:
            self.attn = LoRAAttention(
                dim,
                head_dim,
                num_heads,
                attn_ratio=1,
                resolution=window_resolution,
                rank=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        else:
            self.attn = Attention(
                dim,
                head_dim,
                num_heads,
                attn_ratio=1,
                resolution=window_resolution,
                lora_r=0,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=activation,
            drop=drop,
        )
        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            if pad_b > 0 or pad_r > 0:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            x = (
                x.view(B, nH, self.window_size, nW, self.window_size, C)
                .transpose(2, 3)
                .reshape(B * nH * nW, self.window_size * self.window_size, C)
            )
            x = self.attn(x)
            x = (
                x.view(B, nH, nW, self.window_size, self.window_size, C)
                .transpose(2, 3)
                .reshape(B, pH, pW, C)
            )
            if pad_b or pad_r:
                x = x[:, :H, :W].contiguous()
            x = x.view(B, L, C)
        x = res_x + self.drop_path(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)
        x = x + self.drop_path(self.mlp(x))
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        )


class BasicLayer(nn.Module):
    r"""A basic TinyViT layer for one stage.
    Additional parameters:
      - lora_r, lora_alpha, lora_dropout: parameters for LoRA.
      - block_lora_flags: a list of booleans (length==depth). For each block,
          if True then LoRA is applied, otherwise lora_r is forced to 0.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        local_conv_size=3,
        activation=nn.GELU,
        out_dim=None,
        lora_r=0,
        lora_alpha=1.0,
        lora_dropout=0.0,
        block_lora_flags=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if block_lora_flags is None:
            block_lora_flags = [True if lora_r > 0 else False] * depth

        self.blocks = nn.ModuleList(
            [
                TinyViTBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    local_conv_size=local_conv_size,
                    activation=activation,
                    lora_r=lora_r if block_lora_flags[i] else 0,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation
            )
            if downsample is not None
            else None
        )

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x



class TinyViTLoRA(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=1.0,
        # LoRA parameters:
        lora_r=0,
        lora_alpha=1.0,
        lora_dropout=0.0,
        # List of layer indices to adapt with LoRA.
        lora_layers=None,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            resolution=img_size,
            activation=activation,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Stochastic depth decay rule.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Compute total number of blocks in BasicLayers (excluding the first ConvLayer).
        total_blocks = sum(depths[i] for i in range(1, self.num_layers))
        if lora_layers is None:
            adapted_indices = set(range(self.num_layers)) if lora_r > 0 else set()
        else:
            adapted_indices = set()
            for idx in lora_layers:
                if idx < 0:
                    idx = idx + self.num_layers
                adapted_indices.add(idx)

        self.layers = nn.ModuleList()
        global_block_idx = 0
        for i_layer in range(self.num_layers):
            kwargs = dict(
                dim=embed_dims[i_layer],
                input_resolution=(
                    patches_resolution[0]
                    // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                    patches_resolution[1]
                    // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                ),
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                out_dim=embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                activation=activation,
            )
            if i_layer == 0:
                layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                current_depth = depths[i_layer]
                block_lora_flags = [i_layer in adapted_indices] * current_depth
                global_block_idx += current_depth
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    block_lora_flags=block_lora_flags,
                    **kwargs,
                )
            self.layers.append(layer)

        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = (
            nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dims[-1], 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))
        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, "lr_scale"), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"attention_biases"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = self.layers[i](x)
        B, _, C = x.size()
        x = x.view(B, 64, 64, C)
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


_checkpoint_url_format = (
    "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/{}.pth"
)
_provided_checkpoints = {
    "tiny_vit_5m_224": "tiny_vit_5m_22kto1k_distill",
    "tiny_vit_11m_224": "tiny_vit_11m_22kto1k_distill",
    "tiny_vit_21m_224": "tiny_vit_21m_22kto1k_distill",
    "tiny_vit_21m_384": "tiny_vit_21m_22kto1k_384_distill",
    "tiny_vit_21m_512": "tiny_vit_21m_22kto1k_512_distill",
}


def register_tiny_vit_model(fn):
    def fn_wrapper(pretrained=False, **kwargs):
        model = fn(**kwargs)
        if pretrained:
            model_name = fn.__name__
            assert (
                model_name in _provided_checkpoints
            ), f"Checkpoint for `{model_name}` is not provided."
            url = _checkpoint_url_format.format(_provided_checkpoints[model_name])
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu", check_hash=False
            )
            model.load_state_dict(checkpoint["model"])
        return model

    fn_wrapper.__name__ = fn.__name__
    return register_model(fn_wrapper)


@register_tiny_vit_model
def tiny_vit_5m_224(pretrained=False, num_classes=1000, drop_path_rate=0.0, **kwargs):
    return TinyViTLoRA(
        num_classes=num_classes,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
        **kwargs,
    )


@register_tiny_vit_model
def tiny_vit_11m_224(pretrained=False, num_classes=1000, drop_path_rate=0.1, **kwargs):
    return TinyViTLoRA(
        num_classes=num_classes,
        embed_dims=[64, 128, 256, 448],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 14],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
        **kwargs,
    )


@register_tiny_vit_model
def tiny_vit_21m_224(pretrained=False, num_classes=1000, drop_path_rate=0.2, **kwargs):
    return TinyViTLoRA(
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
        **kwargs,
    )


@register_tiny_vit_model
def tiny_vit_21m_384(pretrained=False, num_classes=1000, drop_path_rate=0.1, **kwargs):
    return TinyViTLoRA(
        img_size=384,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[12, 12, 24, 12],
        drop_path_rate=drop_path_rate,
        **kwargs,
    )


@register_tiny_vit_model
def tiny_vit_21m_512(pretrained=False, num_classes=1000, drop_path_rate=0.1, **kwargs):
    return TinyViTLoRA(
        img_size=512,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[16, 16, 32, 16],
        drop_path_rate=drop_path_rate,
        **kwargs,
    )
