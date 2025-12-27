# src/models/inspectable_encoder_layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

from src.models.positional_encodings import *


class InspectableEncoderLayer(nn.Module):
    """
    Transformer encoder block that can optionally return per-head attention
    weights, for interpretability.

    If need_attn_weights=False, behaves like a standard encoder layer and only
    returns the transformed hidden states.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,  # (B, L, D)
        )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorms and dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu  # or nn.GELU()

    def forward(
        self,
        src: torch.Tensor,                      # (B, L, D)
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        need_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            output: (B, L, D)
            attn_weights: (B, n_heads, L, L) if need_attn_weights=True, else None
        """
        # --- Self-attention ---
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_attn_weights,
            average_attn_weights=False,  # keep per-head weights
        )
        src2 = self.dropout1(attn_output)
        x = self.norm1(src + src2)   # residual + norm

        # --- Feedforward ---
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x2 = self.dropout2(ff)
        x = self.norm2(x + x2)

        if need_attn_weights and attn_weights is not None:
            # attn_weights: (B * n_heads, L, L) -> (B, n_heads, L, L)
            if attn_weights.dim() == 3:
                B, L, _ = src.shape
                n_heads = attn_weights.shape[0] // B
                attn_weights = attn_weights.view(B, n_heads, L, L)
            elif attn_weights.dim() == 4:
                pass
            return x, attn_weights

        return x, None


class BlackboardTransformer(nn.Module):
    """
    Encoder-only Transformer for 2D blackboard sequence (flattened to 1D).

    Positional handling (retro-compatible):
      - Absolute PE modules: pos_enc(x) -> x + pe
      - RelativePositionBias2D: pos_enc() -> (n_heads, L, L) bias
      - Mixed Abs2DPlusRelBias2D (your wrapper):
            pos_enc(x) -> x (+abs)
            pos_enc.get_pos_bias() -> (n_heads, L, L) or None
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        max_len: int = 256,
        dropout: float = 0.1,
        pos_enc: nn.Module = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = pos_enc

        self.layers = nn.ModuleList(
            [
                InspectableEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(d_model, vocab_size)

    def _make_pos_bias(
        self, bias: torch.Tensor, batch_size: int, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        bias: (n_heads, L, L)
        returns: (B*n_heads, L, L) on device
        """
        # Ensure device/dtype consistent
        bias = bias.to(device=device)
        # (B, n_heads, L, L)
        bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        # (B*n_heads, L, L)
        bias = bias.reshape(batch_size * self.nhead, seq_len, seq_len)
        return bias

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, L)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) or None
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Returns:
            logits: (B, L, vocab_size)
            attn_all_layers (optional): list[num_layers] of (B, n_heads, L, L)
        """
        B, L = input_ids.shape
        device = input_ids.device

        # 1) token embeddings
        x = self.token_emb(input_ids) * math.sqrt(self.d_model)  # (B, L, D)

        # 2) positional signals
        pos_bias = None

        if self.pos_enc is not None:
            # (a) old relative-only: RelativePositionBias2D
            if isinstance(self.pos_enc, RelativePositionBias2D):
                bias = self.pos_enc()  # (n_heads, L, L)
                pos_bias = self._make_pos_bias(bias, B, L, device)

            # (b) mixed abs+rel: has get_pos_bias()
            elif hasattr(self.pos_enc, "get_pos_bias") and callable(getattr(self.pos_enc, "get_pos_bias")):
                # absolute part (if any)
                x = self.pos_enc(x)

                # relative part (if any)
                bias = self.pos_enc.get_pos_bias()
                if bias is not None:
                    pos_bias = self._make_pos_bias(bias, B, L, device)

            # (c) absolute-only (sinusoidal / abs2D / etc.)
            else:
                x = self.pos_enc(x)

        # 3) encoder layers
        attn_all_layers: Optional[List[torch.Tensor]] = [] if return_attn else None

        for layer in self.layers:
            if return_attn:
                x, attn = layer(
                    x,
                    src_mask=pos_bias,
                    src_key_padding_mask=src_key_padding_mask,
                    need_attn_weights=True,
                )
                attn_all_layers.append(attn)  # (B, n_heads, L, L)
            else:
                x, _ = layer(
                    x,
                    src_mask=pos_bias,
                    src_key_padding_mask=src_key_padding_mask,
                    need_attn_weights=False,
                )

        # 4) output projection
        logits = self.output_proj(x)  # (B, L, vocab_size)
        return logits, attn_all_layers



class COTTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        max_len: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.nhead = nhead

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        self.layers = nn.ModuleList([
            InspectableEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,                       # (B, L)
        src_mask: torch.Tensor,  # (L, L)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) or None
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            input_ids: (B, L) token ids (flattened board)
            src_key_padding_mask: (B, L) bool, True for PAD positions (if used)
            return_attn: if True, also return attention matrices

        Returns:
            logits: (B, L, vocab_size)
            attn_all_layers (optional): list of length num_layers, where each
                element is a tensor of shape (B, n_heads, L, L) with attention
                weights for that layer. If return_attn=False, this is None.
        """
        # 1) token embeddings + positional encodings
        x = self.token_emb(input_ids) * math.sqrt(self.d_model)  # (B, L, D)
        x = self.pos_enc(x)

        attn_all_layers: Optional[List[torch.Tensor]] = [] if return_attn else None

        mask = None
        if src_mask is not None:
            B, L = input_ids.shape
            mask = src_mask.unsqueeze(0).unsqueeze(0).expand(B, self.nhead, -1, -1)   # (B, n_heads, L, L)
            mask = mask.reshape(B * self.nhead, L, L)  # (B*n_heads, L, L)

        # 2) pass through each encoder layer
        for layer in self.layers:
            if return_attn:
                x, attn = layer(
                    x,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    need_attn_weights=True,
                )
                attn_all_layers.append(attn)   # (B, n_heads, L, L)
            else:
                x, _ = layer(
                    x,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    need_attn_weights=False,
                )

        # 3) project to logits
        logits = self.output_proj(x)  # (B, L, vocab_size)

        if return_attn:
            return logits, attn_all_layers
        return logits, None
    
























