# models/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderBlock(nn.Module):
    """Vanilla Transformer encoder block (batch_first=True)."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, L, D]
        x_att = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + x_att
        x_ff = self.ffn(self.norm2(x))
        x = x + x_ff
        return x


class TransformerEncoder(nn.Module):
    """Stack of TransformerEncoderBlock, returns normalized outputs."""
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        layers = [TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class CrossAttnBlock(nn.Module):
    """
    Cross-attention block where queries are 'masked' tokens and keys/values are visible tokens.
    Uses batch_first MultiheadAttention.
    """
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv):
        # q: [B, M, D] (masked tokens), kv: [B, V, D] (visible tokens)
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out = self.cross_attn(q_norm, kv_norm, kv_norm)[0]
        x = q + attn_out
        ff_out = self.ffn(self.norm2(x))
        x = x + ff_out
        return x


class DecoupledEncoder(nn.Module):
    """
    Container that returns:
      - masked_repr: output of masked branch (cross-attention blocks)
      - visible_repr: output of visible self-attention encoder
    """
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.visible_encoder = TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers, dropout)
        self.masked_blocks = nn.ModuleList([CrossAttnBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, masked_embed, visible_embed):
        """
        Args:
          masked_embed: [B, M, D]  (student masked tokens)
          visible_embed: [B, V, D] (student visible tokens)

        Returns:
          masked_repr: [B, M, D]
          visible_repr: [B, V, D]
        """
        visible_repr = self.visible_encoder(visible_embed)
        x = masked_embed
        for blk in self.masked_blocks:
            x = blk(x, visible_repr)
        masked_repr = x
        return masked_repr, visible_repr
