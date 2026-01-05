# Masked codeword Classification + Masked Representation Regression
import torch
import torch.nn as nn
import torch.nn.functional as F

########################################################
# Tokenizer (codebook) used to generate discrete targets from teacher features
########################################################
class Tokenizer(nn.Module):
    def __init__(self, codebook_size, embed_dim):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codebook_size, embed_dim))

    @torch.no_grad()
    def forward(self, rep):
        # rep: [B, M, D]
        B, M, D = rep.shape
        rep_flat = rep.view(-1, D)   # [B*M, D]
        a2 = (rep_flat**2).sum(dim=1, keepdim=True)           # [B*M,1]
        b2 = (self.codebook**2).sum(dim=1).unsqueeze(0)       # [1, K]
        ab = rep_flat @ self.codebook.t()                     # [B*M, K]
        dist = a2 + b2 - 2.0 * ab
        idx = dist.argmin(dim=1).view(B, M)                   # [B, M]
        return idx

########################################################
# Masked Codeword Classifier
######################################################## 
class MCC(nn.Module):
    def __init__(self, codebook_size, embed_dim):
        super().__init__()
        self.center = nn.Linear(embed_dim, codebook_size)

    def forward(self, rep):
        B, M, D = rep.shape
        logits = self.center(rep.view(-1, D))
        return logits.view(B, M, -1)

########################################################
# Masked representation Regression
######################################################## 
class MRR(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, layers, dropout):
        super().__init__()
        from .encoder import CrossAttnBlock
        self.layers = nn.ModuleList([
            CrossAttnBlock(d_model, attn_heads, d_ffn, dropout)
            for _ in range(layers)
        ])

    def forward(self, masked_rep, visible_rep):
        x = masked_rep
        for block in self.layers:
            x = block(x, visible_rep)
        return x
