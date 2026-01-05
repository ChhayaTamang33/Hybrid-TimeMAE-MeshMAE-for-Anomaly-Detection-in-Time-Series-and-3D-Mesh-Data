# models/timemae.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import DecoupledEncoder
from .preprocessing import TimeMaePreProcessing
from .mrr_mcc import MCC, MRR, Tokenizer


########################################################
# Full TimeMAE model
#######################################################
class TimeMAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.momentum = getattr(args, 'momentum')

        input_dim = args.window_size * args.num_features
        embed_dim = args.embed_dim

        # Preprocessing module
        self.preproc = TimeMaePreProcessing(
            input_dim=input_dim,
            embed_dim=embed_dim,
            mask_ratio=args.mask_ratio,
            window_size=args.window_size,
            max_len=args.max_len,
            device=args.device
        )

        # Student and teacher decoupled encoders
        self.encoder = DecoupledEncoder(embed_dim=embed_dim,
                                        num_heads=args.attn_heads,
                                        ff_dim=args.d_ffn,
                                        num_layers=args.encoder_layers,
                                        dropout=0.1)

        self.teacher = DecoupledEncoder(embed_dim=embed_dim,
                                        num_heads=args.attn_heads,
                                        ff_dim=args.d_ffn,
                                        num_layers=args.encoder_layers,
                                        dropout=0.1)

        # copy weights & freeze teacher
        for t_param, s_param in zip(self.teacher.parameters(), self.encoder.parameters()):
            t_param.data.copy_(s_param.data)
            t_param.requires_grad = False

        # MRR (regressor), tokenizer, MCC, decoder
        self.mrr = MRR(d_model=embed_dim, attn_heads=args.attn_heads, d_ffn=args.d_ffn, layers=args.decoder_layers, dropout = 0.1)
        self.tokenizer = Tokenizer(codebook_size=args.codebook_size, embed_dim=embed_dim)
        self.mcc = MCC(codebook_size=args.codebook_size, embed_dim=embed_dim)
        self.decoder = nn.Linear(embed_dim, input_dim)

        # initialize linear layers
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    @torch.no_grad()
    def momentum_update(self, first_time=False):
        """EMA update of teacher parameters from student."""
        for t_param, s_param in zip(self.teacher.parameters(), self.encoder.parameters()):
            if first_time:
                t_param.data.copy_(s_param.data)
                t_param.requires_grad = False
            else:
                #EMA formula: teacher = m * teacher + (1 - m) * student
                t_param.data = self.momentum * t_param.data + (1.0 - self.momentum) * s_param.data

    def pre_train_forward(self, x):
        """
        Pretraining forward. Returns dictionary of items needed by trainer.
        x: [B, T, F]
        returns dict:
          rep_mask_target: [B, M, D] (teacher)
          rep_mask_prediction: [B, M, D] (student)
          token_logits: [B, M, K] (student)
          token_targets: [B, M] (int targets from tokenizer)
          recon_flat: [B, M, input_dim] optional
          v_idx, m_idx: indices
        """
        visible_embed, masked_target, masked_input_tokens, v_idx, m_idx = self.preproc(x)

        # student visible representation
        rep_visible = self.encoder.visible_encoder(visible_embed)  # [B, V, D]

        # teacher target (no gradients)
        with torch.no_grad():
            rep_mask_target = self.teacher.visible_encoder(masked_target)  # [B, M, D]
            token_targets = self.tokenizer(rep_mask_target)               # [B, M]

        # student reconstructs masked representations
        rep_mask_prediction = self.mrr(masked_input_tokens, rep_visible)  # [B, M, D]

        # MCC logits from student predictions
        token_logits = self.mcc(rep_mask_prediction)  # [B, M, K]

        # raw reconstruction (optional) — consistent shape for decoder usage
        recon_flat = self.decoder(rep_mask_prediction)  # [B, M, input_dim]

        return {
            "rep_mask_target": rep_mask_target,
            "rep_mask_prediction": rep_mask_prediction,
            "token_logits": token_logits,
            "token_targets": token_targets,
            "recon_flat": recon_flat,
            "v_idx": v_idx,
            "m_idx": m_idx
        }

    def forward(self, x):
        """Simple forward returning pooled representation (for downstream usage)."""
        # x: [B, T, F]
        windows = self.preproc._slice_windows(x, self.preproc.window_size)  # [B, num_windows, w, F]
        B, num_windows, w, ff = windows.shape
        flat = windows.view(B, num_windows, -1)
        embedded = self.preproc.window_embed(flat) + self.preproc.pos_embed[:, :num_windows, :].to(flat.device)
        encoded = self.encoder.visible_encoder(embedded)
        pooled = encoded.mean(dim=1)  # [B, D]
        return pooled
