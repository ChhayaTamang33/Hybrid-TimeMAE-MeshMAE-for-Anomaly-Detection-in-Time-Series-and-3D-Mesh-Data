# anomaly detection need temporal information from MESHMAE for proper anomaly detection
import torch
import torch.nn.functional as F

@torch.no_grad()
def anomaly_score(model, x, gamma_align=1.0, gamma_cls=1.0, device=None):
    """
    Compute anomaly scores for a batch of time series.

    Args:
        model: TimeMAE model (already pretrained)
        x: Tensor [B, T, F]
        gamma_align: weight for representation alignment error
        gamma_cls: weight for codeword classification error
        device: cuda or cpu

    Returns:
        score:        [B, M] total anomaly score per masked window
        align_err:    [B, M] alignment component
        cls_err:      [B, M] classification component
        v_idx, m_idx: visible & masked window indices (for mapping back)
    """

    device = device or next(model.parameters()).device
    model.eval()
    x = x.to(device)

    # ---------------------------------------------------------
    # 1. Preprocessing: window slicing + masking + embeddings
    # ---------------------------------------------------------
    visible_embed, masked_target, masked_input_tokens, v_idx, m_idx = model.preproc(x)

    # ---------------------------------------------------------
    # 2. Teacher & Student encoders (NO gradient)
    # ---------------------------------------------------------

    # Student visible encoder
    rep_visible = model.encoder.visible_encoder(visible_embed)

    # Teacher encoder runs on ground-truth masked targets
    rep_mask_target = model.teacher.visible_encoder(masked_target)

    # Student MRR reconstructs masked target representations
    rep_mask_pred = model.mrr(masked_input_tokens, rep_visible)

    # ---------------------------------------------------------
    # 3. Token classification (MCC) for discrete codewords
    # ---------------------------------------------------------
    token_logits = model.mcc(rep_mask_pred)                # [B, M, vocab_size]
    token_targets = model.tokenizer(rep_mask_target)       # [B, M]

    # ---------------------------------------------------------
    # 4. Compute per-window errors
    # ---------------------------------------------------------

    # --- Alignment Loss (L_align) ---
    # mse per masked token → [B, M]
    align_err = F.mse_loss(rep_mask_pred, rep_mask_target,
                           reduction="none").mean(dim=-1)

    # --- Classification Loss (L_cls) ---
    B, M, K = token_logits.shape
    cls_flat = F.cross_entropy(
        token_logits.view(B * M, K),
        token_targets.view(B * M),
        reduction="none"
    )
    cls_err = cls_flat.view(B, M)  # back to [B, M]

    # ---------------------------------------------------------
    # 5. Total anomaly score (higher = more abnormal)
    # ---------------------------------------------------------
    score = gamma_align * align_err + gamma_cls * cls_err

    return score.cpu()