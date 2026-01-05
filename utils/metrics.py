# utils/metrics.py
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

def compute_window_scores(model, x: torch.Tensor, gamma_align=1.0, gamma_cls=1.0, device=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper around model to compute per-window anomaly components.
    Returns numpy arrays: (score, align_err, cls_err) each [B, M]
    """
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model.pre_train_forward(x.to(device))
        rep_mask_target = out["rep_mask_target"]
        rep_mask_pred = out["rep_mask_prediction"]
        token_logits = out["token_logits"]
        token_targets = out["token_targets"]

        # alignment per masked token
        align_err = F.mse_loss(rep_mask_pred, rep_mask_target, reduction="none").mean(dim=-1)  # [B, M]

        B, M, K = token_logits.shape
        cls_flat = F.cross_entropy(token_logits.view(B*M, K), token_targets.view(B*M), reduction="none")
        cls_err = cls_flat.view(B, M)

        score = gamma_align * align_err + gamma_cls * cls_err

        return score.cpu().numpy(), align_err.cpu().numpy(), cls_err.cpu().numpy()


def threshold_scores(scores: np.ndarray, method: str = "std", k: float = 3.0) -> Tuple[float, np.ndarray]:
    """
    Simple thresholding helpers.
    - method="std": threshold = mean + k * std
    - method="percentile": k in (0..100) percentile of scores (e.g. 95)
    Returns: (threshold, boolean_mask)
    """
    if method == "std":
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        thr = mean + k * std
        mask = scores > thr
        return thr, mask
    elif method == "percentile":
        thr = float(np.percentile(scores, k))
        mask = scores > thr
        return thr, mask
    else:
        raise ValueError("Unknown threshold method.")
