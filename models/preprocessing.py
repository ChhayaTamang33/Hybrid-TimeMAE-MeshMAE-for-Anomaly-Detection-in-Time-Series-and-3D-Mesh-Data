# Data Preprocessing
# Window Slicing + Embedding + Positional Encoding + Masking
import torch
import torch.nn as nn
import torch.nn.functional as TF
import math

class TimeMaePreProcessing(nn.Module):
    def __init__(self, input_dim, embed_dim, mask_ratio, window_size, max_len, device):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.device = device

        self.window_embed = nn.Linear(input_dim, embed_dim)
        self.mask_token  = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.randn(1, max_len, embed_dim))

    @staticmethod
    def window_slicing(temps, window_size):
        N, T, F = temps.shape
        num_windows = math.ceil(T / window_size)
        pad_len = num_windows * window_size - T
        temps_padded = TF.pad(temps, (0,0,0,pad_len))
        windows = temps_padded.view(N, num_windows, window_size, F)
        return windows

    def forward(self, temps):
        temps_windows = self.window_slicing(temps, self.window_size)
        N, num_windows, w_s, F = temps_windows.shape

        flat = temps_windows.view(N, num_windows, -1)
        embedded = self.window_embed(flat)
        embedded = embedded + self.pos_embed[:, :num_windows, :]

        num_mask = int(self.mask_ratio * num_windows)
        num_keep = num_windows - num_mask

        visible_list = []
        target_list = []
        masked_input_list = []
        v_idx_list = []
        m_idx_list = []

        for i in range(N):
            perm = torch.randperm(num_windows)
            v_idx = perm[:num_keep]
            m_idx = perm[num_keep:]

            v_idx_list.append(v_idx)
            m_idx_list.append(m_idx)

            visible_list.append(embedded[i, v_idx])
            target_list.append(embedded[i, m_idx])

            mask_tok = self.mask_token.repeat(1, len(m_idx), 1).squeeze(0)
            mask_tok += self.pos_embed[0, m_idx]
            masked_input_list.append(mask_tok)

        return ( torch.stack(visible_list),
                 torch.stack(target_list),
                 torch.stack(masked_input_list),
                 torch.stack(v_idx_list),
                 torch.stack(m_idx_list) )
