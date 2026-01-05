# utils/window_utils.py
import math
from typing import List, Tuple

def make_non_overlapping_ranges(T: int, window_size: int) -> List[Tuple[int,int]]:
    """
    Return list of non-overlapping ranges (start, end) covering T timesteps.
    end is exclusive.
    """
    num_windows = math.ceil(T / window_size)
    ranges = []
    for w in range(num_windows):
        start = w * window_size
        end = min((w+1) * window_size, T)
        ranges.append((start, end))
    return ranges


def windows_to_time_ranges(num_windows: int, window_size: int, orig_length: int):
    return make_non_overlapping_ranges(orig_length, window_size)


def expand_window_scores_to_time(scores_per_window, ranges):
    """
    Convert per-window scores to per-timestep score by broadcasting window score to its range.
    scores_per_window: 1D array length num_windows
    ranges: list of (start,end)
    returns: 1D array length = last range end
    """
    import numpy as np
    T = ranges[-1][1]
    out = np.zeros(T, dtype=float)
    for s, (st, ed) in zip(scores_per_window, ranges):
        out[st:ed] = s
    return out
