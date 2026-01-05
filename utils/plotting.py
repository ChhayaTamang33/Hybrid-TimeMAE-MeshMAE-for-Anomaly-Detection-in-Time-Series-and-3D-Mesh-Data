# utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def plot_series_with_anomalies(time_series: np.ndarray,
                               window_scores: np.ndarray,
                               window_ranges: List[Tuple[int,int]],
                               threshold: float = None,
                               figsize=(14,4),
                               ax=None):
    """
    Plot the raw time series and overlay anomalous windows.
    - time_series: 1D array of length T (e.g. temperature)
    - window_scores: 1D array of length num_windows (score per window)
    - window_ranges: list of (start,end) indices per window
    - threshold: optional threshold on window_scores
    """
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = None

    T = len(time_series)
    ax.plot(np.arange(T), time_series, label='signal', lw=1)

    # map window scores to color
    for i, (s, (st, ed)) in enumerate(zip(window_scores, window_ranges)):
        if threshold is None:
            # alpha proportional to score (normalized)
            pass
        else:
            if s > threshold:
                ax.axvspan(st, ed, color='red', alpha=0.25)

    ax.set_xlabel("time index")
    ax.set_ylabel("value")
    ax.set_title("Time series with anomalous windows highlighted")
    ax.legend()
    if fig is not None:
        return fig, ax
    return ax
