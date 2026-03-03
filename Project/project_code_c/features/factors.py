"""Causal factor computation utilities.

All functions here are strictly causal: they only use data up to and
including time t when computing any statistic for time t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def numpy_pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """Top-K principal component loadings from a centred data matrix X.

    Uses numpy SVD — no sklearn dependency.

    Args:
        X:             (n_samples, n_features) *already-centred* matrix.
        n_components:  Number of components to retain.

    Returns:
        B: (n_features, n_components) loading matrix (columns = eigenvectors).
    """
    k = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k, :].T  # (n_features, k)


def rolling_ols_beta(
    y: np.ndarray,
    x: np.ndarray,
    window: int,
) -> np.ndarray:
    """Scalar rolling OLS slope β(t) = Cov(y,x) / Var(x) over a rolling window.

    Output is NaN for the first `window - 1` observations.
    Result at index i uses observations [i-window+1, ..., i] (causal).

    Args:
        y:      Dependent variable time-series (1-D array, length T).
        x:      Independent variable time-series (1-D array, length T).
        window: Rolling estimation window in observations.

    Returns:
        betas: (T,) array of rolling OLS slopes, NaN for the first window-1 obs.
    """
    T = len(y)
    betas = np.full(T, np.nan)
    for i in range(window - 1, T):
        y_w = y[i - window + 1: i + 1]
        x_w = x[i - window + 1: i + 1]
        x_demean = x_w - x_w.mean()
        var_x = float(np.dot(x_demean, x_demean))
        if var_x < 1e-14:
            continue
        betas[i] = float(np.dot(y_w - y_w.mean(), x_demean) / var_x)
    return betas


def rolling_zscore_series(
    s: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Causal rolling z-score of a pandas Series.

    z_t = (x_t - rolling_mean_t) / rolling_std_t
    using only past observations (no look-ahead).

    Args:
        s:           Input time-series.
        window:      Rolling window in observations.
        min_periods: Minimum non-NaN observations required (default: window//2).

    Returns:
        pd.Series of z-scores (NaN where insufficient history).
    """
    if min_periods is None:
        min_periods = max(window // 2, 2)
    roll = s.rolling(window=window, min_periods=min_periods)
    mu = roll.mean()
    sigma = roll.std()
    denom = sigma.where(sigma > 1e-12, np.nan)
    return ((s - mu) / denom).rename(s.name)
