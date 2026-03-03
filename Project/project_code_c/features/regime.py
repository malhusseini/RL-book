"""Causal market-regime labelling.

All estimators use only data up to and including time t (no look-ahead).
Regime labels are binary: 1 = "good" (low vol), 0 = "bad" (high vol).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_regime(
    market_returns: pd.Series,
    window: int = 21,
    long_window_mult: int = 5,
) -> pd.Series:
    """Assign a causal binary regime label from rolling realized volatility.

    Rule: if rolling short-window vol < its own rolling median over a longer
    horizon → "good" (1); otherwise "bad" (0).

    Args:
        market_returns:    Daily return series (e.g. vwretd), date-indexed.
        window:            Short rolling vol window (days).
        long_window_mult:  Long window = window × long_window_mult.

    Returns:
        pd.Series[bool] (True = good, False = bad), same index as input.
    """
    roll_vol = market_returns.rolling(window, min_periods=window // 2).std()
    long_window = window * long_window_mult
    roll_median = roll_vol.rolling(long_window, min_periods=window).median()
    return (roll_vol < roll_median).rename("regime_good")


def build_transition_matrix(regime_series: pd.Series) -> np.ndarray:
    """Estimate empirical 2×2 Markov transition matrix from a boolean/int series.

    Encoding: 0 = bad, 1 = good.
    Entry [i, j] = P(next regime = j | current regime = i).

    Args:
        regime_series:  Boolean or {0,1}-int daily regime labels.

    Returns:
        (2, 2) row-stochastic transition matrix.
    """
    regime_int = regime_series.astype(int).values
    counts = np.zeros((2, 2), dtype=float)
    for curr, nxt in zip(regime_int[:-1], regime_int[1:]):
        counts[curr, nxt] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums
