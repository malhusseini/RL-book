"""Strategy 1 (conservative): Avellaneda-Lee PCA residual mean reversion.

Algorithm
---------
1. Build a wide return matrix (date × PERMNO).
2. At each date t:
   a. Fit PCA on the past `pca_window` days using numpy SVD (causal, no sklearn).
   b. Compute day-t factor-model residuals: ε_i = r_i − B f_t  where f_t = B^T r_t.
3. Per-stock rolling z-score of the residuals (causal).
4. Bang-bang position: +1 when z < −z_threshold (long residual), −1 when z > +z_threshold.
5. Lag all positions by 1 trading day.
6. PnL at t+1 = lagged_position_t × return_{t+1}.
7. Strategy daily return = equal-weighted mean PnL across all active positions.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..features.factors import numpy_pca

logger = logging.getLogger(__name__)

# Defaults (can be overridden via build_conservative kwargs)
_PCA_WINDOW = 126       # estimation window in trading days
_N_COMPONENTS = 5       # number of PCA factors
_Z_THRESHOLD = 1.5      # entry/exit threshold on residual z-score
_Z_WINDOW = 21          # rolling window for per-stock residual z-score


def build_conservative(
    df: pd.DataFrame,
    ret_col: str = "RET_winsor",
    pca_window: int = _PCA_WINDOW,
    n_components: int = _N_COMPONENTS,
    z_threshold: float = _Z_THRESHOLD,
    z_window: int = _Z_WINDOW,
    max_nan_frac: float = 0.50,
    min_stocks: int = 15,
) -> pd.Series:
    """Build the conservative (PCA residual mean-reversion) daily return series.

    Args:
        df:            Cleaned CRSP panel with columns (PERMNO, date, ret_col).
        ret_col:       Return column to use ('RET_winsor' or 'RET').
        pca_window:    Days of history used to fit PCA each date.
        n_components:  Number of PCA factors.
        z_threshold:   |z| above which a position is opened.
        z_window:      Rolling window for per-stock residual z-score.
        max_nan_frac:  Max fraction of NaN allowed in estimation window per stock.
        min_stocks:    Minimum number of valid stocks required to produce a signal.

    Returns:
        pd.Series named 'strat1_ret', date-indexed, daily PnL (no NaN-padding).
    """
    if ret_col not in df.columns:
        ret_col = "RET"
        logger.warning("ret_col not found; falling back to RET.")

    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    if len(dates) < pca_window + z_window + 5:
        raise ValueError(
            f"Not enough dates ({len(dates)}) for pca_window={pca_window} "
            f"+ z_window={z_window}."
        )

    wide = df.pivot(index="date", columns="PERMNO", values=ret_col)

    records: list[dict] = []
    for i in range(pca_window, len(dates)):
        t = dates.iloc[i]
        past_dates = dates.iloc[i - pca_window: i].values

        past = wide.loc[past_dates]
        valid_stocks = past.isna().mean() <= max_nan_frac
        past_clean = past.loc[:, valid_stocks].fillna(0.0)

        if past_clean.shape[1] < n_components + min_stocks:
            continue

        mu = past_clean.mean(axis=0).values
        X = past_clean.values - mu
        B = numpy_pca(X, n_components)

        r_t = wide.loc[t, valid_stocks]
        valid_t = r_t.notna()
        if valid_t.sum() < min_stocks:
            continue

        r_vals = r_t[valid_t].values.astype(float)
        B_t = B[valid_t.values]
        f_t = r_vals @ B_t
        residual = r_vals - B_t @ f_t

        for j, pno in enumerate(r_t.index[valid_t]):
            records.append({"date": t, "PERMNO": pno, "residual": residual[j], "ret": r_vals[j]})

    if not records:
        logger.warning("Strategy 1: no records produced; check data coverage.")
        return pd.Series(dtype=float, name="strat1_ret")

    res_df = pd.DataFrame(records).sort_values(["PERMNO", "date"]).reset_index(drop=True)

    # Causal per-stock rolling z-score of residual
    res_df["res_mean"] = res_df.groupby("PERMNO")["residual"].transform(
        lambda s: s.rolling(z_window, min_periods=z_window // 2).mean()
    )
    res_df["res_std"] = res_df.groupby("PERMNO")["residual"].transform(
        lambda s: s.rolling(z_window, min_periods=z_window // 2).std()
    )
    denom = res_df["res_std"].where(res_df["res_std"] > 1e-12, np.nan)
    res_df["z"] = (res_df["residual"] - res_df["res_mean"]) / denom

    # Bang-bang positions
    res_df["position"] = 0.0
    res_df.loc[res_df["z"] < -z_threshold, "position"] = 1.0
    res_df.loc[res_df["z"] >  z_threshold, "position"] = -1.0

    # Lag positions by 1 trading day within each PERMNO (causality requirement)
    res_df["position_lag"] = res_df.groupby("PERMNO")["position"].shift(1)

    res_df["pnl"] = res_df["position_lag"] * res_df["ret"]

    strat1_ret = res_df.groupby("date")["pnl"].mean().rename("strat1_ret")
    logger.info(
        "Strategy 1 (PCA residual MR, K=%d): %d trading days, z_thresh=%.1f.",
        n_components, int(strat1_ret.dropna().shape[0]), z_threshold,
    )
    return strat1_ret
