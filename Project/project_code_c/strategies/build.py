"""Combine all three strategy return streams into a single aligned DataFrame.

This is the main entry-point for the strategies sub-package.  It calls
build_conservative / build_medium / build_aggressive and aligns their outputs
on the common date index.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .conservative import build_conservative
from .medium import build_medium
from .aggressive import build_aggressive

logger = logging.getLogger(__name__)


def build_strategy_series(
    df: pd.DataFrame,
    use_winsorized: bool = True,
    # Conservative params
    pca_window: int = 126,
    pca_n_components: int = 5,
    pca_z_threshold: float = 1.5,
    pca_z_window: int = 21,
    # Medium params
    medium_universe_size: int = 150,
    medium_n_pairs: int = 25,
    medium_corr_window: int = 63,
    medium_hedge_window: int = 63,
    medium_z_window: int = 21,
    medium_entry_z: float = 2.0,
    medium_exit_z: float = 0.5,
    medium_rebal_freq: int = 21,
    # Aggressive params
    agg_rv_short: int = 21,
    agg_rv_long: int = 63,
    agg_top_quintile: float = 0.80,
    # Shared
    market_col: str = "vwretd",
) -> pd.DataFrame:
    """Build all three causal strategy return streams and align them by date.

    Args:
        df:              Cleaned CRSP panel (with RET_winsor if use_winsorized=True).
        use_winsorized:  Use RET_winsor for signal computation (True) or raw RET.
        pca_*:           Hyperparameters for the conservative strategy.
        medium_*:        Hyperparameters for the medium strategy.
        agg_*:           Hyperparameters for the aggressive strategy.
        market_col:      Column for market benchmark / regime detection.

    Returns:
        Date-indexed DataFrame with columns:
          ['strat1_ret', 'strat2_ret', 'strat3_ret', 'vwretd', 'ewretd', 'sprtrn']
        Rows with NaN in any strategy column are NOT dropped here — callers decide
        how to handle the burn-in period.
    """
    ret_col = (
        "RET_winsor"
        if (use_winsorized and "RET_winsor" in df.columns)
        else "RET"
    )
    logger.info("Building strategy series using return column '%s'.", ret_col)

    strat1 = build_conservative(
        df, ret_col=ret_col,
        pca_window=pca_window,
        n_components=pca_n_components,
        z_threshold=pca_z_threshold,
        z_window=pca_z_window,
    )

    strat2 = build_medium(
        df, ret_col=ret_col,
        universe_size=medium_universe_size,
        n_pairs=medium_n_pairs,
        corr_window=medium_corr_window,
        hedge_window=medium_hedge_window,
        z_window=medium_z_window,
        entry_z=medium_entry_z,
        exit_z=medium_exit_z,
        rebal_freq=medium_rebal_freq,
    )

    strat3 = build_aggressive(
        df, ret_col=ret_col,
        market_col=market_col,
        rv_short=agg_rv_short,
        rv_long=agg_rv_long,
        top_quintile=agg_top_quintile,
    )

    # Collect available market benchmark columns
    market_cols = [c for c in ["vwretd", "ewretd", "sprtrn"] if c in df.columns]
    market = (
        df[["date"] + market_cols]
        .drop_duplicates("date")
        .set_index("date")
        if market_cols
        else pd.DataFrame(index=df["date"].drop_duplicates().sort_values())
    )

    combined = pd.concat([strat1, strat2, strat3, market], axis=1)

    logger.info(
        "Strategy series pre-dropna: strat1=%d, strat2=%d, strat3=%d, overlap=%d dates.",
        int(combined["strat1_ret"].notna().sum()),
        int(combined["strat2_ret"].notna().sum()),
        int(combined["strat3_ret"].notna().sum()),
        int(combined[["strat1_ret", "strat2_ret", "strat3_ret"]].notna().all(axis=1).sum()),
    )

    result = combined.dropna(subset=["strat1_ret", "strat2_ret", "strat3_ret"]).sort_index()

    logger.info(
        "Strategy series: %d trading days | %s to %s",
        len(result),
        result.index.min().date() if len(result) else "N/A",
        result.index.max().date() if len(result) else "N/A",
    )
    return result
