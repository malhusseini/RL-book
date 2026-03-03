"""Missing-data policy, outlier handling, winsorization, and parquet I/O.

Documented policies (applied here, referenced downstream)
----------------------------------------------------------
NA policy
    RET   : never filled.  A missing return is a missing return.
    PRC / VOL / SHROUT : forward-filled *within each PERMNO* for signal use
            (carry yield, market-cap weights).
    Dates with cross-sectional RET coverage < min_coverage are dropped.

Outlier policy
    |RET| > 50 % are flagged and RETAINED.  They are winsorized only when
    building *signals* (z-scores), never for PnL/wealth computations.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Missing-data policy ───────────────────────────────────────────────────────

def apply_missing_data_policy(
    df: pd.DataFrame,
    min_coverage: float = 0.90,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply NA and outlier policies; return (cleaned_df, dropped_dates_log).

    Fills PRC/VOL/SHROUT forward within PERMNO (for signal use only).
    Drops dates where RET cross-sectional coverage < min_coverage.
    Flags but retains |RET| > 50%.
    """
    df = df.copy()

    for col in ["PRC", "VOL", "SHROUT"]:
        if col in df.columns:
            n_before = int(df[col].isna().sum())
            df[col] = df.groupby("PERMNO")[col].ffill()
            n_after = int(df[col].isna().sum())
            if n_before:
                logger.info(
                    "NA POLICY — forward-fill %s: %d NaN → %d remaining (%.1f%% filled).",
                    col, n_before, n_after,
                    100.0 * (n_before - n_after) / max(n_before, 1),
                )

    # Outlier flag (retain; winsorize in build_strategy_series signal paths only)
    if "RET" in df.columns:
        extreme = df["RET"].abs() > 0.50
        df["extreme_ret_flag"] = extreme
        n_ext = int(extreme.sum())
        if n_ext:
            logger.warning(
                "OUTLIER POLICY: %d obs |RET|>50%% flagged but retained.", n_ext
            )

    # Date-level coverage filter
    coverage = (
        df.groupby("date")["RET"]
        .apply(lambda s: s.notna().mean())
        .rename("coverage")
    )
    bad_dates = coverage[coverage < min_coverage]
    dropped_log = bad_dates.reset_index()
    if len(bad_dates):
        logger.warning(
            "NA POLICY — dropping %d dates with RET coverage < %.0f%%.",
            len(bad_dates), min_coverage * 100,
        )
        df = df[~df["date"].isin(bad_dates.index)]

    return df.reset_index(drop=True), dropped_log


# ── Causal winsorization ──────────────────────────────────────────────────────

def winsorize_returns(
    df: pd.DataFrame,
    z_cap: float = 3.0,
    window: int = 63,
) -> pd.DataFrame:
    """Add RET_winsor column: RET clipped at ±z_cap rolling standard deviations.

    The rolling mean/std use only past data (causal).  The original RET column
    is preserved unchanged for PnL/wealth computations.
    """
    df = df.copy()
    rolling_mean = df.groupby("PERMNO")["RET"].transform(
        lambda s: s.rolling(window, min_periods=window // 2).mean()
    )
    rolling_std = df.groupby("PERMNO")["RET"].transform(
        lambda s: s.rolling(window, min_periods=window // 2).std()
    )
    lo = rolling_mean - z_cap * rolling_std
    hi = rolling_mean + z_cap * rolling_std
    df["RET_winsor"] = df["RET"].clip(lower=lo, upper=hi)

    valid = df["RET"].notna()
    frac = float((valid & (df["RET"] != df["RET_winsor"])).sum() / max(valid.sum(), 1))
    logger.info("Winsorized %.3f%% of valid returns (|z| > %.1f).", frac * 100, z_cap)
    return df


# ── Parquet I/O ───────────────────────────────────────────────────────────────

def save_panel(df: pd.DataFrame, path: str) -> None:
    """Persist cleaned panel to a parquet file.  Creates parent dirs as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved cleaned panel to %s  (%d rows).", path, len(df))


def load_panel(path: str) -> pd.DataFrame:
    """Load a previously saved parquet panel."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded panel from %s  (%d rows).", path, len(df))
    return df


# ── Full pipeline convenience wrapper ────────────────────────────────────────

def build_full_pipeline(
    path: str,
    min_coverage: float = 0.90,
    winsor_window: int = 63,
    z_cap: float = 3.0,
    start_date: Optional[str] = None,
    save_cleaned_to: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load CRSP → integrity checks → missing-data policy → winsorize.

    Args:
        path:             Path to raw CRSP CSV (or parquet).
        min_coverage:     Minimum cross-sectional RET coverage to keep a date.
        winsor_window:    Rolling window for return winsorization.
        z_cap:            Z-score cap for winsorization.
        start_date:       If given (e.g. '2002-01-01'), drop earlier rows.
        save_cleaned_to:  If given, persist cleaned panel to this parquet path.

    Returns:
        df_clean:       Stock-level panel after all cleaning.
        dropped_dates:  Log of dates dropped due to low coverage.
    """
    from .ingest import load_crsp, integrity_checks

    df = load_crsp(path)
    df = integrity_checks(df)
    df, dropped_dates = apply_missing_data_policy(df, min_coverage=min_coverage)

    if start_date is not None:
        cutoff = pd.Timestamp(start_date)
        n_before = len(df)
        df = df[df["date"] >= cutoff].reset_index(drop=True)
        logger.info(
            "start_date filter (%s): %d → %d rows.",
            start_date, n_before, len(df),
        )

    df = winsorize_returns(df, z_cap=z_cap, window=winsor_window)

    if save_cleaned_to:
        save_panel(df, save_cleaned_to)

    return df, dropped_dates
