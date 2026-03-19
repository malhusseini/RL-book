"""CRSP daily S&P 500 data loading and preprocessing pipeline.

Follows the five-principle data-quality framework:
  1a. Load with stable types
  1b. Integrity checks: Time → Identifiers → Values
  1c. Missing data policy (never fill returns)
  1d. Causal rolling normalization (no look-ahead)
  1e. Winsorization of signals (rolling z-scores)
  1f. Strategy return series construction (quintile-based, causal)
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

CRSP_INVALID_CODES = {"B", "C", ""}   # CRSP placeholder strings for bad returns
PRIMARY_SHRCD      = {10, 11}          # Primary US common equity share codes
US_EXCHCD          = {1, 2, 3}         # NYSE=1, AMEX=2, NASDAQ=3


# ── 1a. Load ───────────────────────────────────────────────────────────────

def load_crsp(path: str) -> pd.DataFrame:
    """Load raw CRSP CSV with stable dtypes, sorted by (PERMNO, date)."""
    df = pd.read_csv(path, low_memory=False)

    # Parse date — convention: end-of-day exchange date, US Eastern, no timezone needed
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    n_bad_dates = df["date"].isna().sum()
    if n_bad_dates:
        logger.warning("Could not parse %d date values — dropping them.", n_bad_dates)
        df = df[df["date"].notna()]

    # Coerce all return and price columns; replace CRSP invalid strings first
    numeric_cols = [
        "RET", "RETX", "PRC", "VOL", "SHROUT",
        "DLRET", "DLRETX", "DLPRC",
        "vwretd", "vwretx", "ewretd", "ewretx", "sprtrn",
        "CFACPR", "CFACSHR", "SHRCD", "EXCHCD",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].replace(list(CRSP_INVALID_CODES), np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["PERMNO", "date"]).reset_index(drop=True)

    logger.info(
        "Loaded %d rows | %d unique PERMNOs | %s to %s",
        len(df),
        df["PERMNO"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


# ── 1b. Integrity checks: Time → Identifiers → Values ──────────────────────

def integrity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Run all three integrity checks in order; return cleaned DataFrame."""
    n_start = len(df)

    df = _check_time(df)
    df = _check_identifiers(df)
    df = _check_values(df)

    logger.info(
        "Integrity checks complete: %d → %d rows (removed %d).",
        n_start, len(df), n_start - len(df),
    )
    return df.reset_index(drop=True)


def _check_time(df: pd.DataFrame) -> pd.DataFrame:
    """Verify monotonicity within each PERMNO; drop future-dated rows."""
    # Monotonicity check (data is already sorted by PERMNO, date)
    bad_mono = (
        df.groupby("PERMNO")["date"]
        .transform(lambda s: s.diff().dt.days < 0)
    )
    n_bad = int(bad_mono.sum())
    if n_bad:
        logger.warning("Found %d non-monotonic date rows within some PERMNO.", n_bad)

    # No future dates
    today = pd.Timestamp.today().normalize()
    future_mask = df["date"] > today
    if future_mask.any():
        logger.warning("Dropping %d future-dated rows.", future_mask.sum())
        df = df[~future_mask]

    # Sanity: market-level series should have no unexpected gaps (log only)
    if "vwretd" in df.columns:
        market_dates = df.drop_duplicates("date").set_index("date")["vwretd"]
        n_missing_market = market_dates.isna().sum()
        if n_missing_market:
            logger.warning(
                "%d dates have missing vwretd — possible calendar gaps.", n_missing_market
            )

    return df


def _check_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to primary US common equity; deduplicate (PERMNO, date)."""
    n_before = len(df)

    # Restrict to primary common equity on major US exchanges
    if "SHRCD" in df.columns and "EXCHCD" in df.columns:
        mask = df["SHRCD"].isin(PRIMARY_SHRCD) & df["EXCHCD"].isin(US_EXCHCD)
        df = df[mask].copy()
        logger.info(
            "After SHRCD/EXCHCD filter: %d → %d rows (removed %d).",
            n_before, len(df), n_before - len(df),
        )

    # Log ticker instability (PERMNO with multiple tickers over time)
    if "TICKER" in df.columns:
        multi_ticker = (
            df.groupby("PERMNO")["TICKER"].nunique()
        )
        n_unstable = int((multi_ticker > 1).sum())
        if n_unstable:
            logger.info(
                "%d PERMNOs have more than one TICKER in this dataset "
                "(expected — using PERMNO as stable ID).",
                n_unstable,
            )

    # Deduplicate (PERMNO, date): prefer row with non-NaN RET
    dupes = df.duplicated(subset=["PERMNO", "date"], keep=False)
    if dupes.any():
        n_dupes = int(dupes.sum())
        logger.warning(
            "%d rows involved in (PERMNO, date) duplicates — "
            "keeping the row with valid RET.", n_dupes,
        )
        df["_ret_nan"] = df["RET"].isna().astype(int)
        df = (
            df.sort_values(["PERMNO", "date", "_ret_nan"])
              .drop_duplicates(subset=["PERMNO", "date"], keep="first")
              .drop(columns=["_ret_nan"])
        )

    return df


def _check_values(df: pd.DataFrame) -> pd.DataFrame:
    """Merge delisting returns; flag extreme returns."""
    df = _merge_delisting_returns(df)
    return df


def _merge_delisting_returns(df: pd.DataFrame) -> pd.DataFrame:
    """For each PERMNO, merge DLRET into the final observation's RET."""
    if "DLRET" not in df.columns:
        return df

    last_idx = df.groupby("PERMNO").tail(1).index
    has_dlret = df.loc[last_idx, "DLRET"].notna()
    n_affected = int(has_dlret.sum())

    if n_affected:
        logger.info("Merging DLRET into final RET for %d PERMNOs.", n_affected)
        target = last_idx[has_dlret]
        df.loc[target, "RET"] = df.loc[target, "DLRET"]

    return df


# ── 1c. Missing data policy ─────────────────────────────────────────────────

def apply_missing_data_policy(
    df: pd.DataFrame,
    min_coverage: float = 0.90,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enforce the missing-data policy:
      - Never fill RET.
      - Drop any date where valid-RET coverage < min_coverage of the universe.
      - Forward-fill PRC, VOL, SHROUT within PERMNO (for weighting only).
      - Flag extreme |RET| > 0.50 without removing.

    Returns:
        df_clean:     cleaned DataFrame
        dropped_log:  DataFrame of dropped dates and their coverage fractions
    """
    # Forward-fill non-return state variables (never used as returns)
    for col in ["PRC", "VOL", "SHROUT"]:
        if col in df.columns:
            df[col] = df.groupby("PERMNO")[col].ffill()

    # Flag extreme returns as a research decision — do NOT auto-remove
    extreme = df["RET"].abs() > 0.50
    n_extreme = int(extreme.sum())
    if n_extreme:
        logger.warning(
            "%d observations with |RET| > 50%% — "
            "inspect before calibration (not removed automatically).",
            n_extreme,
        )
    df["extreme_ret_flag"] = extreme

    # Coverage check per date (on the filtered universe)
    coverage = (
        df.groupby("date")["RET"]
        .apply(lambda s: s.notna().mean())
        .rename("coverage")
    )
    bad_dates = coverage[coverage < min_coverage]
    dropped_log = bad_dates.reset_index()

    if len(bad_dates):
        logger.warning(
            "Dropping %d dates with RET coverage < %.0f%%: %s … %s.",
            len(bad_dates),
            min_coverage * 100,
            bad_dates.index.min().date(),
            bad_dates.index.max().date(),
        )
        df = df[~df["date"].isin(bad_dates.index)]

    return df.reset_index(drop=True), dropped_log


# ── 1d. Causal rolling volatility ──────────────────────────────────────────

def add_rolling_vol(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Compute per-PERMNO rolling return volatility using only past data.
    Requires at least window // 2 observations; otherwise NaN.
    """
    df["rolling_vol"] = df.groupby("PERMNO")["RET"].transform(
        lambda s: s.rolling(window, min_periods=window // 2).std()
    )
    return df


# ── 1e. Causal winsorization ────────────────────────────────────────────────

def winsorize_returns(
    df: pd.DataFrame,
    z_cap: float = 3.0,
    window: int = 63,
) -> pd.DataFrame:
    """
    Winsorize each PERMNO's RET at |z| > z_cap using a rolling mean/std.
    Causal: rolling window uses only past data.
    Adds column 'RET_winsor'. Logs the fraction winsorized.
    """
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
    winsorized = valid & (df["RET"] != df["RET_winsor"])
    frac = winsorized.sum() / valid.sum()
    logger.info(
        "Winsorized %.3f%% of valid return observations (|z| > %.1f).",
        frac * 100,
        z_cap,
    )
    return df


# ── 1f. Strategy return series construction ─────────────────────────────────

def build_strategy_series(
    df: pd.DataFrame,
    vol_window: int = 21,
    n_quintiles: int = 5,
    use_winsorized: bool = True,
) -> pd.DataFrame:
    """
    Build two causal daily strategy return series:
      - Strategy 1 (conservative): equal-weighted returns of the
        lowest rolling-vol quintile each day.
      - Strategy 2 (aggressive): equal-weighted returns of the
        highest rolling-vol quintile each day.

    Quintile assignments use lagged rolling vol (one-day lag) — fully
    look-ahead-free.

    Returns a date-indexed DataFrame with columns:
      ['strat1_ret', 'strat2_ret', 'vwretd', 'ewretd', 'sprtrn']
    """
    ret_col = (
        "RET_winsor"
        if (use_winsorized and "RET_winsor" in df.columns)
        else "RET"
    )

    if "rolling_vol" not in df.columns:
        df = add_rolling_vol(df, window=vol_window)

    # Lag rolling vol by one day within each PERMNO (causality)
    df["_lagged_vol"] = df.groupby("PERMNO")["rolling_vol"].shift(1)

    # Assign quintile rank per date based on lagged vol
    df["_vol_quintile"] = df.groupby("date")["_lagged_vol"].transform(
        lambda s: pd.qcut(s, q=n_quintiles, labels=False, duplicates="drop")
    )

    # Strategy 1: bottom quintile (rank 0), Strategy 2: top quintile
    top_q = n_quintiles - 1
    strat1 = (
        df[df["_vol_quintile"] == 0]
        .groupby("date")[ret_col]
        .mean()
        .rename("strat1_ret")
    )
    strat2 = (
        df[df["_vol_quintile"] == top_q]
        .groupby("date")[ret_col]
        .mean()
        .rename("strat2_ret")
    )

    # Market-level series for regime construction (one row per date)
    market_cols = [c for c in ["vwretd", "ewretd", "sprtrn"] if c in df.columns]
    market = (
        df[["date"] + market_cols]
        .drop_duplicates("date")
        .set_index("date")
    )

    result = (
        pd.concat([strat1, strat2, market], axis=1)
        .dropna(subset=["strat1_ret", "strat2_ret"])
        .sort_index()
    )

    # Clean up helper columns
    df.drop(columns=["_lagged_vol", "_vol_quintile"], inplace=True, errors="ignore")

    logger.info(
        "Strategy series: %d trading days | %s to %s",
        len(result),
        result.index.min().date(),
        result.index.max().date(),
    )
    return result


# ── Full pipeline convenience function ─────────────────────────────────────

def build_full_pipeline(
    path: str,
    min_coverage: float = 0.90,
    vol_window: int = 21,
    winsor_window: int = 63,
    z_cap: float = 3.0,
    n_quintiles: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the complete loading and preprocessing pipeline.

    Returns:
        df_clean:         stock-level panel after all cleaning steps
        dropped_dates:    log of dates dropped due to low coverage
        strategy_series:  date-indexed strategy and market returns
    """
    df = load_crsp(path)
    df = integrity_checks(df)
    df, dropped_dates = apply_missing_data_policy(df, min_coverage=min_coverage)
    df = add_rolling_vol(df, window=vol_window)
    df = winsorize_returns(df, z_cap=z_cap, window=winsor_window)
    strategy_series = build_strategy_series(
        df, vol_window=vol_window, n_quintiles=n_quintiles
    )
    return df, dropped_dates, strategy_series
