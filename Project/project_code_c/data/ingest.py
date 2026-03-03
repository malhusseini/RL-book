"""CRSP daily panel ingestion and integrity checks.

All checks follow the order: Time → Identifiers → Values (corporate actions).

Convention
----------
  (PERMNO, date) is the primary key.  Tickers are labels only.
  Raw CRSP RET is a close-to-close total return (dividends reinvested).
  Negative PRC in CRSP = bid-ask midpoint (valid); zero PRC = no price (NaN).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# CRSP string codes that signal "no valid return"
CRSP_INVALID_CODES: set = {"B", "C", ""}

# Only ordinary common shares (no ADRs, preferred, etc.)
PRIMARY_SHRCD: set = {10, 11}

# Major US exchanges: NYSE (1), AMEX (2), NASDAQ (3)
US_EXCHCD: set = {1, 2, 3}


# ── Load ─────────────────────────────────────────────────────────────────────

def load_crsp(path: str) -> pd.DataFrame:
    """Load raw CRSP daily CSV with stable numeric dtypes, sorted by (PERMNO, date).

    Accepts .csv or .csv.gz.  All CRSP invalid-return string codes are
    replaced with NaN so downstream arithmetic never touches them.
    """
    df = pd.read_csv(path, low_memory=False)

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    n_bad = int(df["date"].isna().sum())
    if n_bad:
        logger.warning("Could not parse %d date values — dropping them.", n_bad)
        df = df[df["date"].notna()].copy()

    _numeric_cols = [
        "RET", "RETX", "PRC", "VOL", "SHROUT",
        "DLRET", "DLRETX", "DLPRC",
        "vwretd", "vwretx", "ewretd", "ewretx", "sprtrn",
        "CFACPR", "CFACSHR", "SHRCD", "EXCHCD",
        "DIVAMT", "SICCD",
    ]
    for col in _numeric_cols:
        if col in df.columns:
            df[col] = df[col].replace(list(CRSP_INVALID_CODES), np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["PERMNO", "date"]).reset_index(drop=True)
    logger.info(
        "Loaded %d rows | %d unique PERMNOs | %s to %s",
        len(df), df["PERMNO"].nunique(),
        df["date"].min().date(), df["date"].max().date(),
    )
    return df


# ── Integrity sub-checks ──────────────────────────────────────────────────────

def _check_time(df: pd.DataFrame) -> pd.DataFrame:
    """A: tz-normalization  B: weekend guard  C: monotonic sort  D: future dates  E: market-index."""
    # A. Timezone normalization
    if hasattr(df["date"].dtype, "tz") and df["date"].dt.tz is not None:
        logger.warning(
            "Dates carry timezone info (%s) — converting to tz-naive (UTC).",
            df["date"].dt.tz,
        )
        df = df.copy()
        df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)

    # B. Weekend sanity
    weekends = df["date"].dt.dayofweek >= 5
    if weekends.any():
        logger.warning(
            "%d rows fall on Sat/Sun — unexpected for CRSP daily equity data.",
            int(weekends.sum()),
        )

    # C. Non-monotonic ordering within PERMNO
    bad_mono = df.groupby("PERMNO")["date"].transform(lambda s: s.diff().dt.days < 0)
    if int(bad_mono.sum()):
        logger.warning(
            "%d non-monotonic date rows — re-sorting by (PERMNO, date).",
            int(bad_mono.sum()),
        )
        df = df.sort_values(["PERMNO", "date"]).reset_index(drop=True)

    # D. Future dates
    today = pd.Timestamp.today().normalize()
    future_mask = df["date"] > today
    if future_mask.any():
        logger.warning("Dropping %d future-dated rows.", int(future_mask.sum()))
        df = df[~future_mask].copy()

    # E. Market-index availability
    if "vwretd" in df.columns:
        n_miss = int(df.drop_duplicates("date").set_index("date")["vwretd"].isna().sum())
        if n_miss:
            logger.warning("%d dates have missing vwretd.", n_miss)

    return df


def _check_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """A: invalid PERMNO  B: SHRCD/EXCHCD filter  C: ticker audit  D: dedup."""
    # A. Drop null/zero PERMNO rows
    invalid = df["PERMNO"].isna() | (df["PERMNO"] == 0)
    if invalid.any():
        logger.warning(
            "Dropping %d rows with null/zero PERMNO.",
            int(invalid.sum()),
        )
        df = df[~invalid].copy()

    # B. Share class / exchange filter
    n_before = len(df)
    if "SHRCD" in df.columns and "EXCHCD" in df.columns:
        mask = df["SHRCD"].isin(PRIMARY_SHRCD) & df["EXCHCD"].isin(US_EXCHCD)
        df = df[mask].copy()
        logger.info(
            "SHRCD/EXCHCD filter: %d → %d rows (removed %d).",
            n_before, len(df), n_before - len(df),
        )

    # C. Ticker stability audit (PERMNOs with multiple tickers)
    if "TICKER" in df.columns:
        n_multi = int((df.groupby("PERMNO")["TICKER"].nunique() > 1).sum())
        if n_multi:
            logger.info(
                "%d PERMNOs have >1 TICKER (PERMNO used as stable ID).", n_multi
            )

    # D. Deduplicate (PERMNO, date) — keep row with valid RET
    dupes = df.duplicated(subset=["PERMNO", "date"], keep=False)
    if dupes.any():
        logger.warning(
            "%d rows in (PERMNO,date) duplicates — keeping row with valid RET.",
            int(dupes.sum()),
        )
        df = df.copy()
        df["_ret_nan"] = df["RET"].isna().astype(int)
        df = (
            df.sort_values(["PERMNO", "date", "_ret_nan"])
            .drop_duplicates(subset=["PERMNO", "date"], keep="first")
            .drop(columns=["_ret_nan"])
        )

    return df


def _merge_delisting_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Merge CRSP DLRET into RET for the final observation of each PERMNO.

    CRSP's last-traded-day RET excludes the delisting event.  Replacing it
    with DLRET (where available) yields unbiased compounded wealth paths.
    """
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


def _check_values(df: pd.DataFrame) -> pd.DataFrame:
    """A: zero-price → NaN  B: extreme-return audit  C: delisting merge."""
    # A. Zero PRC → NaN (zero = no price recorded; negative = valid midpoint)
    if "PRC" in df.columns:
        zero_price = df["PRC"] == 0.0
        n_zero = int(zero_price.sum())
        if n_zero:
            logger.warning(
                "%d rows with PRC==0 → setting NaN (negative PRC is valid bid-ask midpoint).",
                n_zero,
            )
            df = df.copy()
            df.loc[zero_price, "PRC"] = np.nan

    # B. Infinite returns → NaN (inf is a data error, not an outlier; outliers are retained)
    if "RET" in df.columns:
        inf_mask = np.isinf(df["RET"])
        n_inf = int(inf_mask.sum())
        if n_inf:
            logger.warning(
                "%d rows with ±inf RET — replacing with NaN (genuine data errors).", n_inf
            )
            df = df.copy()
            df.loc[inf_mask, "RET"] = np.nan

    # C. Extreme-return audit (flag, retain — crashes are economically real)
    if "RET" in df.columns:
        extreme = df["RET"].abs() > 0.50
        n_ext = int(extreme.sum())
        if n_ext:
            top_ext = df.loc[extreme, "PERMNO"].value_counts().head(5).to_dict()
            logger.warning(
                "%d observations with |RET|>50%% (flagged, not removed). Top PERMNOs: %s",
                n_ext, top_ext,
            )

    # C. Delisting return correction
    return _merge_delisting_returns(df)


def integrity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Run ordered Time → Identifiers → Values checks; return cleaned DataFrame."""
    n_start = len(df)
    df = _check_time(df)
    df = _check_identifiers(df)
    df = _check_values(df)
    logger.info(
        "Integrity checks: %d → %d rows (removed %d).",
        n_start, len(df), n_start - len(df),
    )
    return df.reset_index(drop=True)
