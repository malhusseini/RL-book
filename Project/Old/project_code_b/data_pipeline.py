"""CRSP daily S&P 500 data pipeline for Phase 2 version b.

Strategy 1 (conservative): Avellaneda–Lee residual mean reversion (PCA factors).
  Implemented with numpy SVD — no sklearn required.
Strategy 2 (aggressive):   Equity carry proxy (trailing dividend-yield quintile).

Data-quality framework §1a–1e is identical to project_code.
Only §1f (strategy construction) differs.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# CRSP invalid return codes
CRSP_INVALID_CODES = {"B", "C", ""}
PRIMARY_SHRCD = {10, 11}
US_EXCHCD = {1, 2, 3}

# Strategy 1 — Avellaneda–Lee PCA parameters
PCA_ROLLING_WINDOW = 126      # trading days of history for each PCA fit
PCA_N_COMPONENTS = 5          # number of principal components (common factors)
RESIDUAL_Z_THRESHOLD = 1.5    # |z| > threshold triggers a position
RESIDUAL_Z_WINDOW = 21        # rolling window for per-stock residual z-score

# Strategy 2 — equity carry parameters
CARRY_LOOKBACK_DAYS = 252     # trailing window to sum dividends
CARRY_N_QUINTILES = 5         # quintile-rank stocks by carry signal each day


# ── 1a. Load ───────────────────────────────────────────────────────────────

def load_crsp(path: str) -> pd.DataFrame:
    """Load raw CRSP CSV with stable dtypes, sorted by (PERMNO, date)."""
    df = pd.read_csv(path, low_memory=False)

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    n_bad = df["date"].isna().sum()
    if n_bad:
        logger.warning("Could not parse %d date values — dropping them.", n_bad)
        df = df[df["date"].notna()]

    numeric_cols = [
        "RET", "RETX", "PRC", "VOL", "SHROUT",
        "DLRET", "DLRETX", "DLPRC",
        "vwretd", "vwretx", "ewretd", "ewretx", "sprtrn",
        "CFACPR", "CFACSHR", "SHRCD", "EXCHCD",
        "DIVAMT",
    ]
    for col in numeric_cols:
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


# ── 1b. Integrity checks: Time → Identifiers → Values ────────────────────

def _check_time(df: pd.DataFrame) -> pd.DataFrame:
    # A. Timezone normalization — all timestamps must live in one consistent
    #    representation.  CRSP daily is always tz-naive; if somehow a tz crept
    #    in (e.g. from a joined source), strip it and warn.
    if hasattr(df["date"].dtype, "tz") and df["date"].dt.tz is not None:
        logger.warning(
            "Dates carry timezone info (%s) — converting to tz-naive "
            "(treating as UTC) to enforce a single consistent representation.",
            df["date"].dt.tz,
        )
        df = df.copy()
        df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)

    # B. Weekend / holiday sanity — US equity daily data should never land on
    #    Saturday (dayofweek==5) or Sunday (6).  A large count suggests a
    #    data-source issue or a timezone shift masquerading as a date shift.
    weekends = df["date"].dt.dayofweek >= 5
    if weekends.any():
        logger.warning(
            "%d rows fall on Saturday/Sunday — unexpected for CRSP daily "
            "equity data (possible date-format or timezone mismatch).",
            int(weekends.sum()),
        )

    # C. Non-monotonic ordering within PERMNO — re-sort to fix rather than
    #    merely warn; downstream rolling windows require sorted dates.
    bad_mono = df.groupby("PERMNO")["date"].transform(
        lambda s: s.diff().dt.days < 0
    )
    if int(bad_mono.sum()):
        logger.warning(
            "%d non-monotonic date rows found — re-sorting by (PERMNO, date).",
            int(bad_mono.sum()),
        )
        df = df.sort_values(["PERMNO", "date"]).reset_index(drop=True)

    # D. Future dates — cannot have been available at time t.
    today = pd.Timestamp.today().normalize()
    future_mask = df["date"] > today
    if future_mask.any():
        logger.warning("Dropping %d future-dated rows.", int(future_mask.sum()))
        df = df[~future_mask]

    # E. Market-index availability check.
    if "vwretd" in df.columns:
        n_missing = df.drop_duplicates("date").set_index("date")["vwretd"].isna().sum()
        if n_missing:
            logger.warning("%d dates have missing vwretd.", int(n_missing))
    return df


def _check_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    # A. PERMNO validity — PERMNO is the stable, vendor-assigned security
    #    identifier.  Tickers are not stable (reused across companies, differ by
    #    vendor), so PERMNO is used throughout.  Rows with null or zero PERMNO
    #    cannot be mapped to any instrument and must be dropped.
    invalid_permno = df["PERMNO"].isna() | (df["PERMNO"] == 0)
    if invalid_permno.any():
        logger.warning(
            "Dropping %d rows with null or zero PERMNO "
            "(PERMNO is the stable security identifier; "
            "zero/null rows cannot be instrument-mapped).",
            int(invalid_permno.sum()),
        )
        df = df[~invalid_permno].copy()

    # B. Share class / exchange filter — restrict to ordinary common shares
    #    (SHRCD 10/11) on major US exchanges (EXCHCD 1/2/3) to avoid mixing
    #    ADRs, preferred shares, or OTC instruments into the universe.
    n_before = len(df)
    if "SHRCD" in df.columns and "EXCHCD" in df.columns:
        mask = df["SHRCD"].isin(PRIMARY_SHRCD) & df["EXCHCD"].isin(US_EXCHCD)
        df = df[mask].copy()
        logger.info(
            "After SHRCD/EXCHCD filter: %d → %d rows (removed %d).",
            n_before, len(df), n_before - len(df),
        )

    # C. Ticker stability audit — tickers are not stable: they are reused across
    #    companies and differ by vendor.  Log PERMNOs that changed ticker so we
    #    know the stable ID (PERMNO) is doing the correct disambiguation.
    if "TICKER" in df.columns:
        n_multi = int((df.groupby("PERMNO")["TICKER"].nunique() > 1).sum())
        if n_multi:
            logger.info(
                "%d PERMNOs have more than one TICKER (using PERMNO as stable ID).",
                n_multi,
            )
    dupes = df.duplicated(subset=["PERMNO", "date"], keep=False)
    if dupes.any():
        logger.warning(
            "%d rows in (PERMNO, date) duplicates — keeping row with valid RET.",
            int(dupes.sum()),
        )
        df["_ret_nan"] = df["RET"].isna().astype(int)
        df = (
            df.sort_values(["PERMNO", "date", "_ret_nan"])
            .drop_duplicates(subset=["PERMNO", "date"], keep="first")
            .drop(columns=["_ret_nan"])
        )
    return df


def _merge_delisting_returns(df: pd.DataFrame) -> pd.DataFrame:
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
    # A. Price sanity — CRSP convention: negative PRC = bid-ask midpoint
    #    (perfectly valid); zero PRC = no price recorded (invalid for any
    #    return or yield computation → set to NaN so downstream fillna/abs
    #    logic sees a genuine missing value rather than a false zero).
    if "PRC" in df.columns:
        zero_price = df["PRC"] == 0.0
        n_zero = int(zero_price.sum())
        if n_zero:
            logger.warning(
                "%d rows with PRC == 0 — setting to NaN. "
                "(CRSP: negative PRC is a valid bid-ask midpoint; "
                "zero PRC means no price was recorded.)",
                n_zero,
            )
            df = df.copy()
            df.loc[zero_price, "PRC"] = np.nan

    # B. Extreme-return pre-audit — identify before any imputation.
    #    Policy decision: keep (outliers are data; crashes, squeezes, and
    #    delisting-adjacent moves are economically real).  Flag for downstream
    #    inspection.  Identifying the top PERMNOs helps distinguish genuine
    #    micro-cap volatility from data errors.
    if "RET" in df.columns:
        extreme = df["RET"].abs() > 0.50
        n_ext = int(extreme.sum())
        if n_ext:
            top_ext = df.loc[extreme, "PERMNO"].value_counts().head(5).to_dict()
            logger.warning(
                "%d raw observations with |RET| > 50%% (will be flagged, not "
                "removed — see OUTLIER POLICY in apply_missing_data_policy). "
                "Top PERMNOs by frequency: %s",
                n_ext, top_ext,
            )

    # C. Delisting return correction — CRSP's last-traded-day return ignores
    #    the delisting event; merge DLRET into RET for the final observation
    #    of each PERMNO so compounded wealth paths are unbiased.
    return _merge_delisting_returns(df)


def integrity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Run Time → Identifiers → Values checks; return cleaned DataFrame."""
    n_start = len(df)
    df = _check_time(df)
    df = _check_identifiers(df)
    df = _check_values(df)
    logger.info(
        "Integrity checks complete: %d → %d rows (removed %d).",
        n_start, len(df), n_start - len(df),
    )
    return df.reset_index(drop=True)


# ── 1c. Missing data policy ────────────────────────────────────────────────

def apply_missing_data_policy(
    df: pd.DataFrame,
    min_coverage: float = 0.90,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Documented missing-data and outlier policy (strategy-aware, causal):

    NA policy
    ---------
    - RET  : never filled.  A missing return is a missing return; filling
             would create phantom stability or fake compounded wealth.
    - PRC / VOL / SHROUT : forward-filled within each PERMNO for *signal*
             use only (carry yield, market-cap weights).  The scope of each
             fill is logged so it can be audited.
    - Dates with cross-sectional RET coverage < min_coverage are dropped
             entirely (not filled) and returned in the audit log.

    Outlier policy
    --------------
    - |RET| > 50 % : FLAGGED but RETAINED.  Outliers are data points —
             crashes (2008, COVID), micro-cap squeezes, and
             delisting-adjacent returns are economically real events.
             Removing them would bias performance statistics.
             They are clipped later in winsorize_returns when forming
             *signals*, preserving their influence in the wealth path.
    """
    # Forward-fill price/volume/shrout; log exact scope of each fill.
    for col in ["PRC", "VOL", "SHROUT"]:
        if col in df.columns:
            n_na_before = int(df[col].isna().sum())
            df[col] = df.groupby("PERMNO")[col].ffill()
            n_na_after = int(df[col].isna().sum())
            if n_na_before:
                logger.info(
                    "NA POLICY — forward-fill %s: %d NaN → %d remaining NaN "
                    "(%.1f%% filled; residual NaN = asset not yet traded).",
                    col, n_na_before, n_na_after,
                    100.0 * (n_na_before - n_na_after) / max(n_na_before, 1),
                )

    # Outlier policy: flag, retain, document.
    extreme = df["RET"].abs() > 0.50
    n_ext = int(extreme.sum())
    if n_ext:
        logger.warning(
            "OUTLIER POLICY: %d observations with |RET| > 50%% are FLAGGED "
            "but RETAINED (crashes and delisting-adjacent returns are real). "
            "They will be clipped during signal winsorization only.",
            n_ext,
        )
    df["extreme_ret_flag"] = extreme

    # Date-level coverage filter.
    coverage = (
        df.groupby("date")["RET"]
        .apply(lambda s: s.notna().mean())
        .rename("coverage")
    )
    bad_dates = coverage[coverage < min_coverage]
    dropped_log = bad_dates.reset_index()
    if len(bad_dates):
        logger.warning(
            "NA POLICY — dropping %d dates with RET coverage < %.0f%% "
            "(missing data would create fake cross-sectional stability).",
            len(bad_dates), min_coverage * 100,
        )
        df = df[~df["date"].isin(bad_dates.index)]
    return df.reset_index(drop=True), dropped_log


# ── 1e. Causal winsorization ────────────────────────────────────────────────

def winsorize_returns(
    df: pd.DataFrame,
    z_cap: float = 3.0,
    window: int = 63,
) -> pd.DataFrame:
    """Winsorize RET at |z| > z_cap using rolling mean/std. Adds RET_winsor."""
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
    frac = (valid & (df["RET"] != df["RET_winsor"])).sum() / valid.sum()
    logger.info(
        "Winsorized %.3f%% of valid return observations (|z| > %.1f).",
        frac * 100, z_cap,
    )
    return df


# ── 1f. Strategy 1: Avellaneda–Lee PCA residual mean reversion ────────────

def _numpy_pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """
    Compute top-K principal component loadings from a centred data matrix X
    using numpy SVD (no sklearn dependency).

    Args:
        X:             (n_samples, n_features) centred matrix.
        n_components:  Number of components to retain.

    Returns:
        B: (n_features, n_components) loading matrix (columns = eigenvectors).
    """
    k = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
    # full_matrices=False gives Vt of shape (min(m,n), n)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k, :].T          # (n_features, k)


def _build_strategy1_pca_residual(
    df: pd.DataFrame,
    ret_col: str,
    window: int = PCA_ROLLING_WINDOW,
    n_components: int = PCA_N_COMPONENTS,
    z_threshold: float = RESIDUAL_Z_THRESHOLD,
    z_window: int = RESIDUAL_Z_WINDOW,
) -> pd.Series:
    """
    Strategy 1: PCA residual mean reversion (fully causal, numpy SVD).

    Pipeline per trading day t:
      1. Fit PCA on past `window` days (causal, no look-ahead).
      2. Project day-t returns onto the factor loadings; residual = actual − factor-fitted.
      3. Rolling z-score of the residual (per stock, causal).
      4. Position: +1 when z < −threshold (long residual), −1 when z > +threshold (short).
      5. Lag positions by 1 day.
      6. PnL = lagged_position × next-day return.
      7. Strategy return = equal-weighted mean PnL across all active positions.

    Factor choice: PCA (top K principal components). No external data needed.
    """
    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    if len(dates) < window + 10:
        raise ValueError(
            f"Only {len(dates)} dates available; need at least {window + 10}."
        )

    # Wide return matrix: rows = date, columns = PERMNO
    wide = df.pivot(index="date", columns="PERMNO", values=ret_col)

    records = []
    for i in range(window, len(dates)):
        t = dates.iloc[i]
        past_dates = dates.iloc[i - window: i].values

        # Drop stocks with > 50 % NaN in the estimation window
        past = wide.loc[past_dates]
        valid_stocks = past.isna().mean() <= 0.5
        past_clean = past.loc[:, valid_stocks].fillna(0.0)

        if past_clean.shape[1] < n_components + 5:
            continue

        # Centre and fit PCA via SVD
        mu = past_clean.mean(axis=0).values
        X = past_clean.values - mu          # (window, n_valid_stocks)
        B = _numpy_pca(X, n_components)     # (n_valid_stocks, K)

        # Day-t residuals
        r_t = wide.loc[t, valid_stocks]     # Series over valid_stocks PERMNOs
        valid_t = r_t.notna()
        if valid_t.sum() < 10:
            continue

        r_vals = r_t[valid_t].values.astype(float)
        B_t = B[valid_t.values]             # (n_valid_t, K)
        f_t = r_vals @ B_t                  # (K,)
        residual = r_vals - B_t @ f_t       # (n_valid_t,)

        for j, pno in enumerate(r_t.index[valid_t]):
            records.append({
                "date": t,
                "PERMNO": pno,
                "residual": residual[j],
                "ret": r_vals[j],
            })

    if not records:
        logger.warning("Strategy 1: no records produced; check data availability.")
        return pd.Series(dtype=float, name="strat1_ret")

    res_df = pd.DataFrame(records).sort_values(["PERMNO", "date"]).reset_index(drop=True)

    # Per-PERMNO rolling z-score of residual (causal)
    res_df["res_mean"] = res_df.groupby("PERMNO")["residual"].transform(
        lambda s: s.rolling(z_window, min_periods=z_window // 2).mean()
    )
    res_df["res_std"] = res_df.groupby("PERMNO")["residual"].transform(
        lambda s: s.rolling(z_window, min_periods=z_window // 2).std()
    )
    denom = res_df["res_std"].where(res_df["res_std"] > 1e-12, np.nan)
    res_df["z"] = (res_df["residual"] - res_df["res_mean"]) / denom

    # Threshold to positions
    res_df["position"] = 0.0
    res_df.loc[res_df["z"] < -z_threshold, "position"] = 1.0
    res_df.loc[res_df["z"] >  z_threshold, "position"] = -1.0

    # Lag positions by 1 trading day within each stock (causality)
    res_df["position_lag"] = res_df.groupby("PERMNO")["position"].shift(1)

    # PnL = lagged position × realised return
    res_df["pnl"] = res_df["position_lag"] * res_df["ret"]

    strat1_ret = res_df.groupby("date")["pnl"].mean().rename("strat1_ret")
    logger.info(
        "Strategy 1 (Avellaneda–Lee PCA, K=%d): %d trading days, "
        "z-threshold=%.1f, causal.",
        n_components, len(strat1_ret.dropna()), z_threshold,
    )
    return strat1_ret


# ── 1f. Strategy 2: Equity carry proxy (dividend-yield quintile) ──────────

def _build_strategy2_equity_carry(
    df: pd.DataFrame,
    ret_col: str,
    lookback: int = CARRY_LOOKBACK_DAYS,
    n_quintiles: int = CARRY_N_QUINTILES,
) -> pd.Series:
    """
    Strategy 2: Equity carry proxy — equal-weighted return of the top
    trailing-dividend-yield quintile each day (long only, causal).

    Carry signal = trailing sum(DIVAMT / |PRC|) over `lookback` days.
    This is the per-share dividend yield, the standard equity carry proxy.

    Ranking uses percentile ranks (robust to ties and degenerate distributions).
    Stocks with no dividends in the lookback window receive NaN carry signal
    and are excluded from ranking.

    Definition: equity proxy for carry (not futures/FX carry); documented explicitly.
    Falls back to equal-weighted all-stock return if DIVAMT or PRC are unavailable.
    """
    required = {"DIVAMT", "PRC"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning(
            "Columns %s missing — using equal-weighted return as carry fallback.",
            missing,
        )
        return df.groupby("date")[ret_col].mean().rename("strat2_ret")

    df = df.copy()
    df["_price"] = df["PRC"].abs().where(df["PRC"].abs() > 0, np.nan)

    # Per-share dividend yield (NaN on non-ex-dividend days, positive on ex-div)
    df["_div_yield"] = df["DIVAMT"] / df["_price"]

    # Trailing annual dividend yield = sum over lookback window (causal, skipna)
    # min_periods=1 so a stock with even one dividend in the window gets a signal
    df["_carry_signal"] = df.groupby("PERMNO")["_div_yield"].transform(
        lambda s: s.rolling(lookback, min_periods=1).sum()
    )

    # Lag carry signal by 1 day (use only past information)
    df["_carry_signal_lag"] = df.groupby("PERMNO")["_carry_signal"].shift(1)

    # Percentile-rank each day (robust to ties and many-zeros)
    # na_option="keep" → NaN carry stays NaN and is excluded from ranking
    top_pct = 1.0 - 1.0 / n_quintiles   # 0.80 for n_quintiles=5
    df["_carry_rank"] = df.groupby("date")["_carry_signal_lag"].rank(
        pct=True, na_option="keep"
    )

    # Top quintile = stocks ranked in the top (1/n_quintiles) fraction
    strat2 = (
        df[df["_carry_rank"] >= top_pct]
        .groupby("date")[ret_col]
        .mean()
        .rename("strat2_ret")
    )

    n_valid = int(strat2.notna().sum())
    logger.info(
        "Strategy 2 (equity carry proxy, top %.0f%% div-yield): %d trading days. "
        "Signal: trailing sum(DIVAMT/|PRC|, %d days), lagged 1 day, pct-ranked.",
        (1.0 / n_quintiles) * 100, n_valid, lookback,
    )
    return strat2


# ── 1f. Combined strategy series ────────────────────────────────────────────

def build_strategy_series(
    df: pd.DataFrame,
    use_winsorized: bool = True,
    pca_window: int = PCA_ROLLING_WINDOW,
    pca_n_components: int = PCA_N_COMPONENTS,
    carry_lookback: int = CARRY_LOOKBACK_DAYS,
    n_quintiles: int = CARRY_N_QUINTILES,
) -> pd.DataFrame:
    """
    Build two causal daily strategy return series:
      strat1_ret — Avellaneda–Lee PCA residual mean reversion (conservative).
      strat2_ret — Equity carry proxy, top dividend-yield quintile (aggressive).

    Returns a date-indexed DataFrame with columns:
      ['strat1_ret', 'strat2_ret', 'vwretd', 'ewretd', 'sprtrn']
    """
    ret_col = (
        "RET_winsor"
        if (use_winsorized and "RET_winsor" in df.columns)
        else "RET"
    )

    strat1 = _build_strategy1_pca_residual(
        df, ret_col,
        window=pca_window,
        n_components=pca_n_components,
    )
    strat2 = _build_strategy2_equity_carry(
        df, ret_col,
        lookback=carry_lookback,
        n_quintiles=n_quintiles,
    )

    market_cols = [c for c in ["vwretd", "ewretd", "sprtrn"] if c in df.columns]
    market = (
        df[["date"] + market_cols]
        .drop_duplicates("date")
        .set_index("date")
    )

    combined = pd.concat([strat1, strat2, market], axis=1)
    logger.info(
        "Pre-dropna non-NaN counts — strat1: %d, strat2: %d, overlapping: %d",
        int(combined["strat1_ret"].notna().sum()),
        int(combined["strat2_ret"].notna().sum()),
        int(combined[["strat1_ret", "strat2_ret"]].notna().all(axis=1).sum()),
    )
    result = (
        combined
        .dropna(subset=["strat1_ret", "strat2_ret"])
        .sort_index()
    )

    logger.info(
        "Strategy series: %d trading days | %s to %s",
        len(result),
        result.index.min().date(),
        result.index.max().date(),
    )

    return result


# ── Full pipeline ────────────────────────────────────────────────────────────

def build_full_pipeline(
    path: str,
    min_coverage: float = 0.90,
    winsor_window: int = 63,
    z_cap: float = 3.0,
    pca_window: int = PCA_ROLLING_WINDOW,
    pca_n_components: int = PCA_N_COMPONENTS,
    carry_lookback: int = CARRY_LOOKBACK_DAYS,
    n_quintiles: int = CARRY_N_QUINTILES,
    start_date: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full version-b pipeline:
      load → integrity → missing-data policy → (optional date filter)
      → winsorize → build_strategy_series (Avellaneda–Lee PCA + equity carry)

    Args:
        start_date: If provided (e.g. '2002-01-01'), only rows with
                    date >= start_date are retained before strategy
                    construction.  Defaults to None (use all data).

    Returns:
        df_clean:        stock-level panel after cleaning
        dropped_dates:   log of dates dropped (low coverage)
        strategy_series: date-indexed strat1_ret, strat2_ret, market returns
    """
    df = load_crsp(path)
    df = integrity_checks(df)
    df, dropped_dates = apply_missing_data_policy(df, min_coverage=min_coverage)

    if start_date is not None:
        cutoff = pd.Timestamp(start_date)
        n_before = len(df)
        df = df[df["date"] >= cutoff].reset_index(drop=True)
        logger.info(
            "start_date filter (%s): %d → %d rows (removed %d).",
            start_date, n_before, len(df), n_before - len(df),
        )
    df = winsorize_returns(df, z_cap=z_cap, window=winsor_window)
    strategy_series = build_strategy_series(
        df,
        use_winsorized=True,
        pca_window=pca_window,
        pca_n_components=pca_n_components,
        carry_lookback=carry_lookback,
        n_quintiles=n_quintiles,
    )
    return df, dropped_dates, strategy_series
