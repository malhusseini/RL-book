"""Strategy 3 (aggressive): Variance-risk-premium proxy (short-vol profile).

Motivation
----------
Options market-makers collect a variance risk premium (VRP) by selling implied
volatility that tends to exceed realized volatility.  Without options data we
create a cross-sectional equity proxy:

  • "Implied vol proxy" ≈ trailing 63-day realized volatility per stock.
  • "Realized vol"      ≈ trailing 21-day realized volatility per stock.
  • VRP signal         = (rv_63 − rv_21) / rv_63   per stock, lagged by 1 day.
    Positive signal  ≡ current vol is below its recent average → VRP is positive
    → go long (collect the premium during calm periods).

Long top-quintile VRP stocks (equal-weighted).  Size is scaled by the
signal strength (clipped at ±1.5 × cross-sectional std).

Regime filter: if the *market-level* short-vol is in the top quartile of its
rolling history, halve all positions.  This provides some protection during
macro vol spikes while preserving the "carry-like" profile in normal regimes.

Risk profile: small/steady gains in calm markets; potentially large drawdowns
when volatility spikes suddenly (the defining feature of short-vol strategies).

Causality guarantees
--------------------
All rolling estimates use data up to t−1 (lagged by 1 day before PnL is realized).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RV_SHORT = 21     # short-window RV (proxy for current vol)
_RV_LONG = 63      # long-window RV (proxy for implied / expected vol)
_REGIME_WINDOW = 63        # market vol window for regime filter
_REGIME_LONG_WINDOW = 252  # long window for regime percentile
_SIGNAL_Z_CAP = 1.5        # cap cross-sectional signal z-score before ranking


def build_aggressive(
    df: pd.DataFrame,
    ret_col: str = "RET_winsor",
    market_col: str = "vwretd",
    rv_short: int = _RV_SHORT,
    rv_long: int = _RV_LONG,
    regime_window: int = _REGIME_WINDOW,
    regime_long_window: int = _REGIME_LONG_WINDOW,
    signal_z_cap: float = _SIGNAL_Z_CAP,
    top_quintile: float = 0.80,
    regime_scale_bad: float = 0.5,
) -> pd.Series:
    """Build the aggressive (VRP proxy) daily return series.

    Args:
        df:               Cleaned CRSP panel.
        ret_col:          Return column to use.
        market_col:       Market return column for regime filter (e.g. 'vwretd').
        rv_short:         Short rolling-vol window in days (proxy for RV).
        rv_long:          Long rolling-vol window in days (proxy for IV / EV).
        regime_window:    Window for market-level vol regime detection.
        regime_long_window: Lookback for regime percentile.
        signal_z_cap:     Cross-sectional z-score cap applied to the VRP signal.
        top_quintile:     Fraction of stocks to go long (e.g. 0.80 → top 20%).
        regime_scale_bad: Position scalar applied when market is in high-vol regime.

    Returns:
        pd.Series named 'strat3_ret', date-indexed, daily PnL.
    """
    if ret_col not in df.columns:
        ret_col = "RET"
        logger.warning("ret_col not found; falling back to RET.")

    df = df.copy()

    # Per-stock rolling realized vol (causal: min_periods = half-window)
    df["_rv_short"] = df.groupby("PERMNO")[ret_col].transform(
        lambda s: s.rolling(rv_short, min_periods=rv_short // 2).std()
    )
    df["_rv_long"] = df.groupby("PERMNO")[ret_col].transform(
        lambda s: s.rolling(rv_long, min_periods=rv_long // 2).std()
    )

    # VRP signal per stock per day
    denom = df["_rv_long"].where(df["_rv_long"] > 1e-12, np.nan)
    df["_vrp_signal"] = (df["_rv_long"] - df["_rv_short"]) / denom

    # Lag signal by 1 day (decision at t, realized at t+1)
    df["_vrp_signal_lag"] = df.groupby("PERMNO")["_vrp_signal"].shift(1)

    # Cross-sectional z-score cap (causal: applied per date after lagging)
    def _cs_cap(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sigma = s.std(ddof=1)
        if sigma < 1e-12 or s.notna().sum() < 5:
            return s
        z = (s - mu) / sigma
        return mu + sigma * z.clip(-signal_z_cap, signal_z_cap)

    df["_vrp_capped"] = df.groupby("date")["_vrp_signal_lag"].transform(_cs_cap)

    # Percentile rank per date (causal — uses only cross-section at t, which is lagged)
    df["_vrp_rank"] = df.groupby("date")["_vrp_capped"].rank(pct=True, na_option="keep")

    # Long top quintile (positive VRP stocks)
    long_mask = df["_vrp_rank"] >= top_quintile

    # Market regime filter: if market short-vol > long-run 75th percentile → bad regime
    if market_col in df.columns:
        market_dates = df[["date", market_col]].drop_duplicates("date").set_index("date")[market_col]
        mkt_vol = market_dates.rolling(regime_window, min_periods=regime_window // 2).std()
        mkt_vol_q75 = mkt_vol.rolling(regime_long_window, min_periods=regime_window).quantile(0.75)
        bad_regime = (mkt_vol > mkt_vol_q75).rename("bad_regime")
        bad_regime_df = bad_regime.reset_index().rename(columns={"index": "date"})
        df = df.merge(bad_regime_df, on="date", how="left")
        df["bad_regime"] = df["bad_regime"].fillna(False)
    else:
        df["bad_regime"] = False
        logger.warning("market_col '%s' not found; no regime filter applied.", market_col)

    # Position size: +1 for long top-quintile, scaled by regime
    df["_position"] = 0.0
    df.loc[long_mask, "_position"] = 1.0
    df.loc[long_mask & df["bad_regime"], "_position"] = regime_scale_bad

    # Equal-weight within selected long stocks; normalize to 1 total exposure
    n_active = df[df["_position"] > 0].groupby("date")["_position"].transform("count")
    df.loc[df["_position"] > 0, "_position_norm"] = (
        df.loc[df["_position"] > 0, "_position"] / n_active
    )
    df["_position_norm"] = df["_position_norm"].fillna(0.0)

    # PnL = normalized_position × next-day return
    # The return at date t is already lagged (positions are set at t-1 via _vrp_signal_lag)
    df["_pnl"] = df["_position_norm"] * df[ret_col]

    strat3_ret = df.groupby("date")["_pnl"].sum().rename("strat3_ret")

    # Drop dates before we have enough history
    min_window = rv_long + regime_long_window + 5
    dates_sorted = strat3_ret.index.sort_values()
    if len(dates_sorted) > min_window:
        strat3_ret = strat3_ret.loc[dates_sorted[min_window]:]

    logger.info(
        "Strategy 3 (VRP proxy, top %.0f%%): %d trading days.",
        (1 - top_quintile) * 100, int(strat3_ret.dropna().shape[0]),
    )
    return strat3_ret
