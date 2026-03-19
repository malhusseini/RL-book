"""Strategy 2 (medium): Pairs trading via rolling correlation and spread z-score.

Algorithm
---------
1. Universe selection: top `universe_size` stocks by rolling trailing market cap.
   Re-selected monthly (causal: use only past market-cap history).
2. Pair selection: at each monthly rebalance, compute pairwise rolling-63d
   correlations among the universe; select top `n_pairs` by correlation.
3. Spread computation: for each selected pair (stock A, stock B)
     S_t = log|P_A,t| − β_t × log|P_B,t|
   where β_t = rolling OLS slope of log|P_A| on log|P_B| over `hedge_window` days.
4. Spread z-score: rolling 21-day mean/std of S_t (causal).
5. Positions: +1 if z < −entry_z (buy spread), −1 if z > +entry_z (sell spread),
              close at |z| < exit_z.
6. Lag all positions by 1 day.
7. Pair PnL (return units): position × (r_A − β_t_normalized × r_B)
   where β_t_normalized adjusts for the ratio of price levels to return
   contributions. For simplicity, we use equal-dollar PnL: 0.5×r_A − 0.5×r_B
   when β ≈ 1, and more generally weight the legs proportionally to |β|.
8. Strategy return = equal-weighted mean PnL across all active pair positions.

Causality guarantees
--------------------
  - Universe selection uses only trailing market cap (SHROUT × |PRC|).
  - Pair selection and hedge ratios use only past `corr_window` / `hedge_window` days.
  - Z-score is computed on a rolling window ending at t−1 (lagged within the group).
  - Positions are lagged by 1 day before PnL is computed.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..features.factors import rolling_ols_beta

logger = logging.getLogger(__name__)

# Defaults
_UNIVERSE_SIZE = 150    # top stocks by mktcap
_N_PAIRS = 25           # number of pairs to trade
_CORR_WINDOW = 63       # correlation estimation window (days)
_HEDGE_WINDOW = 63      # OLS hedge ratio window (days)
_Z_WINDOW = 21          # spread z-score window (days)
_ENTRY_Z = 2.0          # |z| > entry → open position
_EXIT_Z = 0.5           # |z| < exit → close position
_REBAL_FREQ = 21        # rebalance pair selection every N trading days


def _select_universe(
    df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    n: int,
    lookback: int = 252,
) -> list:
    """Return a causal list of top-n PERMNOs by trailing mean market cap."""
    cutoff = as_of_date - pd.Timedelta(days=lookback * 2)
    window_df = df[(df["date"] < as_of_date) & (df["date"] >= cutoff)].copy()
    if "SHROUT" not in window_df.columns or "PRC" not in window_df.columns:
        permnos = window_df["PERMNO"].unique().tolist()
        return permnos[:n]
    window_df["mktcap"] = window_df["SHROUT"].abs() * window_df["PRC"].abs()
    mean_mktcap = window_df.groupby("PERMNO")["mktcap"].mean()
    top_n = mean_mktcap.nlargest(n).index.tolist()
    return top_n


def _select_pairs(
    wide_ret: pd.DataFrame,
    as_of_date: pd.Timestamp,
    dates: pd.DatetimeIndex,
    universe: list,
    corr_window: int,
    n_pairs: int,
    min_corr: float = 0.5,
) -> list[tuple]:
    """Causal top-n_pairs by rolling correlation in the universe.

    Returns a list of (permno_a, permno_b) tuples sorted by correlation (desc).
    """
    idx = dates.get_loc(as_of_date)
    if idx < corr_window:
        return []
    window_dates = dates[max(0, idx - corr_window): idx]
    avail = [p for p in universe if p in wide_ret.columns]
    if len(avail) < 4:
        return []
    sub = wide_ret.loc[window_dates, avail].dropna(axis=1, thresh=corr_window // 2)
    if sub.shape[1] < 4:
        return []
    corr_mat = sub.corr()
    pairs = []
    visited: set = set()
    for col in corr_mat.columns:
        row = corr_mat[col].drop(index=col).sort_values(ascending=False)
        for other, c in row.items():
            key = tuple(sorted([col, other]))
            if key not in visited and c >= min_corr:
                visited.add(key)
                pairs.append((key[0], key[1], float(c)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(a, b) for a, b, _ in pairs[:n_pairs]]


def build_medium(
    df: pd.DataFrame,
    ret_col: str = "RET_winsor",
    universe_size: int = _UNIVERSE_SIZE,
    n_pairs: int = _N_PAIRS,
    corr_window: int = _CORR_WINDOW,
    hedge_window: int = _HEDGE_WINDOW,
    z_window: int = _Z_WINDOW,
    entry_z: float = _ENTRY_Z,
    exit_z: float = _EXIT_Z,
    rebal_freq: int = _REBAL_FREQ,
) -> pd.Series:
    """Build the medium (pairs-trading) daily return series.

    Args:
        df:            Cleaned CRSP panel with columns including PERMNO, date,
                       ret_col, PRC (and optionally SHROUT for universe ranking).
        ret_col:       Return column ('RET_winsor' or 'RET').
        universe_size: Number of large-cap stocks to include in pair candidates.
        n_pairs:       Number of pairs to trade simultaneously.
        corr_window:   Rolling correlation window for pair selection (days).
        hedge_window:  Rolling OLS window for hedge ratio estimation (days).
        z_window:      Rolling z-score window for spread (days).
        entry_z:       |z| threshold to open a position.
        exit_z:        |z| threshold to close a position.
        rebal_freq:    Rebalance pair list every N trading days.

    Returns:
        pd.Series named 'strat2_ret', date-indexed, daily PnL.
    """
    if ret_col not in df.columns:
        ret_col = "RET"
        logger.warning("ret_col not found; falling back to RET.")

    # Need price data for spread computation
    if "PRC" not in df.columns:
        logger.warning("No PRC column; cannot build pairs strategy — returning zeros.")
        dates = df["date"].drop_duplicates().sort_values()
        return pd.Series(0.0, index=dates, name="strat2_ret")

    dates_series = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    dates_idx = pd.DatetimeIndex(dates_series)

    wide_ret = df.pivot(index="date", columns="PERMNO", values=ret_col)
    wide_prc = df.pivot(index="date", columns="PERMNO", values="PRC")
    wide_prc = wide_prc.abs()  # CRSP negative PRC = valid midpoint; abs for log

    # Replace zero prices with NaN (already handled in integrity_checks, but guard)
    wide_prc = wide_prc.replace(0.0, np.nan)
    log_prc = np.log(wide_prc.where(wide_prc > 0, np.nan))

    # Per-pair position state machine (keyed by (permno_a, permno_b))
    pair_positions: dict[tuple, float] = {}   # +1 long spread, -1 short spread, 0 flat
    active_pairs: list[tuple] = []
    last_rebal_i = -rebal_freq  # force rebal on first eligible date

    min_start = max(corr_window, hedge_window, z_window) + 5
    pnl_by_date: dict[pd.Timestamp, list[float]] = {}

    for i in range(len(dates_series)):
        t = dates_series.iloc[i]

        # Monthly pair rebalancing
        if i - last_rebal_i >= rebal_freq and i >= min_start:
            universe = _select_universe(df, t, universe_size)
            new_pairs = _select_pairs(
                wide_ret, t, dates_idx, universe, corr_window, n_pairs
            )
            # Close positions for pairs that drop out
            dropped = set(pair_positions.keys()) - set(new_pairs)
            for p in dropped:
                pair_positions.pop(p, None)
            # Initialize positions for new pairs
            for p in new_pairs:
                if p not in pair_positions:
                    pair_positions[p] = 0.0
            active_pairs = new_pairs
            last_rebal_i = i

        if i < min_start or not active_pairs:
            continue

        pnl_today: list[float] = []

        for (pa, pb) in active_pairs:
            if pa not in log_prc.columns or pb not in log_prc.columns:
                continue

            # Causal hedge ratio: OLS of log(P_A) on log(P_B) over past hedge_window days
            past_slice = slice(max(0, i - hedge_window), i)
            past_dates = dates_idx[past_slice]
            lp_a = log_prc.loc[past_dates, pa].values.astype(float)
            lp_b = log_prc.loc[past_dates, pb].values.astype(float)
            valid_mask = np.isfinite(lp_a) & np.isfinite(lp_b)
            if valid_mask.sum() < hedge_window // 2:
                continue
            lp_a_v, lp_b_v = lp_a[valid_mask], lp_b[valid_mask]
            xdm = lp_b_v - lp_b_v.mean()
            var_x = float(np.dot(xdm, xdm))
            if var_x < 1e-14:
                beta = 1.0
            else:
                beta = float(np.dot(lp_a_v - lp_a_v.mean(), xdm) / var_x)
            beta = float(np.clip(beta, 0.1, 10.0))  # clamp unreasonable hedge ratios

            # Spread z-score: use past z_window spreads
            spread_window_slice = slice(max(0, i - z_window - 1), i)
            sw_dates = dates_idx[spread_window_slice]
            lp_a_sw = log_prc.loc[sw_dates, pa]
            lp_b_sw = log_prc.loc[sw_dates, pb]
            spread_sw = lp_a_sw - beta * lp_b_sw
            valid_sw = spread_sw.dropna()
            if len(valid_sw) < 5:
                continue
            s_mean = float(valid_sw.mean())
            s_std = float(valid_sw.std(ddof=1))
            if s_std < 1e-12:
                continue

            # Current spread at t
            if pa not in log_prc.columns or pb not in log_prc.columns:
                continue
            lp_a_t = log_prc.loc[t, pa] if t in log_prc.index else np.nan
            lp_b_t = log_prc.loc[t, pb] if t in log_prc.index else np.nan
            if not (np.isfinite(lp_a_t) and np.isfinite(lp_b_t)):
                continue
            spread_t = lp_a_t - beta * lp_b_t
            z_t = (spread_t - s_mean) / s_std

            # Position update (bang-bang with exit zone)
            key = (pa, pb)
            pos = pair_positions.get(key, 0.0)
            if abs(z_t) < exit_z:
                pos = 0.0
            elif z_t < -entry_z:
                pos = 1.0
            elif z_t > entry_z:
                pos = -1.0
            pair_positions[key] = pos

            # PnL at t+1 uses the position decided at t (already from previous iteration)
            # We lag by computing PnL for the *previous* position stored before updating
            # Retrieve return at t for computing PnL (lagged position from t-1 is already
            # the position we set at t-1 and stored; here we apply it to r_t)
            r_a = wide_ret.loc[t, pa] if (t in wide_ret.index and pa in wide_ret.columns) else np.nan
            r_b = wide_ret.loc[t, pb] if (t in wide_ret.index and pb in wide_ret.columns) else np.nan
            if not (np.isfinite(r_a) and np.isfinite(r_b)):
                continue

            # Lagged position: the position we determined at t-1 (before current update)
            # The pair PnL: long spread → long A, short B (normalized to dollar-neutral)
            #   PnL = pos_lag × (r_A − r_B) × 0.5   (simplified equal-weight)
            # Note: pos above is set at t (for use at t+1). To get last step's pos:
            # We use the OLD pos (before update above) by reading pair_positions at i-1.
            # Implementation: store lagged pos separately below.
            pass

        # To properly lag, we compute PnL using positions from the previous step.
        # Restructure: store positions from t-1 when looping at t.
        pnl_by_date[t] = pnl_today  # placeholder, overwritten below

    # Cleaner implementation: explicit lag-position loop
    # Reset and redo with explicit lag storage
    pair_positions_lag: dict[tuple, float] = {}
    pair_positions_curr: dict[tuple, float] = {}
    active_pairs = []
    last_rebal_i = -rebal_freq
    pnl_by_date2: dict[pd.Timestamp, float] = {}

    for i in range(len(dates_series)):
        t = dates_series.iloc[i]

        # Monthly pair rebalancing (causal)
        if i - last_rebal_i >= rebal_freq and i >= min_start:
            universe = _select_universe(df, t, universe_size)
            new_pairs = _select_pairs(
                wide_ret, t, dates_idx, universe, corr_window, n_pairs
            )
            for p in set(pair_positions_curr.keys()) - set(new_pairs):
                pair_positions_curr.pop(p, None)
                pair_positions_lag.pop(p, None)
            for p in new_pairs:
                if p not in pair_positions_curr:
                    pair_positions_curr[p] = 0.0
                    pair_positions_lag[p] = 0.0
            active_pairs = new_pairs
            last_rebal_i = i

        if i < min_start or not active_pairs:
            # Shift curr → lag for next iteration
            for p in active_pairs:
                pair_positions_lag[p] = pair_positions_curr.get(p, 0.0)
            continue

        day_pnl: list[float] = []

        for (pa, pb) in active_pairs:
            if pa not in log_prc.columns or pb not in log_prc.columns:
                continue

            # Hedge ratio (causal, using data up to t-1)
            past_slice = slice(max(0, i - hedge_window), i)
            past_dates_arr = dates_idx[past_slice]
            lp_a = log_prc.loc[past_dates_arr, pa].values.astype(float)
            lp_b = log_prc.loc[past_dates_arr, pb].values.astype(float)
            valid_mask = np.isfinite(lp_a) & np.isfinite(lp_b)
            if valid_mask.sum() < hedge_window // 2:
                continue
            lp_a_v, lp_b_v = lp_a[valid_mask], lp_b[valid_mask]
            xdm = lp_b_v - lp_b_v.mean()
            var_x = float(np.dot(xdm, xdm))
            beta = 1.0 if var_x < 1e-14 else float(
                np.dot(lp_a_v - lp_a_v.mean(), xdm) / var_x
            )
            beta = float(np.clip(beta, 0.1, 10.0))

            # Spread z-score (causal, using data up to t-1)
            sw_slice = slice(max(0, i - z_window - 1), i)
            sw_dates_arr = dates_idx[sw_slice]
            spread_sw = (log_prc.loc[sw_dates_arr, pa] - beta * log_prc.loc[sw_dates_arr, pb]).dropna()
            if len(spread_sw) < 5:
                continue
            s_mean = float(spread_sw.mean())
            s_std = float(spread_sw.std(ddof=1))
            if s_std < 1e-12:
                continue

            if t not in log_prc.index:
                continue
            lp_a_t = log_prc.at[t, pa] if pa in log_prc.columns else np.nan
            lp_b_t = log_prc.at[t, pb] if pb in log_prc.columns else np.nan
            if not (np.isfinite(lp_a_t) and np.isfinite(lp_b_t)):
                continue
            z_t = (lp_a_t - beta * lp_b_t - s_mean) / s_std

            # Update position for *next* period (current signal, not yet realized)
            pos_new = pair_positions_curr.get((pa, pb), 0.0)
            if abs(z_t) < exit_z:
                pos_new = 0.0
            elif z_t < -entry_z:
                pos_new = 1.0
            elif z_t > entry_z:
                pos_new = -1.0
            pair_positions_curr[(pa, pb)] = pos_new

            # PnL uses LAGGED position (decided at t−1, realized at t)
            pos_lag = pair_positions_lag.get((pa, pb), 0.0)
            if pos_lag == 0.0:
                continue

            r_a = wide_ret.at[t, pa] if (t in wide_ret.index and pa in wide_ret.columns) else np.nan
            r_b = wide_ret.at[t, pb] if (t in wide_ret.index and pb in wide_ret.columns) else np.nan
            if not (np.isfinite(r_a) and np.isfinite(r_b)):
                continue

            # Equal-dollar-neutral pair PnL: long spread = long A, short beta-units of B
            # Normalized so sum of gross weights = 1 for each leg
            w_a = 1.0 / (1.0 + abs(beta))
            w_b = abs(beta) / (1.0 + abs(beta))
            pair_pnl = pos_lag * (w_a * r_a - w_b * r_b)
            day_pnl.append(pair_pnl)

        # Shift curr → lag for next iteration
        for p in active_pairs:
            pair_positions_lag[p] = pair_positions_curr.get(p, 0.0)

        if day_pnl:
            pnl_by_date2[t] = float(np.mean(day_pnl))

    if not pnl_by_date2:
        logger.warning("Strategy 2: no PnL produced; check data coverage.")
        return pd.Series(dtype=float, name="strat2_ret")

    strat2_ret = pd.Series(pnl_by_date2, name="strat2_ret").sort_index()
    logger.info(
        "Strategy 2 (pairs trading, %d pairs): %d trading days.",
        n_pairs, len(strat2_ret),
    )
    return strat2_ret
