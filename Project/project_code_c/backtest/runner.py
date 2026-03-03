"""Walk-forward runner — enforces the lagged-execution convention.

Convention (encoded in this module, cannot be bypassed)
--------------------------------------------------------
  Decision at time t  →  positions chosen using data up to t.
  PnL at t+1          =  position_t × r_{t+1}.

The runner never accesses r_{t+1} when computing position_t: it passes only
the strategy-series rows up to and including t to the policy function.

Quality guardrails
------------------
  • Signals passed to the policy at time t are verified to contain no index
    values strictly greater than t (would indicate look-ahead).
  • Each step's PnL is computed with the return at t+1, not t.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for the walk-forward backtest runner.

    Attributes:
        initial_wealth:  Starting wealth (normalised to 1.0 by default).
        transaction_cost: Per-unit turnover cost applied at each reallocation.
        gamma_risk_free:  Annualised risk-free rate (daily = / 252).
        enforce_lag_check: If True, assert that signals contain no future data.
    """
    initial_wealth: float = 1.0
    transaction_cost: float = 0.001
    gamma_risk_free: float = 0.025  # 2.5% annual
    enforce_lag_check: bool = True


def _lag_check(signal_index: pd.DatetimeIndex, current_date: pd.Timestamp) -> None:
    """Raise if any signal index entry is strictly after current_date."""
    if (signal_index > current_date).any():
        future_dates = signal_index[signal_index > current_date].tolist()
        raise ValueError(
            f"LEAKAGE DETECTED: policy received signals with future dates: "
            f"{future_dates[:3]}  (current_date={current_date.date()})"
        )


def run_walk_forward(
    strategy_series: pd.DataFrame,
    policy_fn: Callable[[pd.DataFrame, pd.Timestamp], Tuple[float, float, float, float]],
    cfg: Optional[WalkForwardConfig] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run a causal walk-forward simulation using the provided policy function.

    The policy function is called at each date t with:
      (strategy_series up to and including t, current_date = t)
    and must return a weight tuple (w1, w2, w3, w_cash) summing to 1.

    PnL is realised at t+1 using the position chosen at t.

    Args:
        strategy_series: Date-indexed DataFrame with columns
                         ['strat1_ret', 'strat2_ret', 'strat3_ret'].
        policy_fn:       Callable(history_df, current_date) → (w1, w2, w3, w_cash).
        cfg:             WalkForwardConfig (uses defaults if None).
        verbose:         If True, log each step's allocation and PnL.

    Returns:
        DataFrame with columns ['wealth', 'w1', 'w2', 'w3', 'w_cash',
        'port_ret', 'daily_tc'], indexed by date (one row per t+1 realization).
    """
    if cfg is None:
        cfg = WalkForwardConfig()

    rf_daily = cfg.gamma_risk_free / 252.0

    cols = ["strat1_ret", "strat2_ret", "strat3_ret"]
    for c in cols:
        if c not in strategy_series.columns:
            raise ValueError(f"Missing required column '{c}' in strategy_series.")

    dates = strategy_series.index.sort_values()
    n = len(dates)
    if n < 2:
        raise ValueError("Need at least 2 dates in strategy_series.")

    wealth = cfg.initial_wealth
    prev_weights = (1 / 3, 1 / 3, 1 / 3, 0.0)

    records = []

    for i in range(n - 1):
        t = dates[i]
        t_next = dates[i + 1]

        # Pass only history up to t (inclusive) to the policy
        history = strategy_series.loc[:t]

        # Optional lag check
        if cfg.enforce_lag_check:
            _lag_check(history.index, t)

        # Policy decision at t
        weights = policy_fn(history, t)
        w1, w2, w3, w_cash = weights

        # Validate weights
        total = w1 + w2 + w3 + w_cash
        if abs(total - 1.0) > 1e-6:
            logger.warning("Weights sum to %.6f at %s — normalising.", total, t.date())
            w1, w2, w3, w_cash = w1 / total, w2 / total, w3 / total, w_cash / total

        # Transaction cost
        tc = cfg.transaction_cost * (
            abs(w1 - prev_weights[0])
            + abs(w2 - prev_weights[1])
            + abs(w3 - prev_weights[2])
            + abs(w_cash - prev_weights[3])
        )

        # Realise PnL at t+1 (lagged execution)
        r_next = strategy_series.loc[t_next, cols].values.astype(float)
        port_ret = w1 * r_next[0] + w2 * r_next[1] + w3 * r_next[2] + w_cash * rf_daily

        wealth_new = max(wealth * (1.0 + port_ret) - tc * wealth, 1e-8)

        records.append({
            "date": t_next,
            "wealth": wealth_new,
            "w1": w1, "w2": w2, "w3": w3, "w_cash": w_cash,
            "port_ret": port_ret,
            "daily_tc": tc * wealth,
        })

        if verbose:
            logger.debug(
                "%s: w=(%.2f,%.2f,%.2f,%.2f)  r=%.4f  W=%.4f",
                t.date(), w1, w2, w3, w_cash, port_ret, wealth_new,
            )

        prev_weights = (w1, w2, w3, w_cash)
        wealth = wealth_new

    result = pd.DataFrame(records).set_index("date")
    # Prepend starting row for completeness
    start_row = pd.DataFrame(
        [{
            "wealth": cfg.initial_wealth,
            "w1": np.nan, "w2": np.nan, "w3": np.nan, "w_cash": np.nan,
            "port_ret": 0.0, "daily_tc": 0.0,
        }],
        index=[dates[0] - pd.tseries.offsets.BDay(1)],
    )
    result = pd.concat([start_row, result])
    return result


def backtest_fixed_allocation(
    strategy_series: pd.DataFrame,
    weights: Tuple[float, float, float, float],
    cfg: Optional[WalkForwardConfig] = None,
    label: str = "fixed",
) -> pd.Series:
    """Convenience function: simulate a fixed-weight buy-and-hold strategy.

    No transaction costs are applied after the initial allocation.

    Returns:
        pd.Series of cumulative wealth, date-indexed, named `label`.
    """
    def _fixed_policy(history: pd.DataFrame, _: pd.Timestamp) -> Tuple:
        return weights

    result = run_walk_forward(strategy_series, _fixed_policy, cfg=cfg)
    return result["wealth"].rename(label)
