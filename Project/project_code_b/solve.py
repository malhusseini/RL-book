"""Solve the Phase 2 MDP with Value Iteration or Policy Iteration,
and visualize the optimal policy and value function.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl.dynamic_programming import policy_iteration_result, value_iteration_result
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy

from .calibration import assign_regime
from .mdp import ACTION_NAMES, ACTIONS, MDPConfig, _nearest_bin

logger = logging.getLogger(__name__)

_ACTION_COLORS = ["#4e9af1", "#f5a623", "#e74c3c"]
_ACTION_CMAP   = mcolors.ListedColormap(_ACTION_COLORS)


# ── Solver ───────────────────────────────────────────────────────────────────

def solve(
    mdp: FiniteMarkovDecisionProcess,
    gamma: float = 0.99,
    method: str = "value_iteration",
) -> Tuple[dict, FiniteDeterministicPolicy]:
    """
    Solve the MDP for the optimal value function and policy.

    Args:
        mdp:    FiniteMarkovDecisionProcess from mdp.build_mdp().
        gamma:  Discount factor.
        method: 'value_iteration' or 'policy_iteration'.
    """
    logger.info("Solving MDP via %s  (γ = %.4f) …", method, gamma)
    if method == "value_iteration":
        opt_vf, opt_policy = value_iteration_result(mdp, gamma)
    elif method == "policy_iteration":
        opt_vf, opt_policy = policy_iteration_result(mdp, gamma)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'value_iteration' or 'policy_iteration'."
        )
    logger.info("Solver finished.")
    return opt_vf, opt_policy


# ── Extraction helpers ───────────────────────────────────────────────────────

def _extract_no_regime(
    opt_vf: dict,
    opt_policy: FiniteDeterministicPolicy,
    wealth_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(wealth_grid)
    actions = np.full(n, fill_value=-1, dtype=int)
    values  = np.full(n, fill_value=np.nan)
    for s_nt, v in opt_vf.items():
        idx = s_nt.state
        if isinstance(idx, int) and 0 <= idx < n:
            values[idx]  = v
            actions[idx] = opt_policy.action_for.get(idx, -1)
    return actions, values


def _extract_with_regime(
    opt_vf: dict,
    opt_policy: FiniteDeterministicPolicy,
    wealth_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(wealth_grid)
    policy_grid = np.full((2, n), fill_value=-1, dtype=int)
    value_grid  = np.full((2, n), fill_value=np.nan)
    for s_nt, v in opt_vf.items():
        state = s_nt.state
        if isinstance(state, tuple) and len(state) == 2:
            w_idx, regime = state
            if 0 <= w_idx < n and regime in (0, 1):
                value_grid[regime, w_idx]  = v
                policy_grid[regime, w_idx] = opt_policy.action_for.get(state, -1)
    return policy_grid, value_grid


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_no_regime(
    opt_vf: dict,
    opt_policy: FiniteDeterministicPolicy,
    wealth_grid: np.ndarray,
    cfg: Optional[MDPConfig] = None,
    title_suffix: str = "",
) -> None:
    """Two-panel: optimal policy (scatter) and value function."""
    actions, values = _extract_no_regime(opt_vf, opt_policy, wealth_grid)
    W0 = cfg.initial_wealth if cfg is not None else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(
        f"Phase 2b Optimal Policy & Value Function{title_suffix}", fontsize=13
    )

    ax = axes[0]
    valid = actions >= 0
    ax.scatter(
        wealth_grid[valid], actions[valid],
        c=actions[valid], cmap=_ACTION_CMAP, vmin=0, vmax=2, s=55, zorder=3,
    )
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([ACTION_NAMES[i] for i in range(3)])
    ax.set_xlabel("Wealth  W", fontsize=11)
    ax.set_ylabel("Optimal allocation", fontsize=11)
    ax.set_title("Optimal Policy  π*(W)")
    ax.axvline(W0, color="grey", ls="--", lw=1.2, alpha=0.7, label=f"W₀ = {W0}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    valid_v = ~np.isnan(values)
    ax2.plot(wealth_grid[valid_v], values[valid_v], "k-", lw=1.8)
    ax2.set_xlabel("Wealth  W", fontsize=11)
    ax2.set_ylabel("V*(W)", fontsize=11)
    ax2.set_title("Optimal Value Function  V*(W)")
    ax2.axvline(W0, color="grey", ls="--", lw=1.2, alpha=0.7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_with_regime(
    opt_vf: dict,
    opt_policy: FiniteDeterministicPolicy,
    wealth_grid: np.ndarray,
    cfg: Optional[MDPConfig] = None,
    title_suffix: str = "",
) -> None:
    """Two-panel: policy heatmap (wealth × regime) and value functions."""
    policy_grid, value_grid = _extract_with_regime(opt_vf, opt_policy, wealth_grid)
    W0 = cfg.initial_wealth if cfg is not None else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(
        f"Phase 2b Optimal Policy & Value Function (with Regime){title_suffix}",
        fontsize=13,
    )

    ax = axes[0]
    masked = np.ma.masked_where(policy_grid < 0, policy_grid)
    im = ax.imshow(
        masked, cmap=_ACTION_CMAP, vmin=0, vmax=2,
        aspect="auto",
        extent=[wealth_grid[0], wealth_grid[-1], -0.5, 1.5],
        origin="lower",
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Bad regime", "Good regime"], fontsize=10)
    ax.set_xlabel("Wealth  W", fontsize=11)
    ax.set_title("Optimal Policy  π*(W, regime)")
    cbar = plt.colorbar(im, ax=ax, ticks=[0.33, 1.0, 1.67])
    cbar.ax.set_yticklabels([ACTION_NAMES[i] for i in range(3)], fontsize=9)
    ax.axvline(W0, color="white", ls="--", lw=1.2, alpha=0.8)

    ax2 = axes[1]
    for regime, (label, color) in {
        1: ("Good regime", "#4e9af1"),
        0: ("Bad regime",  "#e74c3c"),
    }.items():
        row = value_grid[regime]
        valid = ~np.isnan(row)
        ax2.plot(wealth_grid[valid], row[valid], color=color, lw=1.8, label=label)
    ax2.set_xlabel("Wealth  W", fontsize=11)
    ax2.set_ylabel("V*(W, regime)", fontsize=11)
    ax2.set_title("Optimal Value Function by Regime")
    ax2.axvline(W0, color="grey", ls="--", lw=1.2, alpha=0.7)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Policy summary table ─────────────────────────────────────────────────────

def policy_summary(
    opt_policy: FiniteDeterministicPolicy,
    wealth_grid: np.ndarray,
    use_regime: bool = False,
) -> None:
    """Print a readable table of the optimal allocation per state."""
    n = len(wealth_grid)
    if not use_regime:
        print(f"{'Wealth':>10}  {'Action':>14}  Alloc (S1/S2/Cash)")
        print("-" * 52)
        for w_idx in range(n):
            a = opt_policy.action_for.get(w_idx, -1)
            if a < 0:
                continue
            f1, f2 = (0.8, 0.2) if a == 0 else (0.5, 0.5) if a == 1 else (0.2, 0.8)
            print(
                f"  {wealth_grid[w_idx]:>8.4f}  {ACTION_NAMES[a]:>14}"
                f"  {f1*100:.0f}% / {f2*100:.0f}% / 0%"
            )
    else:
        print(f"{'Wealth':>10}  {'Regime':>8}  {'Action':>14}  Alloc (S1/S2/Cash)")
        print("-" * 62)
        for w_idx in range(n):
            for regime in (1, 0):
                state = (w_idx, regime)
                a = opt_policy.action_for.get(state, -1)
                if a < 0:
                    continue
                f1, f2 = (0.8, 0.2) if a == 0 else (0.5, 0.5) if a == 1 else (0.2, 0.8)
                label = "good" if regime == 1 else "bad "
                print(
                    f"  {wealth_grid[w_idx]:>8.4f}  {label:>8}  "
                    f"{ACTION_NAMES[a]:>14}  {f1*100:.0f}% / {f2*100:.0f}% / 0%"
                )


# ── Out-of-sample backtest ───────────────────────────────────────────────────

def _print_backtest_summary(df: pd.DataFrame, initial_wealth: float = 1.0) -> None:
    """Print annualised return, Sharpe, and max drawdown for each wealth path."""
    daily_rets = df.pct_change().dropna()
    n_days = len(daily_rets)

    header = f"{'Strategy':<18}  {'Ann. Return':>11}  {'Sharpe':>8}  {'Max DD':>8}  {'Final W':>10}"
    print(f"\n{header}")
    print("-" * len(header))
    for col in df.columns:
        r = daily_rets[col].values
        ann_ret = float((df[col].iloc[-1] / initial_wealth) ** (252.0 / n_days) - 1)
        sharpe = float((r.mean() / r.std(ddof=1)) * np.sqrt(252)) if r.std(ddof=1) > 0 else float("nan")
        cum = df[col].values
        roll_max = np.maximum.accumulate(cum)
        max_dd = float(((cum - roll_max) / roll_max).min())
        print(
            f"  {col:<16}  {ann_ret:+11.2%}  {sharpe:+8.3f}  {max_dd:8.2%}  "
            f"{df[col].iloc[-1]:10.4f}"
        )
    print()


def _plot_backtest(df: pd.DataFrame) -> None:
    """Two-panel: cumulative wealth and drawdown for test-period paths."""
    _colors = {
        "MDP Policy":   "black",
        "Conservative": "#4e9af1",
        "Balanced":     "#f5a623",
        "Aggressive":   "#e74c3c",
    }
    _lws = {"MDP Policy": 2.2}

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Out-of-Sample Backtest: Test Period (2019–2023)", fontsize=13)

    ax = axes[0]
    for col in df.columns:
        ax.plot(
            df.index, df[col],
            label=col,
            color=_colors.get(col, "grey"),
            lw=_lws.get(col, 1.3),
            alpha=0.92 if col == "MDP Policy" else 0.75,
        )
    ax.set_title("Cumulative wealth  (W₀ = 1)")
    ax.set_ylabel("Wealth")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for col in df.columns:
        cum = df[col].values
        roll_max = np.maximum.accumulate(cum)
        dd = 100.0 * (cum - roll_max) / roll_max
        ax2.plot(
            df.index, dd,
            label=col,
            color=_colors.get(col, "grey"),
            lw=_lws.get(col, 1.3),
            alpha=0.92 if col == "MDP Policy" else 0.75,
        )
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def backtest_policy(
    opt_policy: FiniteDeterministicPolicy,
    test_series: pd.DataFrame,
    wealth_grid: np.ndarray,
    cfg: Optional[MDPConfig] = None,
    use_regime: bool = False,
    initial_wealth: float = 1.0,
) -> pd.DataFrame:
    """
    Apply the solved MDP policy to held-out test returns.

    The policy was optimised on training data; here it is evaluated on
    the separate test period to measure out-of-sample performance.

    At each test day:
      1. The current wealth is snapped to the nearest wealth-grid bin.
      2. The regime label (if use_regime=True) is read from a causal
         rolling-vol estimate on the test-period market returns.
      3. The action prescribed by opt_policy for that state is looked up.
      4. The realised portfolio return under that allocation is applied.

    The three fixed-allocation benchmarks (Conservative / Balanced / Aggressive)
    are simulated over the same period for comparison.

    Args:
        opt_policy:      Solved FiniteDeterministicPolicy from solve().
        test_series:     Date-indexed DataFrame with columns
                         ['strat1_ret', 'strat2_ret'] (and optionally 'vwretd').
        wealth_grid:     Wealth grid from build_mdp(), same as used at solve time.
        cfg:             MDPConfig; used for initial_wealth if not explicitly given.
        use_regime:      Whether to pass (wealth_idx, regime) tuples to the policy.
        initial_wealth:  Starting wealth for all simulated paths.

    Returns:
        DataFrame of daily cumulative-wealth paths indexed by date
        (columns: 'MDP Policy', 'Conservative', 'Balanced', 'Aggressive').
    """
    if cfg is not None:
        initial_wealth = cfg.initial_wealth

    s1 = test_series["strat1_ret"].values
    s2 = test_series["strat2_ret"].values
    dates = test_series.index
    n = len(dates)

    # Causal regime labels for the test period
    if use_regime and "vwretd" in test_series.columns:
        regime_labels = (
            assign_regime(test_series["vwretd"].dropna())
            .reindex(test_series.index)
            .ffill()
            .fillna(False)
            .astype(int)
            .values
        )
    else:
        regime_labels = None
        use_regime = False

    # Simulate MDP-policy wealth path
    W = float(initial_wealth)
    policy_wealth = np.empty(n + 1)
    policy_wealth[0] = W
    for i in range(n):
        w_idx = int(_nearest_bin(np.array([W]), wealth_grid)[0])
        state = (w_idx, int(regime_labels[i])) if use_regime else w_idx
        a = opt_policy.action_for.get(state, 1)   # fallback: Balanced
        f1, f2, _ = ACTIONS[a]
        port_ret = f1 * s1[i] + f2 * s2[i]
        W = max(W * (1.0 + port_ret), 1e-8)
        policy_wealth[i + 1] = W

    # Fixed-allocation benchmarks (fully vectorised)
    results: dict = {"MDP Policy": policy_wealth}
    for a_idx, name in ACTION_NAMES.items():
        f1, f2, _ = ACTIONS[a_idx]
        port_rets = f1 * s1 + f2 * s2
        W_bench = initial_wealth * np.concatenate(
            [[1.0], np.cumprod(1.0 + port_rets)]
        )
        results[name] = W_bench

    # Prepend a synthetic start-of-period date one business day before test[0]
    start_date = dates[0] - pd.tseries.offsets.BDay(1)
    all_dates = pd.DatetimeIndex([start_date] + list(dates))
    df = pd.DataFrame(results, index=all_dates)

    _print_backtest_summary(df, initial_wealth)
    _plot_backtest(df)

    return df


# ── Convenience wrapper ──────────────────────────────────────────────────────

def solve_and_plot(
    mdp: FiniteMarkovDecisionProcess,
    wealth_grid: np.ndarray,
    cfg: Optional[MDPConfig] = None,
    gamma: float = 0.99,
    method: str = "value_iteration",
    use_regime: Optional[bool] = None,
) -> Tuple[dict, FiniteDeterministicPolicy]:
    """Solve the MDP and immediately produce visualizations."""
    opt_vf, opt_policy = solve(mdp, gamma=gamma, method=method)

    first_state = mdp.non_terminal_states[0].state
    _use_regime = isinstance(first_state, tuple) if use_regime is None else use_regime

    suffix = f"  (γ={gamma}, {method})"
    if _use_regime:
        plot_with_regime(opt_vf, opt_policy, wealth_grid, cfg=cfg, title_suffix=suffix)
    else:
        plot_no_regime(opt_vf, opt_policy, wealth_grid, cfg=cfg, title_suffix=suffix)

    policy_summary(opt_policy, wealth_grid, use_regime=_use_regime)
    return opt_vf, opt_policy
