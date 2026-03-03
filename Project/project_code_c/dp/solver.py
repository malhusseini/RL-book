"""Solve the Phase-2 MDP and visualize the optimal policy.

Wraps the rl.dynamic_programming value_iteration_result / policy_iteration_result
functions and provides plotting and summary utilities for the 3-strategy
10-action MDP with optional 2-state regime.
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

from ..env.mdp import ACTION_NAMES, ACTIONS, MDPConfig, _nearest_bin
from ..features.regime import assign_regime

logger = logging.getLogger(__name__)

# 10 perceptually-distinct colors (blue → orange → red gradient + greens for cash)
_ACTION_COLORS = [
    "#1a6faf",  # 0  Conservative      (deep blue)
    "#4e9af1",  # 1  Conservative+     (medium blue)
    "#74b9ff",  # 2  Moderate-Low      (light blue)
    "#a8d8ea",  # 3  Moderate          (sky blue)
    "#f5a623",  # 4  Balanced          (amber)
    "#e07b39",  # 5  Moderate-High     (warm orange)
    "#e74c3c",  # 6  Aggressive        (red)
    "#6ab187",  # 7  Defensive-20      (medium green)
    "#3d9970",  # 8  Defensive-40      (teal green)
    "#27ae60",  # 9  Cash-heavy        (dark green)
]
_ACTION_CMAP = mcolors.ListedColormap(_ACTION_COLORS)
_N_ACTIONS = len(ACTIONS)


# ── Solver ────────────────────────────────────────────────────────────────────

def solve(
    mdp: FiniteMarkovDecisionProcess,
    gamma: float = 0.99,
    method: str = "value_iteration",
) -> Tuple[dict, FiniteDeterministicPolicy]:
    """Solve the MDP for the optimal value function and policy.

    Args:
        mdp:    FiniteMarkovDecisionProcess from env.mdp.build_mdp().
        gamma:  Discount factor.
        method: 'value_iteration' or 'policy_iteration'.

    Returns:
        (opt_vf, opt_policy) tuple.
    """
    logger.info("Solving MDP via %s  (γ = %.4f) …", method, gamma)
    if method == "value_iteration":
        opt_vf, opt_policy = value_iteration_result(mdp, gamma)
    elif method == "policy_iteration":
        opt_vf, opt_policy = policy_iteration_result(mdp, gamma)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'value_iteration' or 'policy_iteration'.")
    logger.info("Solver finished.")
    return opt_vf, opt_policy


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_no_regime(
    opt_vf: dict,
    opt_policy: FiniteDeterministicPolicy,
    wealth_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(wealth_grid)
    actions = np.full(n, fill_value=-1, dtype=int)
    values = np.full(n, fill_value=np.nan)
    for s_nt, v in opt_vf.items():
        idx = s_nt.state
        if isinstance(idx, int) and 0 <= idx < n:
            values[idx] = v
            actions[idx] = opt_policy.action_for.get(idx, -1)
    return actions, values


def _extract_with_regime(
    opt_vf: dict,
    opt_policy: FiniteDeterministicPolicy,
    wealth_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(wealth_grid)
    policy_grid = np.full((2, n), fill_value=-1, dtype=int)
    value_grid = np.full((2, n), fill_value=np.nan)
    for s_nt, v in opt_vf.items():
        state = s_nt.state
        if isinstance(state, tuple) and len(state) == 2:
            w_idx, regime = state
            if 0 <= w_idx < n and regime in (0, 1):
                value_grid[regime, w_idx] = v
                policy_grid[regime, w_idx] = opt_policy.action_for.get(state, -1)
    return policy_grid, value_grid


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_no_regime(
    opt_vf: dict,
    opt_policy: FiniteDeterministicPolicy,
    wealth_grid: np.ndarray,
    cfg: Optional[MDPConfig] = None,
    title_suffix: str = "",
) -> None:
    """Two-panel figure: optimal policy (scatter) + value function."""
    actions, values = _extract_no_regime(opt_vf, opt_policy, wealth_grid)
    W0 = cfg.initial_wealth if cfg is not None else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Phase-2c Optimal Policy & Value Function{title_suffix}", fontsize=13)

    ax = axes[0]
    valid = actions >= 0
    sc = ax.scatter(
        wealth_grid[valid], actions[valid],
        c=actions[valid], cmap=_ACTION_CMAP, vmin=0, vmax=_N_ACTIONS - 1, s=55, zorder=3,
    )
    ax.set_yticks(range(_N_ACTIONS))
    ax.set_yticklabels([ACTION_NAMES[i] for i in range(_N_ACTIONS)])
    ax.set_xlabel("Wealth  W")
    ax.set_ylabel("Optimal allocation")
    ax.set_title("Optimal Policy  π*(W)")
    ax.axvline(W0, color="grey", ls="--", lw=1.2, alpha=0.7, label=f"W₀ = {W0}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    valid_v = ~np.isnan(values)
    ax2.plot(wealth_grid[valid_v], values[valid_v], "k-", lw=1.8)
    ax2.set_xlabel("Wealth  W")
    ax2.set_ylabel("V*(W)")
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
    """Two-panel figure: policy heatmap (wealth × regime) + value functions per regime."""
    policy_grid, value_grid = _extract_with_regime(opt_vf, opt_policy, wealth_grid)
    W0 = cfg.initial_wealth if cfg is not None else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    fig.suptitle(
        f"Phase-2c Optimal Policy & Value Function (with Regime){title_suffix}", fontsize=13
    )

    ax = axes[0]
    masked = np.ma.masked_where(policy_grid < 0, policy_grid)
    im = ax.imshow(
        masked, cmap=_ACTION_CMAP, vmin=0, vmax=_N_ACTIONS - 1,
        aspect="auto",
        extent=[wealth_grid[0], wealth_grid[-1], -0.5, 1.5],
        origin="lower",
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Bad regime", "Good regime"])
    ax.set_xlabel("Wealth  W")
    ax.set_title("Optimal Policy  π*(W, regime)")
    tick_positions = [(i + 0.5) * _N_ACTIONS / _N_ACTIONS for i in range(_N_ACTIONS)]
    tick_positions = [(_N_ACTIONS - 1) * (i + 0.5) / _N_ACTIONS for i in range(_N_ACTIONS)]
    cbar = plt.colorbar(im, ax=ax, ticks=tick_positions)
    cbar.ax.set_yticklabels([ACTION_NAMES[i] for i in range(_N_ACTIONS)], fontsize=8)
    ax.axvline(W0, color="white", ls="--", lw=1.2, alpha=0.8)

    ax2 = axes[1]
    regime_colors = {1: "#4e9af1", 0: "#e74c3c"}
    regime_labels = {1: "Good regime", 0: "Bad regime"}
    for r in (1, 0):
        row = value_grid[r]
        valid = ~np.isnan(row)
        ax2.plot(
            wealth_grid[valid], row[valid],
            color=regime_colors[r], lw=1.8, label=regime_labels[r],
        )
    ax2.set_xlabel("Wealth  W")
    ax2.set_ylabel("V*(W, regime)")
    ax2.set_title("Value Function by Regime")
    ax2.axvline(W0, color="grey", ls="--", lw=1.2, alpha=0.7)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Policy summary table ──────────────────────────────────────────────────────

def policy_summary(
    opt_policy: FiniteDeterministicPolicy,
    wealth_grid: np.ndarray,
    use_regime: bool = False,
    max_rows: Optional[int] = None,
) -> None:
    """Print a readable table of the optimal allocation per state."""
    n = len(wealth_grid)
    rows_printed = 0

    if not use_regime:
        header = f"{'Wealth':>10}  {'Action':>14}  {'S1':>6}  {'S2':>6}  {'S3':>6}  {'Cash':>6}"
        print(header)
        print("-" * len(header))
        for w_idx in range(n):
            a = opt_policy.action_for.get(w_idx, -1)
            if a < 0:
                continue
            w1, w2, w3, wc = ACTIONS[a]
            print(
                f"  {wealth_grid[w_idx]:>8.4f}  {ACTION_NAMES[a]:>14}"
                f"  {w1*100:>4.0f}%  {w2*100:>4.0f}%  {w3*100:>4.0f}%  {wc*100:>4.0f}%"
            )
            rows_printed += 1
            if max_rows and rows_printed >= max_rows:
                print(f"  ... ({n - rows_printed} more states not shown)")
                break
    else:
        header = (
            f"{'Wealth':>10}  {'Regime':>8}  {'Action':>14}"
            f"  {'S1':>6}  {'S2':>6}  {'S3':>6}  {'Cash':>6}"
        )
        print(header)
        print("-" * len(header))
        for w_idx in range(n):
            for regime in (1, 0):
                state = (w_idx, regime)
                a = opt_policy.action_for.get(state, -1)
                if a < 0:
                    continue
                w1, w2, w3, wc = ACTIONS[a]
                label = "good" if regime == 1 else "bad "
                print(
                    f"  {wealth_grid[w_idx]:>8.4f}  {label:>8}  {ACTION_NAMES[a]:>14}"
                    f"  {w1*100:>4.0f}%  {w2*100:>4.0f}%  {w3*100:>4.0f}%  {wc*100:>4.0f}%"
                )
                rows_printed += 1
                if max_rows and rows_printed >= max_rows:
                    print(f"  ... truncated at {max_rows} rows")
                    return


# ── Convenience wrapper ───────────────────────────────────────────────────────

def solve_and_plot(
    mdp: FiniteMarkovDecisionProcess,
    wealth_grid: np.ndarray,
    cfg: Optional[MDPConfig] = None,
    gamma: float = 0.99,
    method: str = "value_iteration",
    use_regime: Optional[bool] = None,
) -> Tuple[dict, FiniteDeterministicPolicy]:
    """Solve the MDP and immediately produce the visualization."""
    opt_vf, opt_policy = solve(mdp, gamma=gamma, method=method)

    first_state = mdp.non_terminal_states[0].state
    _use_regime = isinstance(first_state, tuple) if use_regime is None else use_regime

    suffix = f"  (γ={gamma}, {method})"
    if _use_regime:
        plot_with_regime(opt_vf, opt_policy, wealth_grid, cfg=cfg, title_suffix=suffix)
    else:
        plot_no_regime(opt_vf, opt_policy, wealth_grid, cfg=cfg, title_suffix=suffix)

    policy_summary(opt_policy, wealth_grid, use_regime=_use_regime, max_rows=20)
    return opt_vf, opt_policy
