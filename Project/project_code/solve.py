"""Solve the Phase 2 MDP with Value Iteration or Policy Iteration,
and visualize the optimal policy and value function.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from rl.dynamic_programming import policy_iteration_result, value_iteration_result
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy

from .mdp import ACTION_NAMES, MDPConfig

logger = logging.getLogger(__name__)

_ACTION_COLORS = ["#4e9af1", "#f5a623", "#e74c3c"]   # conservative / balanced / aggressive
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
        gamma:  Discount factor (e.g. 0.99 for near-infinite horizon).
        method: 'value_iteration' or 'policy_iteration'.

    Returns:
        (opt_vf, opt_policy):
          opt_vf     — dict mapping NonTerminal[S] → float
          opt_policy — FiniteDeterministicPolicy mapping raw state S → action int
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


# ── Helpers to extract arrays from the solution ─────────────────────────────

def _extract_no_regime(
    opt_vf: dict,
    opt_policy: FiniteDeterministicPolicy,
    wealth_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (actions_array, values_array) aligned with wealth_grid indices."""
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
    """
    Return (policy_grid, value_grid) shaped (2, n_wealth_bins).
    Row 0 = bad regime, Row 1 = good regime.
    """
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


# ── Plotting: no-regime ──────────────────────────────────────────────────────

def plot_no_regime(
    opt_vf: dict,
    opt_policy: FiniteDeterministicPolicy,
    wealth_grid: np.ndarray,
    cfg: Optional[MDPConfig] = None,
    title_suffix: str = "",
) -> None:
    """
    Two-panel plot for the no-regime MDP:
      Left  : Optimal action (scatter, color-coded by allocation type)
      Right : Optimal value function V*(W)
    """
    actions, values = _extract_no_regime(opt_vf, opt_policy, wealth_grid)
    W0 = cfg.initial_wealth if cfg is not None else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Phase 2 Optimal Policy & Value Function{title_suffix}", fontsize=13)

    # — Left: policy —
    ax = axes[0]
    valid = actions >= 0
    sc = ax.scatter(
        wealth_grid[valid], actions[valid],
        c=actions[valid], cmap=_ACTION_CMAP, vmin=0, vmax=2,
        s=55, zorder=3,
    )
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([ACTION_NAMES[i] for i in range(3)])
    ax.set_xlabel("Wealth  W", fontsize=11)
    ax.set_ylabel("Optimal allocation", fontsize=11)
    ax.set_title("Optimal Policy  π*(W)")
    ax.axvline(W0, color="grey", ls="--", lw=1.2, alpha=0.7, label=f"W₀ = {W0}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # — Right: value function —
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


# ── Plotting: with regime ────────────────────────────────────────────────────

def plot_with_regime(
    opt_vf: dict,
    opt_policy: FiniteDeterministicPolicy,
    wealth_grid: np.ndarray,
    cfg: Optional[MDPConfig] = None,
    title_suffix: str = "",
) -> None:
    """
    Two-panel plot for the 2-state regime MDP:
      Left  : Heatmap of optimal action over (wealth × regime)
      Right : V*(W) for good and bad regimes overlaid
    """
    policy_grid, value_grid = _extract_with_regime(opt_vf, opt_policy, wealth_grid)
    W0 = cfg.initial_wealth if cfg is not None else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(
        f"Phase 2 Optimal Policy & Value Function (with Regime){title_suffix}",
        fontsize=13,
    )

    # — Left: policy heatmap —
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

    # — Right: value functions —
    ax2 = axes[1]
    regime_labels = {1: ("Good regime", "#4e9af1"), 0: ("Bad regime", "#e74c3c")}
    for regime, (label, color) in regime_labels.items():
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
    """Print a readable table of optimal actions per state."""
    n = len(wealth_grid)

    if not use_regime:
        print(f"{'Wealth':>10}  {'Action':>14}  {'Alloc (S1/S2/Cash)'}")
        print("-" * 52)
        for w_idx in range(n):
            a = opt_policy.action_for.get(w_idx, -1)
            if a < 0:
                continue
            fracs = (0.8, 0.2) if a == 0 else (0.5, 0.5) if a == 1 else (0.2, 0.8)
            print(
                f"  {wealth_grid[w_idx]:>8.4f}  {ACTION_NAMES[a]:>14}"
                f"  {fracs[0]*100:.0f}% / {fracs[1]*100:.0f}% / 0%"
            )
    else:
        print(f"{'Wealth':>10}  {'Regime':>8}  {'Action':>14}  {'Alloc (S1/S2/Cash)'}")
        print("-" * 62)
        for w_idx in range(n):
            for regime in (1, 0):
                state = (w_idx, regime)
                a = opt_policy.action_for.get(state, -1)
                if a < 0:
                    continue
                fracs = (0.8, 0.2) if a == 0 else (0.5, 0.5) if a == 1 else (0.2, 0.8)
                regime_label = "good" if regime == 1 else "bad "
                print(
                    f"  {wealth_grid[w_idx]:>8.4f}  {regime_label:>8}  "
                    f"{ACTION_NAMES[a]:>14}  {fracs[0]*100:.0f}% / {fracs[1]*100:.0f}% / 0%"
                )


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

    # Infer whether regime was used from the state type of the first state
    first_state = mdp.non_terminal_states[0].state
    _use_regime = isinstance(first_state, tuple) if use_regime is None else use_regime

    suffix = f"  (γ={gamma}, {method})"
    if _use_regime:
        plot_with_regime(opt_vf, opt_policy, wealth_grid, cfg=cfg, title_suffix=suffix)
    else:
        plot_no_regime(opt_vf, opt_policy, wealth_grid, cfg=cfg, title_suffix=suffix)

    policy_summary(opt_policy, wealth_grid, use_regime=_use_regime)

    return opt_vf, opt_policy
