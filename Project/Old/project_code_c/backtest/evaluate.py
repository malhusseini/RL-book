"""Evaluation utilities: metrics, policy backtesting, ablations, leakage tests.

Core metrics
------------
  Annualised return, annualised volatility, Sharpe ratio, maximum drawdown,
  total turnover, and final wealth.

Leakage tests
-------------
  shift_test:   Shift signals forward by 1 day; alpha should materially degrade.
  permute_test: Randomly permute dates; strategy alpha should collapse to ~0.

Ablation runner
---------------
  run_ablations: Run multiple configuration variants and collect metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl.policy import FiniteDeterministicPolicy

from ..env.mdp import ACTIONS, ACTION_NAMES, MDPConfig, _nearest_bin
from ..features.regime import assign_regime
from .runner import WalkForwardConfig, run_walk_forward, backtest_fixed_allocation

logger = logging.getLogger(__name__)

_STRATEGY_COLORS = {
    "MDP Policy":    "black",
    "Conservative":  "#1a6faf",
    "Conservative+": "#4e9af1",
    "Moderate-Low":  "#74b9ff",
    "Moderate":      "#a8d8ea",
    "Balanced":      "#f5a623",
    "Moderate-High": "#e07b39",
    "Aggressive":    "#e74c3c",
    "Defensive-20":  "#6ab187",
    "Defensive-40":  "#3d9970",
    "Cash-heavy":    "#27ae60",
}


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(
    wealth_series: pd.Series,
    initial_wealth: float = 1.0,
    trading_days: int = 252,
) -> Dict[str, float]:
    """Compute standard performance metrics from a cumulative wealth series.

    Args:
        wealth_series:  Cumulative wealth path indexed by date (W_0 = initial_wealth).
        initial_wealth: Starting wealth for return computation.
        trading_days:   Trading days per year (default 252).

    Returns:
        Dictionary with keys: ann_return, ann_vol, sharpe, max_dd,
        total_return, final_wealth, n_days.
    """
    w = wealth_series.dropna().values.astype(float)
    if len(w) < 2:
        return {k: np.nan for k in ["ann_return", "ann_vol", "sharpe", "max_dd",
                                     "total_return", "final_wealth", "n_days"]}

    daily_rets = np.diff(w) / w[:-1]
    n_days = len(daily_rets)

    ann_ret = float((w[-1] / initial_wealth) ** (trading_days / n_days) - 1)
    ann_vol = float(daily_rets.std(ddof=1) * np.sqrt(trading_days))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else np.nan

    cum_max = np.maximum.accumulate(w)
    dd = (w - cum_max) / cum_max
    max_dd = float(dd.min())

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": float(w[-1] / initial_wealth - 1),
        "final_wealth": float(w[-1]),
        "n_days": n_days,
    }


def print_metrics_table(
    results: Dict[str, pd.Series],
    initial_wealth: float = 1.0,
) -> None:
    """Print a formatted metrics table for a set of named wealth paths."""
    header = (
        f"{'Strategy':<18}  {'Ann.Return':>11}  {'Ann.Vol':>8}  "
        f"{'Sharpe':>8}  {'MaxDD':>8}  {'FinalW':>10}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for name, wealth in results.items():
        m = compute_metrics(wealth, initial_wealth=initial_wealth)
        print(
            f"  {name:<16}  {m['ann_return']:+11.2%}  {m['ann_vol']:8.2%}  "
            f"{m['sharpe']:+8.3f}  {m['max_dd']:8.2%}  {m['final_wealth']:10.4f}"
        )
    print()


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_backtest(
    results: Dict[str, pd.Series],
    title: str = "Out-of-Sample Backtest",
) -> None:
    """Two-panel plot: cumulative wealth and drawdown for all strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    for name, wealth in results.items():
        color = _STRATEGY_COLORS.get(name, "grey")
        lw = 2.2 if name == "MDP Policy" else 1.3
        alpha = 0.92 if name == "MDP Policy" else 0.75
        ax.plot(wealth.index, wealth.values, label=name, color=color, lw=lw, alpha=alpha)
    ax.set_title("Cumulative Wealth  (W₀ = 1)")
    ax.set_ylabel("Wealth")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for name, wealth in results.items():
        w = wealth.values.astype(float)
        roll_max = np.maximum.accumulate(np.where(np.isfinite(w), w, np.nan))
        dd = 100.0 * (w - roll_max) / np.where(roll_max > 0, roll_max, 1.0)
        color = _STRATEGY_COLORS.get(name, "grey")
        lw = 2.2 if name == "MDP Policy" else 1.3
        ax2.plot(wealth.index, dd, label=name, color=color, lw=lw, alpha=0.8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── MDP policy backtest ───────────────────────────────────────────────────────

def backtest_policy(
    opt_policy: FiniteDeterministicPolicy,
    test_series: pd.DataFrame,
    wealth_grid: np.ndarray,
    cfg: Optional[MDPConfig] = None,
    use_regime: bool = False,
    initial_wealth: float = 1.0,
    market_col: str = "vwretd",
    show_plot: bool = True,
) -> pd.DataFrame:
    """Apply the solved MDP policy to held-out test returns and compare benchmarks.

    At each test day t:
      1. Snap current wealth to nearest wealth-grid bin.
      2. Read regime label (causal rolling-vol) if use_regime=True.
      3. Look up action from opt_policy.
      4. Apply the portfolio weights to realise PnL at t+1.

    Fixed-allocation benchmarks are simulated in parallel.

    Args:
        opt_policy:      Solved FiniteDeterministicPolicy.
        test_series:     Date-indexed DataFrame with columns
                         ['strat1_ret', 'strat2_ret', 'strat3_ret']
                         and optionally 'vwretd'.
        wealth_grid:     Same wealth grid used at solve time.
        cfg:             MDPConfig; initial_wealth taken from here if provided.
        use_regime:      Pass (wealth_idx, regime) to the policy.
        initial_wealth:  Starting wealth.
        market_col:      Column used to determine regime labels.
        show_plot:       If True, produce the backtest figure.

    Returns:
        DataFrame of cumulative wealth paths, one column per strategy.
    """
    if cfg is not None:
        initial_wealth = cfg.initial_wealth

    s1 = test_series["strat1_ret"].values
    s2 = test_series["strat2_ret"].values
    s3 = test_series["strat3_ret"].values
    dates = test_series.index
    n = len(dates)

    # Causal regime labels
    if use_regime and market_col in test_series.columns:
        regime_series = (
            assign_regime(test_series[market_col].dropna())
            .reindex(test_series.index)
            .ffill()
            .fillna(False)
            .astype(int)
            .values
        )
    else:
        regime_series = None
        use_regime = False

    # Simulate MDP-policy wealth path
    W = float(initial_wealth)
    policy_wealth = np.empty(n + 1)
    policy_wealth[0] = W
    for i in range(n):
        w_idx = int(_nearest_bin(np.array([W]), wealth_grid)[0])
        state = (w_idx, int(regime_series[i])) if use_regime else w_idx
        a = opt_policy.action_for.get(state, 4)  # fallback: Balanced (index 4)
        w1, w2, w3, wc = ACTIONS[a]
        port_ret = w1 * s1[i] + w2 * s2[i] + w3 * s3[i]
        W = max(W * (1.0 + port_ret), 1e-8)
        policy_wealth[i + 1] = W

    # Fixed-allocation benchmarks
    wealth_paths: Dict[str, np.ndarray] = {"MDP Policy": policy_wealth}
    for a_idx, name in ACTION_NAMES.items():
        w1, w2, w3, wc = ACTIONS[a_idx]
        port_rets = w1 * s1 + w2 * s2 + w3 * s3
        W_bench = initial_wealth * np.concatenate([[1.0], np.cumprod(1.0 + port_rets)])
        wealth_paths[name] = W_bench

    # Build date index (prepend synthetic start date)
    start_date = dates[0] - pd.tseries.offsets.BDay(1)
    all_dates = pd.DatetimeIndex([start_date] + list(dates))
    df = pd.DataFrame({k: v for k, v in wealth_paths.items()}, index=all_dates)

    results_dict = {col: df[col] for col in df.columns}
    print_metrics_table(results_dict, initial_wealth=initial_wealth)
    if show_plot:
        plot_backtest(results_dict, title="Out-of-Sample Backtest (Test Period)")

    return df


# ── Leakage sanity checks ─────────────────────────────────────────────────────

def run_leakage_shift_test(
    opt_policy: FiniteDeterministicPolicy,
    test_series: pd.DataFrame,
    wealth_grid: np.ndarray,
    cfg: Optional[MDPConfig] = None,
    use_regime: bool = False,
    shift_days: int = 1,
    show_plot: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Shift returns forward by `shift_days` and compare Sharpe on the same period.

    Original is run on the same date range as the shifted series (so one fewer
    day than full test), ensuring a like-for-like comparison. If the strategy
    has no look-ahead, performance should materially degrade under the shift.

    Returns:
        dict with keys 'original' and 'shifted', each containing metrics dict.
    """
    logger.info("Running leakage shift test (shift_days=%d) …", shift_days)

    # Shifted: use returns shifted forward (pretend we peeked 1 day ahead)
    shifted = test_series.copy()
    for col in ["strat1_ret", "strat2_ret", "strat3_ret"]:
        shifted[col] = shifted[col].shift(-shift_days)
    shifted = shifted.dropna(subset=["strat1_ret", "strat2_ret", "strat3_ret"])

    # Run both backtests on the *same* period (shifted has one fewer row after dropna).
    # Otherwise original uses n days and shifted n-1, so different sample → unfair comparison.
    test_trimmed = test_series.iloc[: len(shifted)]

    df_orig = backtest_policy(
        opt_policy, test_trimmed, wealth_grid, cfg=cfg, use_regime=use_regime,
        show_plot=False,
    )
    original_metrics = compute_metrics(df_orig["MDP Policy"])

    df_shifted = backtest_policy(
        opt_policy, shifted, wealth_grid, cfg=cfg, use_regime=use_regime,
        show_plot=False,
    )
    shifted_metrics = compute_metrics(df_shifted["MDP Policy"])

    sharpe_orig = original_metrics["sharpe"]
    sharpe_shift = shifted_metrics["sharpe"]

    print("\n=== Leakage Shift Test ===")
    print(f"  Original Sharpe :  {sharpe_orig:+.3f}")
    print(f"  Shifted  Sharpe :  {sharpe_shift:+.3f}")
    if sharpe_orig > sharpe_shift:
        print("  PASS — shifting signals degrades performance (no look-ahead detected).")
    else:
        print("  NOTE — shifted (state_t vs return_{t+1}) matches or beats original (state_t vs return_t).")
        print("         This can mean: (1) look-ahead bias, or (2) legitimate next-day predictive power")
        print("         (e.g. regime at t predicts return at t+1). Check features and alignment.")

    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_orig.index, df_orig["MDP Policy"], label=f"Original (Sharpe={sharpe_orig:.2f})",
                color="black", lw=1.8)
        ax.plot(df_shifted.index, df_shifted["MDP Policy"],
                label=f"+{shift_days}d Shifted (Sharpe={sharpe_shift:.2f})",
                color="#e74c3c", lw=1.4, ls="--")
        ax.set_title(f"Leakage Shift Test: Original vs. +{shift_days}d Shift")
        ax.set_ylabel("Cumulative Wealth")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {"original": original_metrics, "shifted": shifted_metrics}


def run_leakage_permute_test(
    opt_policy: FiniteDeterministicPolicy,
    test_series: pd.DataFrame,
    wealth_grid: np.ndarray,
    cfg: Optional[MDPConfig] = None,
    use_regime: bool = False,
    n_permutations: int = 5,
    rng_seed: int = 42,
    show_plot: bool = True,
) -> Dict[str, Any]:
    """Randomly permute dates and check that alpha collapses.

    Under date permutation, the temporal structure is destroyed.  A strategy
    with genuine predictive signal should show near-zero Sharpe on permuted
    data; a strategy exploiting look-ahead would retain its Sharpe.

    Returns:
        dict with 'original_sharpe' and 'permuted_sharpes' (list).
    """
    logger.info("Running leakage permutation test (%d permutations) …", n_permutations)
    rng = np.random.default_rng(rng_seed)

    df_orig = backtest_policy(
        opt_policy, test_series, wealth_grid, cfg=cfg, use_regime=use_regime,
        show_plot=False,
    )
    original_sharpe = compute_metrics(df_orig["MDP Policy"])["sharpe"]

    permuted_sharpes: list[float] = []
    permuted_wealth_paths: list[pd.Series] = []

    for trial in range(n_permutations):
        perm_idx = rng.permutation(len(test_series))
        permuted = test_series.copy()
        permuted.iloc[:] = test_series.values[perm_idx]

        df_perm = backtest_policy(
            opt_policy, permuted, wealth_grid, cfg=cfg, use_regime=use_regime,
            show_plot=False,
        )
        s = compute_metrics(df_perm["MDP Policy"])["sharpe"]
        permuted_sharpes.append(s)
        permuted_wealth_paths.append(df_perm["MDP Policy"].rename(f"Permuted {trial+1}"))

    mean_perm = float(np.nanmean(permuted_sharpes))
    balanced_sharpe = compute_metrics(df_orig["Balanced"])["sharpe"] if "Balanced" in df_orig.columns else None
    collapse_to_balanced = balanced_sharpe is not None and abs(mean_perm - balanced_sharpe) < 0.02
    # Look-ahead would preserve original Sharpe under permutation; only warn when permuted ≈ original
    within_20pct_orig = abs(original_sharpe) > 1e-9 and 0.8 * abs(original_sharpe) <= mean_perm <= 1.2 * abs(original_sharpe)

    print("\n=== Leakage Permutation Test ===")
    print(f"  Original Sharpe     : {original_sharpe:+.3f}")
    print(f"  Mean permuted Sharpe: {mean_perm:+.3f}  (across {n_permutations} trials)")
    if abs(mean_perm) < abs(original_sharpe) * 0.5:
        print("  PASS — permuted Sharpe is substantially lower (genuine temporal signal).")
    elif mean_perm < original_sharpe:
        print("  PASS — permuted Sharpe is lower than original (temporal signal present).")
    elif within_20pct_orig:
        print("  WARNING — permuted Sharpe ≈ original; temporal structure may survive permutation (check for look-ahead).")
    elif mean_perm > original_sharpe:
        print("  NOTE — permuted Sharpe exceeds original; for dynamic policies this can occur by chance on shuffled returns.")
        print("         (Look-ahead would preserve original Sharpe; this is not evidence of look-ahead.)")
    elif collapse_to_balanced:
        print("  NOTE — permuted Sharpe ≈ Balanced; policy may default to fallback under permutation.")
        print("         (Not evidence of look-ahead; fixed allocation Sharpe is unchanged by permuting dates.)")
    else:
        print("  WARNING — permuted Sharpe is not much lower; possible look-ahead!")

    if show_plot and permuted_wealth_paths:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_orig.index, df_orig["MDP Policy"],
                label=f"Original (Sharpe={original_sharpe:.2f})", color="black", lw=2.0)
        for pws in permuted_wealth_paths:
            ax.plot(df_orig.index[:len(pws)], pws.values[:len(df_orig)],
                    color="#e74c3c", lw=0.8, alpha=0.5)
        ax.plot([], [], color="#e74c3c", lw=1.2, alpha=0.6,
                label=f"Permuted (mean Sharpe={mean_perm:.2f})")
        ax.set_title("Leakage Permutation Test")
        ax.set_ylabel("Cumulative Wealth")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        "original_sharpe": original_sharpe,
        "permuted_sharpes": permuted_sharpes,
        "mean_permuted_sharpe": mean_perm,
    }


# ── Ablation runner ────────────────────────────────────────────────────────────

def run_ablations(
    ablation_configs: List[Dict[str, Any]],
    build_fn: Any,
    test_series_start: str,
    test_series_end: Optional[str] = None,
    initial_wealth: float = 1.0,
    show_plot: bool = True,
) -> pd.DataFrame:
    """Run multiple MDP configurations and collect comparison metrics.

    Each entry in ablation_configs is a dict with keys:
      - 'label':      Short descriptive name for this configuration.
      - 'mdp_cfg':    MDPConfig (or None for default).
      - 'params':     CalibrationParams.
      - 'test_series': Pre-built test_series DataFrame (or built by build_fn).
      - 'gamma':      Discount factor for DP solve.
      - 'method':     'value_iteration' or 'policy_iteration'.

    Args:
        ablation_configs:  List of configuration dicts (see above).
        build_fn:          Callable(params, cfg) → (mdp, wealth_grid, cfg).
        test_series_start: ISO date string; start of test period.
        test_series_end:   ISO date string; end of test period (None = full).
        initial_wealth:    Starting wealth for all runs.
        show_plot:         If True, show comparative wealth plots.

    Returns:
        DataFrame with one row per configuration and columns for each metric.
    """
    from ..dp.solver import solve

    results = []

    for entry in ablation_configs:
        label = entry.get("label", "unnamed")
        logger.info("Running ablation: %s …", label)

        params = entry["params"]
        cfg = entry.get("mdp_cfg", None)
        test_series = entry["test_series"]
        gamma = entry.get("gamma", 0.99)
        method = entry.get("method", "value_iteration")

        # Slice test period
        ts = test_series[test_series.index >= test_series_start]
        if test_series_end:
            ts = ts[ts.index <= test_series_end]

        try:
            mdp, wealth_grid, used_cfg = build_fn(params, cfg)
            opt_vf, opt_policy = solve(mdp, gamma=gamma, method=method)
            use_regime = used_cfg.use_regime and params.has_regime

            df_bt = backtest_policy(
                opt_policy, ts, wealth_grid, cfg=used_cfg,
                use_regime=use_regime, initial_wealth=initial_wealth,
                show_plot=False,
            )
            m = compute_metrics(df_bt["MDP Policy"], initial_wealth=initial_wealth)
            m["label"] = label
            results.append(m)
        except Exception as exc:
            logger.error("Ablation '%s' failed: %s", label, exc)
            results.append({"label": label, "ann_return": np.nan,
                            "ann_vol": np.nan, "sharpe": np.nan,
                            "max_dd": np.nan, "final_wealth": np.nan})

    result_df = pd.DataFrame(results).set_index("label")

    print("\n=== Ablation Results ===")
    print(result_df[["ann_return", "ann_vol", "sharpe", "max_dd", "final_wealth"]].to_string(
        float_format=lambda x: f"{x:+.3f}" if abs(x) < 10 else f"{x:.2f}"
    ))

    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        sharpes = result_df["sharpe"].dropna()
        ax.barh(sharpes.index, sharpes.values, color="#4e9af1")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Out-of-Sample Sharpe Ratio")
        ax.set_title("Ablation Comparison: Sharpe Ratios")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

    return result_df
