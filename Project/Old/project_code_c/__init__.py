"""project_code_c — Phase 2 version c.

Three-strategy daily return pipeline → discrete-action MDP → DP solution.

Strategies
----------
  Strategy 1 (conservative):  PCA residual mean reversion (Avellaneda-Lee).
  Strategy 2 (medium):        Pairs trading via rolling cointegration spread.
  Strategy 3 (aggressive):    Variance-risk-premium proxy (short-vol profile).

Modules
-------
  data/       CRSP ingest, cleaning, integrity gates, parquet export.
  features/   Causal rolling factors, PCA residuals, regime labels.
  strategies/ Three strategy engines emitting daily return series.
  env/        Discrete-action MDP environment and Gaussian/bootstrap simulator.
  dp/         Value iteration / policy iteration wrappers and policy tools.
  backtest/   Walk-forward runner, metrics, ablations, leakage tests.

Convention (enforced everywhere)
---------------------------------
  Decision at time t  →  positions held over (t, t+1].
  PnL_t+1  =  position_t  ×  r_{t+1}   (lagged exposure).
"""

from .data.ingest import load_crsp, integrity_checks
from .data.clean import (
    apply_missing_data_policy,
    winsorize_returns,
    save_panel,
    load_panel,
    build_full_pipeline,
)
from .features.regime import assign_regime
from .strategies.build import build_strategy_series
from .env.simulator import CalibrationParams, calibrate
from .env.mdp import (
    ACTIONS,
    ACTION_NAMES,
    MDPConfig,
    build_mdp,
    crra_utility,
    make_wealth_grid,
)
from .dp.solver import solve, solve_and_plot, policy_summary
from .backtest.runner import run_walk_forward, WalkForwardConfig
from .backtest.evaluate import (
    backtest_policy,
    compute_metrics,
    print_metrics_table,
    plot_backtest,
    run_leakage_shift_test,
    run_leakage_permute_test,
    run_ablations,
)

__all__ = [
    # data
    "load_crsp",
    "integrity_checks",
    "apply_missing_data_policy",
    "winsorize_returns",
    "save_panel",
    "load_panel",
    "build_full_pipeline",
    # features
    "assign_regime",
    # strategies
    "build_strategy_series",
    # env
    "CalibrationParams",
    "calibrate",
    "ACTIONS",
    "ACTION_NAMES",
    "MDPConfig",
    "build_mdp",
    "crra_utility",
    "make_wealth_grid",
    # dp
    "solve",
    "solve_and_plot",
    "policy_summary",
    # backtest
    "run_walk_forward",
    "WalkForwardConfig",
    "backtest_policy",
    "compute_metrics",
    "print_metrics_table",
    "plot_backtest",
    "run_leakage_shift_test",
    "run_leakage_permute_test",
    "run_ablations",
]
