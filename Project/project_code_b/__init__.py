"""project_code_b — Phase 2 version b.

Strategies:
  Strategy 1 (conservative): Avellaneda–Lee PCA residual mean reversion.
  Strategy 2 (aggressive):   Equity carry proxy (trailing dividend-yield quintile).

Public API mirrors project_code for drop-in notebook use.
"""

from .data_pipeline import (
    build_full_pipeline,
    build_strategy_series,
    apply_missing_data_policy,
    integrity_checks,
    load_crsp,
    winsorize_returns,
)

from .calibration import (
    CalibrationParams,
    assign_regime,
    calibrate,
)

from .mdp import (
    ACTION_NAMES,
    ACTIONS,
    MDPConfig,
    build_mdp,
    crra_utility,
    make_wealth_grid,
)

from .solve import (
    backtest_policy,
    plot_no_regime,
    plot_with_regime,
    policy_summary,
    solve,
    solve_and_plot,
)

__all__ = [
    "build_full_pipeline",
    "build_strategy_series",
    "apply_missing_data_policy",
    "integrity_checks",
    "load_crsp",
    "winsorize_returns",
    "CalibrationParams",
    "assign_regime",
    "calibrate",
    "ACTION_NAMES",
    "ACTIONS",
    "MDPConfig",
    "build_mdp",
    "crra_utility",
    "make_wealth_grid",
    "backtest_policy",
    "plot_no_regime",
    "plot_with_regime",
    "policy_summary",
    "solve",
    "solve_and_plot",
]
