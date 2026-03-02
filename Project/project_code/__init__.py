"""project_code — Phase 2 implementation modules for CME 241 Final Project.

Public API
----------
data_pipeline : load_crsp, integrity_checks, apply_missing_data_policy,
                add_rolling_vol, winsorize_returns, build_strategy_series,
                build_full_pipeline
calibration   : calibrate, assign_regime, CalibrationParams
mdp           : build_mdp, MDPConfig, make_wealth_grid, crra_utility,
                ACTION_NAMES, ACTIONS
solve         : solve, solve_and_plot, plot_no_regime, plot_with_regime,
                policy_summary
"""

from .data_pipeline import (
    build_full_pipeline,
    build_strategy_series,
    add_rolling_vol,
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
    plot_no_regime,
    plot_with_regime,
    policy_summary,
    solve,
    solve_and_plot,
)

__all__ = [
    # data_pipeline
    "build_full_pipeline",
    "build_strategy_series",
    "add_rolling_vol",
    "apply_missing_data_policy",
    "integrity_checks",
    "load_crsp",
    "winsorize_returns",
    # calibration
    "CalibrationParams",
    "assign_regime",
    "calibrate",
    # mdp
    "ACTION_NAMES",
    "ACTIONS",
    "MDPConfig",
    "build_mdp",
    "crra_utility",
    "make_wealth_grid",
    # solve
    "plot_no_regime",
    "plot_with_regime",
    "policy_summary",
    "solve",
    "solve_and_plot",
]
