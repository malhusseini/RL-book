"""backtest — Walk-forward runner, metrics, ablations, and leakage tests."""

from .runner import run_walk_forward, WalkForwardConfig
from .evaluate import (
    backtest_policy,
    compute_metrics,
    print_metrics_table,
    plot_backtest,
    run_leakage_shift_test,
    run_leakage_permute_test,
    run_ablations,
)

__all__ = [
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
