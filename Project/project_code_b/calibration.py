"""Calibrate the parametric bivariate Gaussian return model (and optional
2-state regime) from the strategy return series produced by
data_pipeline.build_strategy_series().

All estimators are computed on the provided sample only.
Regime assignment is causal (rolling windows, no look-ahead).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Parameter dataclass ─────────────────────────────────────────────────────

@dataclass
class CalibrationParams:
    """
    All calibrated parameters for the Phase 2 MDP.

    Unconditional params are always set; regime-conditional params are set
    only when include_regime=True is passed to calibrate().
    """
    mu_1: float
    sigma_1: float
    mu_2: float
    sigma_2: float
    rho: float

    mu_1_good: Optional[float] = None
    sigma_1_good: Optional[float] = None
    mu_2_good: Optional[float] = None
    sigma_2_good: Optional[float] = None
    rho_good: Optional[float] = None

    mu_1_bad: Optional[float] = None
    sigma_1_bad: Optional[float] = None
    mu_2_bad: Optional[float] = None
    sigma_2_bad: Optional[float] = None
    rho_bad: Optional[float] = None

    # 2×2 regime transition matrix: entry [i, j] = P(regime_j | regime_i)
    # Row 0 = bad, Row 1 = good
    regime_transition: Optional[np.ndarray] = None

    n_obs: int = 0
    n_obs_good: int = 0
    n_obs_bad: int = 0
    date_start: Optional[str] = None
    date_end: Optional[str] = None

    def __post_init__(self) -> None:
        assert self.sigma_1 > 0, "sigma_1 must be positive"
        assert self.sigma_2 > 0, "sigma_2 must be positive"
        assert -1.0 < self.rho < 1.0, "rho must be in (-1, 1)"

    @property
    def has_regime(self) -> bool:
        return self.regime_transition is not None

    def describe(self) -> str:
        lines = [
            "=== Calibration Parameters ===",
            f"  Period : {self.date_start} → {self.date_end}  ({self.n_obs} obs)",
            "",
            "  Unconditional",
            f"    Strategy 1 : μ = {self.mu_1:+.5f},  σ = {self.sigma_1:.5f}",
            f"    Strategy 2 : μ = {self.mu_2:+.5f},  σ = {self.sigma_2:.5f}",
            f"    Correlation : ρ = {self.rho:.4f}",
        ]
        if self.has_regime:
            lines += [
                "",
                f"  Regime-conditional  ({self.n_obs_good} good days,"
                f" {self.n_obs_bad} bad days)",
                "    Good regime",
                f"      Strategy 1 : μ = {self.mu_1_good:+.5f},  σ = {self.sigma_1_good:.5f}",
                f"      Strategy 2 : μ = {self.mu_2_good:+.5f},  σ = {self.sigma_2_good:.5f}",
                f"      ρ = {self.rho_good:.4f}",
                "    Bad regime",
                f"      Strategy 1 : μ = {self.mu_1_bad:+.5f},  σ = {self.sigma_1_bad:.5f}",
                f"      Strategy 2 : μ = {self.mu_2_bad:+.5f},  σ = {self.sigma_2_bad:.5f}",
                f"      ρ = {self.rho_bad:.4f}",
                "",
                "  Regime transition matrix  [bad→, good→]",
                f"    bad  row : {self.regime_transition[0]}",
                f"    good row : {self.regime_transition[1]}",
            ]
        return "\n".join(lines)


# ── Internal helpers ────────────────────────────────────────────────────────

def _fit_bivariate(
    s1: pd.Series,
    s2: pd.Series,
) -> Tuple[float, float, float, float, float]:
    valid = s1.notna() & s2.notna()
    r1 = s1[valid].values.astype(float)
    r2 = s2[valid].values.astype(float)
    if len(r1) < 2:
        raise ValueError("Fewer than 2 valid observations for bivariate fit.")
    mu1, sig1 = float(r1.mean()), float(r1.std(ddof=1))
    mu2, sig2 = float(r2.mean()), float(r2.std(ddof=1))
    rho = float(np.clip(np.corrcoef(r1, r2)[0, 1], -0.9999, 0.9999))
    return mu1, sig1, mu2, sig2, rho


# ── Regime assignment ────────────────────────────────────────────────────────

def assign_regime(
    market_returns: pd.Series,
    window: int = 21,
    long_window_mult: int = 5,
) -> pd.Series:
    """
    Assign a daily binary regime label using only past data (causal).

    Rule: rolling realized vol (window days) < its own rolling median
          over a longer horizon  →  "good" (True); otherwise "bad" (False).
    """
    roll_vol = market_returns.rolling(window, min_periods=window // 2).std()
    long_window = window * long_window_mult
    roll_median = roll_vol.rolling(long_window, min_periods=window).median()
    return (roll_vol < roll_median).rename("regime_good")


def _build_transition_matrix(regime_series: pd.Series) -> np.ndarray:
    """Estimate empirical 2×2 Markov transition matrix. Encoding: 0=bad, 1=good."""
    regime_int = regime_series.astype(int).values
    counts = np.zeros((2, 2), dtype=float)
    for curr, nxt in zip(regime_int[:-1], regime_int[1:]):
        counts[curr, nxt] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


# ── Main calibration entry point ─────────────────────────────────────────────

def calibrate(
    strategy_series: pd.DataFrame,
    market_col: str = "vwretd",
    regime_window: int = 21,
    include_regime: bool = True,
) -> CalibrationParams:
    """
    Calibrate all model parameters from the strategy return series.

    Args:
        strategy_series: DataFrame with columns ['strat1_ret', 'strat2_ret']
                         and optionally market columns (e.g. 'vwretd').
        market_col:      Column used to define regime.
        regime_window:   Rolling window (days) for regime volatility estimate.
        include_regime:  Whether to fit regime-conditional parameters.

    Returns:
        CalibrationParams dataclass with all fitted values.
    """
    s1 = strategy_series["strat1_ret"]
    s2 = strategy_series["strat2_ret"]

    mu1, sig1, mu2, sig2, rho = _fit_bivariate(s1, s2)

    params = CalibrationParams(
        mu_1=mu1, sigma_1=sig1,
        mu_2=mu2, sigma_2=sig2,
        rho=rho,
        n_obs=int(s1.notna().sum()),
        date_start=str(strategy_series.index.min().date()),
        date_end=str(strategy_series.index.max().date()),
    )

    if include_regime and market_col in strategy_series.columns:
        market = strategy_series[market_col].dropna()
        regime = assign_regime(market, window=regime_window)

        aligned = pd.concat([s1, s2, regime], axis=1).dropna()
        good_mask = aligned["regime_good"].astype(bool)

        n_good = int(good_mask.sum())
        n_bad  = int((~good_mask).sum())
        logger.info(
            "Regime: %d good days (%.1f%%), %d bad days.",
            n_good, 100 * n_good / len(aligned), n_bad,
        )
        if n_good < 10 or n_bad < 10:
            logger.warning(
                "Too few observations in one regime (%d good, %d bad) — "
                "regime calibration may be unreliable.",
                n_good, n_bad,
            )

        mu1_g, sig1_g, mu2_g, sig2_g, rho_g = _fit_bivariate(
            aligned.loc[good_mask, "strat1_ret"],
            aligned.loc[good_mask, "strat2_ret"],
        )
        mu1_b, sig1_b, mu2_b, sig2_b, rho_b = _fit_bivariate(
            aligned.loc[~good_mask, "strat1_ret"],
            aligned.loc[~good_mask, "strat2_ret"],
        )

        params.mu_1_good, params.sigma_1_good = mu1_g, sig1_g
        params.mu_2_good, params.sigma_2_good = mu2_g, sig2_g
        params.rho_good = rho_g
        params.mu_1_bad,  params.sigma_1_bad  = mu1_b, sig1_b
        params.mu_2_bad,  params.sigma_2_bad  = mu2_b, sig2_b
        params.rho_bad  = rho_b
        params.n_obs_good = n_good
        params.n_obs_bad  = n_bad
        params.regime_transition = _build_transition_matrix(aligned["regime_good"])

    logger.info("\n%s", params.describe())
    return params
