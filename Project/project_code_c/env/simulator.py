"""Return simulators and calibration for the Phase-2 MDP.

Two simulators are provided:
  GaussianSampler       Parametric multivariate Gaussian (optionally regime-switching).
  BlockBootstrapSampler Resample blocks of historical strategy returns to
                        preserve autocorrelation structure.

CalibrationParams
-----------------
Stores all estimated parameters for the 3-strategy model:
  - Unconditional means, vols, and pairwise correlations.
  - Regime-conditional parameter sets (if include_regime=True).
  - Empirical 2×2 Markov transition matrix.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Calibration parameters ────────────────────────────────────────────────────

@dataclass
class CalibrationParams:
    """Calibrated parameters for the Phase-2 three-strategy MDP.

    Unconditional params are always set.
    Regime-conditional params are set only when include_regime=True.
    """
    # Unconditional
    mu_1: float
    sigma_1: float
    mu_2: float
    sigma_2: float
    mu_3: float
    sigma_3: float
    rho_12: float
    rho_13: float
    rho_23: float

    # Regime-conditional (good regime)
    mu_1_good: Optional[float] = None
    sigma_1_good: Optional[float] = None
    mu_2_good: Optional[float] = None
    sigma_2_good: Optional[float] = None
    mu_3_good: Optional[float] = None
    sigma_3_good: Optional[float] = None
    rho_12_good: Optional[float] = None
    rho_13_good: Optional[float] = None
    rho_23_good: Optional[float] = None

    # Regime-conditional (bad regime)
    mu_1_bad: Optional[float] = None
    sigma_1_bad: Optional[float] = None
    mu_2_bad: Optional[float] = None
    sigma_2_bad: Optional[float] = None
    mu_3_bad: Optional[float] = None
    sigma_3_bad: Optional[float] = None
    rho_12_bad: Optional[float] = None
    rho_13_bad: Optional[float] = None
    rho_23_bad: Optional[float] = None

    # 2×2 regime Markov transition matrix (row 0 = bad, row 1 = good)
    regime_transition: Optional[np.ndarray] = None

    n_obs: int = 0
    n_obs_good: int = 0
    n_obs_bad: int = 0
    date_start: Optional[str] = None
    date_end: Optional[str] = None

    def __post_init__(self) -> None:
        assert self.sigma_1 > 0, "sigma_1 must be positive"
        assert self.sigma_2 > 0, "sigma_2 must be positive"
        assert self.sigma_3 > 0, "sigma_3 must be positive"
        for rho_name, rho_val in [
            ("rho_12", self.rho_12),
            ("rho_13", self.rho_13),
            ("rho_23", self.rho_23),
        ]:
            assert -1.0 < rho_val < 1.0, f"{rho_name} must be in (-1, 1)"

    @property
    def has_regime(self) -> bool:
        return self.regime_transition is not None

    def cov_matrix(self, regime: Optional[int] = None) -> np.ndarray:
        """Return 3×3 covariance matrix for unconditional or regime-conditional params."""
        if regime == 1 and self.mu_1_good is not None:
            s1, s2, s3 = self.sigma_1_good, self.sigma_2_good, self.sigma_3_good
            r12, r13, r23 = self.rho_12_good, self.rho_13_good, self.rho_23_good
        elif regime == 0 and self.mu_1_bad is not None:
            s1, s2, s3 = self.sigma_1_bad, self.sigma_2_bad, self.sigma_3_bad
            r12, r13, r23 = self.rho_12_bad, self.rho_13_bad, self.rho_23_bad
        else:
            s1, s2, s3 = self.sigma_1, self.sigma_2, self.sigma_3
            r12, r13, r23 = self.rho_12, self.rho_13, self.rho_23
        return np.array([
            [s1**2,       r12*s1*s2,  r13*s1*s3],
            [r12*s1*s2,   s2**2,      r23*s2*s3],
            [r13*s1*s3,   r23*s2*s3,  s3**2    ],
        ])

    def mean_vector(self, regime: Optional[int] = None) -> np.ndarray:
        if regime == 1 and self.mu_1_good is not None:
            return np.array([self.mu_1_good, self.mu_2_good, self.mu_3_good])
        if regime == 0 and self.mu_1_bad is not None:
            return np.array([self.mu_1_bad, self.mu_2_bad, self.mu_3_bad])
        return np.array([self.mu_1, self.mu_2, self.mu_3])

    def describe(self) -> str:
        lines = [
            "=== CalibrationParams (3-strategy) ===",
            f"  Period : {self.date_start} → {self.date_end}  ({self.n_obs} obs)",
            "",
            "  Unconditional",
            f"    Strategy 1 : μ={self.mu_1:+.5f}  σ={self.sigma_1:.5f}",
            f"    Strategy 2 : μ={self.mu_2:+.5f}  σ={self.sigma_2:.5f}",
            f"    Strategy 3 : μ={self.mu_3:+.5f}  σ={self.sigma_3:.5f}",
            f"    ρ(1,2)={self.rho_12:.4f}  ρ(1,3)={self.rho_13:.4f}  ρ(2,3)={self.rho_23:.4f}",
        ]
        if self.has_regime:
            lines += [
                "",
                f"  Regime-conditional  ({self.n_obs_good} good, {self.n_obs_bad} bad days)",
                "    Good  μ=("
                f"{self.mu_1_good:+.5f}, {self.mu_2_good:+.5f}, {self.mu_3_good:+.5f})",
                "    Bad   μ=("
                f"{self.mu_1_bad:+.5f}, {self.mu_2_bad:+.5f}, {self.mu_3_bad:+.5f})",
                "",
                "  Regime transition matrix  [bad→, good→]",
                f"    bad  row : {self.regime_transition[0]}",
                f"    good row : {self.regime_transition[1]}",
            ]
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Persist calibration parameters to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in asdict(self).items()}
        with open(path, "w") as fh:
            json.dump(d, fh, indent=2)
        logger.info("Saved CalibrationParams to %s.", path)

    @classmethod
    def load(cls, path: str) -> "CalibrationParams":
        """Load calibration parameters from a JSON file."""
        with open(path) as fh:
            d = json.load(fh)
        if d.get("regime_transition") is not None:
            d["regime_transition"] = np.array(d["regime_transition"])
        return cls(**d)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fit_trivariate(
    s1: pd.Series,
    s2: pd.Series,
    s3: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit means and 3×3 covariance for aligned return series."""
    valid = s1.notna() & s2.notna() & s3.notna()
    R = np.column_stack([s1[valid].values, s2[valid].values, s3[valid].values]).astype(float)
    if R.shape[0] < 3:
        raise ValueError("Fewer than 3 valid joint observations.")
    mu = R.mean(axis=0)
    cov = np.cov(R, rowvar=False, ddof=1)
    return mu, cov


def _cov_to_corr(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose covariance into vol vector and correlation matrix."""
    vol = np.sqrt(np.diag(cov))
    vol_outer = np.outer(vol, vol)
    corr = np.clip(cov / np.where(vol_outer > 1e-14, vol_outer, 1.0), -0.9999, 0.9999)
    np.fill_diagonal(corr, 1.0)
    return vol, corr


# ── Main calibration entry point ──────────────────────────────────────────────

def calibrate(
    strategy_series: pd.DataFrame,
    market_col: str = "vwretd",
    regime_window: int = 21,
    include_regime: bool = True,
) -> CalibrationParams:
    """Calibrate all model parameters from a strategy return series DataFrame.

    Args:
        strategy_series:  DataFrame with columns ['strat1_ret', 'strat2_ret',
                          'strat3_ret'] (and optionally market columns).
        market_col:       Column used to define the market regime.
        regime_window:    Rolling window (days) for regime vol estimate.
        include_regime:   Whether to also fit regime-conditional parameters.

    Returns:
        CalibrationParams with all fitted values.
    """
    from ..features.regime import assign_regime, build_transition_matrix

    s1 = strategy_series["strat1_ret"]
    s2 = strategy_series["strat2_ret"]
    s3 = strategy_series["strat3_ret"]

    mu_unc, cov_unc = _fit_trivariate(s1, s2, s3)
    vol_unc, corr_unc = _cov_to_corr(cov_unc)

    params = CalibrationParams(
        mu_1=float(mu_unc[0]), sigma_1=float(vol_unc[0]),
        mu_2=float(mu_unc[1]), sigma_2=float(vol_unc[1]),
        mu_3=float(mu_unc[2]), sigma_3=float(vol_unc[2]),
        rho_12=float(corr_unc[0, 1]),
        rho_13=float(corr_unc[0, 2]),
        rho_23=float(corr_unc[1, 2]),
        n_obs=int((s1.notna() & s2.notna() & s3.notna()).sum()),
        date_start=str(strategy_series.index.min().date()),
        date_end=str(strategy_series.index.max().date()),
    )

    if include_regime and market_col in strategy_series.columns:
        market = strategy_series[market_col].dropna()
        regime = assign_regime(market, window=regime_window)

        aligned = pd.concat([s1, s2, s3, regime], axis=1).dropna()
        good_mask = aligned["regime_good"].astype(bool)

        n_good = int(good_mask.sum())
        n_bad = int((~good_mask).sum())
        logger.info("Regime: %d good, %d bad days.", n_good, n_bad)
        if n_good < 10 or n_bad < 10:
            logger.warning(
                "Too few obs in one regime (%d good, %d bad) — skipping regime fit.",
                n_good, n_bad,
            )
        else:
            for regime_flag, suffix in [(True, "good"), (False, "bad")]:
                mask = good_mask if regime_flag else ~good_mask
                mu_r, cov_r = _fit_trivariate(
                    aligned.loc[mask, "strat1_ret"],
                    aligned.loc[mask, "strat2_ret"],
                    aligned.loc[mask, "strat3_ret"],
                )
                vol_r, corr_r = _cov_to_corr(cov_r)
                setattr(params, f"mu_1_{suffix}", float(mu_r[0]))
                setattr(params, f"sigma_1_{suffix}", float(vol_r[0]))
                setattr(params, f"mu_2_{suffix}", float(mu_r[1]))
                setattr(params, f"sigma_2_{suffix}", float(vol_r[1]))
                setattr(params, f"mu_3_{suffix}", float(mu_r[2]))
                setattr(params, f"sigma_3_{suffix}", float(vol_r[2]))
                setattr(params, f"rho_12_{suffix}", float(corr_r[0, 1]))
                setattr(params, f"rho_13_{suffix}", float(corr_r[0, 2]))
                setattr(params, f"rho_23_{suffix}", float(corr_r[1, 2]))

            params.n_obs_good = n_good
            params.n_obs_bad = n_bad
            params.regime_transition = build_transition_matrix(aligned["regime_good"])

    logger.info("\n%s", params.describe())
    return params


# ── Samplers ──────────────────────────────────────────────────────────────────

class GaussianSampler:
    """Draw return samples from the calibrated parametric Gaussian model.

    Can sample unconditionally or conditioned on a regime.
    """

    def __init__(self, params: CalibrationParams, rng_seed: int = 42):
        self.params = params
        self.rng = np.random.default_rng(rng_seed)

    def sample(self, n: int, regime: Optional[int] = None) -> np.ndarray:
        """Draw n samples of shape (n, 3) from N(mu, Sigma).

        Args:
            n:      Number of samples.
            regime: 1 = good, 0 = bad, None = unconditional.

        Returns:
            (n, 3) array of strategy returns.
        """
        mu = self.params.mean_vector(regime)
        cov = self.params.cov_matrix(regime)
        return self.rng.multivariate_normal(mu, cov, size=n)


class BlockBootstrapSampler:
    """Resample blocks of historical strategy returns (Künsch 1989).

    Preserves autocorrelation structure (e.g., vol clustering, regime
    persistence) that a simple i.i.d. bootstrap would destroy.
    """

    def __init__(
        self,
        strategy_series: pd.DataFrame,
        block_size: int = 20,
        rng_seed: int = 42,
    ):
        """
        Args:
            strategy_series:  DataFrame with columns strat1_ret, strat2_ret, strat3_ret.
            block_size:       Length of each resampled block in trading days.
            rng_seed:         Random seed.
        """
        cols = ["strat1_ret", "strat2_ret", "strat3_ret"]
        data = strategy_series[cols].dropna()
        self._data = data.values.astype(float)  # (T, 3)
        self._T = len(self._data)
        self.block_size = block_size
        self.rng = np.random.default_rng(rng_seed)

    def sample(self, n: int) -> np.ndarray:
        """Draw n strategy-return samples as (n, 3) array via block bootstrap.

        Blocks are drawn with replacement; each block is a contiguous slice
        of length `block_size` from the historical data.  If n is not a
        multiple of block_size the last partial block is truncated.
        """
        result = []
        while len(result) < n:
            start = int(self.rng.integers(0, self._T - self.block_size + 1))
            block = self._data[start: start + self.block_size]
            result.append(block)
        samples = np.vstack(result)[:n]
        return samples
