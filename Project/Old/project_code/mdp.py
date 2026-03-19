"""Build the Phase 2 FiniteMarkovDecisionProcess for multi-strategy allocation.

State encoding
--------------
  No regime : wealth_index  (int)                   ∈ {0, …, N_W - 1}
  Regime     : (wealth_index, regime_int)            where regime_int ∈ {0=bad, 1=good}

Action encoding
---------------
  0 : Conservative  — 80 % Strategy 1 / 20 % Strategy 2 / 0 % cash
  1 : Balanced       — 50 % / 50 % / 0 %
  2 : Aggressive     — 20 % / 80 % / 0 %

Reward
------
  CRRA utility increment:  R(s, a, s') = U(W') - U(W)
  where U(W) = W^(1-γ) / (1-γ)   (log utility when γ = 1)
  W' is snapped to the wealth grid to make the MDP finite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from rl.distribution import Categorical
from rl.markov_decision_process import FiniteMarkovDecisionProcess

from .calibration import CalibrationParams

logger = logging.getLogger(__name__)


# ── Action table ────────────────────────────────────────────────────────────

# Keys: action index.  Values: (frac_strat1, frac_strat2, frac_cash)
ACTIONS: Dict[int, Tuple[float, float, float]] = {
    0: (0.80, 0.20, 0.00),   # Conservative
    1: (0.50, 0.50, 0.00),   # Balanced
    2: (0.20, 0.80, 0.00),   # Aggressive
}

ACTION_NAMES: Dict[int, str] = {
    0: "Conservative",
    1: "Balanced",
    2: "Aggressive",
}

RISK_FREE_DAILY = 0.0001   # ~2.5 % annual, held in cash


# ── MDP configuration ────────────────────────────────────────────────────────

@dataclass
class MDPConfig:
    """Hyperparameters for the Phase 2 MDP."""
    n_wealth_bins:   int   = 40       # number of discrete wealth grid points
    wealth_min:      float = 0.5      # minimum wealth (fraction of W₀)
    wealth_max:      float = 2.0      # maximum wealth (fraction of W₀)
    initial_wealth:  float = 1.0      # W₀ used for plotting reference
    risk_aversion:   float = 2.0      # CRRA γ  (1.0 → log utility)
    transaction_cost: float = 0.0     # proportional cost per unit turnover
    n_mc_samples:    int   = 3000     # Monte Carlo draws for transition discretization
    use_regime:      bool  = True     # include 2-state regime in state space
    log_spaced_wealth: bool = True     # log-spaced grid (better coverage near wealth_min)
    rng_seed:        int   = 42


# ── Utility function ────────────────────────────────────────────────────────

def crra_utility(wealth: float, gamma: float) -> float:
    """CRRA utility U(W) = W^(1-γ)/(1-γ); log utility when γ = 1."""
    w = max(float(wealth), 1e-8)
    if abs(gamma - 1.0) < 1e-9:
        return float(np.log(w))
    return float(w ** (1.0 - gamma) / (1.0 - gamma))


# ── Wealth grid ─────────────────────────────────────────────────────────────

def make_wealth_grid(cfg: MDPConfig) -> np.ndarray:
    if cfg.log_spaced_wealth:
        return np.exp(
            np.linspace(np.log(cfg.wealth_min), np.log(cfg.wealth_max), cfg.n_wealth_bins)
        )
    return np.linspace(cfg.wealth_min, cfg.wealth_max, cfg.n_wealth_bins)


def _nearest_bin(wealth_vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Snap an array of continuous wealth values to the nearest grid index."""
    # searchsorted gives insertion points; check both neighbors
    idx = np.searchsorted(grid, wealth_vals, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)
    left = np.maximum(idx - 1, 0)
    use_left = (idx > 0) & (
        np.abs(wealth_vals - grid[left]) <= np.abs(wealth_vals - grid[idx])
    )
    idx[use_left] = left[use_left]
    return idx.astype(int)


# ── Portfolio return sampling ────────────────────────────────────────────────

def _sample_portfolio_returns(
    a1: float, a2: float, a_cash: float,
    mu1: float, sig1: float,
    mu2: float, sig2: float,
    rho: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n portfolio return samples from a bivariate Gaussian."""
    cov = np.array([
        [sig1 ** 2,       rho * sig1 * sig2],
        [rho * sig1 * sig2, sig2 ** 2      ],
    ])
    r12 = rng.multivariate_normal([mu1, mu2], cov, size=n)
    return a1 * r12[:, 0] + a2 * r12[:, 1] + a_cash * RISK_FREE_DAILY


# ── Transition distribution builders ────────────────────────────────────────

def _make_categorical(
    W_curr: float,
    W_next_samples: np.ndarray,
    wealth_grid: np.ndarray,
    cfg: MDPConfig,
    regime_weights: Optional[Dict[int, float]] = None,
) -> Categorical:
    """
    Build a Categorical distribution over (next_state, reward) pairs.

    For each next-wealth bin, the reward is  U(grid_value) - U(W_curr),
    which is deterministic given the grid.  If regime_weights is provided,
    each wealth bin is replicated across next-regime values (bad=0, good=1)
    with the given regime transition probabilities.

    Args:
        W_curr:          Current wealth (scalar).
        W_next_samples:  Array of n MC next-wealth values.
        wealth_grid:     Wealth grid array.
        cfg:             MDPConfig.
        regime_weights:  {next_regime_int: probability} for regime transitions;
                         None for no-regime MDP.

    Returns:
        Categorical over (next_state, reward) tuples where next_state is
        an int (no regime) or Tuple[int, int] (with regime).
    """
    U_curr = crra_utility(W_curr, cfg.risk_aversion)

    # Snap all MC draws to grid
    next_bins = _nearest_bin(W_next_samples, wealth_grid)

    # Count frequency per bin (and compute per-bin mean reward)
    n_total = len(W_next_samples)
    bin_counts: Dict[int, int] = {}
    for b in next_bins:
        bin_counts[b] = bin_counts.get(b, 0) + 1

    dist: Dict = {}

    if regime_weights is None:
        # No-regime state: next_state is just the wealth index
        for bin_idx, count in bin_counts.items():
            prob = count / n_total
            reward = crra_utility(wealth_grid[bin_idx], cfg.risk_aversion) - U_curr
            key = (int(bin_idx), float(reward))
            dist[key] = dist.get(key, 0.0) + prob
    else:
        # Regime state: next_state = (wealth_index, regime_int)
        for bin_idx, count in bin_counts.items():
            p_wealth = count / n_total
            reward = crra_utility(wealth_grid[bin_idx], cfg.risk_aversion) - U_curr
            for next_regime, p_regime in regime_weights.items():
                if p_regime == 0.0:
                    continue
                key = ((int(bin_idx), int(next_regime)), float(reward))
                dist[key] = dist.get(key, 0.0) + p_wealth * p_regime

    return Categorical(dist)


def _build_no_regime_mapping(
    params: CalibrationParams,
    cfg: MDPConfig,
    wealth_grid: np.ndarray,
    rng: np.random.Generator,
) -> Dict:
    """Build the full transition mapping for the no-regime MDP."""
    mapping: Dict[int, Dict[int, Categorical]] = {}

    for w_idx in range(cfg.n_wealth_bins):
        W = wealth_grid[w_idx]
        action_map: Dict[int, Categorical] = {}

        for a_idx, (a1, a2, a_cash) in ACTIONS.items():
            W_next = W * (
                1.0 + _sample_portfolio_returns(
                    a1, a2, a_cash,
                    params.mu_1, params.sigma_1,
                    params.mu_2, params.sigma_2,
                    params.rho,
                    cfg.n_mc_samples, rng,
                )
            )
            action_map[a_idx] = _make_categorical(W, W_next, wealth_grid, cfg)

        mapping[w_idx] = action_map

    return mapping


def _build_regime_mapping(
    params: CalibrationParams,
    cfg: MDPConfig,
    wealth_grid: np.ndarray,
    rng: np.random.Generator,
) -> Dict:
    """Build the full transition mapping for the 2-state regime MDP."""
    trans = params.regime_transition   # shape (2, 2); row = current regime (0=bad,1=good)
    mapping: Dict[Tuple[int, int], Dict[int, Categorical]] = {}

    for w_idx in range(cfg.n_wealth_bins):
        W = wealth_grid[w_idx]

        for regime in (0, 1):   # 0 = bad, 1 = good
            state = (w_idx, regime)

            # Select regime-conditional parameters
            if regime == 1:
                mu1, sig1 = params.mu_1_good, params.sigma_1_good
                mu2, sig2 = params.mu_2_good, params.sigma_2_good
                rho        = params.rho_good
            else:
                mu1, sig1 = params.mu_1_bad,  params.sigma_1_bad
                mu2, sig2 = params.mu_2_bad,  params.sigma_2_bad
                rho        = params.rho_bad

            # Regime transition weights for next step
            regime_weights = {
                0: float(trans[regime, 0]),
                1: float(trans[regime, 1]),
            }

            action_map: Dict[int, Categorical] = {}
            for a_idx, (a1, a2, a_cash) in ACTIONS.items():
                W_next = W * (
                    1.0 + _sample_portfolio_returns(
                        a1, a2, a_cash,
                        mu1, sig1, mu2, sig2, rho,
                        cfg.n_mc_samples, rng,
                    )
                )
                action_map[a_idx] = _make_categorical(
                    W, W_next, wealth_grid, cfg, regime_weights=regime_weights
                )

            mapping[state] = action_map

    return mapping


# ── Public builder ───────────────────────────────────────────────────────────

def build_mdp(
    params: CalibrationParams,
    cfg: Optional[MDPConfig] = None,
) -> Tuple[FiniteMarkovDecisionProcess, np.ndarray, MDPConfig]:
    """
    Build the FiniteMarkovDecisionProcess and wealth grid.

    Args:
        params:  CalibrationParams from calibration.calibrate().
        cfg:     MDPConfig; a default MDPConfig() is used if None.

    Returns:
        mdp:          FiniteMarkovDecisionProcess
        wealth_grid:  np.ndarray of wealth values (length n_wealth_bins)
        cfg:          The MDPConfig actually used
    """
    if cfg is None:
        cfg = MDPConfig()

    rng = np.random.default_rng(cfg.rng_seed)
    wealth_grid = make_wealth_grid(cfg)

    use_regime = cfg.use_regime and params.has_regime

    if use_regime:
        n_states = cfg.n_wealth_bins * 2
        logger.info(
            "Building MDP with 2-state regime: %d × 2 = %d states, %d actions.",
            cfg.n_wealth_bins, n_states, len(ACTIONS),
        )
        mapping = _build_regime_mapping(params, cfg, wealth_grid, rng)
    else:
        logger.info(
            "Building MDP without regime: %d states, %d actions.",
            cfg.n_wealth_bins, len(ACTIONS),
        )
        mapping = _build_no_regime_mapping(params, cfg, wealth_grid, rng)

    mdp = FiniteMarkovDecisionProcess(mapping)
    logger.info(
        "MDP built: %d non-terminal states.", len(mdp.non_terminal_states)
    )
    return mdp, wealth_grid, cfg
