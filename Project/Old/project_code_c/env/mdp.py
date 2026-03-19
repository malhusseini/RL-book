"""Phase-2 discrete-action MDP for three-strategy allocation + cash.

State encoding
--------------
  No regime : wealth_index  (int)             ∈ {0, …, N_W - 1}
  Regime     : (wealth_index, regime_int)     regime_int ∈ {0=bad, 1=good}

Action encoding  (10 actions)
------------------------------
  Risk ladder — 0% cash:
    0 : Conservative      — 70% S1 / 20% S2 / 10% S3 /  0% cash
    1 : Conservative+     — 60% S1 / 25% S2 / 15% S3 /  0% cash
    2 : Moderate-Low      — 50% S1 / 30% S2 / 20% S3 /  0% cash
    3 : Moderate          — 35% S1 / 35% S2 / 30% S3 /  0% cash
    4 : Balanced          — 25% S1 / 35% S2 / 40% S3 /  0% cash
    5 : Moderate-High     — 20% S1 / 30% S2 / 50% S3 /  0% cash
    6 : Aggressive        — 15% S1 / 25% S2 / 60% S3 /  0% cash

  Cash ladder — mid-risk core with increasing cash:
    7 : Defensive-20      — 20% S1 / 28% S2 / 32% S3 / 20% cash
    8 : Defensive-40      — 15% S1 / 21% S2 / 24% S3 / 40% cash
    9 : Cash-heavy        — 10% S1 / 15% S2 / 15% S3 / 60% cash

Actions 0–6 form a continuous risk ladder from conservative (high S1) to
aggressive (high S3) with no cash.  Actions 7–9 progressively shift exposure
into the risk-free asset while keeping the risky mix at a moderate weighting,
giving the agent the option to de-risk without fully abandoning equity exposure.

Reward
------
  CRRA utility increment: R(s, a, s') = U(W') − U(W)
  where U(W) = W^{1−γ} / (1−γ)   (log utility when γ = 1).
  W' is snapped to the wealth grid to make the MDP finite.

Transaction costs
-----------------
  TC(a_curr, a_prev) = tc_rate × |a_curr − a_prev|_1 × W
  Applied on every reallocation, encouraging policy stability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from rl.distribution import Categorical
from rl.markov_decision_process import FiniteMarkovDecisionProcess

from .simulator import CalibrationParams

logger = logging.getLogger(__name__)

# ── Action table ─────────────────────────────────────────────────────────────
# Each entry: (w1, w2, w3, w_cash) summing to 1.0
# Risk ladder (actions 0–6): 0% cash, increasing S3 weight.
# Cash ladder (actions 7–9): moderate risky mix with 20/40/60% cash.

ACTIONS: Dict[int, Tuple[float, float, float, float]] = {
    0: (0.70, 0.20, 0.10, 0.00),  # Conservative
    1: (0.60, 0.25, 0.15, 0.00),  # Conservative+
    2: (0.50, 0.30, 0.20, 0.00),  # Moderate-Low
    3: (0.35, 0.35, 0.30, 0.00),  # Moderate
    4: (0.25, 0.35, 0.40, 0.00),  # Balanced
    5: (0.20, 0.30, 0.50, 0.00),  # Moderate-High
    6: (0.15, 0.25, 0.60, 0.00),  # Aggressive
    7: (0.20, 0.28, 0.32, 0.20),  # Defensive-20
    8: (0.15, 0.21, 0.24, 0.40),  # Defensive-40
    9: (0.10, 0.15, 0.15, 0.60),  # Cash-heavy
}

ACTION_NAMES: Dict[int, str] = {
    0: "Conservative",
    1: "Conservative+",
    2: "Moderate-Low",
    3: "Moderate",
    4: "Balanced",
    5: "Moderate-High",
    6: "Aggressive",
    7: "Defensive-20",
    8: "Defensive-40",
    9: "Cash-heavy",
}

RISK_FREE_DAILY = 0.0001  # ~2.5 % annual risk-free rate


# ── MDP configuration ─────────────────────────────────────────────────────────

@dataclass
class MDPConfig:
    """Hyperparameters controlling the Phase-2 MDP discretisation."""
    n_wealth_bins:     int   = 40
    wealth_min:        float = 0.5
    wealth_max:        float = 2.0
    initial_wealth:    float = 1.0
    risk_aversion:     float = 2.0
    transaction_cost:  float = 0.001  # 10 bps per unit turnover
    n_mc_samples:      int   = 3000
    use_regime:        bool  = True
    log_spaced_wealth: bool  = True
    rng_seed:          int   = 42


# ── Utility function ──────────────────────────────────────────────────────────

def crra_utility(wealth: float, gamma: float) -> float:
    """CRRA utility: U(W) = W^(1−γ)/(1−γ).  Log utility when γ = 1."""
    w = max(float(wealth), 1e-8)
    if abs(gamma - 1.0) < 1e-9:
        return float(np.log(w))
    return float(w ** (1.0 - gamma) / (1.0 - gamma))


# ── Wealth grid ───────────────────────────────────────────────────────────────

def make_wealth_grid(cfg: MDPConfig) -> np.ndarray:
    """Create the discretised wealth grid (log-spaced or linear)."""
    if cfg.log_spaced_wealth:
        return np.exp(
            np.linspace(np.log(cfg.wealth_min), np.log(cfg.wealth_max), cfg.n_wealth_bins)
        )
    return np.linspace(cfg.wealth_min, cfg.wealth_max, cfg.n_wealth_bins)


def _nearest_bin(wealth_vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Snap a vector of wealth values to the nearest grid index."""
    idx = np.searchsorted(grid, wealth_vals, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)
    left = np.maximum(idx - 1, 0)
    use_left = (idx > 0) & (
        np.abs(wealth_vals - grid[left]) <= np.abs(wealth_vals - grid[idx])
    )
    idx[use_left] = left[use_left]
    return idx.astype(int)


# ── Portfolio return sampling ─────────────────────────────────────────────────

def _sample_portfolio_returns(
    weights: Tuple[float, float, float, float],
    mu: np.ndarray,
    cov: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n portfolio return samples under a trivariate Gaussian + cash.

    Args:
        weights: (w1, w2, w3, w_cash) — must sum to 1.
        mu:      (3,) mean vector.
        cov:     (3, 3) covariance matrix.
        n:       Number of samples.
        rng:     NumPy random Generator.

    Returns:
        (n,) portfolio return samples.
    """
    w1, w2, w3, w_cash = weights
    r123 = rng.multivariate_normal(mu, cov, size=n)
    return w1 * r123[:, 0] + w2 * r123[:, 1] + w3 * r123[:, 2] + w_cash * RISK_FREE_DAILY


def _transaction_cost(
    new_weights: Tuple[float, float, float, float],
    old_weights: Tuple[float, float, float, float],
    tc_rate: float,
) -> float:
    """Proportional transaction cost = tc_rate × L1 turnover."""
    return tc_rate * sum(abs(n - o) for n, o in zip(new_weights, old_weights))


# ── Categorical distribution builder ─────────────────────────────────────────

def _make_categorical(
    W_curr: float,
    W_next_samples: np.ndarray,
    wealth_grid: np.ndarray,
    cfg: MDPConfig,
    regime_weights: Optional[Dict[int, float]] = None,
) -> Categorical:
    """Build Categorical over (next_state, reward) from Monte-Carlo wealth samples."""
    U_curr = crra_utility(W_curr, cfg.risk_aversion)
    next_bins = _nearest_bin(W_next_samples, wealth_grid)
    n_total = len(W_next_samples)

    bin_counts: Dict[int, int] = {}
    for b in next_bins:
        bin_counts[b] = bin_counts.get(b, 0) + 1

    dist: Dict = {}
    if regime_weights is None:
        for bin_idx, count in bin_counts.items():
            prob = count / n_total
            reward = crra_utility(wealth_grid[bin_idx], cfg.risk_aversion) - U_curr
            key = (int(bin_idx), float(reward))
            dist[key] = dist.get(key, 0.0) + prob
    else:
        for bin_idx, count in bin_counts.items():
            p_wealth = count / n_total
            reward = crra_utility(wealth_grid[bin_idx], cfg.risk_aversion) - U_curr
            for next_regime, p_regime in regime_weights.items():
                if p_regime == 0.0:
                    continue
                key = ((int(bin_idx), int(next_regime)), float(reward))
                dist[key] = dist.get(key, 0.0) + p_wealth * p_regime

    return Categorical(dist)


# ── Transition mapping builders ───────────────────────────────────────────────

def _build_no_regime_mapping(
    params: CalibrationParams,
    cfg: MDPConfig,
    wealth_grid: np.ndarray,
    rng: np.random.Generator,
) -> Dict:
    mu = params.mean_vector()
    cov = params.cov_matrix()

    # Baseline allocation for TC computation (Balanced action = index 4)
    baseline_weights = ACTIONS[4]

    mapping: Dict = {}
    for w_idx in range(cfg.n_wealth_bins):
        W = wealth_grid[w_idx]
        action_map: Dict = {}
        for a_idx, weights in ACTIONS.items():
            tc = _transaction_cost(weights, baseline_weights, cfg.transaction_cost)
            r_port = _sample_portfolio_returns(weights, mu, cov, cfg.n_mc_samples, rng)
            W_next = np.maximum(W * (1.0 + r_port) - tc * W, 1e-8)
            action_map[a_idx] = _make_categorical(W, W_next, wealth_grid, cfg)
        mapping[w_idx] = action_map
    return mapping


def _build_regime_mapping(
    params: CalibrationParams,
    cfg: MDPConfig,
    wealth_grid: np.ndarray,
    rng: np.random.Generator,
) -> Dict:
    trans = params.regime_transition
    baseline_weights = ACTIONS[4]  # Balanced action = index 4

    mapping: Dict = {}
    for w_idx in range(cfg.n_wealth_bins):
        W = wealth_grid[w_idx]
        for regime in (0, 1):
            state = (w_idx, regime)
            mu = params.mean_vector(regime)
            cov = params.cov_matrix(regime)
            regime_weights = {
                0: float(trans[regime, 0]),
                1: float(trans[regime, 1]),
            }
            action_map: Dict = {}
            for a_idx, weights in ACTIONS.items():
                tc = _transaction_cost(weights, baseline_weights, cfg.transaction_cost)
                r_port = _sample_portfolio_returns(weights, mu, cov, cfg.n_mc_samples, rng)
                W_next = np.maximum(W * (1.0 + r_port) - tc * W, 1e-8)
                action_map[a_idx] = _make_categorical(
                    W, W_next, wealth_grid, cfg, regime_weights=regime_weights
                )
            mapping[state] = action_map
    return mapping


# ── Public builder ────────────────────────────────────────────────────────────

def build_mdp(
    params: CalibrationParams,
    cfg: Optional[MDPConfig] = None,
) -> Tuple[FiniteMarkovDecisionProcess, np.ndarray, MDPConfig]:
    """Build the FiniteMarkovDecisionProcess from calibrated parameters.

    Args:
        params:  CalibrationParams from calibrate().
        cfg:     MDPConfig (uses defaults if None).

    Returns:
        mdp:          FiniteMarkovDecisionProcess.
        wealth_grid:  np.ndarray of wealth values (length n_wealth_bins).
        cfg:          The MDPConfig actually used.
    """
    if cfg is None:
        cfg = MDPConfig()
    rng = np.random.default_rng(cfg.rng_seed)
    wealth_grid = make_wealth_grid(cfg)
    use_regime = cfg.use_regime and params.has_regime

    if use_regime:
        logger.info(
            "Building MDP with 2-state regime: %d × 2 = %d states, %d actions.",
            cfg.n_wealth_bins, cfg.n_wealth_bins * 2, len(ACTIONS),
        )
        mapping = _build_regime_mapping(params, cfg, wealth_grid, rng)
    else:
        logger.info(
            "Building MDP without regime: %d states, %d actions.",
            cfg.n_wealth_bins, len(ACTIONS),
        )
        mapping = _build_no_regime_mapping(params, cfg, wealth_grid, rng)

    mdp = FiniteMarkovDecisionProcess(mapping)
    logger.info("MDP built: %d non-terminal states.", len(mdp.non_terminal_states))
    return mdp, wealth_grid, cfg
