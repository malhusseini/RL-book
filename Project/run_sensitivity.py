# === CELL 4 ===
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must be set before importing pyplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

# ── Strategy index constants ──────────────────────────────────────────────
VFSTX   = 0
VBMFX   = 1
VFINX   = 2

# ── Market regime constants ───────────────────────────────────────────────
BULL    = 0
NEUTRAL = 1
BEAR    = 2

# ══════════════════════════════════════════════════════════════════════════
# CONFIG — all hyperparameters in one place; modify here to re-run
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Horizon ──────────────────────────────────────────────────────────
    start_age:      int   = 30
    retirement_age: int   = 65
    birth_year:     int   = 1960

    # ── Finances ─────────────────────────────────────────────────────────
    initial_wealth:      float = 100_000.0
    annual_contribution: float = 10_000.0
    wealth_min:          float = 1_000.0       # floor to avoid log(0)
    wealth_max:          float = 5_000_000.0   # cap for state normalization

    # ── Switching cost ────────────────────────────────────────────────────
    # Applied when the chosen action differs from the previous year.
    switching_cost_pct: float = 0.005  # 0.5% of wealth

    # ── Risk aversion (CRRA) ─────────────────────────────────────────────
    gamma_0:     float = 2.0   # risk aversion at age 30
    gamma_slope: float = 0.03  # incremental increase per year of age
    w_ref:       float = 1_000_000.0  # CRRA normalization reference ($1M target)

    # ── Intermediate reward shaping ──────────────────────────────────────
    # Scaled log-wealth change; tiny relative to terminal reward.
    # Acts as a training aid; does NOT change the optimal policy.
    intermediate_alpha: float = 0.01

    # ── Market regime transition matrix ──────────────────────────────────
    # Entry [i, j] = probability of moving from regime i to regime j.
    # Row order: [bull, neutral, bear]
    regime_transition: np.ndarray = field(default_factory=lambda: np.array([
        [0.80, 0.15, 0.05],  # bull  -> bull / neutral / bear
        [0.20, 0.60, 0.20],  # neutral -> ...
        [0.05, 0.25, 0.70],  # bear  -> ...
    ]))

    # ── Return model ─────────────────────────────────────────────────────
    # Stylized regime-dependent normal distributions (annual nominal returns).
    # Initial values are overwritten by empirical calibration from real
    # VFINX/VBMFX/VFSTX returns (1990-2024). See Cell 6.
    # Format: {action: {regime: (mean, std)}}
    return_params: Dict = field(default_factory=lambda: {
        VFSTX: {
            BULL:    (0.045, 0.005),  # High-rate environment in bull
            NEUTRAL: (0.030, 0.005),  # Mid-cycle Fed rate
            BEAR:    (0.010, 0.005),  # Low-rate recession environment
        },
        VBMFX: {
            BULL:    (0.020, 0.045),  # Modest; rates may rise in bull
            NEUTRAL: (0.040, 0.045),  # Normal yield-driven return
            BEAR:    (0.100, 0.055),  # Flight-to-safety; price appreciation
        },
        VFINX: {
            BULL:    (0.270, 0.150),  # Strong equity bull market
            NEUTRAL: (0.100, 0.150),  # Average equity year
            BEAR:    (-0.150, 0.190), # Bear market drawdown
        },
    })

    # ── DQN hyperparameters ───────────────────────────────────────────────
    n_episodes:          int   = 5000
    batch_size:          int   = 128
    replay_buffer_size:  int   = 50_000
    lr:                  float = 1e-3
    gamma_discount:      float = 0.99   # RL discount factor (not CRRA gamma)
    eps_start:           float = 1.0
    eps_end:             float = 0.05
    eps_decay:           float = 0.995  # multiplicative per-episode decay
    target_update_freq:  int   = 500    # episodes between hard target-net updates
    hidden_size:         int   = 64

    # ── Evaluation ────────────────────────────────────────────────────────
    n_eval_episodes: int = 5000

    # ── Strategy and regime names ─────────────────────────────────────────
    strategy_names: List[str] = field(default_factory=lambda: [
        "VFSTX", "VBMFX", "VFINX"
    ])
    regime_names: List[str] = field(default_factory=lambda: [
        "Bull", "Neutral", "Bear"
    ])


cfg = Config()
T_STEPS   = cfg.retirement_age - cfg.start_age  # 35
STATE_DIM = 5   # norm_age + norm_log_wealth + 3 one-hot regime
N_ACTIONS = 3

print('Configuration loaded.')
print(f'  Horizon           : {T_STEPS} years (ages {cfg.start_age}-{cfg.retirement_age})')
print(f'  State dimension   : {STATE_DIM}')
print(f'  Actions           : {N_ACTIONS}  {cfg.strategy_names}')
print(f'  Regimes           : {cfg.regime_names}')
print(f'  Initial wealth    : ${cfg.initial_wealth:,.0f}')
print(f'  Annual contrib.   : ${cfg.annual_contribution:,.0f}/yr')
print(f'  Switching cost    : {cfg.switching_cost_pct:.1%} of wealth if action changes')
print(f'  gamma_0           : {cfg.gamma_0}  |  gamma_slope: {cfg.gamma_slope}/yr')
print(f'  gamma at age 65   : {cfg.gamma_0 + cfg.gamma_slope * T_STEPS:.3f}')
print()
print('Return model summary (mean +/- std per regime):')
for a in range(N_ACTIONS):
    print(f'  {cfg.strategy_names[a]:<16}', end='')
    for r in range(3):
        mu, sig = cfg.return_params[a][r]
        print(f'  {cfg.regime_names[r]}: {mu:+.1%}+/-{sig:.1%}', end='')
    print()

# === CELL 5 ===
# ─────────────────────────────────────────────────────────────────────────────
# Historical Market Data — VFINX, VBMFX, VFSTX annual returns via yfinance
# ─────────────────────────────────────────────────────────────────────────────
# Tickers:
#   VFINX — Vanguard Total Stock Mkt Index (Inv) (equity, inception 1992)
#   VBMFX — Vanguard Total Bond Market Index      (bonds, inception 1987)
#   VFSTX — Vanguard Short-Term Investment-Grade  (savings proxy, inception 1982)
#
# Annual returns are computed from calendar-year-end adjusted close prices.
# The three series are aligned to their common overlap (1990–2024, 35 years).

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call([__import__('sys').executable, '-m', 'pip',
                           'install', 'yfinance', '-q'])
    import yfinance as yf

print('Downloading historical annual returns (1990–2024)…')
_dl_p3 = {}
for _ticker in ['VFINX', 'VBMFX', 'VFSTX']:
    _raw   = yf.download(_ticker, start='1989-01-01', end='2024-12-31',
                         auto_adjust=True, progress=False)
    _close = _raw['Close'].squeeze()
    _dl_p3[_ticker] = _close.resample('YE').last().pct_change().dropna()

_hist_df_p3  = pd.DataFrame(_dl_p3).dropna()
hist_years   = _hist_df_p3.index.year.tolist()
hist_vfinx_ret = _hist_df_p3['VFINX'].values  # equity  (VFINX)
hist_vbmfx_ret = _hist_df_p3['VBMFX'].values  # bonds   (VBMFX)
hist_vfstx_ret = _hist_df_p3['VFSTX'].values  # savings proxy (VFSTX)

print(f'  Aligned sample: {len(hist_years)} annual observations '
      f'({hist_years[0]}–{hist_years[-1]})')
print()
print(f"  {'Asset':<14}  {'Mean':>7}  {'Std':>7}  {'Min':>8}  {'Max':>8}")
print('  ' + '─' * 52)
for _lbl, _arr in [('VFINX (equity)',   hist_vfinx_ret),
                   ('VBMFX (bonds)',    hist_vbmfx_ret),
                   ('VFSTX (savings)',  hist_vfstx_ret)]:
    print(f'  {_lbl:<14}  {_arr.mean():>7.2%}  {_arr.std():>7.2%}'
          f'  {_arr.min():>8.2%}  {_arr.max():>8.2%}')
print()
print('  VFSTX = Vanguard Short-Term Investment-Grade Fund (savings rate proxy).')
print('  These returns are used to (1) calibrate cfg.regime_transition /  ')
print('  cfg.return_params, and (2) generate empirical bootstrap baselines.')

# === CELL 6 ===
# ─────────────────────────────────────────────────────────────────────────────
# Regime Calibration — fit Markov regime model to real VFINX return history
# ─────────────────────────────────────────────────────────────────────────────
# Market regimes are labeled using VFINX annual-return thresholds:
#   Bull   : VFINX annual return > +15 %
#   Neutral: −5 % ≤ VFINX ≤ +15 %
#   Bear   : VFINX annual return < −5 %
#
# The regime transition matrix and per-regime (μ, σ) for each asset are
# estimated from the 1990–2024 sample.  Laplace smoothing (+ 0.5 per cell)
# prevents zero-probability transitions that arise with a short history.

_vti = hist_vfinx_ret
_ief = hist_vbmfx_ret
_shy = hist_vfstx_ret

# ── Label each year's regime ─────────────────────────────────────────────
_bull_mask    = _vti >  0.15
_neutral_mask = (_vti >= -0.05) & (_vti <=  0.15)
_bear_mask    = _vti < -0.05
_regime_seq   = np.where(_bull_mask, BULL, np.where(_bear_mask, BEAR, NEUTRAL))

# ── Empirical transition matrix with Laplace smoothing ───────────────────
_trans_counts = np.zeros((3, 3))
for _t in range(len(_regime_seq) - 1):
    _trans_counts[_regime_seq[_t], _regime_seq[_t + 1]] += 1
_trans_smooth   = _trans_counts + 0.5
_emp_transition = _trans_smooth / _trans_smooth.sum(axis=1, keepdims=True)

# ── Per-regime (mean, std) for each asset ────────────────────────────────
_orig_params = {k: dict(v) for k, v in cfg.return_params.items()}
_emp_params  = {}
for _action, _arr in [(VFSTX, _shy), (VBMFX, _ief), (VFINX, _vti)]:
    _emp_params[_action] = {}
    for _regime, _mask in [(BULL, _bull_mask), (NEUTRAL, _neutral_mask), (BEAR, _bear_mask)]:
        _sub = _arr[_mask]
        if len(_sub) > 1:
            _mu  = float(_sub.mean())
            _sig = float(_sub.std(ddof=1))
        else:
            _mu  = float(_arr.mean())
            _sig = float(_arr.std(ddof=1))
        _emp_params[_action][_regime] = (_mu, _sig)

# ── Update cfg in-place (before RetirementEnv is instantiated for training) ──
cfg.regime_transition = _emp_transition
cfg.return_params     = _emp_params

# ── Summary printout ─────────────────────────────────────────────────────
_n_b, _n_n, _n_br = _bull_mask.sum(), _neutral_mask.sum(), _bear_mask.sum()
print("Regime calibration complete — cfg.return_params and cfg.regime_transition updated.")
print()
print(f"  Regime counts (1990–2024):  Bull={_n_b}, Neutral={_n_n}, Bear={_n_br}")
print()
print("  Empirical transition matrix  [from → to: Bull / Neutral / Bear]")
_prior_rows = [[0.80, 0.15, 0.05], [0.20, 0.60, 0.20], [0.05, 0.25, 0.70]]
for _i, _rname in enumerate(cfg.regime_names):
    _row = _emp_transition[_i]
    _pr  = _prior_rows[_i]
    print(f"    {_rname:8s}: [{_row[0]:.2f}, {_row[1]:.2f}, {_row[2]:.2f}]"
          f"  (was [{_pr[0]:.2f}, {_pr[1]:.2f}, {_pr[2]:.2f}])")
print()
print("  Return params per regime  (mean ± std):")
for _a in range(N_ACTIONS):
    print(f"    {cfg.strategy_names[_a]:<16}", end='')
    for _r in range(3):
        _mu, _sig = cfg.return_params[_a][_r]
        _om, _os  = _orig_params[_a][_r]
        print(f"  {cfg.regime_names[_r]}: {_mu:+.1%}±{_sig:.1%} (was {_om:+.1%}±{_os:.1%})", end='')
    print()
print()
print("  DQN training will use these empirically-calibrated parameters.")

# === CELL 8 ===
# ── Must define utility helpers before using them in the environment ─────
# (gamma_from_age and crra_utility are defined in the next code cell;
#  they are referenced here via forward call — fine since Python resolves
#  names at call time, not definition time.)


class RetirementEnv:
    """
    Finite-horizon retirement investing environment.

    State vector (5-D):
        [norm_age, norm_log_wealth, is_bull, is_neutral, is_bear]

    Actions:
        0 = VFSTX  |  1 = VBMFX  |  2 = VFINX

    Episode:
        Starts at age 30, ends at age 65 (35 annual steps).

    Wealth dynamics:
        w_{t+1} = max( (w_t + contribution) * (1 + return) - switch_cost, w_min )
    """

    def __init__(self, cfg: Config, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        # Internal state (set by reset())
        self.age         = cfg.start_age
        self.wealth      = cfg.initial_wealth
        self.regime      = NEUTRAL
        self.prev_action = None
        self.step_count  = 0

    def reset(self, start_regime: Optional[int] = None) -> np.ndarray:
        """Reset to age 30 and return the initial state vector."""
        self.age         = self.cfg.start_age
        self.wealth      = self.cfg.initial_wealth
        self.regime      = int(self.rng.integers(0, 3)) if start_regime is None else start_regime
        self.prev_action = None
        self.step_count  = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take one year-long step.

        Returns:
            next_state : np.ndarray (5-D)
            reward     : float
            done       : bool  (True at age 65)
        """
        assert 0 <= action < N_ACTIONS, f'Invalid action {action}'

        # ── Switching cost ────────────────────────────────────────────────
        if self.prev_action is not None and action != self.prev_action:
            switch_cost = self.cfg.switching_cost_pct * self.wealth
        else:
            switch_cost = 0.0

        # ── Sample annual return (regime-dependent clipped normal) ────────
        mu, sig = self.cfg.return_params[action][self.regime]
        annual_return = float(np.clip(self.rng.normal(mu, sig), -0.60, 0.60))

        # ── Wealth update ─────────────────────────────────────────────────
        prev_wealth = self.wealth
        self.wealth = (self.wealth + self.cfg.annual_contribution) * (1.0 + annual_return)
        self.wealth -= switch_cost
        self.wealth  = max(self.wealth, self.cfg.wealth_min)

        # ── Regime transition ─────────────────────────────────────────────
        self.regime = int(self.rng.choice(3, p=self.cfg.regime_transition[self.regime]))

        # ── Advance age and step counter ──────────────────────────────────
        self.prev_action = action
        self.age        += 1
        self.step_count += 1
        done = (self.age >= self.cfg.retirement_age)

        # ── Reward ────────────────────────────────────────────────────────
        if done:
            # Terminal reward: CRRA utility of final wealth at retirement
            reward = crra_utility(self.wealth, gamma_from_age(self.cfg.retirement_age))
        else:
            # Small intermediate shaped reward: log-wealth change
            # This is a training-stability aid, NOT the true economic objective.
            # The alpha=0.01 scale keeps it much smaller than the terminal utility.
            delta_log = np.log(max(self.wealth, 1.0)) - np.log(max(prev_wealth, 1.0))
            reward = self.cfg.intermediate_alpha * delta_log

        return self._get_state(), reward, done

    def _get_state(self) -> np.ndarray:
        """Build the 5-D normalized state vector."""
        norm_age   = (self.age - self.cfg.start_age) / T_STEPS
        log_w      = np.log10(max(self.wealth, self.cfg.wealth_min))
        norm_log_w = log_w / np.log10(self.cfg.wealth_max)
        regime_onehot = np.zeros(3, dtype=np.float32)
        regime_onehot[self.regime] = 1.0
        return np.array([norm_age, norm_log_w, *regime_onehot], dtype=np.float32)


# ── Sanity check ──────────────────────────────────────────────────────────
# (Runs after utility functions are defined in the next cell.)
print('RetirementEnv class defined.')
print('  State vector shape: 5-D')
print('  [norm_age, norm_log_wealth, is_bull, is_neutral, is_bear]')

# === CELL 10 ===
def gamma_from_age(age: int) -> float:
    """CRRA risk aversion: increases linearly with age."""
    return cfg.gamma_0 + cfg.gamma_slope * (age - cfg.start_age)


def crra_utility(w: float, gamma: float) -> float:
    """
    Normalized CRRA utility:  U(w; gamma) = ((w/w_ref)^(1-gamma) - 1) / (1-gamma)

    Normalizing by w_ref prevents numerical saturation for large wealth values.
    If gamma == 1 (log utility):  U = log(w/w_ref).
    """
    w_norm = max(float(w), 1e-6) / cfg.w_ref
    if abs(gamma - 1.0) < 1e-9:
        return float(np.log(w_norm))
    return float((w_norm ** (1.0 - gamma) - 1.0) / (1.0 - gamma))


def terminal_utility(wealth: float) -> float:
    """Convenience wrapper: CRRA utility at retirement age (age 65)."""
    return crra_utility(wealth, gamma_from_age(cfg.retirement_age))


# ── Now run the environment sanity check ──────────────────────────────────
env_test = RetirementEnv(cfg, seed=0)
s0 = env_test.reset(start_regime=BULL)
print(f'Initial state : age={env_test.age}, wealth=${env_test.wealth:,.0f},'
      f' regime={cfg.regime_names[env_test.regime]}')
print(f'State vector  : {s0}')
s1, r1, done1 = env_test.step(VFINX)
print(f'After VFINX step: age={env_test.age}, wealth=${env_test.wealth:,.0f},'
      f' reward={r1:.6f}, done={done1}')
print(f'Next state    : {s1}')
print()
print('Risk aversion gamma(age):')
for age in range(30, 66, 5):
    print(f'  age {age:2d}: gamma = {gamma_from_age(age):.3f}')
print()
print(f'CRRA utility at retirement (gamma={gamma_from_age(65):.2f}):')
for w in [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]:
    print(f'  ${w:>9,.0f}: U = {terminal_utility(w):>8.4f}')

# === CELL 12 ===
def policy_always_vfstx(state: np.ndarray, age: int) -> int:
    return VFSTX

def policy_always_vbmfx(state: np.ndarray, age: int) -> int:
    return VBMFX

def policy_always_vfinx(state: np.ndarray, age: int) -> int:
    return VFINX

def policy_random(state: np.ndarray, age: int) -> int:
    return np.random.randint(0, N_ACTIONS)

def policy_glide_path(state: np.ndarray, age: int) -> int:
    """
    Simple target-date fund glide path:
      ages 30-44  -> VFINX  (aggressive growth)
      ages 45-54  -> VBMFX  (de-risking)
      ages 55-64  -> VFSTX  (capital preservation)
    """
    if age < 45:
        return VFINX
    elif age < 55:
        return VBMFX
    else:
        return VFSTX


BASELINES = {
    "Always VFSTX":  policy_always_vfstx,
    "Always VBMFX":  policy_always_vbmfx,
    "Always VFINX":  policy_always_vfinx,
    "Glide Path":    policy_glide_path,
    "Random":        policy_random,
}

print('Baseline policies defined:', list(BASELINES.keys()))

# === CELL 14 ===
class QNetwork(nn.Module):
    """
    MLP approximating Q(state, action) for all actions simultaneously.

    Input  : 5-D normalized state vector
    Output : 3 Q-values (one per action)
    """
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """
    Circular experience replay buffer.
    Stores (state, action, reward, next_state, done) tuples.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(np.array(actions),     dtype=torch.long),
            torch.tensor(np.array(rewards),     dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones),       dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def select_action(state: np.ndarray, q_net: QNetwork, epsilon: float) -> int:
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(N_ACTIONS)  # explore
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return int(q_net(s).argmax(dim=1).item())  # exploit


def optimize_model(
    q_net: QNetwork,
    target_net: QNetwork,
    replay_buffer: ReplayBuffer,
    optimizer: optim.Optimizer,
    cfg: Config,
) -> Optional[float]:
    """
    One gradient update using a minibatch from the replay buffer.

    DQN target:  y = r + gamma * max_a' Q_target(s', a') * (1 - done)
    Loss       :  MSE(Q(s, a), y)

    Returns the loss value, or None if the buffer is not yet large enough.
    """
    if len(replay_buffer) < cfg.batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(cfg.batch_size)

    # Q-values for the actions that were actually taken
    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q-values — no gradient through target network
    with torch.no_grad():
        max_next_q = target_net(next_states).max(dim=1).values
        targets = rewards + cfg.gamma_discount * max_next_q * (1.0 - dones)

    loss = nn.MSELoss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)  # gradient clipping
    optimizer.step()
    return loss.item()


print('QNetwork, ReplayBuffer, select_action, optimize_model — all defined.')
# Quick architecture summary
demo_net = QNetwork(STATE_DIM, N_ACTIONS, hidden=cfg.hidden_size)
total_params = sum(p.numel() for p in demo_net.parameters())
print(f'QNetwork total parameters: {total_params:,}')

# === CELL 16 ===
def train_dqn(cfg: Config):
    """
    Main DQN training loop.

    Returns:
        q_net           : trained Q-network (policy network)
        episode_rewards : list of total reward per episode
        losses          : list of per-optimization-step losses
    """
    env = RetirementEnv(cfg, seed=SEED)

    q_net      = QNetwork(STATE_DIM, N_ACTIONS, hidden=cfg.hidden_size)
    target_net = QNetwork(STATE_DIM, N_ACTIONS, hidden=cfg.hidden_size)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer     = optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay_buffer = ReplayBuffer(cfg.replay_buffer_size)

    epsilon         = cfg.eps_start
    episode_rewards = []
    losses          = []

    for episode in range(cfg.n_episodes):
        state     = env.reset()
        ep_reward = 0.0
        done      = False

        while not done:
            action                        = select_action(state, q_net, epsilon)
            next_state, reward, done      = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state     = next_state
            ep_reward += reward

            loss_val = optimize_model(q_net, target_net, replay_buffer, optimizer, cfg)
            if loss_val is not None:
                losses.append(loss_val)

        # Multiplicative epsilon decay after each episode
        epsilon = max(cfg.eps_end, epsilon * cfg.eps_decay)

        # Hard copy to target network periodically
        if (episode + 1) % cfg.target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        episode_rewards.append(ep_reward)

        if (episode + 1) % 500 == 0:
            recent = np.mean(episode_rewards[-200:])
            print(f'  Episode {episode+1:>5}/{cfg.n_episodes} | '
                  f'eps={epsilon:.3f} | '
                  f'mean reward (last 200)={recent:.4f} | '
                  f'buffer={len(replay_buffer):,}')

    print('Training complete.')
    return q_net, episode_rewards, losses


print('Starting DQN training ...')
print(f'  Episodes       : {cfg.n_episodes:,}')
print(f'  Batch size     : {cfg.batch_size}')
print(f'  Replay capacity: {cfg.replay_buffer_size:,}')
print(f'  Learning rate  : {cfg.lr}')
print(f'  Target update  : every {cfg.target_update_freq} episodes')
print(f'  Epsilon decay  : {cfg.eps_start} -> {cfg.eps_end} (x{cfg.eps_decay}/ep)')
print()
q_net, episode_rewards, losses = train_dqn(cfg)

# === CELL 19 ===
def run_evaluation(policy_fn, n_episodes: int, seed: int = 0) -> dict:
    """
    Evaluate a policy over n_episodes and collect statistics.

    Returns a dict with terminal wealth distribution, utility, action frequencies,
    and 200 stored wealth trajectories.
    """
    env = RetirementEnv(cfg, seed=seed)
    terminal_wealths = []
    terminal_utils   = []
    action_by_age    = {age: [] for age in range(cfg.start_age, cfg.retirement_age)}
    wealth_paths     = []

    for ep in range(n_episodes):
        state = env.reset()
        done  = False
        path  = [env.wealth]
        while not done:
            current_age = env.age
            action      = policy_fn(state, current_age)
            action_by_age[current_age].append(action)
            state, _, done = env.step(action)
            path.append(env.wealth)
        terminal_wealths.append(env.wealth)
        terminal_utils.append(terminal_utility(env.wealth))
        if ep < 200:
            wealth_paths.append(np.array(path))

    tw = np.array(terminal_wealths)
    return {
        'terminal_wealth':  tw,
        'terminal_utility': np.array(terminal_utils),
        'action_by_age':    action_by_age,
        'wealth_paths':     wealth_paths,
        'mean_wealth':      float(np.mean(tw)),
        'median_wealth':    float(np.median(tw)),
        'std_wealth':       float(np.std(tw)),
        'p10':              float(np.percentile(tw, 10)),
        'p25':              float(np.percentile(tw, 25)),
        'p75':              float(np.percentile(tw, 75)),
        'p90':              float(np.percentile(tw, 90)),
        'mean_utility':     float(np.mean(terminal_utils)),
    }


# ── Build DQN policy callable ─────────────────────────────────────────────
def make_dqn_policy(q_net: QNetwork):
    def dqn_policy(state: np.ndarray, age: int) -> int:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return int(q_net(s).argmax(dim=1).item())
    return dqn_policy

dqn_policy_fn = make_dqn_policy(q_net)

ALL_POLICIES = {**BASELINES, 'DQN': dqn_policy_fn}

print(f'Evaluating {len(ALL_POLICIES)} policies x {cfg.n_eval_episodes:,} episodes each ...')
eval_results = {}
for name, fn in ALL_POLICIES.items():
    print(f'  {name} ...', end=' ', flush=True)
    eval_results[name] = run_evaluation(fn, cfg.n_eval_episodes, seed=0)
    print(f'done.  Median TW = ${eval_results[name]["median_wealth"]:,.0f}')

# ── Summary DataFrame ─────────────────────────────────────────────────────
rows = []
for name, res in eval_results.items():
    rows.append({
        'Policy':       name,
        'Mean ($)':     f'${res["mean_wealth"]:>11,.0f}',
        'Median ($)':   f'${res["median_wealth"]:>11,.0f}',
        'Std Dev ($)':  f'${res["std_wealth"]:>11,.0f}',
        'P10 ($)':      f'${res["p10"]:>11,.0f}',
        'P25 ($)':      f'${res["p25"]:>11,.0f}',
        'P75 ($)':      f'${res["p75"]:>11,.0f}',
        'P90 ($)':      f'${res["p90"]:>11,.0f}',
        'Mean Utility': f'{res["mean_utility"]:>9.4f}',
    })

df_eval = pd.DataFrame(rows)
print()
print('=' * 110)
print(f'  Evaluation Summary — {cfg.n_eval_episodes:,} Monte Carlo episodes per policy')
print('=' * 110)
print(df_eval.to_string(index=False))
print('=' * 110)

# === CELL 32 ===
# ─────────────────────────────────────────────────────────────────────────────
# Section 16a: Action Schemes + Fractional Environment
# ─────────────────────────────────────────────────────────────────────────────

# ── Define Allocation Schemes ─────────────────────────────────────────────────

def generate_allocation_schemes():
    """
    Returns an ordered dict mapping scheme label -> list of allocation vectors.

    Each allocation vector is a tuple (w_VFSTX, w_VBMFX, w_VFINX) summing to 1.0.

    Scheme A:  3 pure-pick actions  (original baseline action space)
    Scheme B:  6 actions = pure picks + all binary 50/50 blends
    Scheme C: 15 actions = all (w0,w1,w2) with 25%-increment weights
    """
    # ── Scheme A: pure picks ──────────────────────────────────────────────────
    scheme_a = [
        (1.00, 0.00, 0.00),   # 100% VFSTX
        (0.00, 1.00, 0.00),   # 100% VBMFX
        (0.00, 0.00, 1.00),   # 100% VFINX
    ]

    # ── Scheme B: pure picks + binary 50/50 blends ────────────────────────────
    scheme_b = list(scheme_a) + [
        (0.50, 0.50, 0.00),   # 50% VFSTX + 50% VBMFX
        (0.50, 0.00, 0.50),   # 50% VFSTX + 50% VFINX
        (0.00, 0.50, 0.50),   # 50% VBMFX + 50% VFINX
    ]

    # ── Scheme C: all 25%-increment weight combinations ───────────────────────
    # Enumerate all (a, b, c) with a+b+c=1 and a,b,c in {0, 0.25, 0.50, 0.75, 1.0}
    levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    scheme_c = []
    for a in levels:
        for b in levels:
            c = round(1.0 - a - b, 10)
            if 0.0 <= c <= 1.0 + 1e-9:
                c = min(c, 1.0)           # clamp floating-point overshoot
                c = round(c, 4)
                scheme_c.append((a, b, c))

    return {
        'Scheme A\n(3 pure picks)':          scheme_a,
        'Scheme B\n(6: pure + 50/50)':       scheme_b,
        'Scheme C\n(15: 25% grid)':           scheme_c,
    }


SCHEMES = generate_allocation_schemes()

# Print scheme details
print('Allocation schemes defined:\n')
for name, vecs in SCHEMES.items():
    label = name.replace('\n', ' ')
    print(f'  {label}  — {len(vecs)} actions')
    for i, v in enumerate(vecs):
        print(f'    action {i:>2d}: VFSTX={v[0]:.2f}  VBMFX={v[1]:.2f}  VFINX={v[2]:.2f}')
    print()


# ── Fractional Retirement Environment ─────────────────────────────────────────

class RetirementEnvFractional:
    """
    Retirement environment supporting blended/fractional strategy allocations.

    Each action index maps to an allocation vector (w_VFSTX, w_VBMFX, w_VFINX)
    summing to 1.0. The blended annual return is:
        r_blend = sum_i  w_i * N(mu_i(regime), sigma_i(regime))
    Switching cost applies when the action index (hence allocation) changes.
    State encoding is identical to RetirementEnv (5-D normalized vector).
    """

    def __init__(self, cfg: Config, action_vectors: list,
                 seed: Optional[int] = None):
        self.cfg            = cfg
        self.action_vectors = action_vectors        # list of (w0, w1, w2)
        self.n_actions      = len(action_vectors)
        self.rng            = np.random.default_rng(seed)
        self.age            = cfg.start_age
        self.wealth         = cfg.initial_wealth
        self.regime         = NEUTRAL
        self.prev_action    = None
        self.step_count     = 0

    def reset(self, start_regime: Optional[int] = None) -> np.ndarray:
        self.age         = self.cfg.start_age
        self.wealth      = self.cfg.initial_wealth
        self.regime      = (int(self.rng.integers(0, 3))
                            if start_regime is None else start_regime)
        self.prev_action = None
        self.step_count  = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        assert 0 <= action < self.n_actions, f'Invalid action {action}'
        allocation = self.action_vectors[action]

        # Switching cost when allocation index changes
        if self.prev_action is not None and action != self.prev_action:
            switch_cost = self.cfg.switching_cost_pct * self.wealth
        else:
            switch_cost = 0.0

        # Blended return: independently sample each strategy, weight by allocation
        blended_return = 0.0
        for strat_idx, weight in enumerate(allocation):
            if weight > 1e-10:
                mu, sig = self.cfg.return_params[strat_idx][self.regime]
                r = float(np.clip(self.rng.normal(mu, sig), -0.60, 0.60))
                blended_return += weight * r

        # Wealth dynamics
        prev_wealth  = self.wealth
        self.wealth  = (self.wealth + self.cfg.annual_contribution) * (1.0 + blended_return)
        self.wealth -= switch_cost
        self.wealth  = max(self.wealth, self.cfg.wealth_min)

        # Regime transition
        self.regime = int(self.rng.choice(3, p=self.cfg.regime_transition[self.regime]))

        # Advance
        self.prev_action = action
        self.age        += 1
        self.step_count += 1
        done = (self.age >= self.cfg.retirement_age)

        # Reward
        if done:
            reward = crra_utility(self.wealth, gamma_from_age(self.cfg.retirement_age))
        else:
            delta_log = np.log(max(self.wealth, 1.0)) - np.log(max(prev_wealth, 1.0))
            reward    = self.cfg.intermediate_alpha * delta_log

        return self._get_state(), reward, done

    def _get_state(self) -> np.ndarray:
        norm_age   = (self.age - self.cfg.start_age) / T_STEPS
        log_w      = np.log10(max(self.wealth, self.cfg.wealth_min))
        norm_log_w = log_w / np.log10(self.cfg.wealth_max)
        regime_oh  = np.zeros(3, dtype=np.float32)
        regime_oh[self.regime] = 1.0
        return np.array([norm_age, norm_log_w, *regime_oh], dtype=np.float32)


print('RetirementEnvFractional defined.')
print(f'STATE_DIM = {STATE_DIM} (unchanged from baseline)')
print()

# Quick sanity check on Scheme C env
_env_c = RetirementEnvFractional(cfg, SCHEMES['Scheme C\n(15: 25% grid)'], seed=99)
_s0 = _env_c.reset(start_regime=BULL)
_s1, _r1, _d1 = _env_c.step(7)   # action 7: some mixed allocation
print(f'Scheme C sanity: age={_env_c.age}, wealth=${_env_c.wealth:,.0f}, '
      f'reward={_r1:.5f}, done={_d1}')

# === CELL 33 ===
# ─────────────────────────────────────────────────────────────────────────────
# Section 16b: Modified Training + Evaluation Functions
# ─────────────────────────────────────────────────────────────────────────────

def train_dqn_scheme(cfg: Config,
                     action_vectors: list,
                     n_episodes: Optional[int] = None,
                     seed: int = SEED,
                     verbose: bool = True) -> Tuple:
    """
    Train DQN for the given fractional action scheme.

    Uses RetirementEnvFractional; all other hyperparameters from cfg.
    optimize_model() is reused unchanged (it is action-count–agnostic).

    Returns: (q_net, episode_rewards, losses)
    """
    if n_episodes is None:
        n_episodes = cfg.n_episodes

    n_actions = len(action_vectors)
    env = RetirementEnvFractional(cfg, action_vectors, seed=seed)

    q_net      = QNetwork(STATE_DIM, n_actions, hidden=cfg.hidden_size)
    target_net = QNetwork(STATE_DIM, n_actions, hidden=cfg.hidden_size)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer     = optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay_buffer = ReplayBuffer(cfg.replay_buffer_size)

    epsilon         = cfg.eps_start
    episode_rewards = []
    losses          = []

    for episode in range(n_episodes):
        state     = env.reset()
        ep_reward = 0.0
        done      = False

        while not done:
            # ε-greedy with scheme-specific n_actions
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = int(q_net(s).argmax(dim=1).item())

            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state      = next_state
            ep_reward += reward

            loss_val = optimize_model(q_net, target_net, replay_buffer, optimizer, cfg)
            if loss_val is not None:
                losses.append(loss_val)

        epsilon = max(cfg.eps_end, epsilon * cfg.eps_decay)

        if (episode + 1) % cfg.target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        episode_rewards.append(ep_reward)

        if verbose and (episode + 1) % 500 == 0:
            recent = np.mean(episode_rewards[-200:])
            print(f'    ep {episode+1:>5}/{n_episodes} | ε={epsilon:.3f} | '
                  f'mean reward (last 200)={recent:.4f}')

    return q_net, episode_rewards, losses


def evaluate_scheme(q_net: QNetwork,
                    action_vectors: list,
                    n_episodes: int = 2000,
                    seed: int = 0) -> dict:
    """
    Evaluate a trained DQN policy under the given fractional action scheme.

    Records: terminal wealth distribution, CRRA utilities, action indices by age,
    and up to 200 wealth trajectories for path-plot purposes.
    """
    env = RetirementEnvFractional(cfg, action_vectors, seed=seed)

    terminal_wealths = []
    terminal_utils   = []
    action_by_age    = {age: [] for age in range(cfg.start_age, cfg.retirement_age)}
    wealth_paths     = []

    for ep in range(n_episodes):
        state = env.reset()
        done  = False
        path  = [env.wealth]

        while not done:
            age = env.age
            with torch.no_grad():
                s      = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = int(q_net(s).argmax(dim=1).item())
            action_by_age[age].append(action)
            state, _, done = env.step(action)
            path.append(env.wealth)

        terminal_wealths.append(env.wealth)
        terminal_utils.append(terminal_utility(env.wealth))
        if ep < 200:
            wealth_paths.append(np.array(path))

    tw = np.array(terminal_wealths)
    return {
        'terminal_wealth':  tw,
        'terminal_utility': np.array(terminal_utils),
        'action_by_age':    action_by_age,
        'wealth_paths':     wealth_paths,
        'action_vectors':   action_vectors,
        'mean_wealth':      float(np.mean(tw)),
        'median_wealth':    float(np.median(tw)),
        'std_wealth':       float(np.std(tw)),
        'p10':              float(np.percentile(tw, 10)),
        'p25':              float(np.percentile(tw, 25)),
        'p75':              float(np.percentile(tw, 75)),
        'p90':              float(np.percentile(tw, 90)),
        'mean_utility':     float(np.mean(np.array(terminal_utils))),
    }


print('train_dqn_scheme() and evaluate_scheme() defined.')

# === CELL 34 ===
# ─────────────────────────────────────────────────────────────────────────────
# Section 16c: Run Sensitivity Experiments
# ─────────────────────────────────────────────────────────────────────────────
# Each scheme is trained from scratch for cfg.n_episodes (5,000) episodes,
# then evaluated for N_EVAL_SENSITIVITY (2,000) episodes.
# ─────────────────────────────────────────────────────────────────────────────

N_EVAL_SENSITIVITY = 2000   # evaluation episodes per scheme (medium budget)

sensitivity_results = {}    # scheme_label -> result dict

for scheme_name, action_vectors in SCHEMES.items():
    label = scheme_name.replace('\n', ' ')
    n_act = len(action_vectors)
    print(f'\n{"=" * 65}')
    print(f'  Scheme : {label}  ({n_act} actions)')
    print(f'  Training {cfg.n_episodes:,} episodes ...')

    # Reset global seeds for reproducibility
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    q_net_s, ep_rewards_s, losses_s = train_dqn_scheme(
        cfg, action_vectors,
        n_episodes=cfg.n_episodes,
        seed=SEED,
        verbose=True,
    )

    print(f'  Evaluating {N_EVAL_SENSITIVITY:,} episodes ...')
    res = evaluate_scheme(q_net_s, action_vectors,
                          n_episodes=N_EVAL_SENSITIVITY, seed=0)

    res['episode_rewards'] = ep_rewards_s
    res['losses']          = losses_s
    res['q_net']           = q_net_s
    sensitivity_results[label] = res

    print(f'  → Median TW = ${res["median_wealth"]:>12,.0f} | '
          f'Mean Utility = {res["mean_utility"]:.4f}')

print(f'\n{"=" * 65}')
print('All sensitivity experiments complete.')

# ── Quick summary table ───────────────────────────────────────────────────────
print(f'\n{"=" * 95}')
print('  Sensitivity Analysis — Quick Summary')
print('=' * 95)
hdr = (f'  {"Scheme":<30}  {"N Actions":>9}  '
       f'{"Mean ($)":>14}  {"Median ($)":>14}  {"Mean Utility":>13}')
print(hdr)
print('-' * 95)
for label, res in sensitivity_results.items():
    n_act = len(res['action_vectors'])
    short = label.split('(')[0].strip()
    print(f'  {short:<30}  {n_act:>9}  '
          f'${res["mean_wealth"]:>13,.0f}  ${res["median_wealth"]:>13,.0f}  '
          f'{res["mean_utility"]:>13.4f}')
print('=' * 95)

# === CELL 35 ===
# ─────────────────────────────────────────────────────────────────────────────
# Section 16d: Comparative Plots
# ─────────────────────────────────────────────────────────────────────────────

_SCHEME_LABELS  = list(sensitivity_results.keys())
_SCHEME_COLORS  = ['#1E88E5', '#43A047', '#E53935']  # blue, green, red
_STRAT_NAMES    = cfg.strategy_names                  # ['VFSTX','VBMFX','VFINX']
_STRAT_COLORS   = ['#29B6F6', '#FFA726', '#66BB6A']   # per-strategy palette
_AGES           = np.arange(cfg.start_age, cfg.retirement_age + 1)  # 30..65
_AGES_STEPS     = list(range(cfg.start_age, cfg.retirement_age))     # 30..64

# ── Optional scipy KDE (graceful fallback to numpy histogram) ─────────────────
try:
    from scipy.stats import gaussian_kde as _scipy_kde
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _density_plot(ax, data, color, label, cap_pct=99):
    """Plot a density curve for terminal wealth (capped at cap_pct percentile)."""
    cap = np.percentile(data, cap_pct)
    d   = data[data <= cap]
    if _HAS_SCIPY:
        kde    = _scipy_kde(d, bw_method='scott')
        x_grid = np.linspace(d.min(), d.max(), 400)
        ax.plot(x_grid / 1e6, kde(x_grid) * 1e6, color=color, linewidth=2.0,
                label=label)
    else:
        counts, edges = np.histogram(d / 1e6, bins=60, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, counts, color=color, linewidth=2.0, label=label)
    ax.axvline(np.median(data) / 1e6, color=color, linestyle='--',
               linewidth=1.2, alpha=0.65)


# ── Figure SA: Mean Utility + Key Wealth Stats per Scheme ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# (a) Mean CRRA Utility
ax = axes[0]
mean_utils = [sensitivity_results[s]['mean_utility'] for s in _SCHEME_LABELS]
bars = ax.bar(range(len(_SCHEME_LABELS)), mean_utils,
              color=_SCHEME_COLORS, width=0.5, edgecolor='white')
ax.set_xticks(range(len(_SCHEME_LABELS)))
ax.set_xticklabels([s.replace('\n', '\n') for s in _SCHEME_LABELS], fontsize=9)
ax.set_ylabel('Mean CRRA Utility')
ax.set_title('(a) Mean CRRA Utility', fontweight='bold')
for bar, val in zip(bars, mean_utils):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002, f'{val:.4f}',
            ha='center', va='bottom', fontsize=9)

# (b) Median Terminal Wealth
ax = axes[1]
medians = [sensitivity_results[s]['median_wealth'] / 1e6 for s in _SCHEME_LABELS]
bars2 = ax.bar(range(len(_SCHEME_LABELS)), medians,
               color=_SCHEME_COLORS, width=0.5, edgecolor='white')
ax.set_xticks(range(len(_SCHEME_LABELS)))
ax.set_xticklabels([s.replace('\n', '\n') for s in _SCHEME_LABELS], fontsize=9)
ax.set_ylabel('Median Terminal Wealth ($M)')
ax.set_title('(b) Median Terminal Wealth', fontweight='bold')
for bar, val in zip(bars2, medians):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05, f'${val:.2f}M',
            ha='center', va='bottom', fontsize=9)

# (c) P10 / Median / P90 range (downside + upside)
ax = axes[2]
x_pos = np.arange(len(_SCHEME_LABELS))
p10s  = [sensitivity_results[s]['p10']  / 1e6 for s in _SCHEME_LABELS]
p90s  = [sensitivity_results[s]['p90']  / 1e6 for s in _SCHEME_LABELS]
for i, (label, p10, med, p90, col) in enumerate(
        zip(_SCHEME_LABELS, p10s, medians, p90s, _SCHEME_COLORS)):
    ax.plot([i, i], [p10, p90], color=col, linewidth=6, alpha=0.30, solid_capstyle='round')
    ax.plot(i, med, 'o', color=col, markersize=8, zorder=5)
ax.set_xticks(x_pos)
ax.set_xticklabels([s.replace('\n', '\n') for s in _SCHEME_LABELS], fontsize=9)
ax.set_ylabel('Terminal Wealth ($M)')
ax.set_title('(c) P10–Median–P90 Range', fontweight='bold')

fig.suptitle('Sensitivity Analysis — Utility & Wealth Summary by Action Scheme',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('fig_s3_sens_summary.png', dpi=110, bbox_inches='tight')
plt.close()
print('Figure SA (summary stats) saved → fig_s3_sens_summary.png')


# ── Figure SB: Terminal Wealth Distributions ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for i, (label, col) in enumerate(zip(_SCHEME_LABELS, _SCHEME_COLORS)):
    res  = sensitivity_results[label]
    tw   = res['terminal_wealth']
    short = label.split('(')[0].strip()
    _density_plot(ax, tw, col, f'{short}  (med=${np.median(tw)/1e6:.2f}M)')

ax.set_xlabel('Terminal Wealth ($M)')
ax.set_ylabel('Density')
ax.set_title('Sensitivity Analysis — Terminal Wealth Distributions\n'
             '(dashed = scheme median)', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('fig_s3_sens_wealth_dist.png', dpi=110, bbox_inches='tight')
plt.close()
print('Figure SB (wealth distributions) saved → fig_s3_sens_wealth_dist.png')


# ── Figure SC: Wealth Paths (Median + IQR + P10–P90) ─────────────────────────
fig, axes = plt.subplots(1, len(_SCHEME_LABELS),
                         figsize=(14, 5), sharey=True)
for i, (label, col) in enumerate(zip(_SCHEME_LABELS, _SCHEME_COLORS)):
    ax    = axes[i]
    paths = np.array(sensitivity_results[label]['wealth_paths'])   # (≤200, T+1)
    med   = np.median(paths, axis=0)
    p25   = np.percentile(paths, 25, axis=0)
    p75   = np.percentile(paths, 75, axis=0)
    p10   = np.percentile(paths, 10, axis=0)
    p90   = np.percentile(paths, 90, axis=0)

    ax.fill_between(_AGES, p10 / 1e6, p90 / 1e6, alpha=0.12, color=col)
    ax.fill_between(_AGES, p25 / 1e6, p75 / 1e6, alpha=0.28, color=col)
    ax.plot(_AGES, med / 1e6, color=col, linewidth=2.0, label='Median')
    ax.set_xlabel('Age')
    ax.set_title(label, fontsize=9)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))

axes[0].set_ylabel('Wealth')
fig.suptitle('Sensitivity Analysis — Wealth Paths\n'
             '(median + IQR shaded + P10–P90 light band)',
             fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('fig_s3_sens_wealth_paths.png', dpi=110, bbox_inches='tight')
plt.close()
print('Figure SC (wealth paths) saved → fig_s3_sens_wealth_paths.png')


# ── Figure SD: Effective Allocation by Age ────────────────────────────────────
# For each scheme, compute the mean portfolio weight assigned to each strategy
# at each age (averaging over evaluation episodes).

fig, axes = plt.subplots(1, len(_SCHEME_LABELS),
                         figsize=(14, 4.5), sharey=True)
for i, (label, col) in enumerate(zip(_SCHEME_LABELS, _SCHEME_COLORS)):
    ax         = axes[i]
    res        = sensitivity_results[label]
    act_vecs   = res['action_vectors']
    n_strats   = 3

    eff = np.zeros((len(_AGES_STEPS), n_strats))   # (35, 3)
    for t, age in enumerate(_AGES_STEPS):
        acts = res['action_by_age'].get(age, [])
        if not acts:
            continue
        for a in acts:
            eff[t] += np.array(act_vecs[a])
        eff[t] /= len(acts)

    bottom = np.zeros(len(_AGES_STEPS))
    for s in range(n_strats):
        ax.bar(_AGES_STEPS, eff[:, s], bottom=bottom,
               color=_STRAT_COLORS[s], label=_STRAT_NAMES[s], width=1.0)
        bottom += eff[:, s]

    ax.set_xlabel('Age')
    ax.set_title(label, fontsize=9)
    ax.set_ylim(0, 1.02)
    if i == 0:
        ax.set_ylabel('Mean Allocation Fraction')
    if i == len(_SCHEME_LABELS) - 1:
        ax.legend(loc='lower right', fontsize=8)

fig.suptitle('Sensitivity Analysis — Effective Portfolio Allocation by Age',
             fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('fig_s3_sens_alloc_age.png', dpi=110, bbox_inches='tight')
plt.close()
print('Figure SD (allocation by age) saved → fig_s3_sens_alloc_age.png')


# ── Figure SE: Training Reward Curves ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for i, (label, col) in enumerate(zip(_SCHEME_LABELS, _SCHEME_COLORS)):
    ep_rews = sensitivity_results[label]['episode_rewards']
    short   = label.split('(')[0].strip()
    ax.plot(ep_rews, alpha=0.12, color=col, linewidth=0.5)
    if len(ep_rews) >= 100:
        sm = np.convolve(ep_rews, np.ones(100) / 100, mode='valid')
        ax.plot(np.arange(99, len(ep_rews)), sm, color=col,
                linewidth=2.0, label=f'{short} (100-ep avg)')

ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('Sensitivity Analysis — Training Reward Curves', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('fig_s3_sens_training.png', dpi=110, bbox_inches='tight')
plt.close()
print('Figure SE (training curves) saved → fig_s3_sens_training.png')


print('\nAll sensitivity figures saved.')
print('Files: fig_s3_sens_summary.png, fig_s3_sens_wealth_dist.png,')
print('       fig_s3_sens_wealth_paths.png, fig_s3_sens_alloc_age.png,')
print('       fig_s3_sens_training.png')

