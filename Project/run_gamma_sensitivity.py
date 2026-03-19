#!/usr/bin/env python3
"""
Sensitivity analysis on the gamma risk aversion parameter for Phase 3 DQN.
Uses Scheme B (6 actions: pure + 50/50) for all experiments.
"""
import matplotlib
matplotlib.use('Agg')
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

VFSTX, VBMFX, VFINX = 0, 1, 2
BULL, NEUTRAL, BEAR = 0, 1, 2

@dataclass
class Config:
    start_age: int = 30
    retirement_age: int = 65
    initial_wealth: float = 100_000.0
    annual_contribution: float = 10_000.0
    wealth_min: float = 1_000.0
    wealth_max: float = 5_000_000.0
    switching_cost_pct: float = 0.005
    gamma_0: float = 2.0
    gamma_slope: float = 0.03
    w_ref: float = 1_000_000.0
    intermediate_alpha: float = 0.01
    regime_transition: np.ndarray = field(default_factory=lambda: np.array([
        [0.80, 0.15, 0.05],
        [0.20, 0.60, 0.20],
        [0.05, 0.25, 0.70],
    ]))
    return_params: Dict = field(default_factory=lambda: {
        VFSTX: {BULL: (0.045, 0.005), NEUTRAL: (0.030, 0.005), BEAR: (0.010, 0.005)},
        VBMFX: {BULL: (0.020, 0.045), NEUTRAL: (0.040, 0.045), BEAR: (0.100, 0.055)},
        VFINX: {BULL: (0.270, 0.150), NEUTRAL: (0.100, 0.150), BEAR: (-0.150, 0.190)},
    })
    n_episodes: int = 5000
    batch_size: int = 128
    replay_buffer_size: int = 50_000
    lr: float = 1e-3
    gamma_discount: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995
    target_update_freq: int = 500
    hidden_size: int = 64
    strategy_names: List[str] = field(default_factory=lambda: ["VFSTX", "VBMFX", "VFINX"])
    regime_names: List[str] = field(default_factory=lambda: ["Bull", "Neutral", "Bear"])

T_STEPS = 65 - 30 # standard 35
STATE_DIM = 5

def gamma_from_age(age: int, cfg: Config) -> float:
    return cfg.gamma_0 + cfg.gamma_slope * (age - cfg.start_age)

def crra_utility(w: float, gamma: float, cfg: Config) -> float:
    w_norm = max(float(w), 1e-6) / cfg.w_ref
    if abs(gamma - 1.0) < 1e-9:
        return float(np.log(w_norm))
    return float((w_norm ** (1.0 - gamma) - 1.0) / (1.0 - gamma))

def terminal_utility(wealth: float, cfg: Config) -> float:
    return crra_utility(wealth, gamma_from_age(cfg.retirement_age, cfg), cfg)

class RetirementEnvFractional:
    def __init__(self, cfg: Config, action_vectors: list, seed: Optional[int] = None):
        self.cfg = cfg
        self.action_vectors = action_vectors
        self.n_actions = len(action_vectors)
        self.rng = np.random.default_rng(seed)
        self.age = cfg.start_age
        self.wealth = cfg.initial_wealth
        self.regime = NEUTRAL
        self.prev_action = None
        self.step_count = 0

    def reset(self, start_regime: Optional[int] = None) -> np.ndarray:
        self.age = self.cfg.start_age
        self.wealth = self.cfg.initial_wealth
        self.regime = int(self.rng.integers(0, 3)) if start_regime is None else start_regime
        self.prev_action = None
        self.step_count = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        assert 0 <= action < self.n_actions
        allocation = self.action_vectors[action]
        if self.prev_action is not None and action != self.prev_action:
            switch_cost = self.cfg.switching_cost_pct * self.wealth
        else:
            switch_cost = 0.0
        blended_return = 0.0
        for strat_idx, weight in enumerate(allocation):
            if weight > 1e-10:
                mu, sig = self.cfg.return_params[strat_idx][self.regime]
                r = float(np.clip(self.rng.normal(mu, sig), -0.60, 0.60))
                blended_return += weight * r
        prev_wealth = self.wealth
        self.wealth = (self.wealth + self.cfg.annual_contribution) * (1.0 + blended_return)
        self.wealth -= switch_cost
        self.wealth = max(self.wealth, self.cfg.wealth_min)
        self.regime = int(self.rng.choice(3, p=self.cfg.regime_transition[self.regime]))
        self.prev_action = action
        self.age += 1
        self.step_count += 1
        done = (self.age >= self.cfg.retirement_age)
        if done:
            reward = crra_utility(self.wealth, gamma_from_age(self.cfg.retirement_age, self.cfg), self.cfg)
        else:
            delta_log = np.log(max(self.wealth, 1.0)) - np.log(max(prev_wealth, 1.0))
            reward = self.cfg.intermediate_alpha * delta_log
        return self._get_state(), reward, done

    def _get_state(self) -> np.ndarray:
        norm_age = (self.age - self.cfg.start_age) / (self.cfg.retirement_age - self.cfg.start_age)
        log_w = np.log10(max(self.wealth, self.cfg.wealth_min))
        norm_log_w = log_w / np.log10(self.cfg.wealth_max)
        regime_oh = np.zeros(3, dtype=np.float32)
        regime_oh[self.regime] = 1.0
        return np.array([norm_age, norm_log_w, *regime_oh], dtype=np.float32)

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )
    def __len__(self):
        return len(self.buffer)

def optimize_model(q_net, target_net, replay_buffer, optimizer, cfg):
    if len(replay_buffer) < cfg.batch_size:
        return None
    states, actions, rewards, next_states, dones = replay_buffer.sample(cfg.batch_size)
    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target = rewards + cfg.gamma_discount * next_q * (1 - dones)
    loss = nn.functional.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())

# Scheme B is the chosen action space
SCHEME_B = [
    (1.00, 0.00, 0.00), (0.00, 1.00, 0.00), (0.00, 0.00, 1.00),
    (0.50, 0.50, 0.00), (0.50, 0.00, 0.50), (0.00, 0.50, 0.50)
]

def train_dqn_scheme(cfg, action_vectors, n_episodes=None, seed=SEED, verbose=True):
    n_episodes = n_episodes or cfg.n_episodes
    n_actions = len(action_vectors)
    env = RetirementEnvFractional(cfg, action_vectors, seed=seed)
    q_net = QNetwork(STATE_DIM, n_actions, hidden=cfg.hidden_size)
    target_net = QNetwork(STATE_DIM, n_actions, hidden=cfg.hidden_size)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay_buffer = ReplayBuffer(cfg.replay_buffer_size)
    epsilon = cfg.eps_start
    episode_rewards = []
    losses = []
    for episode in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = int(q_net(s).argmax(dim=1).item())
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
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
            print(f'    ep {episode+1:>5}/{n_episodes} | eps={epsilon:.3f} | mean reward (last 200)={recent:.4f}')
    return q_net, episode_rewards, losses

def evaluate_scheme(q_net, cfg, action_vectors, n_episodes=2000, seed=0):
    env = RetirementEnvFractional(cfg, action_vectors, seed=seed)
    terminal_wealths = []
    terminal_utils = []
    action_by_age = {age: [] for age in range(cfg.start_age, cfg.retirement_age)}
    wealth_paths = []
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        path = [env.wealth]
        while not done:
            age = env.age
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = int(q_net(s).argmax(dim=1).item())
            action_by_age[age].append(action)
            state, _, done = env.step(action)
            path.append(env.wealth)
        terminal_wealths.append(env.wealth)
        terminal_utils.append(terminal_utility(env.wealth, cfg))
        if ep < 200:
            wealth_paths.append(np.array(path))
    tw = np.array(terminal_wealths)
    return {
        'terminal_wealth': tw,
        'terminal_utility': np.array(terminal_utils),
        'action_by_age': action_by_age,
        'wealth_paths': wealth_paths,
        'action_vectors': action_vectors,
        'mean_wealth': float(np.mean(tw)),
        'median_wealth': float(np.median(tw)),
        'std_wealth': float(np.std(tw)),
        'p10': float(np.percentile(tw, 10)),
        'p25': float(np.percentile(tw, 25)),
        'p75': float(np.percentile(tw, 75)),
        'p90': float(np.percentile(tw, 90)),
        'mean_utility': float(np.mean(np.array(terminal_utils))),
    }

# --- Define the 6 Gamma Profiles ---
PROFILES = {
    'Log Utility\n(γ=1.0, slope=0.0)': Config(gamma_0=1.0, gamma_slope=0.0),
    'Constant Mod\n(γ=2.0, slope=0.0)': Config(gamma_0=2.0, gamma_slope=0.0),
    'Baseline Inc\n(γ0=2.0, slope=0.03)': Config(gamma_0=2.0, gamma_slope=0.03),
    'Constant High\n(γ=4.0, slope=0.0)': Config(gamma_0=4.0, gamma_slope=0.0),
    'Inc High\n(γ0=3.0, slope=0.03)': Config(gamma_0=3.0, gamma_slope=0.03),
    'Inc V. High\n(γ0=4.0, slope=0.03)': Config(gamma_0=4.0, gamma_slope=0.03),
}

N_EVAL = 2000
base_cfg = Config()
AGES = np.arange(base_cfg.start_age, base_cfg.retirement_age + 1)
AGES_STEPS = list(range(base_cfg.start_age, base_cfg.retirement_age))

sensitivity_results = {}

print('Running Gamma Sensitivity Analysis (6 profiles x 5000 train + 2000 eval)...')
for label, p_cfg in PROFILES.items():
    print(f'\n{"="*65}\n  Profile: {label.replace(chr(10), " ")}')
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    q_net_s, ep_rewards_s, losses_s = train_dqn_scheme(
        p_cfg, SCHEME_B, n_episodes=p_cfg.n_episodes, seed=SEED, verbose=True)
    print(f'  Evaluating {N_EVAL} episodes ...')
    res = evaluate_scheme(q_net_s, p_cfg, SCHEME_B, n_episodes=N_EVAL, seed=0)
    res['episode_rewards'] = ep_rewards_s
    res['losses'] = losses_s
    res['q_net'] = q_net_s
    sensitivity_results[label] = res
    print(f'  -> Median TW = ${res["median_wealth"]:>12,.0f} | Mean Utility = {res["mean_utility"]:.4f}')

print('\nAll gamma sensitivity experiments complete.')

_PROFILE_LABELS = list(sensitivity_results.keys())
# Define a set of colors for the profiles
_PROFILE_COLORS = plt.cm.tab10(np.linspace(0, 1, len(_PROFILE_LABELS)))

_STRAT_NAMES = base_cfg.strategy_names
_STRAT_COLORS = ['#29B6F6', '#FFA726', '#66BB6A']

try:
    from scipy.stats import gaussian_kde as _scipy_kde
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

def _density_plot(ax, data, color, label, cap_pct=99):
    cap = np.percentile(data, cap_pct)
    d = data[data <= cap]
    if _HAS_SCIPY:
        kde = _scipy_kde(d, bw_method='scott')
        x_grid = np.linspace(d.min(), d.max(), 400)
        ax.plot(x_grid / 1e6, kde(x_grid) * 1e6, color=color, linewidth=2.0, label=label)
    else:
        counts, edges = np.histogram(d / 1e6, bins=60, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, counts, color=color, linewidth=2.0, label=label)
    ax.axvline(np.median(data) / 1e6, color=color, linestyle='--', linewidth=1.2, alpha=0.65)

# --- Plot 1: Wealth Summary (No Utility plot due to incomparability) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

medians = [sensitivity_results[s]['median_wealth']/1e6 for s in _PROFILE_LABELS]
bars2 = axes[0].bar(range(len(_PROFILE_LABELS)), medians, color=_PROFILE_COLORS, width=0.6, edgecolor='white')
axes[0].set_xticks(range(len(_PROFILE_LABELS)))
axes[0].set_xticklabels([s for s in _PROFILE_LABELS], fontsize=8, rotation=45, ha='right')
axes[0].set_ylabel('Median Terminal Wealth ($M)')
axes[0].set_title('(a) Median Terminal Wealth', fontweight='bold')
for bar, val in zip(bars2, medians):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.05, f'${val:.2f}M', ha='center', va='bottom', fontsize=9)

x_pos = np.arange(len(_PROFILE_LABELS))
p10s = [sensitivity_results[s]['p10']/1e6 for s in _PROFILE_LABELS]
p90s = [sensitivity_results[s]['p90']/1e6 for s in _PROFILE_LABELS]
for i, (label, p10, med, p90, col) in enumerate(zip(_PROFILE_LABELS, p10s, medians, p90s, _PROFILE_COLORS)):
    axes[1].plot([i, i], [p10, p90], color=col, linewidth=6, alpha=0.30, solid_capstyle='round')
    axes[1].plot(i, med, 'o', color=col, markersize=8, zorder=5)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([s for s in _PROFILE_LABELS], fontsize=8, rotation=45, ha='right')
axes[1].set_ylabel('Terminal Wealth ($M)')
axes[1].set_title('(b) P10-Median-P90 Range', fontweight='bold')

fig.suptitle('Gamma Sensitivity — Wealth Outcomes by Risk Aversion Profile', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('fig_gamma_sens_summary.png', dpi=110, bbox_inches='tight')
plt.close()
print('Saved fig_gamma_sens_summary.png')

# --- Plot 2: Terminal Wealth Distributions ---
fig, ax = plt.subplots(figsize=(10, 5))
for i, (label, col) in enumerate(zip(_PROFILE_LABELS, _PROFILE_COLORS)):
    res = sensitivity_results[label]
    tw = res['terminal_wealth']
    short = label.split('\n')[0].strip()
    _density_plot(ax, tw, col, f'{short}  (med=${np.median(tw)/1e6:.2f}M)')
ax.set_xlabel('Terminal Wealth ($M)')
ax.set_ylabel('Density')
ax.set_title('Gamma Sensitivity — Terminal Wealth Distributions', fontweight='bold')
ax.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('fig_gamma_sens_wealth_dist.png', dpi=110, bbox_inches='tight')
plt.close()
print('Saved fig_gamma_sens_wealth_dist.png')

# --- Plot 3: Wealth Paths ---
fig, axes = plt.subplots(1, len(_PROFILE_LABELS), figsize=(18, 5), sharey=True)
if len(_PROFILE_LABELS) == 1:
    axes = [axes]
for i, (label, col) in enumerate(zip(_PROFILE_LABELS, _PROFILE_COLORS)):
    ax = axes[i]
    paths = np.array(sensitivity_results[label]['wealth_paths'])
    med = np.median(paths, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    ax.fill_between(AGES, p10/1e6, p90/1e6, alpha=0.12, color=col)
    ax.fill_between(AGES, p25/1e6, p75/1e6, alpha=0.28, color=col)
    ax.plot(AGES, med/1e6, color=col, linewidth=2.0, label='Median')
    ax.set_xlabel('Age')
    ax.set_title(label, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))
axes[0].set_ylabel('Wealth')
fig.suptitle('Gamma Sensitivity — Wealth Paths Over Time', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('fig_gamma_sens_wealth_paths.png', dpi=110, bbox_inches='tight')
plt.close()
print('Saved fig_gamma_sens_wealth_paths.png')

# --- Plot 4: Allocation by Age ---
fig, axes = plt.subplots(1, len(_PROFILE_LABELS), figsize=(18, 5), sharey=True)
if len(_PROFILE_LABELS) == 1:
    axes = [axes]
for i, (label, col) in enumerate(zip(_PROFILE_LABELS, _PROFILE_COLORS)):
    ax = axes[i]
    res = sensitivity_results[label]
    act_vecs = res['action_vectors']
    n_strats = 3
    eff = np.zeros((len(AGES_STEPS), n_strats))
    for t, age in enumerate(AGES_STEPS):
        acts = res['action_by_age'].get(age, [])
        if not acts:
            continue
        for a in acts:
            eff[t] += np.array(act_vecs[a])
        eff[t] /= len(acts)
    bottom = np.zeros(len(AGES_STEPS))
    for s in range(n_strats):
        ax.bar(AGES_STEPS, eff[:, s], bottom=bottom, color=_STRAT_COLORS[s], label=_STRAT_NAMES[s], width=1.0)
        bottom += eff[:, s]
    ax.set_xlabel('Age')
    ax.set_title(label, fontsize=9)
    ax.set_ylim(0, 1.02)
    if i == 0:
        ax.set_ylabel('Mean Allocation Fraction')
    if i == len(_PROFILE_LABELS) - 1:
        ax.legend(loc='lower right', fontsize=8)
fig.suptitle('Gamma Sensitivity — Effective Portfolio Allocation by Age', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('fig_gamma_sens_alloc_age.png', dpi=110, bbox_inches='tight')
plt.close()
print('Saved fig_gamma_sens_alloc_age.png')

print('\nAll gamma sensitivity figures saved successfully.')