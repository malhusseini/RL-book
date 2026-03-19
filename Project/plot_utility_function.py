"""
CRRA Utility Consumption Function — Visualization
=================================================
Generates a 3-panel figure illustrating the normalized CRRA utility used in
the Phase 3 DQN retirement project.

  U(w; γ) = [(w / w_ref)^(1−γ) − 1] / (1−γ),   γ ≠ 1
  U(w; γ) = log(w / w_ref),                        γ = 1

  γ(age)  = γ₀ + γ_slope × (age − age_start)
           = 2.0 + 0.03 × (age − 30)
  → γ(30) = 2.00,  γ(65) = 3.05

  w_ref   = $1,000,000
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive — save to file only
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Parameters (match project Config) ────────────────────────────────────────
W_REF       = 1_000_000.0   # normalization reference wealth
GAMMA_0     = 2.0            # risk aversion at age 30
GAMMA_SLOPE = 0.03           # increment per year
AGE_START   = 30
AGE_RETIRE  = 65

KEY_AGES    = [30, 40, 50, 60, 65]
PALETTE     = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]


def gamma_from_age(age: int) -> float:
    return GAMMA_0 + GAMMA_SLOPE * (age - AGE_START)


def crra_utility(w: np.ndarray, gamma: float) -> np.ndarray:
    w_norm = np.maximum(w, 1e-6) / W_REF
    if abs(gamma - 1.0) < 1e-9:
        return np.log(w_norm)
    return (w_norm ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def marginal_utility(w: np.ndarray, gamma: float) -> np.ndarray:
    """dU/dw = (w/w_ref)^(−γ) / w_ref"""
    w_norm = np.maximum(w, 1e-6) / W_REF
    return w_norm ** (-gamma) / W_REF


# ── Wealth grid ───────────────────────────────────────────────────────────────
w_vals = np.linspace(100_000, 1_000_000, 2_000)

# ── Figure layout: single panel — combined A+C (dual axis) ───────────────────
fig = plt.figure(figsize=(8, 5))
fig.patch.set_facecolor("white")

ax1 = fig.add_subplot(1, 1, 1)  # left y-axis: CRRA utility
ax3 = ax1.twinx()               # right y-axis: marginal utility

# ─────────────────────────────────────────────────────────────────────────────
# Panel A+C — CRRA utility (left) and marginal utility (right) vs wealth
# ─────────────────────────────────────────────────────────────────────────────
lines_legend = []
for age, color in zip(KEY_AGES, PALETTE):
    g  = gamma_from_age(age)
    u  = crra_utility(w_vals, g)
    mu = marginal_utility(w_vals, g)

    ln1, = ax1.plot(w_vals / 1e6, u,        color=color, lw=2.2, ls="-")
    ln2, = ax3.plot(w_vals / 1e6, mu * 1e6, color=color, lw=1.6, ls="--", alpha=0.75)
    lines_legend.append((ln1, ln2, f"Age {age}  (γ = {g:.2f})"))

# Reference wealth line
ax1.axvline(1.0, color="gray", lw=1.0, ls=":", alpha=0.55)
ax1.axhline(0.0, color="gray", lw=0.6, ls=":", alpha=0.4)
ax1.text(1.03, -48, r"$w_{\mathrm{ref}}$", fontsize=8.5, color="gray", va="bottom")

ax1.set_xlabel("Wealth  ($M)", fontsize=11)
ax1.set_ylabel("CRRA Utility  U(w; γ)  [solid]", fontsize=10.5, color="#333333")
ax3.set_ylabel(r"Marginal Utility  $U'(w;\,\gamma)$  [dashed, ×10⁻⁶]",
               fontsize=10.5, color="#555555")
ax3.set_ylim(bottom=0)

ax1.set_title("A+C — CRRA Utility & Diminishing Marginal Utility",
              fontsize=11.5, fontweight="bold")
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))
ax1.set_xlim(w_vals[0] / 1e6, w_vals[-1] / 1e6)
ax1.grid(True, alpha=0.22)

# Combined legend: one entry per age grouping both line styles
from matplotlib.lines import Line2D
legend_handles = []
for ln1, ln2, label in lines_legend:
    handle = Line2D([0], [0], color=ln1.get_color(), lw=2.0,
                    label=label)
    legend_handles.append(handle)

# Style indicators
solid_patch  = Line2D([0], [0], color="black", lw=2.0, ls="-",  label="— Utility U(w)")
dashed_patch = Line2D([0], [0], color="black", lw=1.6, ls="--", label="-- Marginal U′(w)")
legend_handles = [solid_patch, dashed_patch] + legend_handles

ax1.legend(handles=legend_handles, fontsize=8, framealpha=0.9,
           loc="lower right", ncol=1)


# ─────────────────────────────────────────────────────────────────────────────
# Overall title
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Normalized CRRA Utility Function  —  Phase 3 Retirement Project\n"
    r"$U(w;\,\gamma)=\dfrac{(w/w_{\mathrm{ref}})^{1-\gamma}-1}{1-\gamma}$"
    r",  $w_{\mathrm{ref}} = \$1\mathrm{M}$",
    fontsize=13, fontweight="bold", y=1.04
)

plt.savefig("fig_utility_consumption.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved → fig_utility_consumption.png")
