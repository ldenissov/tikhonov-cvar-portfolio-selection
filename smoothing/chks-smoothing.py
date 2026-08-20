################################################################################
# Plots the CHKS smoothing function
################################################################################

import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

def phi(t, eps):
    """CHKS smoothing function"""
    return (t + np.sqrt(t**2 + 4.0 * eps**2)) / 2.0

def plus(t):
    """Function (t)^+  """
    return np.maximum(t, 0.0)


EPSILONS = [0.5, 0.25, 0.1]
T_MIN, T_MAX = -2.0, 2.0

t = np.linspace(T_MIN, T_MAX, 2001)

fig, ax = plt.subplots(figsize=(5.2, 3.4))

# nonsmooth plus function
ax.plot(t, plus(t), color="black", linestyle="--", linewidth=1.2,
        label=r"$(t)^+$", zorder=3)

# smoothed versions
for eps in EPSILONS:
    ax.plot(t, phi(t, eps), linewidth=1.4,
            label=rf"$\phi_\varepsilon,\ \varepsilon={eps}$")

# vertical gap at the origin
eps_max = max(EPSILONS)
ax.annotate(
    "",
    xy=(0.0, eps_max), xytext=(0.0, 0.0),
    arrowprops=dict(arrowstyle="<->", linewidth=0.8, color="0.35"),
)
ax.text(0.06, eps_max / 2.0, r"$\varepsilon$", color="0.35", fontsize=9,
        va="center")

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\phi_\varepsilon(t)$")
ax.set_xlim(T_MIN, T_MAX)
ax.set_ylim(-0.1, 2.0)
ax.legend(frameon=False, loc="upper left")

fig.tight_layout()

os.makedirs("figures", exist_ok=True)
out = os.path.join("figures", "chks_smoothing.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"written: {out}")