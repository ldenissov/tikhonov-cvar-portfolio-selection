################################################################################
#  Standard Example
################################################################################

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Regularization_Methods_for_HVIs import Scheduler, Algorithm

from model.operators import (
    build_problem, objective_nu, pseudogradient
)
from model.projection import (
    pack, unpack, project_Z, clip_stats, reset_clip_stats
)
from model.sampling import simple_cov, draw_normal


################################################################################
#  Model Parameters
################################################################################

N = 2                       # number of accounts
K = 3                       # number of assets
b = np.array([1.0, 1.5])    # budgets

# Current portfolios v_1, v_2
v = np.array([[0.45, 0.15, 0.40],
              [0.10, 0.30, 0.60]])

# CVaR confidence levels and CHKS smoothing parameter
alpha_cvar = np.array([0.90, 0.90])
eps = 0.1

# Return distribution
sigma = 0.20
rho = 0.3
mu = np.array([[0.05, 0.04, 0.03],     # account 1
               [0.02, 0.04, 0.07]])    # account 2

# Monte Carlo sample size per account
S = 100
seed_returns = 42

# Truncation of the auxiliary CVaR variables
u_lo, u_hi = -1, 1

# Market impact
Lambda = np.array([[0.4, 0.2, 0.2],
                   [0.2, 0.4, 0.2],
                   [0.2, 0.2, 0.4]])


################################################################################
#  Selection Parameters
################################################################################

xi = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
eta = 0.1 / np.sqrt(14)


################################################################################
#  Algorithm Parameters
################################################################################

max_iterations = 50000   # number of outer iterations
alpha_prox = 38.0        # proximal parameter
theta = 0.7              # relaxation parameter

# Polynomial sequences
beta = Scheduler(0.55, 1.0)
epsilon = Scheduler(0.55 + 1.45, 1e-3)


################################################################################
#  Output Parameters for creating the plots
################################################################################

# Evaluate the residual only every few outer iterations
residual_every = 25

# Sample size study
sample_sizes = [5, 10, 50, 100, 200, 500]
sample_study_iterations = 10000
sample_study_selection = "turnover"

figure_dir = Path("figures")
table_dir = Path("tables")
figure_dir.mkdir(parents=True, exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)


################################################################################
#  LaTeX Tables and formatting of the numbers
################################################################################

def sci(x, digits=2):
    """
    Format a number as d.dd * 10^e.
    """
    if x == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    y = x / 10.0 ** exponent
    return f"{y:.{digits}f} \\cdot 10^{{{exponent}}}"


def write_latex_table(path, columns, rows, caption, label, align=None,
                      placement="htbp"):
    """
    Write a LaTeX table.
    """
    align = align or "l" + "r" * (len(columns) - 1)
    lines = [r"\begin{table}[" + placement + "]",
             r"  \centering",
             r"  \begin{tabular}{" + align + "}",
             r"    \toprule",
             "    " + " & ".join(columns) + r" \\",
             r"    \midrule"]
    for row in rows:
        lines.append("    " + (row if isinstance(row, str)
                               else " & ".join(row) + r" \\"))
    lines += [r"    \bottomrule",
              r"  \end{tabular}",
              rf"  \caption{{{caption}}}",
              rf"  \label{{{label}}}",
              r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


################################################################################
#  Problem Definition
################################################################################

# Both accounts use the same market impact matrix
Lam = np.array([Lambda for _ in range(N)])

# Return scenarios, drawn once and then kept fixed for both runs
r = draw_normal(N, K, S, mu, simple_cov(K, sigma, rho), seed=seed_returns)

# Starting point: equally weighted portfolios and u = 0
w0 = pack(np.full((N, K), 1.0 / K), np.zeros(N))


################################################################################
#  Run Experiments
################################################################################

# Same instance and same sample for both runs, only the selection function changes

selections = ["sparse", "turnover"]
labels = {"sparse": r"$\Phi_{\mathrm{sp}}$",
          "turnover": r"$\Phi_{\mathrm{turn}}$"}

trajectories = {}
solutions = {}
truncation_report = {}

for selection in selections:
    problem = build_problem(r, b, v, Lam, alpha_cvar, eps, xi, eta,
                            u_lo, u_hi, selection)
    algorithm = Algorithm(problem, max_iterations, beta, epsilon)

    reset_clip_stats()
    trajectories[selection] = algorithm.solve(w0, theta, alpha_prox,
                                              progress=tqdm)
    solutions[selection] = trajectories[selection][-1]

    # Diagnostics for the truncation
    u_max = max(np.abs(unpack(z, N, K)[1]).max()
                for z in trajectories[selection])
    active = clip_stats["active"]
    calls = max(clip_stats["calls"], 1)

    truncation_report[selection] = (u_max, active, calls)
    print(f"{selection}: max|u|={u_max:.4f}, clip active in "
          f"{active} of {calls} projections")

################################################################################
#  Residual Trajectories
################################################################################

def natural_residual(z):
    """Lower-level residual """
    F = pseudogradient(z, r, b, v, Lam, alpha_cvar, eps)
    return np.linalg.norm(z - project_Z(z - F, N, K, u_lo, u_hi))

residual_curves = {}

for selection in selections:
    history = trajectories[selection]
    indices = np.arange(0, len(history), residual_every, dtype=int)
    if indices[-1] != len(history) - 1:
        indices = np.append(indices, len(history) - 1)

    residuals = np.array([
        natural_residual(history[n])
        for n in tqdm(indices, desc=f"Residuals: {selection}", leave=False)
    ])

    residual_curves[selection] = (indices, residuals)


################################################################################
#  Figure of the residual per iterations
################################################################################

plt.figure(figsize=(7.2, 4.6))

for selection in selections:
    indices, residuals = residual_curves[selection]
    plt.semilogy(indices, residuals, label=f"{selection} selection")

plt.xlabel("Outer iteration $n$")
plt.ylabel(r"Natural residual $\|z-P_{Z^R}(z-\widehat F_S(z))\|$")
plt.title("Lower-level residual (standard example)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(figure_dir / "standard_residual.pdf", bbox_inches="tight")
plt.savefig(figure_dir / "standard_residual.png", dpi=300, bbox_inches="tight")
plt.close()


################################################################################
#  Table of computed Portfolios
################################################################################
rows_portfolios = []

for nu in range(N):
    for selection in selections:
        y, u = unpack(solutions[selection], N, K)
        rows_portfolios.append(
            [f"Account {nu + 1}, {labels[selection]}"]
            + [f"{val:.4f}" for val in y[nu]]
        )

write_latex_table(
    table_dir / "table3_standard_portfolios.tex",
    ["", r"$y_{\nu,1}$", r"$y_{\nu,2}$", r"$y_{\nu,3}$"],
    rows_portfolios,
    "Computed portfolios in the standard example.",
    "tab:standard-portfolios",
)

################################################################################
#  Table of distances and lower level Residuals
################################################################################

diff = solutions["sparse"] - solutions["turnover"]
distance_1 = np.abs(diff).sum()
distance_inf = np.abs(diff).max()
residual_final = {s: natural_residual(solutions[s]) for s in selections}

write_latex_table(
    table_dir / "table4_standard_distances.tex",
    ["Quantity", "Value"],
    [[r"$\|z^*_{\mathrm{sp}} - z^*_{\mathrm{turn}}\|_1$", f"${sci(distance_1)}$"],
     [r"$\|z^*_{\mathrm{sp}} - z^*_{\mathrm{turn}}\|_\infty$", f"${sci(distance_inf)}$"],
     [r"Residual at $z^*_{\mathrm{sp}}$", f"${sci(residual_final['sparse'])}$"],
     [r"Residual at $z^*_{\mathrm{turn}}$", f"${sci(residual_final['turnover'])}$"],
     [r"$\max_n \max_\nu |u_\nu^n|$ for $\Phi_{\mathrm{sp}}$",
      f"{truncation_report['sparse'][0]:.4f}"],
     [r"$\max_n \max_\nu |u_\nu^n|$ for $\Phi_{\mathrm{turn}}$",
      f"{truncation_report['turnover'][0]:.4f}"]],
    "Distances and lower-level residuals in the standard example.",
    "tab:standard-distances",
)


################################################################################
#  Table of Expected Objective Values
################################################################################

rows_objectives = []

for selection in selections:
    values = [objective_nu(solutions[selection], nu, r, b, v, Lam,
                           alpha_cvar, eps) for nu in range(N)]
    rows_objectives.append([labels[selection]]
                           + [f"{val:.6f}" for val in values])

write_latex_table(
    table_dir / "table5_standard_objectives.tex",
    ["", r"$\mathbb{E}[f_1]$", r"$\mathbb{E}[f_2]$"],
    rows_objectives,
    "Expected objective values in the standard example.",
    "tab:standard-objectives",
)


################################################################################
#  Sample Size Study
################################################################################

r_pool = draw_normal(N, K, max(sample_sizes), mu,
                     simple_cov(K, sigma, rho), seed=seed_returns)

study = {}

for S_test in sample_sizes:
    r_S = r_pool[:S_test]

    problem_S = build_problem(r_S, b, v, Lam, alpha_cvar, eps, xi, eta,
                              u_lo, u_hi, sample_study_selection)
    algorithm_S = Algorithm(problem_S, sample_study_iterations, beta, epsilon)

    z_S = algorithm_S.solve(w0, theta, alpha_prox, progress=tqdm)[-1]

    # Residual of z_S with respect to its own sample average F_S
    F_S = pseudogradient(z_S, r_S, b, v, Lam, alpha_cvar, eps)
    residual_S = np.linalg.norm(z_S - project_Z(z_S - F_S, N, K, u_lo, u_hi))

    study[S_test] = {
        "z": z_S,
        "residual": residual_S,
        "u_max": np.abs(unpack(z_S, N, K)[1]).max(),
    }

# The largest sample serves as the reference point
z_reference = study[max(sample_sizes)]["z"]

for S_test in sample_sizes:
    study[S_test]["distance"] = np.linalg.norm(study[S_test]["z"] - z_reference)


################################################################################
#  Table: Sample Size Study
################################################################################

rows_samples = []

for S_test in sample_sizes:
    entry = study[S_test]
    is_reference = S_test == max(sample_sizes)

    rows_samples.append([
        f"${S_test}$",
        f"${sci(entry['residual'])}$",
        "reference" if is_reference else f"${sci(entry['distance'])}$",
        f"{entry['u_max']:.4f}",
    ])

write_latex_table(
    table_dir / "table_standard_sample_size.tex",
    ["$S$", "Residual",
     r"$\|z^*_S - z^*_{S_{\max}}\|$", r"$\max_\nu |u_\nu|$"],
    rows_samples,
    "Effect of the sample size in the standard example.",
    "tab:standard-sample-size",
)

print(f"\nFigures saved in: {figure_dir.resolve()}")
print(f"Tables saved in:  {table_dir.resolve()}")