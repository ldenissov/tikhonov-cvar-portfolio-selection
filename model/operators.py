
import numpy as np

from model.projection import pack, unpack, project_Z
from Regularization_Methods_for_HVIs import (
    LipschitzMonotoneOperator, MaximallyMonotoneOperator, Problem
)

################################################################################
#  CHKS smoothing of the plus function
################################################################################

def phi(t, eps):
    """Smoothed plus function"""
    return 0.5 * (t + np.sqrt(t ** 2 + 4.0 * eps ** 2))

def phi_prime(t, eps):
    """Derivative"""
    return 0.5 * (1.0 + t / np.sqrt(t ** 2 + 4.0 * eps ** 2))

################################################################################
#  Pseudogradient and objective values
################################################################################

def pseudogradient(z, r, b, v, Lam, alpha, eps):
    """
    Sample average approximation of the pseudogradient

    z:     (N(K+1),)  flat strategy profile
    r:     (S, N, K)  return scenarios
    b:     (N,)       budgets
    v:     (N, K)     current portfolios
    Lam:   (N, K, K)  market impact matrices
    alpha: (N,)       CVaR confidence levels
    eps:              CHKS smoothing parameter
    """
    S, N, K = r.shape
    y, u = unpack(z, N, K)

    # Losses
    L = -b * np.einsum('snk,nk->sn', r, y)
    w = phi_prime(L - u, eps)

    # Return and CVaR terms, both averaged over the S scenarios
    coef = 1.0 + w / (1.0 - alpha)
    grad_y = -b[:, None] * np.einsum('sn,snk->nk', coef, r) / S

    # Trading volumes and their aggregate over accounts
    d = b[:, None] * (y - v)
    aggregate = d.sum(axis=0)

    # Transaction costs
    for nu in range(N):
        grad_y[nu] += b[nu] * (Lam[nu] @ aggregate + Lam[nu].T @ d[nu])

    grad_u = 1.0 - w.mean(axis=0) / (1.0 - alpha)

    return pack(grad_y, grad_u)


def objective_nu(z, nu, r, b, v, Lam, alpha, eps):
    """
    Sample average approximation of E[f_nu]
    """
    N, K = v.shape
    y, u = unpack(z, N, K)

    income = -b[nu] * r[:, nu, :].mean(axis=0) @ y[nu]

    L = -b[nu] * (r[:, nu, :] @ y[nu])
    risk = u[nu] + phi(L - u[nu], eps).mean() / (1.0 - alpha[nu])

    d = b[:, None] * (y - v)
    costs = b[nu] * (y[nu] - v[nu]) @ (Lam[nu] @ d.sum(axis=0))

    return income + risk + costs


################################################################################
#  Lipschitz constant of the pseudogradient
################################################################################

def lipschitz_bound(r, b, Lam, alpha, eps):
    """
    Upper bound for the Lipschitz constant L_F of F_S.

    The pseudogradient splits into a transaction cost part with constant Jacobian
    and a smoothed CVaR part. The two are bounded separately and added.
    """
    S, N, K = r.shape
    dim = N * K + N

    # Transaction costs
    J = np.zeros((dim, dim))
    for nu in range(N):
        for lam in range(N):
            if lam == nu:
                block = b[nu] ** 2 * (Lam[nu] + Lam[nu].T)
            else:
                block = b[nu] * b[lam] * Lam[nu]
            J[nu * K:(nu + 1) * K, lam * K:(lam + 1) * K] = block
    L_tc = np.linalg.norm(J, 2)

    # Smoothed CVaR
    L_cvar = 0.0
    for nu in range(N):
        norm2 = (r[:, nu, :] ** 2).sum(axis=1)
        value = (b[nu] ** 2 * norm2 + 1.0).mean() / (4.0 * eps * (1.0 - alpha[nu]))
        L_cvar = max(L_cvar, value)

    return L_tc + L_cvar

################################################################################
#  Upper-level selection functions
################################################################################

def phi_sparse(z, N, K, xi, eta, v=None):
    """Phi_sp(z)"""
    y, u = unpack(z, N, K)
    return float(xi @ y.sum(axis=0) + 0.5 * eta * np.sum(y ** 2))

def grad_phi_sparse(z, N, K, xi, eta, v=None):
    """Gradient of Phi_sp"""
    y, u = unpack(z, N, K)
    return pack(xi + eta * y, np.zeros(N))

def phi_turnover(z, N, K, xi=None, eta=None, v=None):
    """Phi_turn(z)"""
    y, u = unpack(z, N, K)
    return float(0.5 * np.sum((y - v) ** 2))

def grad_phi_turnover(z, N, K, xi=None, eta=None, v=None):
    """Gradient of Phi_turn"""
    y, u = unpack(z, N, K)
    return pack(y - v, np.zeros(N))


################################################################################
#  Assembly of the hierarchical problem
################################################################################

def build_problem(r, b, v, Lam, alpha, eps, xi, eta, u_lo, u_hi, selection="sparse"):
    """
    Assemble the hierarchical problem VI(grad Phi, Zer(A + F))
    """
    S, N, K = r.shape

    F = LipschitzMonotoneOperator(
        evaluate=lambda z: pseudogradient(z, r, b, v, Lam, alpha, eps),
        L= lipschitz_bound(r, b, Lam, alpha, eps),
    )

    A = MaximallyMonotoneOperator(
        evaluate_resolvent=lambda z, gamma: project_Z(z, N, K, u_lo, u_hi)
    )

    if selection == "sparse":
        G = LipschitzMonotoneOperator(
            evaluate=lambda z: grad_phi_sparse(z, N, K, xi, eta),
            L=eta,
        )

    elif selection == "turnover":
        G = LipschitzMonotoneOperator(
            evaluate=lambda z: grad_phi_turnover(z, N, K, v=v),
            L=1.0,
        )

    else:
        raise ValueError(f"unknown selection function: {selection}")

    return Problem(leader=G, follower=A + F)