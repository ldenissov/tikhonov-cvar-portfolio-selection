
import numpy as np

################################################################################
#  Covariance structure
################################################################################

def simple_cov(K, sigma, rho):
    """
    Simple Covariance matrix
    """
    C = np.full((K, K), rho)
    np.fill_diagonal(C, 1.0)
    return sigma ** 2 * C


################################################################################
#  Scenario generation
################################################################################

def draw_normal(N, K, S, mu, cov, seed=None):
    """
    Draw S return scenarios from a multivariate normal distribution.

    Returns an array of shape (S, N, K), where r[s, nu, k] is the return of asset
    k that account nu observes in scenario s. Accounts may have heterogeneous
    beliefs, so each of them can be assigned its own mu and cov.

    Sample is generated once per experiment and then kept fixed.

    mu:  (K,) for identical expectations, (N, K) for heterogeneous ones
    cov: (K, K) for an identical covariance, (N, K, K) for heterogeneous ones
    """
    rng = np.random.default_rng(seed)

    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)

    # Transfer the homogeneous case to the heterogeneous layout
    if mu.ndim == 1:
        mu = np.tile(mu, (N, 1))
    if cov.ndim == 2:
        cov = np.tile(cov, (N, 1, 1))

    if mu.shape != (N, K):
        raise ValueError(f"mu should have shape ({N}, {K}), but has {mu.shape}")
    if cov.shape != (N, K, K):
        raise ValueError(f"cov should have shape ({N}, {K}, {K}), but has {cov.shape}")

    r = np.empty((S, N, K))
    for nu in range(N):
        r[:, nu, :] = rng.multivariate_normal(mu[nu], cov[nu], size=S)
    return r