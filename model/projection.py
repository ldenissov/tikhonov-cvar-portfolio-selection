import numpy as np

################################################################################
#  Layout of the decision vector
################################################################################

def unpack(z, N, K):
    """Split the vector z into portfolios y (N x K) and auxiliary variables u (N)."""
    y = z[:N * K].reshape(N, K)
    u = z[N * K:]
    return y, u


def pack(y, u):
    """Inverse of unpack: recombine portfolios and auxiliary variables."""
    return np.concatenate([y.ravel(), u])


################################################################################
#  Projections
################################################################################

def project_simplex(v):
    """
    Projection onto the standard simplex.

    Exact sorting algorithm of
    M. Blondel, A. Fujino, N. Ueda, "Large-scale Multiclass Support Vector
    Machine Training via Euclidean Projection onto the Simplex",
    ICPR 2014, Algorithm 2.

    Source: https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    """
    K = len(v)
    w = np.sort(v)[::-1]
    cumsum = np.cumsum(w) - 1
    index = np.arange(1, K + 1)
    rho = index[w - cumsum / index > 0][-1]
    theta = cumsum[rho - 1] / rho
    return np.maximum(v - theta, 0)


# Counter: how many times the u-barriers have been triggered
clip_stats = {"calls": 0, "active": 0, "max_abs_u": 0.0}

def reset_clip_stats():
    """Reset the counter before a run."""
    clip_stats.update(calls=0, active=0, max_abs_u=0.0)

def project_Z(z, N, K, u_lo, u_hi):
    """
    Projection onto the truncated work set Z^R
    """
    y, u = unpack(z, N, K)
    y_proj = np.array([project_simplex(y[nu]) for nu in range(N)])
    u_proj = np.clip(u, u_lo, u_hi)

    # Log for the counter from above
    clip_stats["calls"] += 1
    if not np.array_equal(u, u_proj):
        clip_stats["active"] += 1
    clip_stats["max_abs_u"] = max(clip_stats["max_abs_u"],
                                  float(np.abs(u).max()))
    return pack(y_proj, u_proj)