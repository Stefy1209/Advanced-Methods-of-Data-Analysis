"""Fuzzy C-Means (FCM) and fuzzy cluster validity indices, in plain NumPy.

Notation follows the lecture "05a fuzzy-methods" (H.F. Pop, UBB Cluj):
    X  : data, n points x p features (one point x_k per row)
    c  : number of clusters
    m  : fuzzifier, m > 1
    U  : fuzzy partition, c x n, u_ik in [0, 1], each column sums to 1
    V  : prototypes (cluster centres), c x p, one prototype v_i per row
    d_ik = ||x_k - v_i||  (Euclidean, i.e. A = identity on slide 16)

Objective (slide 16):
    J_m(U, V; X) = sum_i sum_k u_ik^m * d_ik^2

Algorithm (slide 17):
    1. initialise a fuzzy partition U
    2. compute prototypes   v_i = sum_k u_ik^m x_k / sum_k u_ik^m
    3. compute new partition u_ik = 1 / sum_j (d_ik^2 / d_jk^2)^(1/(m-1))
    4. stop if the partitions are close enough (||U_new - U_old||_F < eps), else go to 2
"""

from dataclasses import dataclass, field

import numpy as np

TINY = np.finfo(np.float64).eps  # guards against division by zero when a point coincides with a prototype


@dataclass
class FCMResult:
    U: np.ndarray                 # final fuzzy partition, c x n
    V: np.ndarray                 # final prototypes, c x p
    J_history: list = field(default_factory=list)  # objective value at every iteration
    n_iter: int = 0
    converged: bool = False
    U0: np.ndarray = None         # initial partition (kept for reproducibility / cross-checks)

    @property
    def J(self):
        return self.J_history[-1]

    @property
    def labels(self):
        """Hard assignment: each point goes to the cluster of maximal membership."""
        return self.U.argmax(axis=0)


def random_partition(c, n, rng):
    """Random fuzzy partition: uniform random numbers, columns normalised to sum to 1."""
    U = rng.random((c, n))
    return U / U.sum(axis=0, keepdims=True)


def squared_distances(X, V):
    """d_ik^2 = ||x_k - v_i||^2 for all prototypes i and points k; returns a c x n matrix."""
    return ((V[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)


def prototypes(X, U, m):
    """Step 2 (slide 17): weighted means of the data, with weights u_ik^m."""
    Um = U ** m
    return (Um @ X) / Um.sum(axis=1, keepdims=True)


def memberships(D2, m):
    """Step 3 (slide 16): u_ik = 1 / sum_j (d_ik^2 / d_jk^2)^(1/(m-1)).

    Computed in the equivalent form u_ik = d_ik^(-2/(m-1)) / sum_j d_jk^(-2/(m-1)).
    """
    D2 = np.fmax(D2, TINY)
    # Dividing each column by its smallest distance does not change the result (it cancels in the
    # normalisation) but keeps the powers in floating-point range when m is close to 1.
    W = (D2 / D2.min(axis=0, keepdims=True)) ** (-1.0 / (m - 1.0))
    return W / W.sum(axis=0, keepdims=True)


def fcm(X, c, m=2.0, eps=1e-5, max_iter=1000, seed=None, U0=None):
    """Run Fuzzy C-Means on X (n x p).

    Either give an initial partition U0 (c x n) or a seed for a random one.
    Stopping rule: Frobenius norm of the change in U below eps, or max_iter iterations.
    The objective recorded at each iteration is J_m(U_old, V_new), the value the
    prototype step has just minimised.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if m <= 1:
        raise ValueError("the fuzzifier m must be > 1")

    if U0 is None:
        U0 = random_partition(c, n, np.random.default_rng(seed))
    U = np.fmax(np.array(U0, dtype=float), TINY)

    result = FCMResult(U=U, V=None, U0=np.array(U0, dtype=float))
    for it in range(1, max_iter + 1):
        V = prototypes(X, U, m)                       # step 2
        D2 = squared_distances(X, V)
        result.J_history.append(float(((U ** m) * D2).sum()))
        U_new = memberships(D2, m)                    # step 3
        change = np.linalg.norm(U_new - U)            # step 4
        U = U_new
        if change < eps:
            result.converged = True
            break

    result.U, result.V, result.n_iter = U, V, it
    return result


def fcm_multistart(X, c, m=2.0, n_init=10, seed=0, eps=1e-5, max_iter=1000, rel_tol=1e-6):
    """Run FCM from n_init random initial partitions and keep the run with the lowest J.

    FCM only reaches a local minimum of J, so the result can depend on the start.
    Run r uses seed + r, so every run is reproducible on its own.

    Returns (best_result, info) where info records, for the log:
      J_all            : final J of every run
      n_at_best        : how many runs ended within rel_tol of the best J (same minimum)
      n_not_converged  : runs that hit max_iter before the stopping rule
    """
    runs = [fcm(X, c, m=m, eps=eps, max_iter=max_iter, seed=seed + r) for r in range(n_init)]
    J_all = np.array([r.J for r in runs])
    best = runs[int(J_all.argmin())]
    info = {
        "J_all": J_all,
        "n_at_best": int((J_all <= best.J * (1 + rel_tol)).sum()),
        "n_not_converged": int(sum(not r.converged for r in runs)),
    }
    return best, info


# ----------------------------------------------------------------------------
# Possibilistic C-Means (lecture 05a, slides 21-23; Krishnapuram & Keller 1993, 1996)
# ----------------------------------------------------------------------------

def pcm_eta(X, U, V, m):
    """Scale of each cluster, from an FCM result (slide 21, no a priori information):
    eta_i = sum_k u_ik^m d_ik^2 / sum_k u_ik^m   (a membership-weighted mean squared distance).
    A point at squared distance eta_i from prototype i gets typicality 0.5 in cluster i.
    """
    Um = U ** m
    return (Um * squared_distances(X, V)).sum(axis=1) / Um.sum(axis=1)


def typicalities(D2, eta, m):
    """t_ik = 1 / (1 + (d_ik^2 / eta_i)^(1/(m-1)))  (slide 21). No sum-to-one constraint across clusters."""
    return 1.0 / (1.0 + (D2 / eta[:, None]) ** (1.0 / (m - 1.0)))


def pcm(X, V0, eta, m=2.0, eps=1e-5, max_iter=1000):
    """Possibilistic C-Means started from prototypes V0 (normally the FCM prototypes, slide 22),
    with the cluster scales eta kept fixed.

    Objective: J = sum_i sum_k t_ik^m d_ik^2 + sum_i eta_i sum_k (1 - t_ik)^m.
    Iterates typicalities -> prototypes (same weighted-mean formula as FCM, with t_ik^m as weights)
    until ||T_new - T_old||_F < eps. Result.U holds the typicalities T (c x n).
    """
    X = np.asarray(X, dtype=float)
    V = np.array(V0, dtype=float)
    T = typicalities(squared_distances(X, V), eta, m)
    result = FCMResult(U=T, V=V, U0=T.copy())
    for it in range(1, max_iter + 1):
        V = prototypes(X, T, m)
        D2 = squared_distances(X, V)
        T_new = typicalities(D2, eta, m)
        result.J_history.append(float(((T_new ** m) * D2).sum() + (eta[:, None] * (1 - T_new) ** m).sum()))
        change = np.linalg.norm(T_new - T)
        T = T_new
        if change < eps:
            result.converged = True
            break
    result.U, result.V, result.n_iter = T, V, it
    return result


# ----------------------------------------------------------------------------
# Validity indices (lecture 05a, "Global Quality Measures", slides 75-82)
# ----------------------------------------------------------------------------

def partition_coefficient(U):
    """PC = (1/n) sum_k sum_i u_ik^2. Maximise. 1/c = uniform partition, 1 = crisp (Bezdek 1974)."""
    return float((U ** 2).sum() / U.shape[1])


def normalized_partition_coefficient(U):
    """NPC = (PC * c - 1) / (c - 1). Maximise. 0 = uniform, 1 = crisp."""
    c = U.shape[0]
    return (partition_coefficient(U) * c - 1) / (c - 1)


def partition_entropy(U):
    """PE = -(1/n) sum_k sum_i u_ik ln u_ik. Minimise. 0 = crisp, ln c = uniform (Bezdek 1975)."""
    Uc = np.fmax(U, TINY)
    return float(-(U * np.log(Uc)).sum() / U.shape[1])


def normalized_partition_entropy(U):
    """NPE = PE / ln c. Minimise. 0 = crisp, 1 = uniform."""
    return partition_entropy(U) / np.log(U.shape[0])


def xie_beni(X, U, V, exponent=2.0):
    """XB = sum_i sum_k u_ik^2 ||x_k - v_i||^2 / (n * min_{i != j} ||v_i - v_j||^2). Minimise.

    Compactness divided by the separation of the two closest prototypes (Xie & Beni 1991).
    The lecture (slide 82) uses u_ik^2; pass exponent=m for the generalised version.
    """
    n = X.shape[0]
    compactness = ((U ** exponent) * squared_distances(X, V)).sum()
    VV = squared_distances(V, V)
    separation = VV[~np.eye(len(V), dtype=bool)].min()
    if separation == 0:          # two prototypes coincide exactly: XB is undefined, reported as infinity
        return float("inf")
    return float(compactness / (n * separation))


def fukuyama_sugeno(X, U, V, m):
    """FS = sum_i sum_k u_ik^m (||x_k - v_i||^2 - ||v_i - v_bar||^2). Minimise.

    Compactness minus separation of the prototypes from the grand mean v_bar
    (Fukuyama & Sugeno 1989). Not in the lecture slides; added as a fourth criterion.
    """
    v_bar = X.mean(axis=0, keepdims=True)
    Um = U ** m
    compact = (Um * squared_distances(X, V)).sum()
    separation = (Um * squared_distances(v_bar, V)).sum()   # ||v_i - v_bar||^2 broadcast over k
    return float(compact - separation)


def prototype_diagnostics(X, result, rel_tol=1e-3):
    """Checks whether the prototypes are really distinct (degeneracy check).

    min_proto_dist : smallest Euclidean distance between two prototypes
    n_distinct     : number of prototypes left after merging those closer than rel_tol * data scale,
                     where data scale = sqrt(total variance of X)
    dist_to_cog    : mean distance of the prototypes to the centre of gravity (mean of X), relative to data scale
    mean_max_u     : mean over points of the largest membership (1/c = completely uniform, 1 = crisp)
    """
    V, U = result.V, result.U
    scale = np.sqrt(X.var(axis=0).sum())
    D = np.sqrt(squared_distances(V, V))
    min_dist = D[~np.eye(len(V), dtype=bool)].min()
    groups = []                                   # greedy merging of near-identical prototypes
    for i in range(len(V)):
        if not any(D[i, j] < rel_tol * scale for j in groups):
            groups.append(i)
    cog = X.mean(axis=0, keepdims=True)
    return {
        "min_proto_dist": float(min_dist),
        "n_distinct": len(groups),
        "dist_to_cog": float(np.sqrt(squared_distances(cog, V)).mean() / scale),
        "mean_max_u": float(U.max(axis=0).mean()),
    }


def validity_indices(X, result, m):
    """All indices for one FCM result, as a dict (one row of a results table)."""
    U, V = result.U, result.V
    return {
        "PC": partition_coefficient(U),
        "NPC": normalized_partition_coefficient(U),
        "PE": partition_entropy(U),
        "NPE": normalized_partition_entropy(U),
        "XB": xie_beni(X, U, V),
        "FS": fukuyama_sugeno(X, U, V, m),
    }
