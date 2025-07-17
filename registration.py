import numpy as np

def umeyama_similarity(P: np.ndarray,
                       Q: np.ndarray,
                       weights: np.ndarray = None,
                       with_scaling: bool = False):
    """
    Weighted similarity (rigid+optional scale) fit of P → Q via Umeyama/Kabsch.
    P, Q: (N,3) corresponding points
    weights: (N,) non-negative weights (if None → uniform)
    with_scaling: if True, computes uniform scale; otherwise s = 1.
    Returns (s, R, t) so that:  Q ≈ s * R @ P + t
    """
    assert P.shape == Q.shape
    N = P.shape[0]
    if weights is None:
        w = np.ones(N)
    else:
        w = weights.copy()
    w_sum = w.sum()
    # 1) weighted centroids
    mu_P = (w[:,None] * P).sum(axis=0) / w_sum
    mu_Q = (w[:,None] * Q).sum(axis=0) / w_sum

    # 2) demean
    Pc = P - mu_P
    Qc = Q - mu_Q

    # 3) weighted covariance
    S = (Pc.T * w) @ Qc / w_sum

    # 4) SVD
    U, D, Vt = np.linalg.svd(S)
    S_mat = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_mat[-1,-1] = -1
    R = U @ S_mat @ Vt

    # 5) scale
    if with_scaling:
        var_P = (w[:,None] * (Pc**2)).sum() / w_sum
        s = np.trace(np.diag(D) @ S_mat) / var_P
    else:
        s = 1.0

    # 6) translation
    t = mu_Q - s * (R @ mu_P)

    return s, R, t

# def umeyama_similarity(P, Q, with_scaling=True):
#     """
#     Estimate similarity transform (s, R, t) that maps P to Q:
#         Q ≈ s * R @ P + t

#     P, Q: (N,3) numpy arrays of corresponding points.
#     with_scaling: if False, forces s=1 (pure‐rigid).
#     Returns:
#         s: scalar scale
#         R: (3×3) rotation
#         t: (3,)    translation
#     """
#     # 1. centroids
#     mu_P = P.mean(axis=0)
#     mu_Q = Q.mean(axis=0)
#     P_centered = P - mu_P
#     Q_centered = Q - mu_Q

#     # 2. covariance
#     H = P_centered.T @ Q_centered / P.shape[0]

#     # 3. SVD
#     U, S_values, Vt = np.linalg.svd(H)
#     R = Vt.T @ U.T

#     # Reflection check
#     if np.linalg.det(R) < 0:
#         Vt[-1, :] *= -1
#         R = Vt.T @ U.T

#     # 4. scale
#     if with_scaling:
#         var_P = (P_centered**2).sum() / P.shape[0]
#         # sum of singular values
#         scale = S_values.sum() / var_P
#     else:
#         scale = 1.0

#     # 5. translation: mu_Q = scale*R*mu_P + t  →  t = mu_Q - scale*R*mu_P
#     t = mu_Q - scale * (R @ mu_P)

#     return scale, R, t

from sklearn.neighbors import KDTree

def icp_umeyama(source, target, max_iterations=50, tol=1e-6,
               init_R=None, init_t=None):
    """
    Align `source` to `target` using ICP with Kabsch at each iteration.
    Returns aligned_source, R_total, t_total, rmse_history.
    """
    src = source.copy()

    # Optional initial guess
    if init_R is not None and init_t is not None:
        src = (init_R @ src.T).T + init_t

    tree = KDTree(target)
    s_total = 1.0
    R_total = np.eye(3)
    t_total = np.zeros(3)
    prev_error = np.inf
    rmse_history = []

    for i in range(max_iterations):
        dists, idxs = tree.query(src, k=1)
        # corr = target[idxs.ravel()]
        max_corr_dist = 0.005        # e.g. 5 cm
        mask = (dists.ravel() < max_corr_dist)
        src_valid = src[mask]
        corr_valid = target[idxs.ravel()[mask]]

        # estimate similarity
        s, R, t = umeyama_similarity(src_valid, corr_valid, with_scaling=False)

        # apply transform
        src = (s * (R @ src.T)).T + t

        # accumulate
        s_total *= s
        R_total = R @ R_total
        t_total = s * (R @ t_total) + t

        rmse = np.sqrt((dists**2).mean())
        rmse_history.append(rmse)
        if abs(prev_error - rmse) < tol:
            break
        prev_error = rmse

    return src, s_total, R_total, t_total, rmse_history

def icp_color_trimmed(src_pts: np.ndarray,
                      src_col: np.ndarray,
                      tgt_pts: np.ndarray,
                      tgt_col: np.ndarray,
                      max_iterations: int = 50,
                      tol: float = 1e-6,
                      max_corr_dist: float = 0.05,
                      keep_frac: float = 0.75,
                      sigma_color: float = 10.0):
    """
    ICP loop that fuses geometry + color, rejects outliers, and trims worst residuals.

    src_pts, tgt_pts: (N,3)/(M,3) float point coordinates
    src_col, tgt_col: (N,3)/(M,3) color in Lab (or RGB) space
    max_corr_dist: max geometric correspondence distance (meters)
    keep_frac: fraction of best correspondences to keep each iter
    sigma_color: color-kernel scale (same units as src_col)
    Returns:
      - aligned_src: (N,3) transformed source
      - R_total, t_total: cumulative rigid transform (no scale)
      - rmse_history: list of RMSE per iteration
    """
    # copy to avoid mutating inputs
    src = src_pts.copy()
    # build KD-tree on target geometry (xyz)
    tree = KDTree(tgt_pts)

    R_total = np.eye(3)
    t_total = np.zeros(3)
    s_total = 1.0
    prev_rmse = np.inf
    rmse_history = []

    for it in range(max_iterations):
        # 1) find nearest neighbor
        dists, idxs = tree.query(src, k=1)
        dists = dists.ravel()
        corr_pts = tgt_pts[idxs.ravel()]
        corr_col = tgt_col[idxs.ravel()]

        # 2) geometry‐based masking
        mask = dists < max_corr_dist
        if mask.sum() < 3:
            break

        # 3) color‐based weights (Gaussian)
        col_diff = np.linalg.norm(src_col - corr_col, axis=1)
        w_color = np.exp(-0.5 * (col_diff / sigma_color)**2)

        # restrict to masked subset
        src_m = src[mask]
        q_m   = corr_pts[mask]
        w_m   = w_color[mask]
        d_m   = dists[mask]

        # 4) trimmed ICP: keep best keep_frac by geometric residual
        k = max(3, int(len(d_m) * keep_frac))
        idx_keep = np.argsort(d_m)[:k]
        P_trim = src_m[idx_keep]
        Q_trim = q_m[idx_keep]
        w_trim = w_m[idx_keep]

        # 5) weighted Umeyama (rigid)
        s, R, t = umeyama_similarity(P_trim, Q_trim, weights=w_trim,
                                     with_scaling=False)

        # 6) apply to full src
        src = (s * (R @ src.T)).T + t

        # 7) accumulate
        s_total *= s
        R_total = R @ R_total
        t_total = s * (R @ t_total) + t

        # 8) compute RMSE
        rmse = np.sqrt((d_m[idx_keep]**2).mean())
        rmse_history.append(rmse)
        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

    return src, s_total, R_total, t_total, rmse_history
