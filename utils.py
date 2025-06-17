import numpy as np
import math
try:                                           # Hungarian assignment
    from scipy.optimize import linear_sum_assignment
    _has_scipy = True
except ImportError:
    _has_scipy = False                         # will fall back to greedy match


# ---------------------------- helper geometry --------------------------------
def iou_matrix(a, b):
    """
    Return IoU matrix between two lists of boxes.
    a, b: (N,4) and (M,4) arrays of xyxy int
    """
    N, M = len(a), len(b)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=float)
    a = a.astype(float); b = b.astype(float)
    # areas
    area_a = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
    area_b = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])

    # broadcast intersection
    xa1 = np.maximum(a[:, None, 0], b[None, :, 0])
    ya1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xa2 = np.minimum(a[:, None, 2], b[None, :, 2])
    ya2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(xa2 - xa1, 0, None) * np.clip(ya2 - ya1, 0, None)

    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def greedy_match(cost):
    """Simple greedy min-cost match if SciPy is absent."""
    matches = []
    used_rows, used_cols = set(), set()
    while True:
        # find smallest unused cost
        min_val, min_rc = math.inf, None
        for r in range(cost.shape[0]):
            if r in used_rows: continue
            c_idx = np.argmin(cost[r])
            if c_idx in used_cols: continue
            val = cost[r, c_idx]
            if val < min_val:
                min_val, min_rc = (r, c_idx)
        if min_rc is None:
            break
        r, c = min_rc
        matches.append((r, c))
        used_rows.add(r); used_cols.add(c)
    return matches