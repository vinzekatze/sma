"""
Market-phase labels for regime-aware neighbour search.

A phase captures WHERE ratio sits relative to the MA trend AND which
direction it is moving.  Two identical Δratio patterns in different phases
(e.g. one while ratio > up_thr and one while ratio < dn_thr) likely have
different outcomes — so we restrict the neighbour pool to the same phase.

Encoding
--------
  position   0 = ratio < dn_thr   ("below MA")
             1 = dn_thr ≤ ratio ≤ up_thr  ("neutral")
             2 = ratio > up_thr   ("above MA")
  direction  0 = falling   (sum of last smooth_k Δratio ≤ 0)
             1 = rising

  label = position * 2 + direction   →  0 … 5
"""

from __future__ import annotations
import numpy as np

PHASE_NAMES = [
    "ниже MA ↓", "ниже MA ↑",
    "нейтрал  ↓", "нейтрал  ↑",
    "выше MA  ↓", "выше MA  ↑",
]


def pool_phase_labels(
    ratio: np.ndarray,
    dratio: np.ndarray,
    p: int,
    dn_thr: float = 0.98,
    up_thr: float = 1.02,
    smooth_k: int = 5,
) -> np.ndarray:
    """
    Phase label (int8, 0–5) for each of the N = len(dratio)–p rows of the
    delay matrix built from *dratio*.

    For row k the "current bar" is ratio[k+p] and the direction is the sign
    of sum(dratio[k+p−smooth_k : k+p]).
    """
    N = len(dratio) - p
    if N <= 0:
        return np.empty(0, dtype=np.int8)

    k = np.arange(N)
    ratio_idx = k + p                               # index into ratio
    pos = np.where(
        ratio[ratio_idx] < dn_thr, 0,
        np.where(ratio[ratio_idx] > up_thr, 2, 1),
    )

    # vectorised smoothed direction via cumsum
    cs = np.concatenate([[0.0], np.cumsum(dratio)])  # length len(dratio)+1
    end_d   = k + p                                  # exclusive end into dratio
    start_d = np.maximum(0, end_d - smooth_k)
    dirs = (cs[end_d] - cs[start_d] > 0).astype(np.int8)

    return (pos * 2 + dirs).astype(np.int8)


def query_phase_label(
    ratio: np.ndarray,
    dratio: np.ndarray,
    origin_k: int,
    dn_thr: float = 0.98,
    up_thr: float = 1.02,
    smooth_k: int = 5,
) -> int:
    """Phase label for the query bar at *origin_k*."""
    r = float(ratio[origin_k])
    pos = 0 if r < dn_thr else (2 if r > up_thr else 1)
    tail = dratio[max(0, origin_k - smooth_k): origin_k]
    direction = 1 if (len(tail) > 0 and float(tail.sum()) > 0) else 0
    return int(pos * 2 + direction)
