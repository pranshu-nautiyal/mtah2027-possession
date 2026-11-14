"""Turn per-frame probabilities into a clean H/A/D sequence.

Viterbi over three states with a fixed cost for every change of state, so a label only changes
when the evidence for the new state outweighs the cost. Separate costs for a team-to-team switch,
a stoppage starting and play restarting, because they fail in different ways. The costs are
chosen on the proxy leaderboard, never on the provided match.

Decoded independently inside each graded run (each half), as the scorer does.
"""
import numba
import numpy as np

H, A, D, X = 0, 1, 2, 3


@numba.njit(cache=True)
def _viterbi(logp, cost):
    n = logp.shape[0]
    back = np.zeros((n, 3), np.int8)
    score = logp[0].copy()
    new = np.empty(3)
    for t in range(1, n):
        for j in range(3):
            best = -1e18
            arg = 0
            for i in range(3):
                v = score[i] - cost[i, j]
                if v > best:
                    best = v
                    arg = i
            new[j] = best + logp[t, j]
            back[t, j] = arg
        for j in range(3):
            score[j] = new[j]
    path = np.empty(n, np.int8)
    k = 0
    for j in range(1, 3):
        if score[j] > score[k]:
            k = j
    path[n - 1] = k
    for t in range(n - 1, 0, -1):
        k = back[t, k]
        path[t - 1] = k
    return path


@numba.njit(cache=True)
def _viterbi_tv(logp, nlp, base, alpha):
    """Viterbi with time-varying change costs. nlp[t] = -log P(change of each type at t) for
    (team switch, stoppage starts, restart); cost = base + alpha * nlp. Cheap where the change
    model sees a real touch or restart, expensive elsewhere."""
    n = logp.shape[0]
    back = np.zeros((n, 3), np.int8)
    score = logp[0].copy()
    new = np.empty(3)
    for t in range(1, n):
        c_sw = base[0] + alpha[0] * nlp[t, 0]
        c_st = base[1] + alpha[1] * nlp[t, 1]
        c_rs = base[2] + alpha[2] * nlp[t, 2]
        for j in range(3):
            best = -1e18
            arg = 0
            for i in range(3):
                if i == j:
                    c = 0.0
                elif j == 2:
                    c = c_st
                elif i == 2:
                    c = c_rs
                else:
                    c = c_sw
                v = score[i] - c
                if v > best:
                    best = v
                    arg = i
            new[j] = best + logp[t, j]
            back[t, j] = arg
        for j in range(3):
            score[j] = new[j]
    path = np.empty(n, np.int8)
    k = 0
    for j in range(1, 3):
        if score[j] > score[k]:
            k = j
    path[n - 1] = k
    for t in range(n - 1, 0, -1):
        k = back[t, k]
        path[t - 1] = k
    return path


def decode_tv(prob, change, graded, base, alpha, floor=1e-4):
    """prob (n, 3) frame probabilities; change (n, 4) P(none, switch, stop, restart) near t."""
    logp = np.log(np.clip(prob, floor, 1.0))
    nlp = -np.log(np.clip(change[:, 1:4], floor, 1.0))
    base = np.asarray(base, np.float64)
    alpha = np.asarray(alpha, np.float64)
    out = np.full(len(prob), X, np.int8)
    for a, b in graded_runs(graded):
        out[a:b] = _viterbi_tv(logp[a:b], nlp[a:b], base, alpha)
    return out


def cost_matrix(c_switch, c_stop, c_restart):
    return np.array([[0.0, c_switch, c_stop],
                     [c_switch, 0.0, c_stop],
                     [c_restart, c_restart, 0.0]])


def graded_runs(graded):
    e = np.diff(np.concatenate(([0], graded.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def decode(prob, graded, c_switch, c_stop, c_restart, floor=1e-4):
    """prob (n, 3) -> int8 codes; X outside graded runs."""
    logp = np.log(np.clip(prob, floor, 1.0))
    cost = cost_matrix(c_switch, c_stop, c_restart)
    out = np.full(len(prob), X, np.int8)
    for a, b in graded_runs(graded):
        out[a:b] = _viterbi(logp[a:b], cost)
    return out
