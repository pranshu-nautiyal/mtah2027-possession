"""The proxy leaderboard: score a predictor on every proxy match with the official metric code.

A predictor is any function  predict(match_dict) -> int8 codes (0 H, 1 A, 2 D, 3 X), one per frame.
Matches are the dirtied proxy .npz files. Scores use evaluation/possession_metrics.py unchanged,
graded on the frames the proxy key labels (not X).

Reports per match, the pooled mean, and a bootstrap interval over matches -- the number used to
decide whether a change ships (PLAN.md, ground rule 2).

usage: python proxy/board.py baseline <dirty_dir>      # the A6 honesty check
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "repo", "evaluation"))
import possession_metrics as pm  # noqa: E402

from common import load  # noqa: E402

H, A, D, X = 0, 1, 2, 3


def score(gt, pred):
    r = pm.evaluate(gt.astype(np.int8), pred.astype(np.int8))
    return r["spell_f1_mean"], r["macro_f1"], r["n_gt_spells"], r["n_pred_spells"]


def board(predictors, match_paths, verbose=True):
    """predictors: {name: fn}. Returns {name: (spell array, macro array)} over matches."""
    out = {name: ([], []) for name in predictors}
    for path in match_paths:
        d = load(path)
        gt = d["label"]
        row = []
        for name, fn in predictors.items():
            s, m, ng, npred = score(gt, fn(d))
            out[name][0].append(s)
            out[name][1].append(m)
            row.append("%s %.3f/%.3f (%d vs %d)" % (name, s, m, npred, ng))
        if verbose:
            print("%-14s %s" % (os.path.basename(path)[:-4], " | ".join(row)))
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in out.items()}


def summary(results, seed=0, n_boot=2000):
    rng = np.random.RandomState(seed)
    print("\n%-22s %8s %16s %8s" % ("predictor", "spellF1", "90% interval", "macroF1"))
    for name, (s, m) in results.items():
        boot = [s[rng.randint(0, len(s), len(s))].mean() for _ in range(n_boot)]
        print("%-22s %8.4f   [%.3f, %.3f] %8.4f" % (name, s.mean(), *np.percentile(boot, [5, 95]), m.mean()))


# -- the official baseline, re-implemented on the common format --------------------------------

def nearest_team(d):
    P, B, pitch, team = d["players"], d["ball"], d["pitch"], d["slot_team"]
    dx = (P[:, :, 0] - B[:, None, 0]) * pitch[0]
    dy = (P[:, :, 1] - B[:, None, 1]) * pitch[1]
    dist = np.where(np.isnan(dx), np.inf, dx * dx + dy * dy)
    best = np.argmin(dist, axis=1)
    codes = team[best].astype(np.int8)
    codes[np.isnan(B[:, 0]) | np.isinf(dist.min(axis=1))] = -1
    return codes


def smooth(codes, window):
    """Same as starter/baseline.py: centred majority vote, D where nobody is seen, ties to H."""
    if window <= 0:
        return np.where(codes == 0, H, np.where(codes == 1, A, D)).astype(np.int8)
    hc = np.concatenate(([0], np.cumsum(codes == 0)))
    ac = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    out = np.full(len(codes), D, np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], H, A)
    return out


def baseline(window):
    return lambda d: smooth(nearest_team(d), window)


if __name__ == "__main__":
    if sys.argv[1] == "baseline":
        paths = sorted(glob.glob(os.path.join(sys.argv[2], "*.npz")))
        res = board({"raw": baseline(0), "smooth25": baseline(25), "smooth50": baseline(50),
                     "smooth125": baseline(125)}, paths)
        summary(res)
        print("\npublished on the real full match:  raw 0.0486/0.6532  smooth25 0.1996/0.6752  "
              "smooth50 0.2433/0.6724  smooth125 0.2242/0.6356")
