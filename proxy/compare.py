"""Paired comparison of two stacked versions on the proxy board -- the ship/no-ship test.

A change ships only if it wins by more than the noise (PLAN.md, ground rule 2). Both versions are
scored on the same held-out matches, so the honest test is on the per-match differences: mean
difference, a bootstrap interval over matches, and how many matches it wins.

usage: python proxy/compare.py <work_dir_A> <work_dir_B>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_v2  # noqa: E402


def per_match(work):
    dec = json.load(open(os.path.join(work, "stack_decoder.json")))
    data = exp_v2.load_preds(work)
    stacked = [(nm, np.load(os.path.join(work, "stack", nm + ".npy")).astype(np.float64), chg, y)
               for nm, _, chg, y in data]
    names = [s[0] for s in stacked]
    return names, np.array(exp_v2.board_score(stacked, dec["base"], dec["alpha"]))


def main(a, b):
    na, sa = per_match(a)
    nb, sb = per_match(b)
    assert na == nb, "different match sets"
    diff = sb - sa
    rng = np.random.RandomState(0)
    boot = [diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [5, 95])
    print("%-14s %8s %8s %8s" % ("match", "A", "B", "B-A"))
    for nm, x, y in zip(na, sa, sb):
        print("%-14s %8.3f %8.3f %+8.3f" % (nm, x, y, y - x))
    print("\nA %.4f  B %.4f  mean B-A %+.4f  90%% [%+.4f, %+.4f]  B wins %d/%d" % (
        sa.mean(), sb.mean(), diff.mean(), lo, hi, int((diff > 0).sum()), len(diff)))
    verdict = "B is better" if lo > 0 else ("A is better" if hi < 0 else "no clear difference")
    print("verdict:", verdict)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
