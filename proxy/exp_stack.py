"""Second stage on top of a version's held-out predictions ("stacking").

The first-stage frame model sees each frame through hand-made features. The second stage sees the
first stage's own H/A/D probabilities and change probabilities over windows before and after each
frame, and learns which patterns of evidence mean what -- e.g. a short dip toward the other team
right after a change spike is a real short possession, a lone dip is noise.

Trained leave-one-match-out on held-out first-stage outputs, so it never sees predictions the
first stage made on its own training data.

usage: python proxy/exp_stack.py lomo|grid <work_dir>
"""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import model  # noqa: E402
from possession.features import swap_labels  # noqa: E402
from possession.stack import stack_features, views  # noqa: E402,F401

import exp_v2  # noqa: E402

SUBSAMPLE = 5


def lomo(work):
    data = exp_v2.load_preds(work)
    out_dir = os.path.join(work, "stack")
    os.makedirs(out_dir, exist_ok=True)
    cached = {}
    for nm, prob, chg, y in data:
        z = np.load(os.path.join(work, "feat", nm + ".npz"))
        names = [str(s) for s in z["names"]]
        bvis = z["F"][:, names.index("bvis")].astype(np.float64)
        cached[nm] = views(prob, chg, bvis) + (y,)
    for i, (nm, prob, chg, y) in enumerate(data):
        t0 = time.time()
        Xs, ys = [], []
        for j, (other, _, _, yo) in enumerate(data):
            if j == i:
                continue
            F, Fs, _ = cached[other]
            g = np.flatnonzero(yo != 3)[::SUBSAMPLE]
            Xs += [F[g], Fs[g]]
            ys += [yo[g], swap_labels(yo)[g]]
        booster = model.train(np.concatenate(Xs), np.concatenate(ys))
        F, Fs, _ = cached[nm]
        p2 = model.predict(booster, F, Fs)
        np.save(os.path.join(out_dir, nm + ".npy"), p2.astype(np.float32))
        g = y != 3
        print("  held out %-14s stage-2 frame acc %.3f (stage 1 %.3f)  (%.0fs)" % (
            nm, np.mean(np.argmax(p2[g], 1) == y[g]), np.mean(np.argmax(prob[g], 1) == y[g]),
            time.time() - t0), flush=True)


def grid(work):
    data = exp_v2.load_preds(work)
    stacked = [(nm, np.load(os.path.join(work, "stack", nm + ".npy")).astype(np.float64), chg, y)
               for nm, _, chg, y in data]
    best = None
    for sw in (8, 12, 16, 24):
        for st in (16, 32, 96):
            for rs in (16, 32):
                s = np.mean(exp_v2.board_score(stacked, (sw, st, rs), (0, 0, 0)))
                if best is None or s > best[0]:
                    best = (s, [sw, st, rs], [0.0, 0.0, 0.0])
    print("stacked, constant costs: %.4f at %s" % (best[0], best[1]), flush=True)
    for _ in range(2):
        for k, (bases, alphas) in enumerate((([0, 2, 4, 6, 8, 12], [0, 0.5, 1, 1.5, 2, 3, 4]),
                                             ([0, 4, 8, 16, 32], [0, 0.5, 1, 2, 3, 4]),
                                             ([0, 4, 8, 16, 32], [0, 0.5, 1, 2, 3, 4]))):
            for b in bases:
                for a in alphas:
                    base, alpha = list(best[1]), list(best[2])
                    base[k], alpha[k] = b, a
                    s = np.mean(exp_v2.board_score(stacked, base, alpha))
                    if s > best[0] + 1e-6:
                        best = (s, base, alpha)
            print("  after %s: %.4f  base %s alpha %s" % (("switch", "stop", "restart")[k],
                                                         best[0], best[1], best[2]), flush=True)
    per = np.array(exp_v2.board_score(stacked, best[1], best[2]))
    rng = np.random.RandomState(0)
    boot = [per[rng.randint(0, len(per), len(per))].mean() for _ in range(2000)]
    print("\nstacked proxy board: spellF1 %.4f  90%% [%.3f, %.3f]" % (per.mean(), *np.percentile(boot, [5, 95])))
    for (nm, _, _, _), s in zip(stacked, per):
        print("  %-14s %.3f" % (nm, s))
    json.dump({"base": best[1], "alpha": best[2], "proxy_spellF1": float(per.mean())},
              open(os.path.join(work, "stack_decoder.json"), "w"), indent=1)


if __name__ == "__main__":
    {"lomo": lomo, "grid": grid}[sys.argv[1]](sys.argv[2])
