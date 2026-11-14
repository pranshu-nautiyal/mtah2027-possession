"""Model v1 on the proxy leaderboard.

  feat    features for every dirty proxy match, normal and team-swapped views
  lomo    leave-one-match-out: train on 9 matches, predict the 10th, for each match
  grid    choose the decoder's three costs on the pooled held-out predictions, report the board

usage: python proxy/exp_v1.py feat|lomo|grid <dirty_dir> <work_dir>
"""
import glob
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "..", "repo", "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession import decode as dec  # noqa: E402
from possession import features, model  # noqa: E402

from common import load  # noqa: E402

SUBSAMPLE = 5


def _feat_one(args):
    path, out = args
    d = load(path)
    F, names = features.compute(d, version="v1")
    Fs, _ = features.compute(d, swap=True, version="v1")
    y = d["label"].astype(np.int8)
    np.savez(out, F=F, Fs=Fs, y=y, ys=features.swap_labels(y), names=np.array(names))
    return os.path.basename(path), F.shape


def feat(dirty_dir, work):
    os.makedirs(os.path.join(work, "feat"), exist_ok=True)
    jobs = [(p, os.path.join(work, "feat", os.path.basename(p))) for p in
            sorted(glob.glob(os.path.join(dirty_dir, "*.npz")))]
    t0 = time.time()
    with Pool(4) as pool:
        for name, shape in pool.imap(_feat_one, jobs):
            print("  %s %s" % (name, shape), flush=True)
    print("features done in %.0fs" % (time.time() - t0))


def _train_rows(paths):
    Xs, ys = [], []
    for p in paths:
        z = np.load(p)
        g = np.flatnonzero(z["y"] != 3)[::SUBSAMPLE]
        Xs += [z["F"][g], z["Fs"][g]]
        ys += [z["y"][g], z["ys"][g]]
    return np.concatenate(Xs), np.concatenate(ys)


def lomo(work):
    paths = sorted(glob.glob(os.path.join(work, "feat", "*.npz")))
    os.makedirs(os.path.join(work, "probs"), exist_ok=True)
    for i, held in enumerate(paths):
        t0 = time.time()
        X, y = _train_rows([p for j, p in enumerate(paths) if j != i])
        booster = model.train(X, y)
        z = np.load(held)
        prob = model.predict(booster, z["F"], z["Fs"])
        np.save(os.path.join(work, "probs", os.path.basename(held)[:-4] + ".npy"), prob.astype(np.float32))
        g = z["y"] != 3
        acc = float(np.mean(np.argmax(prob[g], 1) == z["y"][g]))
        print("  held out %-14s train rows %d  frame accuracy %.3f  (%.0fs)" % (
            os.path.basename(held)[:-4], len(y), acc, time.time() - t0), flush=True)


def spell(gt, pred):
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    return pm.spell_scores(gt, filled, runs)["mean_f1"]


def grid(work, out_json):
    names = [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(work, "probs", "*.npy")))]
    data = []
    for nm in names:
        prob = np.load(os.path.join(work, "probs", nm + ".npy")).astype(np.float64)
        y = np.load(os.path.join(work, "feat", nm + ".npz"))["y"]
        data.append((nm, prob, y))
    # argmax, no decoder: what the frame model alone gives
    raw = [spell(y, np.where(y == 3, 3, np.argmax(p, 1)).astype(np.int8)) for _, p, y in data]
    print("frame model alone (argmax): spellF1 %.4f" % np.mean(raw))
    grid_sw = [8, 12, 16, 20, 24, 32, 48, 64]
    grid_st = [16, 24, 32, 48, 64, 96]
    grid_rs = [16, 24, 32, 48, 64, 96]
    best = None
    rows = []
    for c in itertools.product(grid_sw, grid_st, grid_rs):
        s = [spell(y, dec.decode(p, y != 3, *c)) for _, p, y in data]
        rows.append((float(np.mean(s)), c, s))
        if best is None or rows[-1][0] > best[0]:
            best = rows[-1]
            print("  new best %.4f at switch %s stop %s restart %s" % (best[0], *c), flush=True)
    score, c, per = best
    rng = np.random.RandomState(0)
    per = np.array(per)
    boot = [per[rng.randint(0, len(per), len(per))].mean() for _ in range(2000)]
    print("\nv1 on the proxy board: spellF1 %.4f  90%% [%.3f, %.3f]   (baseline smooth50: 0.2607)" % (
        score, *np.percentile(boot, [5, 95])))
    for nm, s in zip(names, per):
        print("  %-14s %.3f" % (nm, s))
    # how flat is the optimum? neighbours within 0.005
    near = [r for r in rows if r[0] > score - 0.005]
    print("%d of %d cost settings within 0.005 of the best" % (len(near), len(rows)))
    json.dump({"c_switch": c[0], "c_stop": c[1], "c_restart": c[2], "proxy_spellF1": score},
              open(out_json, "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "feat":
        feat(sys.argv[2], sys.argv[3])
    elif cmd == "lomo":
        lomo(sys.argv[3])
    elif cmd == "grid":
        grid(sys.argv[3], os.path.join(ROOT, "weights", "v1_decoder.json"))
