"""Model v2 on the proxy leaderboard: v2 features, a frame model, a change-point model, and a
decoder whose switch / stop / restart costs follow the change-point model.

  feat    v2 features for every dirty proxy match (normal and team-swapped views)
  lomo    leave-one-match-out: frame model + change model trained on 9 matches, 10th predicted
  grid    choose the decoder settings on the pooled held-out predictions, report the board

usage: python proxy/exp_v2.py feat|lomo|grid <dirty_dir> <work_dir> [version]
"""
import glob
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
CHANGE_SUBSAMPLE = 3
CHANGE_HALF = 3          # a frame is "at a change" if one happens within +/-3 frames (0.12 s)
CHANGE_PARAMS = dict(num_class=4, min_data_in_leaf=100)


def change_target(y, half=CHANGE_HALF):
    """0 none, 1 team switch, 2 stoppage starts, 3 restart -- around every label change."""
    out = np.zeros(len(y), np.int8)
    idx = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] != 3) & (y[:-1] != 3)) + 1
    for t in idx:
        a, b = y[t - 1], y[t]
        kind = 2 if b == 2 else (3 if a == 2 else 1)
        out[max(0, t - half):t + half + 1] = kind
    return out


def _feat_one(args):
    path, out, version = args
    d = load(path)
    F, names = features.compute(d, version=version)
    Fs, _ = features.compute(d, swap=True, version=version)
    y = d["label"].astype(np.int8)
    np.savez(out, F=F, Fs=Fs, y=y, ys=features.swap_labels(y), c=change_target(y),
             names=np.array(names))
    return os.path.basename(path), F.shape


def feat(dirty_dir, work, version):
    os.makedirs(os.path.join(work, "feat"), exist_ok=True)
    jobs = [(p, os.path.join(work, "feat", os.path.basename(p)), version)
            for p in sorted(glob.glob(os.path.join(dirty_dir, "*.npz")))]
    t0 = time.time()
    with Pool(4) as pool:
        for name, shape in pool.imap(_feat_one, jobs):
            print("  %s %s" % (name, shape), flush=True)
    print("features done in %.0fs" % (time.time() - t0))


def train_rows(paths):
    """Frame-model rows and change-model rows from a set of proxy feature files."""
    Xf, yf, Xc, yc = [], [], [], []
    for p in paths:
        z = np.load(p)
        g = np.flatnonzero(z["y"] != 3)
        gf, gc = g[::SUBSAMPLE], g[::CHANGE_SUBSAMPLE]
        Xf += [z["F"][gf], z["Fs"][gf]]
        yf += [z["y"][gf], z["ys"][gf]]
        Xc += [z["F"][gc], z["Fs"][gc]]
        yc += [z["c"][gc], z["c"][gc]]          # change type does not depend on team naming
    return np.concatenate(Xf), np.concatenate(yf), np.concatenate(Xc), np.concatenate(yc)


def predict_change(booster, F, Fs):
    return (booster.predict(F, num_threads=8) + booster.predict(Fs, num_threads=8)) / 2


def lomo(work):
    paths = sorted(glob.glob(os.path.join(work, "feat", "*.npz")))
    os.makedirs(os.path.join(work, "probs"), exist_ok=True)
    for i, held in enumerate(paths):
        t0 = time.time()
        Xf, yf, Xc, yc = train_rows([p for j, p in enumerate(paths) if j != i])
        fm = model.train(Xf, yf)
        cm = model.train(Xc, yc, params=CHANGE_PARAMS)
        z = np.load(held)
        prob = model.predict(fm, z["F"], z["Fs"])
        chg = predict_change(cm, z["F"], z["Fs"])
        nm = os.path.basename(held)[:-4]
        np.save(os.path.join(work, "probs", nm + ".npy"), prob.astype(np.float32))
        np.save(os.path.join(work, "probs", nm + "_change.npy"), chg.astype(np.float32))
        g = z["y"] != 3
        acc = float(np.mean(np.argmax(prob[g], 1) == z["y"][g]))
        sw = z["c"] == 1
        print("  held out %-14s frame acc %.3f | change model: mean P(switch) at true switches %.3f, elsewhere %.4f  (%.0fs)" % (
            nm, acc, chg[sw, 1].mean(), chg[~sw & g, 1].mean(), time.time() - t0), flush=True)


def spell(gt, pred):
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"]


def load_preds(work):
    out = []
    for p in sorted(glob.glob(os.path.join(work, "probs", "*_change.npy"))):
        nm = os.path.basename(p)[:-len("_change.npy")]
        prob = np.load(os.path.join(work, "probs", nm + ".npy")).astype(np.float64)
        chg = np.load(p).astype(np.float64)
        y = np.load(os.path.join(work, "feat", nm + ".npz"))["y"]
        out.append((nm, prob, chg, y))
    return out


def board_score(data, base, alpha):
    return [spell(y, dec.decode_tv(p, c, y != 3, base, alpha)) for _, p, c, y in data]


def grid(work, out_json):
    data = load_preds(work)
    # start: v1-style constant costs (alpha = 0), best on this version's frame model
    best = None
    for sw in (8, 12, 16, 24):
        for st in (16, 32, 96):
            for rs in (16, 32):
                s = np.mean(board_score(data, (sw, st, rs), (0, 0, 0)))
                if best is None or s > best[0]:
                    best = (s, [sw, st, rs], [0.0, 0.0, 0.0])
    print("constant costs (v1-style decoder): %.4f at %s" % (best[0], best[1]), flush=True)
    # coordinate search: each change type's (base, alpha) in turn, two passes
    for _ in range(2):
        for k, (bases, alphas) in enumerate((([0, 2, 4, 6, 8, 12], [0, 0.5, 1, 1.5, 2, 3, 4]),
                                             ([0, 4, 8, 16, 32], [0, 0.5, 1, 2, 3, 4]),
                                             ([0, 4, 8, 16, 32], [0, 0.5, 1, 2, 3, 4]))):
            for b in bases:
                for a in alphas:
                    base, alpha = list(best[1]), list(best[2])
                    base[k], alpha[k] = b, a
                    s = np.mean(board_score(data, base, alpha))
                    if s > best[0] + 1e-6:
                        best = (s, base, alpha)
            print("  after %s: %.4f  base %s alpha %s" % (("switch", "stop", "restart")[k], best[0],
                                                         best[1], best[2]), flush=True)
    per = np.array(board_score(data, best[1], best[2]))
    rng = np.random.RandomState(0)
    boot = [per[rng.randint(0, len(per), len(per))].mean() for _ in range(2000)]
    print("\nproxy board: spellF1 %.4f  90%% [%.3f, %.3f]" % (per.mean(), *np.percentile(boot, [5, 95])))
    for (nm, _, _, _), s in zip(data, per):
        print("  %-14s %.3f" % (nm, s))
    json.dump({"base": best[1], "alpha": best[2], "proxy_spellF1": float(per.mean())},
              open(out_json, "w"), indent=1)


if __name__ == "__main__":
    cmd, dirty, work = sys.argv[1:4]
    version = sys.argv[4] if len(sys.argv) > 4 else "v2"
    if cmd == "feat":
        feat(dirty, work, version)
    elif cmd == "lomo":
        lomo(work)
    elif cmd == "grid":
        grid(work, os.path.join(work, "stage1_decoder.json"))
