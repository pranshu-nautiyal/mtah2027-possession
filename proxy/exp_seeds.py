"""v6 experiment: train on several simulated-noise versions of each proxy match.

Leave-one-match-out exactly as exp_v2, but the 9 training matches contribute rows from every
noise seed's feature dir; the held-out match is always scored on seed 0 (the same test as v5).
Writes probs/ and feat/ (seed 0 copies) into <out_work> so exp_stack.py and compare.py work.

usage: python proxy/exp_seeds.py <out_work> <seed0_work> <seed_work>...
"""
import glob
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from possession import model  # noqa: E402

import exp_v2  # noqa: E402


def main(out, seed0, *others):
    os.makedirs(os.path.join(out, "probs"), exist_ok=True)
    os.makedirs(os.path.join(out, "feat"), exist_ok=True)
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(seed0, "feat", "*.npz")))
    exp_v2.SUBSAMPLE *= len(others) + 1          # same row count as v5, more noise variety
    exp_v2.CHANGE_SUBSAMPLE *= len(others) + 1
    for nm in names:
        dst = os.path.join(out, "feat", nm)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(os.path.join(seed0, "feat", nm)), dst)
    for held in names:
        t0 = time.time()
        train = [os.path.join(w, "feat", n) for w in (seed0,) + others for n in names if n != held]
        Xf, yf, Xc, yc = exp_v2.train_rows(train)
        fm = model.train(Xf, yf)
        cm = model.train(Xc, yc, params=exp_v2.CHANGE_PARAMS)
        z = np.load(os.path.join(seed0, "feat", held))
        prob = model.predict(fm, z["F"], z["Fs"])
        chg = exp_v2.predict_change(cm, z["F"], z["Fs"])
        stem = held[:-4]
        np.save(os.path.join(out, "probs", stem + ".npy"), prob.astype(np.float32))
        np.save(os.path.join(out, "probs", stem + "_change.npy"), chg.astype(np.float32))
        g = z["y"] != 3
        print("  held out %-14s frame acc %.3f (%.0fs)" % (
            stem, np.mean(np.argmax(prob[g], 1) == z["y"][g]), time.time() - t0), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
