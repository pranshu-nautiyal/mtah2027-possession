"""Train the committed frame classifier on all proxy matches (none from the provided game).

    python training/train_final.py <proxy feature dir> <version>

Writes weights/<version>_model.txt and weights/<version>_features.txt. The proxy feature dir is
produced by proxy/exp_v1.py feat (proxy matches: IDSSE + Metrica open data, converted, harmonised
and dirtied by the scripts in proxy/).
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from possession import model  # noqa: E402

SUBSAMPLE = 5


def main(feat_dir, version):
    Xs, ys, names = [], [], None
    for p in sorted(glob.glob(os.path.join(feat_dir, "*.npz"))):
        z = np.load(p)
        names = [str(s) for s in z["names"]]
        g = np.flatnonzero(z["y"] != 3)[::SUBSAMPLE]
        Xs += [z["F"][g], z["Fs"][g]]
        ys += [z["y"][g], z["ys"][g]]
    X, y = np.concatenate(Xs), np.concatenate(ys)
    booster = model.train(X, y)
    model.save(booster, os.path.join(ROOT, "weights", "%s_model.txt" % version))
    open(os.path.join(ROOT, "weights", "%s_features.txt" % version), "w").write("\n".join(names) + "\n")
    print("trained on %d rows from %d proxy matches -> weights/%s_model.txt" % (
        len(y), len(glob.glob(os.path.join(feat_dir, "*.npz"))), version))


if __name__ == "__main__":
    main(*sys.argv[1:3])
