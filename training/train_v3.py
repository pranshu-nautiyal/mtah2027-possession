"""Train the committed v3 models on all proxy matches (none from the provided game).

    python training/train_v3.py <proxy work dir> [version] [feature version]

v3 = v2 features -> frame model + change model -> second stage on the first stage's held-out
outputs -> time-varying Viterbi. Writes into weights/:
  v3_frame.txt, v3_change.txt, v3_stack.txt   LightGBM models
  v3_features.txt, v3_stack_features.txt      feature lists the models were trained on
  v3_decoder.json                             decoder settings chosen on the proxy board
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "proxy"))
from possession import model, stack  # noqa: E402
from possession.features import swap_labels  # noqa: E402

import exp_v2  # noqa: E402

W = os.path.join(ROOT, "weights")


def main(work, version="v3", feature_version="v2"):
    feats = sorted(glob.glob(os.path.join(work, "feat", "*.npz")))
    names = [str(s) for s in np.load(feats[0])["names"]]
    # stage 1: frame and change models on all proxy matches
    extra = [d for d in os.environ.get("EXTRA_FEAT_DIRS", "").split(":") if d]
    if extra:  # v6: also train on other simulated-noise versions of the same proxy matches
        exp_v2.SUBSAMPLE *= len(extra) + 1
        exp_v2.CHANGE_SUBSAMPLE *= len(extra) + 1
    rows = feats + [os.path.join(d, "feat", os.path.basename(f)) for d in extra for f in feats]
    Xf, yf, Xc, yc = exp_v2.train_rows(rows)
    model.save(model.train(Xf, yf), os.path.join(W, "%s_frame.txt" % version))
    model.save(model.train(Xc, yc, params=exp_v2.CHANGE_PARAMS), os.path.join(W, "%s_change.txt" % version))
    open(os.path.join(W, "%s_features.txt" % version), "w").write("\n".join(names) + "\n")
    print("stage 1 trained: frame rows %d, change rows %d" % (len(yf), len(yc)))
    # stage 2: on the held-out (leave-one-match-out) stage-1 outputs of every proxy match
    Xs, ys, snames = [], [], None
    bvis_col = names.index("bvis")
    for nm, prob, chg, y in exp_v2.load_preds(work):
        bvis = np.load(os.path.join(work, "feat", nm + ".npz"))["F"][:, bvis_col].astype(np.float64)
        F, Fs = stack.views(prob, chg, bvis)
        _, snames = stack.stack_features(prob[:2], chg[:2], bvis[:2])
        g = np.flatnonzero(y != 3)[::5]
        Xs += [F[g], Fs[g]]
        ys += [y[g], swap_labels(y)[g]]
    model.save(model.train(np.concatenate(Xs), np.concatenate(ys)), os.path.join(W, "%s_stack.txt" % version))
    open(os.path.join(W, "%s_stack_features.txt" % version), "w").write("\n".join(snames) + "\n")
    shutil.copy(os.path.join(work, "stack_decoder.json"), os.path.join(W, "%s_decoder.json" % version))
    json.dump({"feature_version": feature_version}, open(os.path.join(W, "%s_meta.json" % version), "w"))
    print("stage 2 trained on %d rows; decoder %s" % (
        sum(len(v) for v in ys), json.load(open(os.path.join(W, "%s_decoder.json" % version)))))


if __name__ == "__main__":
    main(*sys.argv[1:4])
