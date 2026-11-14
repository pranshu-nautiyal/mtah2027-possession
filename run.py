"""Provided data in, submission CSV out. The one command that regenerates our file.

    python run.py --data <folder with tactical/> --out <team>.csv

Steps: read the raw tactical ATD -> per-frame features (normal and team-swapped views) -> frame
classifier with the committed weights -> Viterbi decoder with the committed costs -> CSV.
Everything runs on the CPU and is deterministic; the SHA-256 of the output is printed so two runs
can be compared byte for byte.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from possession import atd, decode, features, model, stack  # noqa: E402

LETTERS = np.array(["H", "A", "D", "X"])
N_FRAMES = 199319


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="folder holding tactical/BAR-ATH_tactical_atd.*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=os.path.join(HERE, "weights"))
    ap.add_argument("--version", default="v6")
    args = ap.parse_args()

    t0 = time.time()
    tac = os.path.join(args.data, "tactical")
    d = atd.load(os.path.join(tac, "BAR-ATH_tactical_atd.xml"),
                 os.path.join(tac, "BAR-ATH_tactical_atd.txt"), n_frames=N_FRAMES)
    print("tracking read: %d frames, %d track slots (%.0fs)" % (len(d["ball"]), len(d["slot_team"]),
                                                               time.time() - t0))
    w = lambda name: os.path.join(args.weights, "%s_%s" % (args.version, name))  # noqa: E731
    meta = w("meta.json")
    fversion = ("v1" if args.version == "v1" else
                json.load(open(meta))["feature_version"] if os.path.exists(meta) else "v2")
    F, names = features.compute(d, version=fversion)
    Fs, _ = features.compute(d, swap=True, version=fversion)
    if names != open(w("features.txt")).read().split():
        raise SystemExit("feature list differs from the one the weights were trained on")
    costs = json.load(open(w("decoder.json")))
    if args.version == "v1":
        prob = model.predict(model.load(w("model.txt")), F, Fs)
        codes = decode.decode(prob, d["graded"], costs["c_switch"], costs["c_stop"],
                              costs["c_restart"])
    else:
        # stage 1: frame probabilities and change-point probabilities
        prob1 = model.predict(model.load(w("frame.txt")), F, Fs)
        cm = model.load(w("change.txt"))
        chg = (cm.predict(F, num_threads=8) + cm.predict(Fs, num_threads=8)) / 2
        # stage 2: refine the frame probabilities from their own context
        bvis = F[:, names.index("bvis")].astype(np.float64)
        S, Ss = stack.views(prob1.astype(np.float64), chg.astype(np.float64), bvis)
        _, snames = stack.stack_features(prob1[:2], chg[:2], bvis[:2])
        if snames != open(w("stack_features.txt")).read().split():
            raise SystemExit("stage-2 feature list differs from the trained one")
        prob = model.predict(model.load(w("stack.txt")), S, Ss)
        codes = decode.decode_tv(prob, chg, d["graded"], costs["base"], costs["alpha"])

    labels = LETTERS[codes]
    with open(args.out, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("%d,%s\n" % (i, lab) for i, lab in enumerate(labels))
    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    g = d["graded"]
    counts = {k: int(np.sum(labels[g] == k)) for k in "HAD"}
    print("wrote %s  (%d rows)  graded H %d A %d D %d  in %.0fs" % (
        args.out, len(labels), counts["H"], counts["A"], counts["D"], time.time() - t0))
    print("sha256 %s" % digest)


if __name__ == "__main__":
    main()
