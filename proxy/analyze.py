"""Where does a version lose points? Error breakdown on the proxy board's held-out predictions.

usage: python proxy/analyze.py <work_dir> <decoder.json>
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "repo", "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession import decode as dec  # noqa: E402

BUCKETS = [0, 1.5, 4, 8, 16, 1e9]


def bucket_counts(durs):
    return np.histogram(durs, [b * 25 for b in BUCKETS])[0]


def analyze(work, costs):
    tot = dict(gt=np.zeros(5), missed=np.zeros(5), pred=np.zeros(5), spurious=np.zeros(5))
    lat, ad, team, trans12, trans25, n_gt, n_pred = [], [], [], [], [], 0, 0
    why = dict(no_switch=0, d_confusion=0, boundary=0, wrong_team=0)
    for p in sorted(glob.glob(os.path.join(work, "probs", "*.npy"))):
        nm = os.path.basename(p)[:-4]
        prob = np.load(p).astype(np.float64)
        y = np.load(os.path.join(work, "feat", nm + ".npz"))["y"]
        pred = dec.decode(prob, y != 3, costs["c_switch"], costs["c_stop"], costs["c_restart"])
        r = pm.evaluate(y, pred)
        ad.append(r["alive_dead_f1"])
        team.append(r["team_acc_both_alive"])
        trans12.append(r["transitions"][12]["f1"])
        trans25.append(r["transitions"][25]["f1"])
        if r["transitions"][25]["mean_latency"] is not None:
            lat.append(r["transitions"][25]["mean_latency"])
        runs = pm.valid_runs(y)
        gs, ps = pm.possession_spells(y, runs), pm.possession_spells(pred, runs)
        n_gt += len(gs)
        n_pred += len(ps)
        flags = pm.matched_gt_flags(pm.spell_overlaps(gs, ps), len(gs), 0.5)
        # which predictions were matched, at IoU 0.5
        ov = pm.spell_overlaps(gs, ps)
        used_p, used_g = set(), set()
        for iou, gi, pi in ov:
            if iou < 0.5:
                break
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
        gd = np.array([b - a for a, b, _ in gs])
        pdur = np.array([b - a for a, b, _ in ps])
        tot["gt"] += bucket_counts(gd)
        tot["missed"] += bucket_counts(gd[~flags])
        tot["pred"] += bucket_counts(pdur)
        tot["spurious"] += bucket_counts(np.array([pdur[i] for i in range(len(ps)) if i not in used_p]))
        # why was each missed GT spell missed?
        for gi in np.flatnonzero(~flags):
            a, b, k = gs[gi]
            seg = pred[a:b]
            same = np.mean(seg == k)
            if np.mean(seg == 2) > 0.5:
                why["d_confusion"] += 1
            elif np.mean(seg == 1 - k) > 0.5:
                why["wrong_team"] += 1 if same < 0.2 else 0
                why["no_switch"] += 1 if same >= 0.2 else 0
            elif same >= 0.5:
                why["boundary"] += 1
            else:
                why["no_switch"] += 1
    print("possessions: true %d, predicted %d" % (n_gt, n_pred))
    print("alive/dead F1 %.3f | team accuracy when both alive %.3f | switch F1 +/-0.5s %.3f, +/-1s %.3f | mean switch lag %+.1f frames" % (
        np.mean(ad), np.mean(team), np.mean(trans12), np.mean(trans25), np.mean(lat)))
    print("\n%-12s %8s %8s %9s %8s %9s %9s" % ("duration", "true", "missed", "miss %", "pred", "spurious", "spur %"))
    labels = ["<1.5s", "1.5-4s", "4-8s", "8-16s", ">16s"]
    for i, lab in enumerate(labels):
        print("%-12s %8d %8d %8.0f%% %8d %9d %8.0f%%" % (
            lab, tot["gt"][i], tot["missed"][i], 100 * tot["missed"][i] / max(1, tot["gt"][i]),
            tot["pred"][i], tot["spurious"][i], 100 * tot["spurious"][i] / max(1, tot["pred"][i])))
    print("\nmissed possessions (IoU<0.5), main cause:", why)


if __name__ == "__main__":
    analyze(sys.argv[1], json.load(open(sys.argv[2])))
