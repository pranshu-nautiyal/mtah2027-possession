"""Possession-length profile of an answer key: how many spells per alive minute, how long, how many
short ones. Used only to check which labelling convention the proxy keys should follow (A3)."""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "repo", "evaluation"))
import possession_metrics as pm

def profile(lab):
    runs = pm.valid_runs(lab)
    spells = pm.possession_spells(lab, runs)
    segs = pm.segments(lab, runs)
    dur = np.array([(b - a) / 25 for a, b, _ in spells])
    alive_min = np.isin(lab, (0, 1)).sum() / 25 / 60
    dead = [(b - a) / 25 for a, b, k in segs if k == 2]
    return dict(n=len(dur), per_alive_min=len(dur) / alive_min, mean=dur.mean(), med=np.median(dur),
                lt1_5=(dur < 1.5).mean(), lt4=(dur < 4).mean(), dead_per_min=len(dead) / alive_min,
                dead_med=np.median(dead) if dead else 0, dead_share=(lab == 2).sum() / (lab != 3).sum())

def show(name, lab):
    p = profile(lab)
    print(f"{name:<22} spells {p['n']:4d}  /alive-min {p['per_alive_min']:.2f}  mean {p['mean']:5.1f}s  med {p['med']:4.1f}s"
          f"  <1.5s {p['lt1_5']:.0%}  <4s {p['lt4']:.0%}  dead/alive-min {p['dead_per_min']:.2f}  dead med {p['dead_med']:4.1f}s  dead share {p['dead_share']:.0%}")

if __name__ == "__main__":
    gt = pm.label_codes(pm.read_label_csv(sys.argv[1]))
    show("BAR-ATH public 5 min", gt)
    for f in sorted(glob.glob(os.path.join(sys.argv[2], "*.npz"))):
        d = np.load(f)
        name = os.path.basename(f)[:-4]
        if "label_control" in d:
            show(name + " control", d["label_control"]); show(name + " touch", d["label_touch"])
        else:
            show(name, d["label"])
