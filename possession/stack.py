"""Second-stage features: the first stage's own outputs over windows before and after each frame.

Team-symmetric by construction: the team-swapped view is the same function with the H and A
probability columns exchanged.
"""
import numpy as np

from .features import _roll, _shift

WINDOWS = ((5, 0), (0, 5), (12, 0), (0, 12), (25, 0), (0, 25), (50, 0), (0, 50), (125, 0),
           (0, 125), (12, 12), (50, 50), (250, 250))
SHIFTS = (-25, -12, -5, 5, 12, 25)


def stack_features(prob, chg, bvis):
    team = prob[:, 0] - prob[:, 1]                    # > 0 leans home
    cols = {"team": team, "dead": prob[:, 2], "sw": chg[:, 1], "st": chg[:, 2], "rs": chg[:, 3],
            "bvis": bvis}
    feats = dict(cols)
    for name in ("team", "dead", "sw", "st", "rs"):
        for b, a in WINDOWS:
            feats["%s_%d_%d" % (name, b, a)] = _roll(cols[name], b, a)
    for name in ("team", "dead"):
        for k in SHIFTS:
            feats["%s_at%+d" % (name, k)] = _shift(cols[name], k)
    feats["team_flip_12"] = feats["team_0_12"] - feats["team_12_0"]
    feats["team_flip_50"] = feats["team_0_50"] - feats["team_50_0"]
    names = sorted(feats)
    return np.stack([feats[k] for k in names], 1).astype(np.float32), names


def views(prob, chg, bvis):
    """(normal view, team-swapped view) of the second-stage features."""
    F, _ = stack_features(prob, chg, bvis)
    Fs, _ = stack_features(prob[:, [1, 0, 2]], chg, bvis)
    return F, Fs
