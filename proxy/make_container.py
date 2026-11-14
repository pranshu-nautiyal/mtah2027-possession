"""Package a dirty proxy match so it looks exactly like the BAR-ATH data bundle.

Any pipeline written for the provided match can then run on a proxy match unchanged, and its
output is scored against the proxy's answer key with the official scorer. Used to compare other
teams' public pipelines with ours on the proxy board.

Layout written to <out>/:
  tactical/BAR-ATH_tactical_atd.xml          BAR-ATH's metadata, attack direction set from the data
  tactical/BAR-ATH_tactical_atd.txt          199,319 frames; the proxy's halves start at BAR-ATH's
                                             half starts (11185, 110808), padding frames are empty
  tactical/BAR-ATH_tactical_homography.csv   a constant, valid matrix (BAR-ATH's own median)
  tactical/BAR-ATH_tactical_smart_tagging.xml  no instances (proxies have no Smart Tagging)
  gt.csv                                     the proxy answer key on the same 199,319 frames, X elsewhere

Slots follow BAR-ATH's metadata: 24 home slots (index 23 = goalkeeper), 22 away (index 45 = GK).

usage: python proxy/make_container.py <bar_meta.xml> <bar_homography.csv> <dirty.npz> <out_dir>
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load  # noqa: E402

N = 199319
HALF_STARTS = (11185, 110808)
HALF_LENGTHS = (90651 - 11185 + 1, 190197 - 110808 + 1)
HOME_SLOTS, AWAY_SLOTS = list(range(0, 24)), list(range(24, 46))
HOME_GK, AWAY_GK = 23, 45
LETTERS = np.array(["H", "A", "D", "X"])


def slot_map(slot_team, slot_gk):
    """proxy slot -> BAR slot (or -1 if it does not fit)."""
    out = np.full(len(slot_team), -1)
    for team, slots, gk in ((0, HOME_SLOTS, HOME_GK), (1, AWAY_SLOTS, AWAY_GK)):
        idx = np.flatnonzero(slot_team == team)
        free = [s for s in slots if s != gk]
        for j in idx:
            if slot_gk[j] and gk not in out:
                out[j] = gk
            elif free:
                out[j] = free.pop(0)
    return out


def write_smart_tagging(path):
    """Only the two kick-off tags, as 5 s windows straddling each half's start (BAR-ATH's own
    file has its first kick-off window 443.88-448.88 s around graded frame 11185). Half starts
    are published boundaries, not labels. Some pipelines refuse to run without them; nothing
    else is tagged, because the proxy matches have no Smart Tagging."""
    inst = []
    for i, (start, half) in enumerate(zip(HALF_STARTS, ("1st Half", "2nd Half")), start=1):
        t0 = (start - 88) / 25.0
        inst.append(
            "    <instance>\n      <ID>%d</ID>\n      <start>%.2f</start>\n      <end>%.2f</end>\n"
            "      <code>SET PIECES</code>\n"
            "      <label>\n        <group>Team</group>\n        <text>N/A</text>\n      </label>\n"
            "      <label>\n        <group>Type</group>\n        <text>Kick Off (Start)</text>\n      </label>\n"
            "      <label>\n        <group>Half</group>\n        <text>%s</text>\n      </label>\n"
            "    </instance>\n" % (i, t0, t0 + 5.0, half))
    with open(path, "w") as fh:
        fh.write('<?xml version="1.0" ?>\n<file>\n  <ALL_INSTANCES>\n%s  </ALL_INSTANCES>\n</file>\n'
                 % "".join(inst))


def main(meta_path, homog_path, npz_path, out_dir):
    d = load(npz_path)
    P, B, lab, period = d["players"], d["ball"], d["label"], d["period"]
    pitch = d["pitch"]
    smap = slot_map(d["slot_team"], d["slot_gk"])
    # place halves
    Pc = np.full((N, 46, 3), np.nan, np.float32)
    Bc = np.full((N, 3), np.nan, np.float32)
    gt = np.full(N, 3, np.int8)
    for h, (start, length) in enumerate(zip(HALF_STARTS, HALF_LENGTHS), start=1):
        src = np.flatnonzero(period == h)[:length]
        dst = start + np.arange(len(src))
        for j, s in enumerate(smap):
            if s >= 0:
                Pc[dst, s, :2] = P[src, j, :2]
        Bc[dst, :2] = B[src, :2]
        gt[dst] = lab[src]
    # player speed channel (m/s) from positions, as in the provided ATD
    x, y = Pc[:, :, 0] * pitch[0], Pc[:, :, 1] * pitch[1]
    sp = np.hypot(np.diff(x, axis=0, prepend=np.nan), np.diff(y, axis=0, prepend=np.nan)) * 25
    sp[sp > 12] = np.nan
    Pc[:, :, 2] = np.where(np.isnan(Pc[:, :, 0]), np.nan, np.nan_to_num(sp, nan=0.0))
    # attack direction of the home team in the first half, from the data
    first = slice(HALF_STARTS[0], HALF_STARTS[0] + HALF_LENGTHS[0])
    with np.errstate(all="ignore"):
        hx = np.nanmean(Pc[first][:, HOME_SLOTS, 0])
        ax = np.nanmean(Pc[first][:, AWAY_SLOTS, 0])
    home_dir = "left_to_right" if hx < ax else "right_to_left"
    away_dir = "right_to_left" if home_dir == "left_to_right" else "left_to_right"

    tac = os.path.join(out_dir, "tactical")
    os.makedirs(tac, exist_ok=True)
    xml = open(meta_path).read()
    dirs = iter([home_dir, away_dir])
    xml = re.sub(r"(<Name>attack_direction_first_half</Name>\s*<Value>)[^<]*(</Value>)",
                 lambda m: m.group(1) + next(dirs) + m.group(2), xml)
    open(os.path.join(tac, "BAR-ATH_tactical_atd.xml"), "w").write(xml)

    def fmt_group(v, ndec):
        return ",".join("NaN" if np.isnan(a) else ("%.*f" % (ndec, a)) for a in v)

    with open(os.path.join(tac, "BAR-ATH_tactical_atd.txt"), "w") as fh:
        for i in range(N):
            players = ";".join(fmt_group(Pc[i, s], 5) for s in range(46))
            ball = fmt_group(Bc[i], 5)
            fh.write("%d:%s:%s\n" % (i + 1, players, ball))

    # homography: BAR-ATH's median valid matrix on every frame
    H = np.genfromtxt(homog_path, delimiter=",", skip_header=1, max_rows=60000)
    H = H[(H[:, -1] == 1) & ~np.isnan(H[:, 1])]
    med = np.median(H[:, 1:10], axis=0)
    with open(os.path.join(tac, "BAR-ATH_tactical_homography.csv"), "w") as fh:
        fh.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
        row = ",".join("%.9g" % v for v in med)
        fh.writelines("%d,%s,0.3,1\n" % (i, row) for i in range(N))

    write_smart_tagging(os.path.join(tac, "BAR-ATH_tactical_smart_tagging.xml"))

    with open(os.path.join(out_dir, "gt.csv"), "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, LETTERS[c]) for i, c in enumerate(gt))
    graded = int((gt != 3).sum())
    print("%s -> %s: %d graded frames, home attacks %s in half 1" % (
        os.path.basename(npz_path), out_dir, graded, home_dir))


if __name__ == "__main__":
    main(*sys.argv[1:5])
