"""Bring IDSSE's possession labels to the competition's convention (PLAN.md step A3).

DFL's per-frame BallPossession flips about twice as often as the competition's answer key and
Metrica's event-based keys (see proxy/label_stats.py). The extra flips are short spells that no
on-ball action of that team supports. This keeps DFL's frame-accurate labels and BallStatus, and
only drops possession spells that the human-logged event data does not confirm:

  a spell of team T is kept if T logged an on-ball action during it (pass, touch, shot, keeper
  claim, set piece, or a won duel), within a small tolerance after correcting the match's
  event-clock offset. An unconfirmed spell goes back to the team that had the ball before it --
  the challenge's own rule for loose balls ("possession stays with the last team that had it").

Adds `label_dfl` (original) and overwrites `label` (harmonised) in the match's .npz.

usage: python proxy/harmonize_idsse.py <idsse_dir> <match_id> <npz>
"""
import glob
import os
import re
import sys
from datetime import datetime

import numpy as np

H, A, D, X = 0, 1, 2, 3
TOL_S = 1.0  # event-vs-tracking tolerance after the offset correction (DFL logs by hand)
ACTION_TAGS = ("Play", "ShotAtGoal", "OtherBallAction", "BallClaiming", "ThrowIn", "FreeKick",
               "GoalKick", "CornerKick", "Penalty", "KickOff")
RESTART_TAGS = ("ThrowIn", "FreeKick", "GoalKick", "CornerKick", "KickOff")


def iso(t):
    return datetime.fromisoformat(t).timestamp()


def ball_frames(positions_path):
    """(N, epoch seconds, possession, status) for every ball frame, in file order per half."""
    rows = []
    in_ball = False
    pat = re.compile(r'N="(\d+)" T="([^"]+)".*?BallPossession="(\d)" BallStatus="(\d)"')
    with open(positions_path) as handle:
        for line in handle:
            if "<FrameSet" in line:
                in_ball = 'TeamId="BALL"' in line
                continue
            if in_ball and "<Frame " in line:
                m = pat.search(line)
                if m:
                    rows.append((int(m.group(1)), iso(m.group(2)), int(m.group(3)), int(m.group(4))))
    rows.sort()
    arr = np.array(rows)
    return arr[:, 0].astype(np.int64), arr[:, 1], arr[:, 2].astype(int), arr[:, 3].astype(int)


def events(events_path, home_id):
    """[(epoch seconds, tag, side)] for on-ball actions; side 0 home, 1 away."""
    txt = open(events_path).read()
    out = []
    for ev in re.finditer(r'<Event [^>]*EventTime="([^"]+)"[^>]*>(.*?)</Event>', txt, re.S):
        t = iso(ev.group(1))
        body = ev.group(2)
        first = re.match(r'\s*<(\w+)([^>]*)>', body)
        if not first:
            continue
        tag, attrs = first.group(1), first.group(2)
        team = None
        if tag == "TacklingGame":
            m = re.search(r'WinnerTeam="([^"]+)"', attrs)
            team = m.group(1) if m else None
        elif tag in ACTION_TAGS:
            m = re.search(r'Team="([^"]+)"', attrs) or re.search(r'Team="([^"]+)"', body)
            team = m.group(1) if m else None
        if team is not None:
            out.append((t, tag, 0 if team == home_id else 1))
    return out


def clock_offset(ev, times, status):
    """Median (event time - tracking restart time) over set pieces, matched to the nearest
    dead->alive switch within 10 s. Returns (offset, spread, n)."""
    starts = times[1:][(status[1:] == 1) & (status[:-1] == 0)]
    diffs = []
    for t, tag, _ in ev:
        if tag in RESTART_TAGS and len(starts):
            j = np.argmin(np.abs(starts - t))
            if abs(starts[j] - t) < 10:
                diffs.append(t - starts[j])
    diffs = np.array(diffs)
    return float(np.median(diffs)), float(np.percentile(np.abs(diffs - np.median(diffs)), 90)), len(diffs)


def harmonise(label, times, ev):
    lab = label.copy()
    ev_t = np.array([e[0] for e in ev])
    ev_side = np.array([e[2] for e in ev])
    n = len(lab)
    i = 0
    prev_team = None
    changed = 0
    while i < n:
        k = lab[i]
        j = i
        while j < n and lab[j] == k:
            j += 1
        if k in (H, A):
            lo, hi = times[i] - TOL_S, times[j - 1] + TOL_S
            sel = (ev_t >= lo) & (ev_t <= hi)
            confirmed = bool(np.any(ev_side[sel] == k))
            if not confirmed and prev_team is not None and prev_team != k:
                lab[i:j] = prev_team
                changed += j - i
            else:
                prev_team = k
        elif k == D:
            prev_team = None  # every restart has a clear possessor; nothing carries over
        i = j
    # merge pass: an unconfirmed spell may have been the first after a stoppage
    return lab, changed


def main(idsse_dir, match_id, npz_path):
    pos = glob.glob(os.path.join(idsse_dir, "*positions_raw_observed*%s.xml" % match_id))[0]
    evp = glob.glob(os.path.join(idsse_dir, "*events_raw*%s.xml" % match_id))[0]
    info = open(glob.glob(os.path.join(idsse_dir, "*matchinformation*%s.xml" % match_id))[0]).read()
    home_id = re.search(r'<Team TeamId="([^"]+)"[^>]*Role="home"', info).group(1)

    nums, times, poss, status = ball_frames(pos)
    data = dict(np.load(npz_path))
    label = data.get("label_dfl", data["label"])
    if len(label) != len(times):
        raise SystemExit("frame count mismatch: npz %d vs ball frames %d" % (len(label), len(times)))
    check = np.where(status == 0, D, np.where(poss == 1, H, A))
    agree = float(np.mean(check == label))
    ev = events(evp, home_id)
    off, spread, n_restarts = clock_offset(ev, times, status)
    ev = [(t - off, tag, side) for t, tag, side in ev]
    new, changed = harmonise(label, times, ev)
    data["label_dfl"] = label
    data["label"] = new
    np.savez_compressed(npz_path, **data)
    print("%s: frames agree with XML %.4f | %d actions | clock offset %+.2fs (90%% within %.2fs, %d restarts) | relabelled %.1f%% of alive frames" % (
        match_id, agree, len(ev), off, spread, n_restarts, 100.0 * changed / max(1, np.isin(label, (H, A)).sum())))


if __name__ == "__main__":
    main(*sys.argv[1:4])
