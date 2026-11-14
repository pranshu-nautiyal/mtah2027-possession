"""Convert Metrica's three open sample games into the common proxy format, with answer keys built
from their human-logged event data (other games, so allowed by RULES.md).

Labels are built under two possession conventions so we can check which one the competition's
answer key follows (PLAN.md, step A3):
  label_control  possession changes when the other team makes an on-ball action (pass, carry,
                 recovery, set piece, shot). Challenges alone never change it.
  label_touch    the same, plus a won challenge (aerial/ground/tackle WON) gives that team the
                 ball from that frame, which creates the short "won header" possessions.
Dead ball (both): from BALL OUT, a foul, or a shot that goes out/scores, until the next restart
(set piece or goal kick). Before each period's kick-off is ungraded (X).

usage: python proxy/convert_metrica.py <metrica sample-data/data dir> <out_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from kloppy import metrica

H, A, D, X = 0, 1, 2, 3
ON_BALL = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT"}
MAX_DEAD_FRAMES = 25 * 90  # a missing restart event must not swallow minutes of play


def tracking_arrays(ds):
    home, away = ds.metadata.teams
    players = [p for t in (home, away) for p in t.players]
    index = {p.player_id: j for j, p in enumerate(players)}
    n, S = len(ds.frames), len(players)
    P = np.full((n, S, 3), np.nan, np.float32)
    B = np.full((n, 3), np.nan, np.float32)
    frame_ids = np.zeros(n, np.int64)
    period = np.zeros(n, np.int8)
    for i, fr in enumerate(ds.frames):
        frame_ids[i] = fr.frame_id
        period[i] = fr.period.id
        for p, pdata in fr.players_data.items():
            if pdata.coordinates is not None and not np.isnan(pdata.coordinates.x):
                j = index[p.player_id]
                P[i, j, 0], P[i, j, 1] = pdata.coordinates.x, pdata.coordinates.y
        if fr.ball_coordinates is not None:
            B[i, 0], B[i, 1] = fr.ball_coordinates.x, fr.ball_coordinates.y
    length = ds.metadata.pitch_dimensions.pitch_length or 105.0
    width = ds.metadata.pitch_dimensions.pitch_width or 68.0
    sp = np.hypot(np.diff(P[:, :, 0], axis=0) * length, np.diff(P[:, :, 1], axis=0) * width) * 25
    P[1:, :, 2] = sp
    B[1:, 2] = np.hypot(np.diff(B[:, 0]) * length, np.diff(B[:, 1]) * width) * 25
    slot_team = np.array([0 if p.team == home else 1 for p in players], np.int8)
    slot_gk = np.array(["goal" in str(p.starting_position).lower() or p.jersey_no == 1
                        for p in players])
    return P, B, frame_ids, period, slot_team, slot_gk, (length, width)


def events_csv(path):
    ev = pd.read_csv(path)
    out = []
    for r in ev.itertuples(index=False):
        out.append(dict(team=H if r.Team == "Home" else A, type=r.Type,
                        sub=str(r.Subtype).upper() if isinstance(r.Subtype, str) else "",
                        start=int(r[4]), end=int(r[6]) if r[6] else int(r[4]), period=int(r.Period)))
    return out


def events_json(path):
    data = json.load(open(path))["data"]
    teams = []
    for e in data:
        if e["team"]["id"] not in teams:
            teams.append(e["team"]["id"])
    out = []
    for e in data:
        subs = e.get("subtypes")
        if isinstance(subs, dict):
            subs = [subs]
        sub = "-".join(s["name"] for s in (subs or [])).upper()
        out.append(dict(team=H if e["team"]["id"] == teams[0] else A, type=e["type"]["name"],
                        sub=sub, start=int(e["start"]["frame"]),
                        end=int(e["end"]["frame"] or e["start"]["frame"]), period=int(e["period"])))
    return out


def build_labels(events, frame_ids, period, touch):
    n = len(frame_ids)
    pos = {f: i for i, f in enumerate(frame_ids)}
    def idx(f):
        return pos.get(f, int(np.searchsorted(frame_ids, f)))
    lab = np.full(n, X, np.int8)
    events = sorted(events, key=lambda e: e["start"])
    for per in sorted(set(period.tolist())):
        evp = [e for e in events if e["period"] == per]
        if not evp:
            continue
        rows = np.flatnonzero(period == per)
        first = idx(evp[0]["start"])  # the kick-off
        last = rows[-1]
        # ownership timeline
        owner = np.full(n, -1, np.int8)
        changes = []
        for e in evp:
            if e["type"] in ON_BALL:
                changes.append((idx(e["start"]), e["team"]))
            elif touch and e["type"] == "CHALLENGE" and "WON" in e["sub"]:
                changes.append((idx(e["start"]), e["team"]))
        changes.sort()
        cur = changes[0][1] if changes else H
        k = 0
        for i in range(first, last + 1):
            while k < len(changes) and changes[k][0] <= i:
                cur = changes[k][1]
                k += 1
            owner[i] = cur
        lab[first:last + 1] = owner[first:last + 1]
        # dead-ball stretches
        restarts = sorted(idx(e["start"]) for e in evp
                          if e["type"] == "SET PIECE" or "GOAL KICK" in e["sub"])
        for e in evp:
            stop = (e["type"] == "BALL OUT" or e["type"] == "FAULT RECEIVED"
                    or (e["type"] == "CHALLENGE" and "FAULT" in e["sub"])
                    or (e["type"] == "SHOT" and ("OUT" in e["sub"] or "GOAL" in e["sub"])))
            if not stop:
                continue
            a = idx(e["end"] if e["type"] in ("BALL OUT", "SHOT") else e["start"])
            nxt = [r for r in restarts if r > a]
            b = nxt[0] if nxt else last + 1
            if b - a > MAX_DEAD_FRAMES:
                later = [idx(x["start"]) for x in evp if idx(x["start"]) > a]
                b = later[0] if later else last + 1
            lab[a:b] = D
    return lab


def main(data_dir, out_dir):
    games = {
        "metrica_1": ("csv", "Sample_Game_1"),
        "metrica_2": ("csv", "Sample_Game_2"),
        "metrica_3": ("epts", "Sample_Game_3"),
    }
    for name, (kind, folder) in games.items():
        base = os.path.join(data_dir, folder)
        if kind == "csv":
            ds = metrica.load_tracking_csv(
                home_data=os.path.join(base, folder + "_RawTrackingData_Home_Team.csv"),
                away_data=os.path.join(base, folder + "_RawTrackingData_Away_Team.csv"),
                coordinates="kloppy")
            events = events_csv(os.path.join(base, folder + "_RawEventsData.csv"))
        else:
            ds = metrica.load_tracking_epts(meta_data=os.path.join(base, folder + "_metadata.xml"),
                                            raw_data=os.path.join(base, folder + "_tracking.txt"),
                                            coordinates="kloppy")
            events = events_json(os.path.join(base, folder + "_events.json"))
        P, B, frame_ids, period, slot_team, slot_gk, pitch = tracking_arrays(ds)
        lc = build_labels(events, frame_ids, period, touch=False)
        lt = build_labels(events, frame_ids, period, touch=True)
        np.savez_compressed(os.path.join(out_dir, name + ".npz"), players=P, ball=B,
                            slot_team=slot_team, slot_gk=slot_gk, label=lt, label_control=lc,
                            label_touch=lt, period=period, pitch=np.array(pitch, np.float32))
        print("%s: %d frames, %d slots; control H/A/D/X %s; touch %s" % (
            name, len(B), len(slot_team), np.bincount(lc, minlength=4), np.bincount(lt, minlength=4)))


if __name__ == "__main__":
    main(*sys.argv[1:3])
