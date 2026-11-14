"""Read the provided Metrica ATD (FIFA/EPTS export) into the common per-match format.

Raw line: "<frame_id>:<x,y,speed>;<x,y,speed>;...:<ball x,y,speed>", NaN for unseen objects.
frame_id counts from 1; video frame (= submission row) is frame_id - 1 (DATA.md).
"""
import xml.etree.ElementTree as ET

import numpy as np

PITCH = (105.3, 68.0)                          # metres, from games.py (corrected-tracking measure)
HALVES = ((11185, 90651), (110808, 190197))    # graded video frames, inclusive (DATA.md)


def load(meta_path, raw_path, n_frames=199319, pitch=PITCH, halves=HALVES):
    root = ET.parse(meta_path).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    players = list(root.iter("Player"))
    slot_team = np.array([0 if p.get("teamId") == teams[0] else 1 for p in players], np.int8)
    slot_gk = []
    for p in players:
        vals = {pp.findtext("Name"): pp.findtext("Value") for pp in p.iter("ProviderParameter")}
        slot_gk.append(vals.get("position_index") == "0")
    S = len(players)
    P = np.full((n_frames, S, 3), np.nan, np.float32)
    B = np.full((n_frames, 3), np.nan, np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                continue
            P[row] = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32).reshape(S, 3)
            B[row] = np.array(parts[2].split(","), dtype=np.float32)
    period = np.ones(n_frames, np.int8)
    period[(halves[0][1] + halves[1][0]) // 2:] = 2
    graded = np.zeros(n_frames, bool)
    for a, b in halves:
        graded[a:b + 1] = True
    return dict(players=P, ball=B, slot_team=slot_team, slot_gk=np.array(slot_gk),
                pitch=np.array(pitch, np.float32), period=period, graded=graded)
