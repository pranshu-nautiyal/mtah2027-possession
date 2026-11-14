"""Convert one IDSSE (DFL open data) match into the common proxy format.

Common format (one .npz per match), same layout as the BAR-ATH cache:
  players   (n, S, 3) float32  x, y in 0-1 pitch units (x along the length), speed m/s; NaN = unseen
  ball      (n, 3)    float32  x, y in 0-1, speed m/s
  slot_team (S,)      int8     0 home, 1 away
  slot_gk   (S,)      bool     goalkeeper slot
  label     (n,)      int8     proxy answer key: 0 H, 1 A, 2 D, 3 X (not graded)
  pitch     (2,)      float32  length, width in metres

The answer key comes from DFL's own per-frame BallStatus (alive/dead) and BallPossession (team),
the same kind of tracking-derived key the competition uses. CC BY 4.0, Bassek et al. 2025.

usage: python proxy/convert_idsse.py <idsse_dir> <match_id> <out.npz>
"""
import glob
import os
import sys

import numpy as np
from kloppy import sportec
from kloppy.domain import BallState

H, A, D, X = 0, 1, 2, 3


def convert(idsse_dir, match_id, out_path):
    meta = glob.glob(os.path.join(idsse_dir, "*matchinformation*%s.xml" % match_id))[0]
    raw = glob.glob(os.path.join(idsse_dir, "*positions_raw_observed*%s.xml" % match_id))[0]
    ds = sportec.load_tracking(raw_data=raw, meta_data=meta, coordinates="kloppy",
                               only_alive=False)
    home, away = ds.metadata.teams
    players = [p for t in (home, away) for p in t.players]
    slot_team = np.array([0 if p.team == home else 1 for p in players], np.int8)
    slot_gk = np.array([bool(p.starting_position and "goal" in str(p.starting_position).lower())
                        for p in players])
    length = ds.metadata.pitch_dimensions.pitch_length or 105.0
    width = ds.metadata.pitch_dimensions.pitch_width or 68.0
    n, S = len(ds.frames), len(players)
    P = np.full((n, S, 3), np.nan, np.float32)
    B = np.full((n, 3), np.nan, np.float32)
    lab = np.full(n, X, np.int8)
    period = np.zeros(n, np.int8)
    index = {p.player_id: j for j, p in enumerate(players)}
    for i, fr in enumerate(ds.frames):
        period[i] = fr.period.id
        for p, pd in fr.players_data.items():
            if pd.coordinates is not None:
                j = index[p.player_id]
                P[i, j, 0], P[i, j, 1] = pd.coordinates.x, pd.coordinates.y
                P[i, j, 2] = pd.speed if pd.speed is not None else np.nan
        if fr.ball_coordinates is not None:
            B[i, 0], B[i, 1] = fr.ball_coordinates.x, fr.ball_coordinates.y
        if fr.ball_state == BallState.DEAD:
            lab[i] = D
        elif fr.ball_owning_team == home:
            lab[i] = H
        elif fr.ball_owning_team == away:
            lab[i] = A
    # speeds from positions where the provider gave none (m/s, 25 fps)
    for arr in (P,):
        sp = np.hypot(np.diff(arr[:, :, 0], axis=0) * length, np.diff(arr[:, :, 1], axis=0) * width) * 25
        fill = np.isnan(arr[1:, :, 2])
        arr[1:, :, 2][fill] = sp[fill]
    bs = np.hypot(np.diff(B[:, 0]) * length, np.diff(B[:, 1]) * width) * 25
    B[1:, 2] = bs
    np.savez_compressed(out_path, players=P, ball=B, slot_team=slot_team, slot_gk=slot_gk,
                        label=lab, period=period, pitch=np.array([length, width], np.float32))
    counts = np.bincount(lab, minlength=4)
    print("%s: %d frames, %d slots, H %d A %d D %d X %d, gk slots %s" % (
        match_id, n, S, *counts, np.flatnonzero(slot_gk)))


if __name__ == "__main__":
    convert(*sys.argv[1:4])
