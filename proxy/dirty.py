"""Make clean proxy tracking fail the way BAR-ATH's automatic tracking fails (PLAN.md step A4).

Fitted only to *unlabelled* statistics of BAR-ATH's tracking (no answer key involved):

ball      BAR-ATH loses the ball in long blocks, not in blinks, and mostly when play is slow
          (79% missing when players average <1 m/s, ~10% at full speed). Modelled as an on/off
          Markov chain whose switch rates depend on the collective player speed, so slow phases
          (stoppages, build-up) lose the ball as often as they do in BAR-ATH.
players   ~10.2 of 11 visible per team; track pieces live a median ~35 s. Each player blinks out
          with visible stretches drawn from BAR-ATH's track lifetimes and gaps sized to match the
          visible fraction.
smoothness BAR-ATH's tracker outputs smooth paths: a touch bends the ball's path rather than
          kinking it (ball acceleration p99 ~37 m/s^2 vs 200-560 in the proxies; IDSSE positions
          are also rounded to the centimetre). Each proxy match gets its own white-noise level
          and Gaussian smoothing, chosen so its ball and player acceleration percentiles match
          BAR-ATH's. Applied before the dropouts, so everything downstream sees BAR-like paths.

usage:
  python proxy/dirty.py fit <bar_atd.npz> <profile.json>
  python proxy/dirty.py calibrate <profile.json> <clean.npz>...
  python proxy/dirty.py apply <profile.json> <clean.npz> <dirty.npz> [seed]
  python proxy/dirty.py compare <profile.json> <bar_atd.npz> <dirty.npz>...
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

from common import FPS, collective_speed, load, metres, runs

ACC_PCTS = [50, 90, 99]
SIGMAS = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12]
NOISE_M = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.3]
BALL_SPEED_EDGES = [1.0, 4.0, 10.0]   # still / slow / moving / fast, m/s
HAZARD_PSEUDO = 400                   # shrink sparse hazard cells toward the speed-bin mean
CROWD_M = 3.0                         # players this close to the ball count as crowding it
CROWD_MAX = 3                         # crowd bins 0, 1, 2, 3+


def smooth_nan(arr, sigma):
    """Gaussian smoothing along time that ignores NaN (normalised convolution); NaN stays NaN."""
    if sigma <= 0:
        return arr
    v = ~np.isnan(arr)
    num = gaussian_filter1d(np.where(v, arr, 0.0), sigma, axis=0, mode="nearest")
    den = gaussian_filter1d(v.astype(float), sigma, axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~v] = np.nan
    return out


def acc_pct(x, y, g):
    """Acceleration percentiles (m/s^2) of a path in metres; works on (n,) or (n, S)."""
    acc = np.hypot(x[2:] - 2 * x[1:-1] + x[:-2], y[2:] - 2 * y[1:-1] + y[:-2]) * FPS * FPS
    acc = acc[g[1:-1]]
    acc = acc[~np.isnan(acc)]
    return np.percentile(acc, ACC_PCTS) if len(acc) else np.full(len(ACC_PCTS), np.nan)


def roughen(x, y, noise_m, sigma, rng):
    """White noise of noise_m metres, then Gaussian smoothing of sigma frames."""
    if noise_m > 0:
        x = x + rng.normal(0, noise_m, x.shape)
        y = y + rng.normal(0, noise_m, y.shape)
    return smooth_nan(x, sigma), smooth_nan(y, sigma)


def best_roughening(x, y, g, target, seed=7):
    """(noise_m, sigma) whose acceleration percentiles best match target, in log space."""
    best = None
    for noise in NOISE_M:
        for sigma in SIGMAS:
            xs, ys = roughen(x, y, noise, sigma, np.random.RandomState(seed))
            p = acc_pct(xs, ys, g)
            err = float(np.sum((np.log(np.maximum(p, 1e-3)) - np.log(target)) ** 2))
            if best is None or err < best[0]:
                best = (err, noise, sigma, p)
    return best

SPEED_EDGES = [0.0, 0.75, 1.25, 1.75, 2.25, 3.0, 99.0]
BAR_GRADED = [(11185, 90652), (110808, 190198)]


def graded_mask(n, label=None):
    if label is not None:
        return label != 3
    g = np.zeros(n, bool)
    for a, b in BAR_GRADED:
        g[a:b] = True
    return g


def speed_bin(cs):
    return np.clip(np.digitize(cs, SPEED_EDGES) - 1, 0, len(SPEED_EDGES) - 2)


def roughness(x, y, g):
    acc = np.hypot(x[2:] - 2 * x[1:-1] + x[:-2], y[2:] - 2 * y[1:-1] + y[:-2])[g[1:-1]]
    return float(np.nanmedian(acc))


def fit(bar_path, out_path):
    d = load(bar_path)
    P, B, pitch = d["players"], d["ball"], d["pitch"]
    g = graded_mask(len(B))
    cs = collective_speed(P, pitch)
    sb = speed_bin(cs)
    vis = ~np.isnan(B[:, 0])
    prof = {"speed_edges": SPEED_EDGES, "ball_vm": [], "ball_mv": [], "ball_missing_share": []}
    for b in range(len(SPEED_EDGES) - 1):
        sel = g[:-1] & (sb[:-1] == b)
        v, m = sel & vis[:-1], sel & ~vis[:-1]
        prof["ball_vm"].append(float((v & ~vis[1:]).sum() / max(1, v.sum())))
        prof["ball_mv"].append(float((m & vis[1:]).sum() / max(1, m.sum())))
        prof["ball_missing_share"].append(float(m.sum() / max(1, sel.sum())))
    life = []
    for j in range(P.shape[1]):
        a, b = runs(~np.isnan(P[:, j, 0]) & g)
        life += list(b - a)
    life = np.array(life)
    team = d["slot_team"]
    visible_per_team = float(np.mean([(~np.isnan(P[:, team == k, 0])).sum(1)[g].mean() for k in (0, 1)]))
    a, z = runs(~vis & g)
    prof["ball_gap_median_frames"] = float(np.median(z - a))
    # loss hazard by what the ball is doing at the moment it is lost (observed: it is visible).
    # BAR-ATH loses a still ball 5-10x faster than a moving one, and faster again off the pitch.
    sbin, out = ball_state(B, pitch)
    table = np.zeros((len(SPEED_EDGES) - 1, len(BALL_SPEED_EDGES) + 1, 2))
    seen = g[:-1] & vis[:-1]
    lost = ~vis[1:]
    for c in range(table.shape[0]):
        prior = prof["ball_vm"][c]
        for s in range(table.shape[1]):
            for o in range(2):
                m = seen & (sb[:-1] == c) & (sbin[:-1] == s) & (out[:-1] == o)
                table[c, s, o] = (lost[m].sum() + HAZARD_PSEUDO * prior) / (m.sum() + HAZARD_PSEUDO)
    prof["ball_vm_table"] = table.tolist()
    crowd = crowd_bin(P, B, pitch)
    expected = table[sb, sbin, out]
    prof["crowd_mult"] = [float(lost[seen & (crowd[:-1] == k)].sum() /
                                max(1e-9, expected[:-1][seen & (crowd[:-1] == k)].sum()))
                          for k in range(CROWD_MAX + 1)]
    prof["crowd_loss_rate"] = [float(lost[seen & (crowd[:-1] == k)].mean())
                               for k in range(CROWD_MAX + 1)]
    prof["visible_ball_outside_share"] = float(out[g & vis].mean())
    prof["track_lifetimes"] = sorted(int(v) for v in life)
    prof["visible_fraction"] = visible_per_team / 11.0
    x, y = metres(P, pitch)
    bx, by = metres(B, pitch)
    prof["player_acc_pct"] = acc_pct(x, y, g[:, None].repeat(P.shape[1], 1)).tolist()
    prof["ball_acc_pct"] = acc_pct(bx, by, g).tolist()
    if os.path.exists(out_path):  # the smoothness search is slow and independent of the rest
        old = json.load(open(out_path))
        if "roughening" in old and old.get("ball_acc_pct") == prof["ball_acc_pct"]:
            prof["roughening"] = old["roughening"]
    json.dump(prof, open(out_path, "w"))
    print("fitted: ball accel p50/p90/p99 %s, players %s" % (
        np.round(prof["ball_acc_pct"], 2), np.round(prof["player_acc_pct"], 2)))
    print("fitted: ball missing by speed bin", np.round(prof["ball_missing_share"], 2),
          "| visible/team %.2f | track pieces %d (median %.0fs)" % (
              visible_per_team, len(life), np.median(life) / FPS))


def fill_ball(B):
    """Linear interpolation across every gap. Metrica's open data blanks the ball whenever it is
    dead, which would make 'ball missing' a perfect dead-ball cue that BAR-ATH does not offer.
    Filling first means only the simulated tracker below decides when the ball disappears."""
    B = B.copy()
    t = np.arange(len(B))
    ok = ~np.isnan(B[:, 0])
    if ok.sum() < 2 or ok.all():
        return B
    for c in (0, 1):
        B[:, c] = np.interp(t, t[ok], B[ok, c])
    return B


def ball_state(B, pitch):
    """(speed bin 0..3: still / slow / moving / fast, outside-the-pitch flag) per frame."""
    x, y = metres(B.astype(np.float64), pitch)
    sp = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)) * FPS
    sp[0] = sp[1] if len(sp) > 1 else 0.0
    sbin = np.digitize(np.nan_to_num(sp, nan=0.0), BALL_SPEED_EDGES)
    with np.errstate(invalid="ignore"):
        out = ((B[:, 0] < 0) | (B[:, 0] > 1) | (B[:, 1] < 0) | (B[:, 1] > 1)).astype(np.int8)
    return sbin, out


def crowd_bin(P, B, pitch):
    """Players (either team) within CROWD_M of the ball, capped at CROWD_MAX. BAR-ATH loses the
    ball 3x more often with 3+ players around it -- duels and tackles hide it."""
    x, y = metres(P.astype(np.float64), pitch)
    bx, by = metres(B.astype(np.float64), pitch)
    with np.errstate(invalid="ignore"):
        n = (np.hypot(x - bx[:, None], y - by[:, None]) < CROWD_M).sum(1)
    return np.minimum(n, CROWD_MAX)


def simulate_missing(lose, recover, rng):
    """On/off chain with per-frame hazards: lose[t] while seen, recover[t] while lost."""
    u = rng.random_sample(len(lose))
    missing = np.zeros(len(lose), bool)
    state = False
    for t in range(len(lose)):
        state = (u[t] >= recover[t]) if state else (u[t] < lose[t])
        missing[t] = state
    return missing


def hide_outside_episodes(out, q, rng):
    """Each contiguous stretch with the ball off the pitch is hidden whole with probability q:
    BAR-ATH's tracker typically drops the ball as it crosses the line."""
    hide = np.zeros(len(out), bool)
    a, b = runs(out.astype(bool))
    draw = rng.random_sample(len(a))
    for s_, e_, u in zip(a, b, draw):
        if u < q:
            hide[s_:e_] = True
    return hide


def hazards(prof, cs_bin, sbin, out, crowd=None):
    table = np.array(prof["ball_vm_table"])
    mult = np.array(prof.get("ball_vm_mult", np.ones(table.shape[0])))
    lose = table[cs_bin, sbin, out] * mult[cs_bin]
    if crowd is not None and "crowd_mult" in prof:
        # shift losses toward crowded moments by BAR-ATH's measured ratios, keeping the match's
        # average loss rate unchanged so the occupancy calibration stays meaningful
        cm = np.array(prof["crowd_mult"])[crowd]
        lose = lose * cm / max(1e-9, float(np.mean(cm)))
    recover = np.array(prof["ball_mv"])[cs_bin]
    # BAR-ATH rarely re-acquires a ball that is off the pitch; one multiplier, tuned so the share
    # of visible-ball frames outside the pitch matches BAR-ATH's
    recover = recover * np.where(out.astype(bool), prof.get("ball_mv_out_mult", 1.0), 1.0)  # stays 1.0
    return np.clip(lose, 0, 0.9), np.clip(recover, 0, 0.9)


def calibrate(prof_path, clean_paths, iters=14):
    """Tune the per-speed-bin loss rate (vm) and a global recovery scale (mv) on the pooled proxy
    matches until simulated ball loss matches BAR-ATH: share missing in each speed bin, and the
    median gap length. Unlabelled statistics only."""
    prof = json.load(open(prof_path))
    target = np.array(prof["ball_missing_share"])
    gap_target = prof["ball_gap_median_frames"]
    # 1) per-match smoothness, on a 30k-frame slice to keep the search quick (cached)
    todo = [p for p in clean_paths if os.path.basename(p) not in prof.get("roughening", {})]
    prof.setdefault("roughening", {})
    for p in todo:
        d = load(p)
        g = graded_mask(len(d["ball"]), d.get("label"))
        sl = slice(len(g) // 3, len(g) // 3 + 30000)
        bx, by = metres(fill_ball(d["ball"])[sl], d["pitch"])
        b = best_roughening(bx, by, g[sl], np.array(prof["ball_acc_pct"]))
        px, py = metres(d["players"][sl], d["pitch"])
        q = best_roughening(px, py, g[sl][:, None].repeat(px.shape[1], 1), np.array(prof["player_acc_pct"]))
        prof["roughening"][os.path.basename(p)] = {"ball": [b[1], b[2]], "players": [q[1], q[2]]}
        print("  %-18s ball noise %.3fm sigma %s -> accel %s | players noise %.3fm sigma %s -> %s" % (
            os.path.basename(p)[:-4], b[1], b[2], np.round(b[3], 1), q[1], q[2], np.round(q[3], 2)))
    # 2) ball-loss hazards, with contexts from the roughened players and ball (as apply() sees
    #    them). The measured table keeps its shape (still vs moving, inside vs outside); one
    #    multiplier per players'-speed bin and one global recovery scale are tuned so occupancy
    #    and gap length match BAR-ATH.
    ctx = []
    for p in clean_paths:
        d = load(p)
        g = graded_mask(len(d["ball"]), d.get("label"))
        setting = prof["roughening"][os.path.basename(p)]
        P = roughened_players(d, setting["players"], seed=0)
        Bc = roughened_ball(fill_ball(d["ball"]), d["pitch"], setting["ball"], seed=0)
        sbin, out = ball_state(Bc, d["pitch"])
        ctx.append((speed_bin(collective_speed(P, d["pitch"])), sbin, out, g, crowd_bin(P, Bc, d["pitch"])))
    nb = len(SPEED_EDGES) - 1
    prof["ball_vm_mult"] = [1.0] * nb
    for it in range(iters):
        occ_num, occ_den, gaps, out_vis, vis_n = np.zeros(nb), np.zeros(nb), [], 0, 0
        c_lost, c_seen = np.zeros(CROWD_MAX + 1), np.zeros(CROWD_MAX + 1)
        for i, (cb, sbin, out, g, crowd) in enumerate(ctx):
            lose, rec = hazards(prof, cb, sbin, out, crowd)
            miss = simulate_missing(lose, rec, np.random.RandomState(1000 + i))
            miss |= hide_outside_episodes(out, prof.get("ball_out_hide_q", 0.0),
                                          np.random.RandomState(2000 + i))
            for b in range(nb):
                sel = g & (cb == b)
                occ_num[b] += (miss & sel).sum()
                occ_den[b] += sel.sum()
            a, z = runs(miss & g)
            gaps += list(z - a)
            out_vis += (out.astype(bool) & ~miss & g).sum()
            vis_n += (~miss & g).sum()
            seen_ = ~miss[:-1] & g[:-1]
            lost_ = miss[1:]
            for k in range(CROWD_MAX + 1):
                m = seen_ & (crowd[:-1] == k)
                c_lost[k] += (lost_ & m).sum()
                c_seen[k] += m.sum()
        occ = occ_num / np.maximum(occ_den, 1)
        gm = float(np.median(gaps))
        print("  iter %d: missing by bin %s  gap median %.1fs  visible ball outside pitch %.1f%% (BAR %.1f%%)" % (
            it, np.round(occ, 2), gm / FPS, 100 * out_vis / max(1, vis_n),
            100 * prof["visible_ball_outside_share"]))
        prof["ball_vm_mult"] = np.clip(np.array(prof["ball_vm_mult"]) *
                                       (target / np.maximum(occ, 1e-3)) ** 0.8, 0.05, 50).tolist()
        prof["ball_mv"] = np.clip(np.array(prof["ball_mv"]) * (gm / gap_target) ** 0.8,
                                  1e-5, 0.5).tolist()
        sim_rate = c_lost / np.maximum(c_seen, 1)
        print("           loss rate x1000 by crowd 0/1/2/3+: sim %s  BAR %s" % (
            np.round(1000 * sim_rate, 2), np.round(1000 * np.array(prof["crowd_loss_rate"]), 2)))
        sim_out = out_vis / max(1, vis_n)
        # raise the hiding probability while too many visible frames sit off the pitch
        q = prof.get("ball_out_hide_q", 0.0)
        ratio = sim_out / max(prof["visible_ball_outside_share"], 1e-4)
        prof["ball_out_hide_q"] = float(np.clip(1 - (1 - q) / max(ratio, 1e-3) ** 0.8, 0.0, 0.995))
    json.dump(prof, open(prof_path, "w"))


def roughened_players(d, setting, seed):
    noise, sigma = setting
    x, y = metres(d["players"].astype(np.float64), d["pitch"])
    x, y = roughen(x, y, noise, sigma, np.random.RandomState(seed))
    P = d["players"].copy()
    P[..., 0], P[..., 1] = x / d["pitch"][0], y / d["pitch"][1]
    return P


def roughened_ball(B, pitch, setting, seed):
    noise, sigma = setting
    x, y = metres(B.astype(np.float64), pitch)
    x, y = roughen(x, y, noise, sigma, np.random.RandomState(seed))
    out = B.copy()
    out[:, 0], out[:, 1] = x / pitch[0], y / pitch[1]
    return out


def apply(prof_path, clean_path, out_path, seed=0):
    prof = json.load(open(prof_path))
    rng = np.random.RandomState(int(seed))
    d = load(clean_path)
    setting = prof["roughening"][os.path.basename(clean_path)]
    pitch = d["pitch"]
    P = roughened_players(d, setting["players"], seed=int(seed) + 11)
    B = roughened_ball(fill_ball(d["ball"]), pitch, setting["ball"], seed=int(seed) + 13)
    d = dict(d, players=P.copy())   # the ball-loss context below uses the roughened players
    n = len(B)
    # players: on/off per player
    life = np.array(prof["track_lifetimes"])
    vf = prof["visible_fraction"]
    gap_mean = max(1.0, np.mean(life) * (1 - vf) / vf)
    for j in range(P.shape[1]):
        present = ~np.isnan(P[:, j, 0])
        t = int(rng.randint(0, max(1, int(np.median(life)))))
        while t < n:
            t += int(rng.choice(life))
            gap = int(rng.geometric(1.0 / gap_mean))
            P[t:t + gap, j, :] = np.nan
            t += gap
        P[~present, j, :] = np.nan
    # ball: on/off chain whose loss hazard depends on the players' speed and on what the ball is
    # doing (still / moving, inside / outside the pitch), as measured on BAR-ATH
    sbin, outside = ball_state(B, pitch)
    lose, rec = hazards(prof, speed_bin(collective_speed(d["players"], pitch)), sbin, outside,
                        crowd_bin(P, B, pitch))
    missing = simulate_missing(lose, rec, rng)
    missing |= hide_outside_episodes(outside, prof.get("ball_out_hide_q", 0.0), rng)
    B[missing] = np.nan
    out = dict(d)
    out["players"], out["ball"] = P.astype(np.float32), B.astype(np.float32)
    np.savez_compressed(out_path, **out)


def describe(d, label=None):
    P, B, pitch = d["players"], d["ball"], d["pitch"]
    g = graded_mask(len(B), label)
    vis = ~np.isnan(B[:, 0])
    a, b = runs(vis & g)
    a2, b2 = runs(~vis & g)
    vl, ml = (b - a) / FPS, (b2 - a2) / FPS
    sb = speed_bin(collective_speed(P, pitch))
    miss = [float((~vis[g & (sb == k)]).mean()) if (g & (sb == k)).any() else float("nan")
            for k in range(len(SPEED_EDGES) - 1)]
    team = d["slot_team"]
    vpt = np.mean([(~np.isnan(P[:, team == k, 0])).sum(1)[g].mean() for k in (0, 1)])
    bx, by = metres(B, pitch)
    px, py = metres(P[::5], pitch)
    ba = acc_pct(bx, by, g)
    pa = acc_pct(px, py, g[::5][:, None].repeat(P.shape[1], 1))
    return ("ball missing %.2f | gaps/h %4.0f | gap med %5.1fs p90 %5.1fs | vis run med %5.1fs | by speed %s | pl/team %.1f | ball acc %s | pl acc %s"
            % ((~vis[g]).mean(), len(ml) / (g.sum() / FPS / 3600), np.median(ml), np.percentile(ml, 90),
               np.median(vl), np.round(miss, 2), vpt, np.round(ba, 1), np.round(pa, 1)))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit(sys.argv[2], sys.argv[3])
    elif cmd == "calibrate":
        calibrate(sys.argv[2], sys.argv[3:])
    elif cmd == "apply":
        apply(*sys.argv[2:6])
    elif cmd == "compare":
        print("%-18s %s" % ("BAR-ATH", describe(load(sys.argv[3]))))
        for p in sys.argv[4:]:
            d = load(p)
            print("%-18s %s" % (p.split("/")[-1][:-4], describe(d, d.get("label"))))
