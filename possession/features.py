"""Per-frame features for possession and dead-ball inference.

Works on the common per-match dict (players (n,S,3), ball (n,3), slot_team, pitch, period) for
the provided match and every proxy match alike. All features are anonymous and collective: no
player identities, only team membership and positions -- the provided tracking has nothing more.

Orientation: in each period the home team (team 0) is turned to attack +X, decided from the data
(the team whose players sit at lower x defends that end). No metadata needed, so proxy matches
and the provided match go through identical code.

Team symmetry: `compute(d, swap=True)` relabels the teams before computing, which mirrors every
feature. Training on both views and averaging both views at prediction time keeps the model from
learning a home/away bias.
"""
import numpy as np

FPS = 25
NEAR_M = 3.0  # a player this close to the ball is treated as on it, for the near-team indicator


def _roll(x, before, after):
    """NaN-aware mean of x over [t - before, t + after]."""
    n = len(x)
    v = ~np.isnan(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(v, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(v)))
    t = np.arange(n)
    lo, hi = np.clip(t - before, 0, n), np.clip(t + after + 1, 0, n)
    cnt = k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def _last_next(valid):
    n = len(valid)
    t = np.arange(n)
    last = np.maximum.accumulate(np.where(valid, t, -1))
    nxt = np.minimum.accumulate(np.where(valid, t, n)[::-1])[::-1]
    return last, nxt


def _take(x, idx):
    out = np.full(len(x), np.nan)
    ok = (idx >= 0) & (idx < len(x))
    out[ok] = x[idx[ok]]
    return out


def _kth(sorted_vals, n_valid, k, from_top):
    """k-th smallest (or largest) valid value per row of a NaN-last sorted array."""
    n = len(sorted_vals)
    idx = (n_valid - 1 - k) if from_top else np.full(n, k)
    ok = (idx >= 0) & (idx < n_valid)
    out = np.full(n, np.nan)
    rows = np.arange(n)
    out[ok] = sorted_vals[rows[ok], idx[ok]]
    return out


def _orientation(P, team, period):
    sign = np.ones(len(P))
    for p in np.unique(period):
        sel = period == p
        with np.errstate(all="ignore"):
            hx = np.nanmedian(np.nanmean(P[sel][:, team == 0, 0], axis=1))
            ax = np.nanmedian(np.nanmean(P[sel][:, team == 1, 0], axis=1))
        sign[sel] = 1.0 if hx < ax else -1.0
    return sign


def _shift(a, k):
    """a[t + k], NaN where out of range."""
    out = np.full(len(a), np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def _roll_max(a, half):
    """NaN-aware max over [t - half, t + half]."""
    out = np.full(len(a), np.nan)
    for k in range(-half, half + 1):
        out = np.fmax(out, _shift(a, k))
    return out


def _roll_min(a, half):
    out = np.full(len(a), np.nan)
    for k in range(-half, half + 1):
        out = np.fmin(out, _shift(a, k))
    return out


def compute(d, swap=False, version="v2"):
    """(features (n, F) float32, names). Row t describes video frame t.
    version "v1" reproduces the frozen v1 feature set exactly; "v2" adds fine touch cues."""
    P = d["players"].astype(np.float64)
    B = d["ball"].astype(np.float64)
    L, W = float(d["pitch"][0]), float(d["pitch"][1])
    team = d["slot_team"].astype(np.int8)
    if swap:
        team = 1 - team
    n = len(P)
    sign = _orientation(P, team, d["period"])
    X = (P[:, :, 0] - 0.5) * L * sign[:, None]
    Y = (P[:, :, 1] - 0.5) * W
    BX = (B[:, 0] - 0.5) * L * sign
    BY = (B[:, 1] - 0.5) * W

    feats = {}

    def add(name, arr):
        feats[name] = np.asarray(arr, np.float64)

    # player speeds from positions (period switches and identity jumps removed)
    sp = np.hypot(np.diff(X, axis=0, prepend=np.nan), np.diff(Y, axis=0, prepend=np.nan)) * FPS
    sp[sp > 12] = np.nan

    # -- team shape, each team in its own attacking direction --------------------------------
    for k, tag in ((0, "h"), (1, "a")):
        cols = team == k
        own = 1.0 if k == 0 else -1.0
        Xo, Yk = X[:, cols] * own, Y[:, cols]
        xs = np.sort(Xo, axis=1)
        nv = (~np.isnan(Xo)).sum(1)
        ys = np.sort(Yk, axis=1)
        with np.errstate(all="ignore"):
            cx = np.nanmean(Xo, 1)
            add("cx_" + tag, cx)
            add("sx_" + tag, np.nanstd(Xo, 1))
            add("sy_" + tag, np.nanstd(Yk, 1))
            add("spd_" + tag, _roll(np.nanmean(sp[:, cols], 1), 12, 12))
        add("nv_" + tag, nv)
        add("deep_" + tag, _kth(xs, nv, 1, False) + L / 2)     # 2nd deepest: the back line
        add("adv_" + tag, _kth(xs, nv, 0, True) + L / 2)       # most advanced player
        add("len_" + tag, _kth(xs, nv, 1, True) - _kth(xs, nv, 1, False))
        add("wid_" + tag, _kth(ys, nv, 1, True) - _kth(ys, nv, 1, False))
        vcx = np.diff(cx, prepend=np.nan) * FPS
        vcx[np.abs(vcx) > 8] = np.nan
        add("vcx1_" + tag, _roll(vcx, 12, 12))
        add("vcx3_" + tag, _roll(vcx, 37, 37))
    block = (feats["cx_h"] - feats["cx_a"]) / 2          # where the play is, home frame
    add("block", block)
    add("block_r5", _roll(block, 125, 125))
    for f in ("sx", "sy", "len", "wid", "deep", "adv", "spd", "vcx3"):
        add(f + "_diff", feats[f + "_h"] - feats[f + "_a"])

    # marking: mean distance from each player to the nearest opponent
    Xh, Yh, Xa, Ya = X[:, team == 0], Y[:, team == 0], X[:, team == 1], Y[:, team == 1]
    mnd_h, mnd_a = np.full(n, np.nan), np.full(n, np.nan)
    for a in range(0, n, 20000):
        b = min(n, a + 20000)
        dist = np.hypot(Xh[a:b, :, None] - Xa[a:b, None, :], Yh[a:b, :, None] - Ya[a:b, None, :])
        with np.errstate(all="ignore"):
            mnd_h[a:b] = np.nanmean(np.nanmin(dist, axis=2), 1)
            mnd_a[a:b] = np.nanmean(np.nanmin(dist, axis=1), 1)
    add("mark_h", _roll(mnd_h, 12, 12))
    add("mark_a", _roll(mnd_a, 12, 12))
    add("mark_diff", feats["mark_h"] - feats["mark_a"])

    # -- ball ------------------------------------------------------------------------------------
    bvis = ~np.isnan(BX)
    add("bvis", bvis)
    add("bx", BX)
    add("by_abs", np.abs(BY))
    bvx = np.concatenate(([np.nan], (BX[2:] - BX[:-2]) / 2 * FPS, [np.nan]))
    bvy = np.concatenate(([np.nan], (BY[2:] - BY[:-2]) / 2 * FPS, [np.nan]))
    bsp = np.hypot(bvx, bvy)
    bsp[bsp > 45] = np.nan
    add("bsp", bsp)
    add("bvx", bvx)
    add("bsp_r", _roll(bsp, 5, 5))
    bacc = np.concatenate(([np.nan], np.hypot(BX[2:] - 2 * BX[1:-1] + BX[:-2],
                                              BY[2:] - 2 * BY[1:-1] + BY[:-2]) * FPS * FPS, [np.nan]))
    add("bacc", bacc)
    # largest kick in the ball's path within +/-3 frames: a touch, a deflection, a shot
    add("bacc_max", np.where(bvis, np.maximum.reduce(
        [np.roll(np.nan_to_num(bacc), s) for s in range(-3, 4)]), np.nan))
    with np.errstate(invalid="ignore"):
        add("bout", np.where(bvis, np.maximum.reduce([np.abs(BX) - L / 2, np.abs(BY) - W / 2,
                                                       np.zeros(n)]), np.nan))
    add("d_gl", L / 2 - np.abs(BX))
    add("d_tl", W / 2 - np.abs(BY))
    add("d_corner", np.hypot(L / 2 - np.abs(BX), W / 2 - np.abs(BY)))
    add("d_centre", np.hypot(BX, BY))
    add("d_pspot", np.hypot(L / 2 - 11 - np.abs(BX), BY))
    t = np.arange(n)
    moving = ~bvis | (np.nan_to_num(bsp, nan=99) > 1.0)
    still = t - np.maximum.accumulate(np.where(moving, t, -1))
    add("still", np.where(bvis, np.minimum(still, 500) / FPS, np.nan))

    # nearest players to the ball, per team
    for k, tag in ((0, "h"), (1, "a")):
        cols = team == k
        dist = np.hypot(X[:, cols] - BX[:, None], Y[:, cols] - BY[:, None])
        s = np.sort(dist, axis=1)
        add("d1_" + tag, s[:, 0])
        add("d2_" + tag, s[:, 1])
        with np.errstate(invalid="ignore"):
            add("c5_" + tag, np.where(bvis, (dist < 5).sum(1), np.nan))
            add("c10_" + tag, np.where(bvis, (dist < 10).sum(1), np.nan))
    dd = feats["d1_a"] - feats["d1_h"]                  # > 0: home nearer
    add("dd", np.clip(dd, -30, 30))
    near = np.minimum(feats["d1_h"], feats["d1_a"])
    ni = np.where(bvis, np.where(near < NEAR_M, np.sign(dd), 0.0), np.nan)
    for b, a in ((25, 0), (0, 25), (75, 0), (0, 75), (250, 250)):
        add("ni_%d_%d" % (b, a), _roll(ni, b, a))
    add("dd_r1", _roll(feats["dd"], 12, 12))
    add("dd_r5", _roll(feats["dd"], 125, 125))
    add("bx_r5", _roll(BX, 125, 125))

    # -- ball gaps: what was true when the ball was last / next seen ------------------------------
    last, nxt = _last_next(bvis)
    add("t_since", np.where(last >= 0, (t - last) / FPS, np.nan))
    add("t_until", np.where(nxt < n, (nxt - t) / FPS, np.nan))
    for name in ("dd", "bx", "bout", "still", "bsp", "d_gl", "d_tl", "ni_25_0", "ni_0_25"):
        add(name + "_last", _take(feats[name], last))
        add(name + "_next", _take(feats[name], nxt))

    # -- stoppage cues: everyone slowing down -----------------------------------------------------
    with np.errstate(all="ignore"):
        cs = np.nanmean(sp, 1)
    for b, a, name in ((12, 12, "cs1"), (37, 37, "cs3"), (125, 125, "cs10"), (125, 0, "csp5"),
                       (0, 125, "csf5"), (250, 250, "cs20"), (25, 0, "csp1"), (0, 25, "csf1")):
        add(name, _roll(cs, b, a))
    add("cs_delta", feats["csf5"] - feats["csp5"])
    add("nvis", (~np.isnan(X)).sum(1))

    if version != "v1":
        # -- fine touch cues -------------------------------------------------------------------
        # BAR-ATH's ball path is smooth, so a touch shows as the path turning or the ball
        # speeding up / slowing down near a player, not as a one-frame kink. Measured over
        # 0.16 s either side of each frame.
        k = 4
        vbx, vby = (BX - _shift(BX, -k)) * FPS / k, (BY - _shift(BY, -k)) * FPS / k
        vax, vay = (_shift(BX, k) - BX) * FPS / k, (_shift(BY, k) - BY) * FPS / k
        sb_, sa_ = np.hypot(vbx, vby), np.hypot(vax, vay)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosang = (vbx * vax + vby * vay) / (sb_ * sa_)
            turn = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        turn[(sb_ < 0.8) | (sa_ < 0.8)] = np.nan
        add("turn", turn)
        add("turn_max", _roll_max(turn, 4))
        add("dspeed", sa_ - sb_)
        add("dspeed_abs_max", _roll_max(np.abs(sa_ - sb_), 4))
        # which team is closest within a few frames of now
        for tag in ("h", "a"):
            add("d1min5_" + tag, _roll_min(feats["d1_" + tag], 5))
            add("d1min12_" + tag, _roll_min(feats["d1_" + tag], 12))
        add("ddmin5", feats["d1min5_a"] - feats["d1min5_h"])
        for b, a in ((5, 0), (0, 5), (12, 0), (0, 12)):
            add("ni_%d_%d" % (b, a), _roll(ni, b, a))
        add("ni_flip", feats["ni_0_12"] - feats["ni_12_0"])         # near team changing now
        add("dd_m5", _shift(feats["dd"], -5))
        add("dd_p5", _shift(feats["dd"], 5))
        add("bvis_r1", _roll(bvis.astype(float), 12, 12))

    if version not in ("v1", "v2"):
        # v3: drop the visible-player counts. They describe the tracker, not the football (the
        # proxies' dropouts are random), and they are the clearest BAR-vs-proxy tell.
        for name in ("nvis", "nv_h", "nv_a"):
            feats.pop(name)

    names = sorted(feats)
    return np.stack([feats[k] for k in names], axis=1).astype(np.float32), names


def swap_labels(y):
    """H <-> A, D unchanged (codes 0 H, 1 A, 2 D, 3 X)."""
    out = y.copy()
    out[y == 0], out[y == 1] = 1, 0
    return out
