"""Shared helpers for the common per-match format (see convert_idsse.py)."""
import numpy as np

FPS = 25


def load(path):
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    if "pitch" not in out:
        out["pitch"] = np.array([105.3, 68.0], np.float32)
    return out


def metres(arr, pitch):
    """x, y (0-1) -> metres. Works on (..., >=2) arrays."""
    return arr[..., 0] * pitch[0], arr[..., 1] * pitch[1]


def collective_speed(players, pitch, smooth=FPS):
    """Mean speed (m/s) of the visible players, from positions, smoothed over `smooth` frames.
    Computed the same way for every dataset, so it can be a shared context variable."""
    x, y = metres(players, pitch)
    v = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)) * FPS
    v = np.where(v > 12, np.nan, v)  # identity jumps are not running
    with np.errstate(all="ignore"):
        m = np.nanmean(v, axis=1)
    m = np.concatenate(([m[0] if len(m) else 0.0], m))
    m = np.where(np.isnan(m), np.nanmedian(m), m)
    k = np.ones(smooth) / smooth
    return np.convolve(m, k, mode="same")


def runs(mask):
    """[start, stop) of True runs."""
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)
