"""Peak-relative boundary extraction.

postproc.decode places both boundaries where the probability curve crosses one global
threshold. The curve's height varies with how loud the noise is, so the same threshold
sits near the foot of a confident event (boundaries too wide) and near the shoulder of a
faint one (boundaries too narrow). On validation only 12.9% of natural reference events
are truly missed while 52% overlap a prediction that misses a boundary tolerance, and
the median boundary bias is already ~0, so the loss is spread, not offset: the fix has
to be per event, not a global shift.

decode_rel therefore detects a region with a low floor, requires its peak to clear a
detection threshold, and then re-places both boundaries at alpha * peak of that region.
"""
import numpy as np
from scipy.ndimage import median_filter

HOP = 0.02


def _cross(p, c, i0, i1, level):
    a, b = p[i0], p[i1]
    f = 0.5 if a == b else (level - a) / (b - a)
    return c[i0] + float(np.clip(f, 0, 1)) * (c[i1] - c[i0])


def decode_rel(p, thr_det, alpha, floor, med=1, min_dur=0.0, gap=0.0, dur=None, with_peak=False):
    """Regions above `floor` whose peak clears `thr_det`; boundaries at alpha * peak."""
    if med > 1:
        p = median_filter(p, size=med, mode="nearest")
    on = p > floor
    if not on.any():
        return []
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]      # ends exclusive
    c = (np.arange(len(p)) + 0.5) * HOP
    evs = []
    for s, e in zip(starts, ends):
        peak = float(p[s:e].max())
        if peak < thr_det:
            continue
        level = max(floor, alpha * peak)
        # walk out from the peak to the first frame below `level`; the region is above
        # `floor` throughout, so this stays inside [s, e)
        k = s + int(np.argmax(p[s:e]))
        i = k
        while i > s and p[i - 1] >= level:
            i -= 1
        j = k
        while j < e - 1 and p[j + 1] >= level:
            j += 1
        t0 = 0.0 if i == 0 else _cross(p, c, i - 1, i, level)
        t1 = len(p) * HOP if j == len(p) - 1 else _cross(p, c, j, j + 1, level)
        evs.append([t0, t1, peak])
    merged = []
    for ev in sorted(evs):
        if merged and ev[0] - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], ev[1])
            merged[-1][2] = max(merged[-1][2], ev[2])   # a merged event is as strong as its best part
        else:
            merged.append(list(ev))
    out = []
    for a, b, peak in merged:
        if dur is not None:
            b = min(b, dur)
        if b - a >= max(min_dur, 1e-3):
            out.append((round(a, 3), round(b, 3), peak) if with_peak else (round(a, 3), round(b, 3)))
    return out
