"""Frame probabilities -> onset/offset events, plus the grid search that tunes it."""
import itertools

import numpy as np
from scipy.ndimage import median_filter

from reldecode import decode_rel
from scorer import score

HOP = 0.02


def decode(p, thr, med=1, min_dur=0.0, gap=0.0, dur=None):
    """p: (T,) probabilities at 50 fps -> list of (onset, offset) seconds.

    Boundaries are placed where the smoothed curve crosses thr, linearly interpolated
    between frame centres, which is finer than the 20 ms frame grid.
    """
    if med > 1:
        p = median_filter(p, size=med, mode="nearest")
    on = p > thr
    if not on.any():
        return []
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]  # ends exclusive
    c = (np.arange(len(p)) + 0.5) * HOP

    def cross(i0, i1):  # time where p crosses thr between frames i0 and i1
        a, b = p[i0], p[i1]
        f = 0.5 if a == b else (thr - a) / (b - a)
        return c[i0] + float(np.clip(f, 0, 1)) * (c[i1] - c[i0])

    evs = []
    for s, e in zip(starts, ends):
        t0 = 0.0 if s == 0 else cross(s - 1, s)
        t1 = len(p) * HOP if e == len(p) else cross(e - 1, e)
        evs.append([t0, t1])
    merged = []
    for ev in evs:
        if merged and ev[0] - merged[-1][1] <= gap:
            merged[-1][1] = ev[1]
        else:
            merged.append(ev)
    out = []
    for a, b in merged:
        if dur is not None:
            b = min(b, dur)
        if b - a >= max(min_dur, 1e-3):
            out.append((round(a, 3), round(b, 3)))
    return out


def decode_clip(probs, cfg, dur=None):
    """probs: (T, 8). cfg['mode'] 'agnostic' uses channel 0 only; 'percat' decodes each
    category channel and keeps overlapping events of different categories."""
    if cfg.get("mode", "agnostic") == "agnostic":
        evs = decode(probs[:, 0], cfg["thr"], cfg["med"], cfg["min_dur"], cfg["gap"], dur)
    elif cfg["mode"] == "split":
        evs = decode_split(probs, cfg, dur)
    elif cfg["mode"] == "rel_split":
        evs = decode_rel_split(probs, cfg, dur)
    elif cfg["mode"] == "split2":
        evs = decode_split2(probs, cfg, dur)
    else:
        evs = []
        for ch in range(1, probs.shape[1]):
            # gate categories by the agnostic channel so they cannot fire alone
            p = np.minimum(probs[:, ch], probs[:, 0])
            evs += decode(p, cfg["thr"], cfg["med"], cfg["min_dur"], cfg["gap"], dur)
        evs = dedupe(sorted(evs), cfg.get("iou", 0.5))
    if not evs and cfg.get("fallback", True):
        # 99.9% of reference clips contain an event; an empty list scores Dice 0 for sure
        evs = decode(probs[:, 0], float(probs[:, 0].max()) * 0.5, cfg["med"], 0.0, cfg["gap"], dur)[:1]
        if not evs:
            # smoothing can erase a lone peak: fall back to +-0.25 s around the argmax
            c = (int(np.argmax(probs[:, 0])) + 0.5) * HOP
            end = dur if dur is not None else len(probs) * HOP
            evs = [(round(max(0.0, c - 0.25), 3), round(min(end, c + 0.25), 3))]
    return evs


HNS = 7  # channel of human_non_speech (sed.CATEGORIES index 6, +1 for the agnostic channel)


def decode_split(probs, cfg, dur):
    """Two label conventions live in this data. Square-bracket tags ([breathing],
    [inhaling]) are standalone sounds whose boundaries follow the sound, and the SED gets
    them within ~40 ms. Angle-bracket tags (<horn> ... </horn>) wrap the spoken words
    the noise is heard over, so a long one runs from phrase start to phrase end while the
    detector only fires where the noise is loud: on validation, events >= 1.5 s were
    predicted ~1 s late at onset and ~1 s early at offset, half of them fragmented.

    human_non_speech is decoded tightly; the other categories are merged across wider
    gaps and dilated, and snapped to the clip edge when they get close to it."""
    agn = probs[:, 0]
    p_h = np.minimum(probs[:, HNS], agn)
    p_o = np.minimum(np.delete(probs[:, 1:], HNS - 1, axis=1).max(1), agn)
    evs = decode(p_h, cfg["thr"], cfg["med"], cfg["min_dur"], cfg["gap"], dur)
    wide = decode(p_o, cfg["o_thr"], cfg["o_med"], cfg["o_min_dur"], cfg["o_gap"], dur)
    end = dur if dur is not None else len(agn) * HOP
    for a, b in wide:
        a, b = max(0.0, a - cfg["o_dil"]), min(end, b + cfg["o_dil"])
        if a <= cfg["o_snap"]:
            a = 0.0
        if end - b <= cfg["o_snap"]:
            b = end
        evs.append((round(a, 3), round(b, 3)))
    return dedupe(sorted(evs), cfg.get("iou", 0.5))


def decode_rel_split(probs, cfg, dur):
    """decode_split with both branches re-cut by reldecode.decode_rel: each event keeps
    its own boundary level (alpha * that event's peak) instead of one global threshold.
    Fitted by tune_rel.py; worth +0.029 event F1 on the natural out-of-fold clips."""
    agn = probs[:, 0]
    p_h = np.minimum(probs[:, HNS], agn)
    p_o = np.minimum(np.delete(probs[:, 1:], HNS - 1, axis=1).max(1), agn)
    out = decode_rel(p_h, cfg["thr_det"], cfg["alpha"], cfg["floor"], cfg["med"],
                     cfg["min_dur"], cfg["gap"], dur)
    wide = decode_rel(p_o, cfg["o_thr_det"], cfg["o_alpha"], cfg["o_floor"], cfg["o_med"],
                      cfg["o_min_dur"], cfg["o_gap"], dur)
    end = dur if dur is not None else len(agn) * HOP
    out = list(out)
    for a, b in wide:
        a, b = max(0.0, a - cfg["o_dil"]), min(end, b + cfg["o_dil"])
        if a <= cfg["o_snap"]:
            a = 0.0
        if end - b <= cfg["o_snap"]:
            b = end
        out.append((round(a, 3), round(b, 3)))
    return dedupe(sorted(out), cfg.get("iou", 0.5))


def decode_split2(probs, cfg, dur):
    """Like decode_split, but the wide (word-wrapping) branch is only trusted when the
    merged event is actually long. Validation showed short <tag> events (<0.6 s) land
    inside the tolerance only 13% of the time because merging and edge-snapping make
    them far too long (offset +0.17 s median), while long ones need exactly that merge.
    So: merge widely, then replace any short merged event with the tight decode inside it.
    """
    agn = probs[:, 0]
    p_h = np.minimum(probs[:, HNS], agn)
    p_o = np.minimum(np.delete(probs[:, 1:], HNS - 1, axis=1).max(1), agn)
    evs = decode(p_h, cfg["thr"], cfg["med"], cfg["min_dur"], cfg["gap"], dur)
    end = dur if dur is not None else len(agn) * HOP
    tight = decode(p_o, cfg["t_thr"], cfg["med"], cfg["t_min_dur"], cfg["t_gap"], dur)
    wide = decode(p_o, cfg["o_thr"], cfg["o_med"], cfg["o_min_dur"], cfg["o_gap"], dur)
    for a, b in wide:
        a2, b2 = max(0.0, a - cfg["o_dil"]), min(end, b + cfg["o_dil"])
        if a2 <= cfg["o_snap"]:
            a2 = 0.0
        if end - b2 <= cfg["o_snap"]:
            b2 = end
        inside = [(c, d) for c, d in tight if min(b, d) - max(a, c) > 0]
        if b2 - a2 >= cfg["long_s"]:
            evs.append((round(a2, 3), round(b2, 3)))
            if cfg.get("keep_inner"):
                # a long merged event can hide several short reference events; emitting the
                # tight detections inside it as well trades precision for their recall
                evs.extend((round(c, 3), round(d, 3)) for c, d in inside
                           if d - c <= cfg.get("inner_max", 0.6))
        else:
            evs.extend((round(c, 3), round(d, 3)) for c, d in (inside or [(a, b)]))
    return dedupe(sorted(evs), cfg.get("iou", 0.5))


def dedupe(evs, iou_thr):
    kept = []
    for a, b in evs:
        if any((min(b, d) - max(a, c)) / (max(b, d) - min(a, c)) > iou_thr for c, d in kept):
            continue
        kept.append((a, b))
    return kept


def grid_search(probs, ref, durs, grid, keys=None, log=print):
    keys = list(ref) if keys is None else keys
    best = None
    names = list(grid)
    for vals in itertools.product(*grid.values()):
        cfg = dict(zip(names, vals))
        pred = {k: decode_clip(probs[k], cfg, durs[k]) for k in keys}
        s = score({k: ref[k] for k in keys}, pred)
        if best is None or s["combined_score"] > best[0]["combined_score"]:
            best = (s, cfg)
    log(f"  best {fmt(best[0])} cfg={best[1]}")
    return best


def fmt(s):
    return " ".join(f"{k.replace('_score', '')}={v:.4f}" for k, v in s.items())
