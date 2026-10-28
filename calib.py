"""Boundary calibration for natural-clip SED events.

The metric accepts an event only when BOTH boundaries land within
max(0.2 * ref_duration, 0.05) s, so a systematic onset/offset bias costs F1 directly
even when the event is found. decode_split already applies one fixed dilation and an
edge snap; this module fits a per-branch affine correction instead:

    onset'  = onset  + a0 + a1 * L
    offset' = offset + b0 + b1 * L          (L = predicted event length)

The two branches of decode_split have opposite biases -- the tight human_non_speech
branch is already close, the merged branch that covers <tag>-wraps-words events starts
late and ends early -- so they get their own coefficients. Coefficients are fitted by
coordinate descent on the out-of-fold natural predictions and are applied at predict
time in run.py.
"""
import numpy as np

from postproc import HOP, decode, dedupe
from scorer import score

HNS = 7


def decode_tagged(probs, cfg, dur):
    """decode_split, but every event carries the branch it came from (0 tight, 1 wide)."""
    agn = probs[:, 0]
    p_h = np.minimum(probs[:, HNS], agn)
    p_o = np.minimum(np.delete(probs[:, 1:], HNS - 1, axis=1).max(1), agn)
    out = [(a, b, 0) for a, b in decode(p_h, cfg["thr"], cfg["med"], cfg["min_dur"], cfg["gap"], dur)]
    wide = decode(p_o, cfg["o_thr"], cfg["o_med"], cfg["o_min_dur"], cfg["o_gap"], dur)
    end = dur if dur is not None else len(agn) * HOP
    for a, b in wide:
        a, b = max(0.0, a - cfg["o_dil"]), min(end, b + cfg["o_dil"])
        if a <= cfg["o_snap"]:
            a = 0.0
        if end - b <= cfg["o_snap"]:
            b = end
        out.append((a, b, 1))
    return out, end


def apply_corr(tagged, end, corr, iou=0.5):
    """corr: {branch: (a0, a1, b0, b1)} -> corrected, deduped (onset, offset) list."""
    evs = []
    for a, b, br in tagged:
        a0, a1, b0, b1 = corr[br]
        L = b - a
        a2 = min(max(0.0, a + a0 + a1 * L), end)
        b2 = min(max(0.0, b + b0 + b1 * L), end)
        if b2 - a2 >= 1e-3:
            evs.append((round(a2, 3), round(b2, 3)))
    return dedupe(sorted(evs), iou)


def fallback(probs, cfg, dur, end):
    """postproc.decode_clip's empty-clip fallback, kept identical."""
    evs = decode(probs[:, 0], float(probs[:, 0].max()) * 0.5, cfg["med"], 0.0, cfg["gap"], dur)[:1]
    if not evs:
        c = (int(np.argmax(probs[:, 0])) + 0.5) * HOP
        evs = [(round(max(0.0, c - 0.25), 3), round(min(end, c + 0.25), 3))]
    return evs


def evaluate(tagged, ref, corr, iou=0.5):
    """tagged: {key: (events, end, probs, dur)} -> metric dict."""
    pred = {}
    for k, (tg, end, probs, dur, cfg) in tagged.items():
        e = apply_corr(tg, end, corr, iou)
        pred[k] = e or fallback(probs, cfg, dur, end)
    return score(ref, pred)


def bias_report(tagged, ref, log=print):
    """Median boundary error of the closest prediction per reference event, by branch
    and by reference length -- this is what the affine form has to absorb."""
    rows = {}
    for k, (tg, _end, _p, _d, _c) in tagged.items():
        for r_on, r_off in ref[k]:
            best = None
            for a, b, br in tg:
                ov = min(b, r_off) - max(a, r_on)
                if ov > 0 and (best is None or ov > best[0]):
                    best = (ov, a, b, br)
            if best is None:
                continue
            _, a, b, br = best
            bucket = "short(<0.6s)" if r_off - r_on < 0.6 else ("mid(0.6-1.5s)" if r_off - r_on < 1.5 else "long(>=1.5s)")
            rows.setdefault((br, bucket), []).append((a - r_on, b - r_off, r_off - r_on))
    for key in sorted(rows):
        v = np.array(rows[key])
        log(f"  branch {key[0]} {key[1]:14s} n={len(v):5d} onset err med {np.median(v[:, 0]):+.3f} "
            f"offset err med {np.median(v[:, 1]):+.3f} (ref len med {np.median(v[:, 2]):.2f})")
