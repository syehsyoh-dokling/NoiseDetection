"""Combine matched-filter detections with SED events, and tune the combination."""
import itertools

import numpy as np

from postproc import decode_clip, fmt
from scorer import score
from template_match import null_levels, select_events


def fuse(tpl_events, probs, cfg, dur):
    """tpl_events: [(on, off, ncc, tidx)] already thresholded. probs: (T, 8) SED output."""
    if tpl_events:
        # A confident library hit means this is a synthetic clip: pasted samples do not
        # overlap and ~94% of them are in the library, so the SED only fills gaps.
        out = [(round(a, 3), round(b, 3)) for a, b, _, _ in tpl_events]
        if cfg["syn_sed"]:
            sed = decode_clip(probs, {**cfg["sed_syn"], "fallback": False}, dur)
            for a, b in sed:
                if all(min(b, d) - max(a, c) <= cfg["syn_overlap"] * (b - a) for c, d in out):
                    out.append((a, b))
        return sorted(out)
    return decode_clip(probs, cfg["sed_nat"], dur)


def crossfit_nulls(cands, nat_keys, syn_keys, n_templates, q, seed=0):
    """Null level per clip without letting a natural clip calibrate its own threshold:
    natural clips are split in two halves that calibrate each other; synthetic clips
    (never part of calibration) use all natural clips, as the test set will."""
    nat_keys = sorted(nat_keys)
    perm = np.random.default_rng(seed).permutation(len(nat_keys))
    a = [nat_keys[i] for i in perm[: len(nat_keys) // 2]]
    b = [nat_keys[i] for i in perm[len(nat_keys) // 2:]]
    na, nb = null_levels(cands, a, n_templates, q), null_levels(cands, b, n_templates, q)
    nall = null_levels(cands, nat_keys, n_templates, q)
    out = {k: nb for k in a}
    out.update({k: na for k in b})
    out.update({k: nall for k in syn_keys})
    return out


def tune_fusion(cands, lib, lens, probs, ref, durs, nat_keys, syn_keys, sed_nat, sed_syn_grid, log=print):
    """Grid-search the template margin, the null quantile, and synthetic gap filling.
    The library is used leave-one-clip-out so a clip never matches its own noise."""
    best = None
    nulls = {q: crossfit_nulls(cands, nat_keys, syn_keys, len(lib), q) for q in (0.99, 0.999)}
    for q, thr, syn_sed, ov, sed_syn in itertools.product(
            (0.99, 0.999), [0.0, 0.05, 0.1, 0.15], [False, True], [0.0, 0.2], sed_syn_grid):
        if not syn_sed and (ov != 0.0 or sed_syn is not sed_syn_grid[0]):
            continue
        cfg = {"null_q": q, "tpl_thr": thr, "syn_sed": syn_sed, "syn_overlap": ov, "sed_syn": sed_syn,
               "sed_nat": sed_nat}
        pred = {}
        for k in ref:
            tpl = select_events(cands[k], lib, lens, thr, exclude_src=k, null=nulls[q][k])
            pred[k] = fuse(tpl, probs[k], cfg, durs[k])
        s = score(ref, pred)
        if best is None or s["combined_score"] > best[0]["combined_score"]:
            best = (s, cfg)
    c = best[1]
    log(f"  fusion best {fmt(best[0])} null_q={c['null_q']} tpl_thr={c['tpl_thr']} syn_sed={c['syn_sed']} "
        f"ov={c['syn_overlap']} sed_syn={c['sed_syn']}")
    return best[0], best[1], nulls[c["null_q"]]
