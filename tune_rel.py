"""Tune the peak-relative split decode (reldecode.decode_rel) on natural OOF clips.

Coordinate descent from several starting points, because the two branches have six
parameters each (a full grid would be 10^6 evaluations) and descent only finds the basin
it starts in. Reports the current global-threshold decode as the baseline.
"""
import argparse
import itertools
import json
import os
import pickle

import numpy as np

from calib import fallback
from postproc import HOP, decode_clip, dedupe
from reldecode import decode_rel
from scorer import score
from sed import val_items

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
HNS = 7


def parse_tags(spec):
    tags, w = [], []
    for part in spec.split(","):
        t, _, k = part.partition(":")
        tags.append(t); w.append(float(k) if k else 1.0)
    return tags, w


def branch_probs(probs):
    agn = probs[:, 0]
    return (np.minimum(probs[:, HNS], agn),
            np.minimum(np.delete(probs[:, 1:], HNS - 1, axis=1).max(1), agn))


def decode_rel_split(probs, cfg, dur):
    p_h, p_o = branch_probs(probs)
    evs = decode_rel(p_h, cfg["thr_det"], cfg["alpha"], cfg["floor"], cfg["med"],
                     cfg["min_dur"], cfg["gap"], dur)
    wide = decode_rel(p_o, cfg["o_thr_det"], cfg["o_alpha"], cfg["o_floor"], cfg["o_med"],
                      cfg["o_min_dur"], cfg["o_gap"], dur)
    end = dur if dur is not None else len(probs) * HOP
    out = list(evs)
    for a, b in wide:
        a, b = max(0.0, a - cfg["o_dil"]), min(end, b + cfg["o_dil"])
        if a <= cfg["o_snap"]:
            a = 0.0
        if end - b <= cfg["o_snap"]:
            b = end
        out.append((round(a, 3), round(b, 3)))
    return dedupe(sorted(out), cfg["iou"])


def evaluate(oof, ref, durs, cfg, base_cfg):
    pred = {}
    for k in ref:
        e = decode_rel_split(oof[k], cfg, durs[k])
        pred[k] = e or fallback(oof[k], base_cfg, durs[k], durs[k])
    return score(ref, pred)


def descend(oof, ref, durs, cur, best, base_cfg, grids, sweeps, si):
    for sweep in range(sweeps):
        improved = False
        for name, vals in grids:
            old, keep = cur[name], None
            for v in vals:
                if v == old:
                    continue
                cur[name] = v
                s = evaluate(oof, ref, durs, cur, base_cfg)["combined_score"]
                if s > best + 1e-9:
                    best, keep = s, v
            cur[name] = keep if keep is not None else old
            if keep is not None:
                improved = True
        print(f"  start {si} sweep {sweep}: combined {best:.4f}", flush=True)
        if not improved:
            break
    return cur, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--restarts", type=int, default=0,
                    help="extra random starting points; coordinate descent is local, and the"
                         " decode grid has several basins")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = val_items(os.path.join(ROOT, "data", "validation"))
    nat = [it for it in items if not it["synthetic"]]
    ref = {it["key"]: [(s, e) for s, e, _ in it["events"]] for it in nat}
    durs = {it["key"]: it["dur"] for it in nat}
    tags, w = parse_tags(args.tags)
    oofs = [pickle.load(open(os.path.join(CACHE, f"oof_{t}.pkl"), "rb")) for t in tags]
    oof = {k: np.average([o[k] for o in oofs], axis=0, weights=w) for k in ref}
    base_cfg = json.load(open(args.cfg))["sed_nat"]

    base = score(ref, {k: decode_clip(oof[k], base_cfg, durs[k]) for k in ref})
    print(f"baseline (global threshold): {json.dumps({k: round(v, 4) for k, v in base.items()})}", flush=True)

    cur = {"thr_det": 0.4, "alpha": 0.5, "floor": 0.15, "med": 5, "min_dur": 0.1, "gap": 0.1,
           "o_thr_det": 0.3, "o_alpha": 0.5, "o_floor": 0.12, "o_med": 5, "o_min_dur": 0.3,
           "o_gap": 1.5, "o_dil": 0.0, "o_snap": 0.3, "iou": 0.5}
    grids = [
        ("alpha", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        ("thr_det", [0.3, 0.4, 0.5, 0.6, 0.7]),
        ("floor", [0.05, 0.1, 0.15, 0.2, 0.3]),
        ("o_alpha", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        ("o_thr_det", [0.2, 0.3, 0.4, 0.5, 0.6]),
        ("o_floor", [0.05, 0.1, 0.15, 0.2, 0.3]),
        ("o_gap", [0.3, 0.6, 1.0, 1.5, 2.5]),
        ("o_min_dur", [0.1, 0.3, 0.5, 0.8]),
        ("o_snap", [0.0, 0.3, 0.6]),
        ("o_dil", [0.0, 0.05, 0.1]),
        ("med", [3, 5, 7]),
        ("o_med", [3, 5, 7]),
        ("min_dur", [0.06, 0.1, 0.15]),
        ("gap", [0.04, 0.1, 0.2]),
        ("iou", [0.3, 0.5, 0.7]),
    ]
    starts = [dict(cur)]
    rng = np.random.default_rng(args.seed)
    for _ in range(args.restarts):
        s0 = dict(cur)
        for name, vals in grids:
            s0[name] = vals[int(rng.integers(len(vals)))]
        starts.append(s0)
    overall = None
    for si, start in enumerate(starts):
        cur = dict(start)
        best = evaluate(oof, ref, durs, cur, base_cfg)["combined_score"]
        print(f"start {si}: {best:.4f}", flush=True)
        cur, best = descend(oof, ref, durs, cur, best, base_cfg, grids, args.sweeps, si)
        if overall is None or best > overall[0]:
            overall = (best, dict(cur))
            print(f"  start {si} is the best so far: {best:.4f}", flush=True)
    best, cur = overall
    final = evaluate(oof, ref, durs, cur, base_cfg)
    print(f"tuned natural: {json.dumps({k: round(v, 4) for k, v in final.items()})}")
    print(f"gain vs global threshold {final['combined_score'] - base['combined_score']:+.4f}")
    print("cfg =", json.dumps(cur))
    if args.out:
        full = json.load(open(args.cfg))
        full["sed_nat"] = {"mode": "rel_split", **cur}
        full["nat_rel"] = {k: round(v, 4) for k, v in final.items()}
        json.dump(full, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
