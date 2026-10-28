"""Re-tune the ensemble weights under a fixed natural decode.

The weights in fusion_S_*.json were chosen while the natural clips were decoded with one
global threshold. The peak-relative decode reads the same probabilities differently --
it cares about the shape of each peak rather than its absolute height -- so the mixture
that suited the old decode need not suit this one. Coordinate descent over a coarse
weight grid, natural clips only (the synthetic half is decided by the matched filter).
"""
import argparse
import json
import os
import pickle

import numpy as np

from calib import fallback
from postproc import decode_clip
from scorer import score
from sed import val_items

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
GRID = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="comma-separated tags, each optionally tag:start_weight")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--sweeps", type=int, default=3)
    args = ap.parse_args()

    items = val_items(os.path.join(ROOT, "data", "validation"))
    nat = [it for it in items if not it["synthetic"]]
    ref = {it["key"]: [(s, e) for s, e, _ in it["events"]] for it in nat}
    durs = {it["key"]: it["dur"] for it in nat}
    keys = list(ref)
    tags, w0 = [], []
    for part in args.pool.split(","):
        t, _, x = part.partition(":")
        tags.append(t); w0.append(float(x) if x else 0.0)
    stack = {}
    for k in keys:
        stack[k] = None
    for i, t in enumerate(tags):
        o = pickle.load(open(os.path.join(CACHE, f"oof_{t}.pkl"), "rb"))
        for k in keys:
            a = np.asarray(o[k], dtype=np.float32)
            if stack[k] is None:
                stack[k] = np.empty((len(tags),) + a.shape, np.float32)
            stack[k][i] = a
        del o
        print(f"  loaded oof_{t}", flush=True)
    cfg = json.load(open(args.cfg))["sed_nat"]

    def run(w):
        w = np.asarray(w, np.float32)
        if w.sum() <= 0:
            return {"combined_score": 0.0}
        pred = {}
        for k in keys:
            p = np.tensordot(w, stack[k], axes=(0, 0)) / w.sum()
            e = decode_clip(p, cfg, durs[k])
            pred[k] = e or fallback(p, cfg, durs[k], durs[k])
        return score(ref, pred)

    cur = list(w0)
    best = run(cur)["combined_score"]
    print(f"start {dict(zip(tags, cur))} -> {best:.4f}", flush=True)
    for sweep in range(args.sweeps):
        moved = False
        for i, t in enumerate(tags):
            old, keep = cur[i], None
            for v in GRID:
                if v == old:
                    continue
                cur[i] = v
                s = run(cur)["combined_score"]
                if s > best + 1e-9:
                    best, keep = s, v
            cur[i] = keep if keep is not None else old
            if keep is not None:
                moved = True
                print(f"  sweep {sweep} {t} -> {cur[i]}  combined {best:.4f}", flush=True)
        if not moved:
            print(f"  sweep {sweep}: no further gain", flush=True)
            break
    final = run(cur)
    spec = ",".join(f"{t}:{int(v) if v == int(v) else v}" for t, v in zip(tags, cur) if v > 0)
    print(f"tuned natural: {json.dumps({k: round(v, 4) for k, v in final.items()})}")
    print("tags =", spec)


if __name__ == "__main__":
    main()
