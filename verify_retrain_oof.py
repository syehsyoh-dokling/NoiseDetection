"""Compare the retrained aFG/bFG out-of-fold predictions with the originals that produced submission 994586.

Merges kaggle_t1/out_retrain/{ast,beats}/cache/oof_<tag>_fold<k>.pkl into cache/oof_<tag>RT.pkl, then reports:
  - per-clip agreement of frame probabilities (Pearson r, mean |diff|) between original and retrain
  - each member alone, and the Q1 fusion (the validated part of the final config; bFS has no OOF) with the
    original members vs. the retrained ones, pooled and per district fold.
Pass: |pooled diff| <= 0.005 for the Q1 fusion (seed-noise level measured earlier).

    python verify_retrain_oof.py
"""
import json
import os
import pickle

import numpy as np
from sklearn.model_selection import GroupKFold

from fusion import crossfit_nulls, fuse
from run import load_library, parse_tags
from scorer import score
from sed import val_items
from template_match import select_events

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
RT = os.path.join(ROOT, "kaggle_t1", "out_retrain")
SRC = {"aFG": "ast", "bFG": "beats"}

for tag, kind in SRC.items():
    merged = {}
    for f in range(5):
        d = pickle.load(open(os.path.join(RT, kind, "cache", f"oof_{tag}_fold{f}.pkl"), "rb"))
        assert not set(d) & set(merged)
        merged.update(d)
    pickle.dump(merged, open(os.path.join(CACHE, f"oof_{tag}RT.pkl"), "wb"))

q1 = json.load(open(os.path.join(ROOT, "models", "fusion_Q1_P2_og20.json")))
items = val_items(os.path.join(ROOT, "data", "validation"))
keys = [it["key"] for it in items]
ref = {it["key"]: [(s, e) for s, e, _ in it["events"]] for it in items}
durs = {it["key"]: it["dur"] for it in items}
folds = list(GroupKFold(n_splits=5).split(items, groups=[it["group"] for it in items]))
cands = pickle.load(open(os.path.join(CACHE, "val_template_cands.pkl"), "rb"))
lib, lens = load_library()
nulls = crossfit_nulls(cands, [it["key"] for it in items if not it["synthetic"]],
                       [it["key"] for it in items if it["synthetic"]], len(lib), q1["null_q"])
tpl = {k: select_events(cands[k], lib, lens, q1["tpl_thr"], exclude_src=k, null=nulls[k]) for k in keys}
cache = {}


def load(t):
    if t not in cache:
        cache[t] = pickle.load(open(os.path.join(CACHE, f"oof_{t}.pkl"), "rb"))
    return cache[t]


def run(spec):
    tags, w = parse_tags(spec)
    pr = {k: np.average([load(t)[k] for t in tags], axis=0, weights=w) for k in keys}
    pred = {k: fuse(tpl[k], pr[k], q1, durs[k]) for k in keys}
    pf = [score({k: ref[k] for k in ks}, {k: pred[k] for k in ks})["combined_score"]
          for ks in ([keys[i] for i in te] for _, te in folds)]
    return score(ref, pred), pf


for tag in SRC:
    a, b = load(tag), load(tag + "RT")
    assert set(a) == set(b) == set(keys), f"{tag}: clip set differs"
    r = [np.corrcoef(a[k].ravel(), b[k].ravel())[0, 1] for k in keys if a[k].std() > 0 and b[k].std() > 0]
    md = np.mean([np.abs(a[k] - b[k]).mean() for k in keys])
    print(f"{tag}: per-clip r median {np.median(r):.4f} p10 {np.percentile(r, 10):.4f} | mean |diff| {md:.4f}", flush=True)

rows = [("aFG alone", "aFG:1", "aFGRT:1"), ("bFG alone", "bFG:1", "bFGRT:1"),
        ("Q1 fusion", q1["tags"], q1["tags"].replace("aFG:", "aFGRT:").replace("bFG:", "bFGRT:"))]
verdict = None
for name, orig, new in rows:
    o, of = run(orig)
    n, nf = run(new)
    d = n["combined_score"] - o["combined_score"]
    print(f"{name:10s} orig {o['combined_score']:.5f} | retrain {n['combined_score']:.5f} ({d:+.5f}) "
          f"F1 {o['event_f1']:.4f}->{n['event_f1']:.4f} Dice {o['segment_dice']:.4f}->{n['segment_dice']:.4f} "
          f"| folds {np.round(np.subtract(nf, of), 4).tolist()}", flush=True)
    if name == "Q1 fusion":
        verdict = "PASS" if d >= -0.005 else "FAIL"
print("VERDICT (Q1 fusion within -0.005 or better):", verdict, flush=True)
