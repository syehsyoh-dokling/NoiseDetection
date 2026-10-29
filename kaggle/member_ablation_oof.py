"""Leave-one-member-out OOF of the Q1 configuration and a breakdown of the natural validation clips with a
library hit. Uses only cached OOF probabilities and validation template candidates (no training).

    python kaggle_t1/member_ablation_oof.py
"""
import json
import os
import pickle
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
os.chdir(R)
sys.stdout.reconfigure(encoding="utf-8")
import run  # noqa: E402
from fusion import crossfit_nulls, fuse  # noqa: E402
from scorer import match_events, score  # noqa: E402
from template_match import select_events  # noqa: E402

cfg = json.load(open(os.path.join(R, "models", "fusion_Q1_P2_og20.json")))
items = run.val_items(run.VAL)
ref = {it["key"]: [(s, e) for s, e, _ in it["events"]] for it in items}
durs = {it["key"]: it["dur"] for it in items}
nat = [it["key"] for it in items if not it["synthetic"]]
syn = [it["key"] for it in items if it["synthetic"]]
cands = pickle.load(open(os.path.join(R, "cache", "val_template_cands.pkl"), "rb"))
lib, lens = run.load_library()
nulls = crossfit_nulls(cands, nat, syn, len(lib), cfg["null_q"])
tpl = {k: select_events(cands[k], lib, lens, cfg["tpl_thr"], exclude_src=k, null=nulls[k]) for k in ref}
groups = np.array([it["group"] for it in items])
folds = [[items[i]["key"] for i in te] for _, te in GroupKFold(n_splits=5).split(items, groups=groups)]
tags, w = run.parse_tags(cfg["tags"])
oof = {t: pickle.load(open(os.path.join(R, "cache", f"oof_{t}.pkl"), "rb")) for t in tags}


def system(keep):
    ws = [(t, x) for t, x in zip(tags, w) if t in keep]
    probs = {k: np.average([oof[t][k] for t, _ in ws], axis=0, weights=[x for _, x in ws]) for k in ref}
    return {k: fuse(tpl[k], probs[k], cfg, durs[k]) for k in ref}


def fold_scores(pred):
    return np.array([score({k: ref[k] for k in f}, {k: pred[k] for k in f})["combined_score"] for f in folds])


full = system(set(tags))
base, base_f = score(ref, full)["combined_score"], fold_scores(full)
print(f"Q1 full pooled {base:.4f}")
print("leave-one-member-out: member weight | pooled without | delta (full - without) | folds where full is better")
for t, x in zip(tags, w):
    p = system(set(tags) - {t})
    s, f = score(ref, p)["combined_score"], fold_scores(p)
    print(f"  {t:5s} w={x:<4} {s:.4f} {base - s:+.4f} {int(((base_f - f) > 0).sum())}/5", flush=True)

# natural clips with a library hit
hit = [k for k in nat if tpl[k]]
sed_only = {k: fuse([], np.average([oof[t][k] for t in tags], axis=0, weights=w), cfg, durs[k]) for k in hit}
s_fused = score({k: ref[k] for k in hit}, {k: full[k] for k in hit})
s_sed = score({k: ref[k] for k in hit}, sed_only)
tp = fp = 0
for k in hit:
    a, b, _ = match_events(ref[k], [(on, off) for on, off, _, _ in tpl[k]])
    tp += a; fp += b
print(f"natural clips with a library hit: {len(hit)}; template events {tp + fp}, matched to a reference event "
      f"{tp} ({100 * tp / max(1, tp + fp):.1f} %)")
print(f"  these clips: fused {s_fused['combined_score']:.4f} ({s_fused['event_f1']:.4f}/{s_fused['segment_dice']:.4f})"
      f" vs SED-only decode {s_sed['combined_score']:.4f} ({s_sed['event_f1']:.4f}/{s_sed['segment_dice']:.4f})")
