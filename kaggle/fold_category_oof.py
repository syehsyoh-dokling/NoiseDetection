"""Per-fold spread and per-category recall of the Q1 configuration, out-of-fold on validation.

Folds: the same GroupKFold(5) over state/district groups that train_cv.py used (items in val_items order).
Per-category recall: fraction of natural reference events of each category that the metric's matcher
(scorer.match_events, 20 % / 50 ms boundary tolerance) pairs with a prediction; predictions are category-agnostic.

    python kaggle_t1/fold_category_oof.py
"""
import collections
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


def ensemble(spec):
    tags, w = run.parse_tags(spec)
    oofs = [pickle.load(open(os.path.join(R, "cache", f"oof_{t}.pkl"), "rb")) for t in tags]
    return {k: np.average([o[k] for o in oofs], axis=0, weights=w) for k in oofs[0]}


spec = cfg["tags"]
systems = {"Q1 full": (ensemble(spec), True),
           "without wlP": (ensemble(",".join(p for p in spec.split(",") if not p.startswith("wlP"))), True),
           "SED only": (ensemble(spec), False)}
groups = np.array([it["group"] for it in items])
folds = list(GroupKFold(n_splits=5).split(items, groups=groups))
print("fold sizes:", [len(te) for _, te in folds], "| districts per fold:",
      [len(set(groups[te])) for _, te in folds])
preds = {}
for name, (probs, use_tpl) in systems.items():
    pred = {k: fuse(tpl[k] if use_tpl else [], probs[k], cfg, durs[k]) for k in ref}
    preds[name] = pred
    per = []
    for _, te in folds:
        keys = [items[i]["key"] for i in te]
        s = score({k: ref[k] for k in keys}, {k: pred[k] for k in keys})
        per.append(s["combined_score"])
    pooled = score(ref, pred)["combined_score"]
    print(f"{name:12s} pooled {pooled:.4f} | folds " + " ".join(f"{v:.4f}" for v in per) +
          f" | mean {np.mean(per):.4f} sd {np.std(per, ddof=1):.4f} min {min(per):.4f} max {max(per):.4f}", flush=True)

# per-category recall on natural clips (Q1 full)
pred = preds["Q1 full"]
hit, tot, dur = collections.Counter(), collections.Counter(), collections.defaultdict(list)
for it in items:
    if it["synthetic"]:
        continue
    k = it["key"]
    evs = it["events"]                      # (start, end, category)
    r = [(s, e) for s, e, _ in evs]
    p = pred[k]
    # replicate match_events to know which reference indices matched
    matched_ref, matched_pred, cand = set(), set(), []
    for ri, (a, b) in enumerate(r):
        tol = max(0.2 * (b - a), 0.05)
        for pi, (c, d) in enumerate(p):
            if abs(c - a) <= tol and abs(d - b) <= tol:
                cand.append((abs(c - a) + abs(d - b), ri, pi))
    for _, ri, pi in sorted(cand):
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri); matched_pred.add(pi)
    for ri, (a, b, c) in enumerate(evs):
        tot[c] += 1; hit[c] += ri in matched_ref; dur[c].append(b - a)
print("natural reference events: recall per category (Q1 full, metric tolerance)")
for c, n in sorted(tot.items(), key=lambda x: -x[1]):
    print(f"  {c:18s} n={n:5d} recall {hit[c] / n:.3f} median dur {np.median(dur[c]):.2f} s")
print(f"  {'all':18s} n={sum(tot.values()):5d} recall {sum(hit.values()) / sum(tot.values()):.3f}")
# duration buckets
b = collections.Counter(); h = collections.Counter()
for it in items:
    if it["synthetic"]:
        continue
    k = it["key"]; r = [(s, e) for s, e, _ in it["events"]]; p = pred[k]
    tp, fp, fn = match_events(r, p)
for c in dur:
    pass
