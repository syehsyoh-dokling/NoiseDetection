"""Out-of-fold component scores of the final Track 1 configuration on the validation release (5 district folds,
leave-one-clip-out library, cross-fitted null levels, exactly as run.tune evaluates):

  full        Q1 configuration (all members with an OOF estimate, i.e. without bFS)
  no_wlP      the same without the pseudo-label member wlP
  sed_only    SED branch alone on every clip (no library events)
  tpl_only    library matched filter alone (no SED events anywhere)
  retrained   full, with the retrained aFG/bFG out-of-fold probabilities (oof_aFGRT / oof_bFGRT)

Each row reports combined / F1 / Dice on all clips, on the 1,806 natural and on the 695 synthetic clips.

    python kaggle_t1/component_oof.py
"""
import json
import os
import pickle
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
os.chdir(R)
sys.stdout.reconfigure(encoding="utf-8")
import run  # noqa: E402
from fusion import crossfit_nulls, fuse  # noqa: E402
from postproc import decode_clip  # noqa: E402
from scorer import score  # noqa: E402
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
print(f"clips {len(ref)} (natural {len(nat)}, synthetic {len(syn)}); library {len(lib)}; "
      f"clips with a library event: natural {sum(bool(tpl[k]) for k in nat)}, synthetic {sum(bool(tpl[k]) for k in syn)}",
      flush=True)


def ensemble(spec):
    tags, w = run.parse_tags(spec)
    oofs = [pickle.load(open(os.path.join(R, "cache", f"oof_{t}.pkl"), "rb")) for t in tags]
    return {k: np.average([o[k] for o in oofs], axis=0, weights=w) for k in oofs[0]}


def report(name, pred):
    rows = []
    for sub, keys in (("all", list(ref)), ("natural", nat), ("synthetic", syn)):
        s = score({k: ref[k] for k in keys}, {k: pred[k] for k in keys})
        rows.append(f"{sub} {s['combined_score']:.4f} ({s['event_f1']:.4f}/{s['segment_dice']:.4f})")
    print(f"{name:10s} " + " | ".join(rows), flush=True)


spec = cfg["tags"]
full = ensemble(spec)
report("full", {k: fuse(tpl[k], full[k], cfg, durs[k]) for k in ref})
no_wlp = ensemble(",".join(p for p in spec.split(",") if not p.startswith("wlP")))
report("no_wlP", {k: fuse(tpl[k], no_wlp[k], cfg, durs[k]) for k in ref})
report("sed_only", {k: fuse([], full[k], cfg, durs[k]) for k in ref})
report("tpl_only", {k: [(round(a, 3), round(b, 3)) for a, b, _, _ in tpl[k]] for k in ref})
rt = ensemble(spec.replace("aFG:", "aFGRT:").replace("bFG:", "bFGRT:"))
report("retrained", {k: fuse(tpl[k], rt[k], cfg, durs[k]) for k in ref})
