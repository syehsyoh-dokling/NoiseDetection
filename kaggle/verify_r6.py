"""Round-6 checks: library-hit clip set of the rebuild vs the original run, test audio hours, wlP paired stats."""
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
from template_match import null_levels, select_events  # noqa: E402

cfg = json.load(open(os.path.join(R, "models", "fusion_Q1_bFSadd4.json")))
lib, lens = run.load_library()
cands = pickle.load(open(os.path.join(R, "cache", "test_template_cands.pkl"), "rb"))
val_cands = pickle.load(open(os.path.join(R, "cache", "val_template_cands.pkl"), "rb"))
nat = [it["key"] for it in run.val_items(run.VAL) if not it["synthetic"]]
null = null_levels(val_cands, nat, len(lib), cfg["null_q"])
hit = {k for k in cands if select_events(cands[k], lib, lens, cfg["tpl_thr"], null=null)}
orig = set(json.load(open(os.path.join(R, "cache", "t1_syn_keys.json"))))
norm = lambda s: {x if x.endswith(".wav") else x + ".wav" for x in s}
hit, orig = norm(hit), norm(orig)
print(f"rebuild library-hit clips {len(hit)} | original t1_syn_keys {len(orig)} | identical set {hit == orig} "
      f"| only rebuild {len(hit - orig)} | only original {len(orig - hit)}")
durs = json.load(open(os.path.join(R, "cache", "test_durs.json")))
print(f"test audio: {len(durs)} clips, {sum(durs.values()) / 3600:.2f} h")
d = np.array([0.0066, 0.0004, -0.0008, 0.0010, 0.0056])
d = np.array([1.4912 - 1.4920, 1.4412 - 1.4408, 1.5895 - 1.5829, 1.3787 - 1.3777, 1.4806 - 1.4750])
t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
print("paired diffs", np.round(d, 4), f"mean {d.mean():.4f} sd {d.std(ddof=1):.4f} t {t:.2f}")
