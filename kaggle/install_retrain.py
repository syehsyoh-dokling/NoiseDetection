"""Assemble the reproduction of the final Track 1 submission (994586, fusion_Q1_bFSadd4) from the recovered parts:

  cache/test_sed_probs_<tag>.pkl      nine frozen members, recomputed on Kaggle (kernel indoml-t1-frozen)
  cache/test_probs_aFG_f<k>.pkl       retrained AST fine-tune, per fold (kernel indoml-t1-retrain-ast)
  cache/test_probs_bFG_f<k>.pkl       retrained BEATs fine-tune, per fold (kernel indoml-t1-retrain-beats)
  cache/test_sed_probs_bFS.pkl        retrained full-data BEATs+Silver member, padded to aH's frame count

then builds submission/submission_v20_Q1_bFSadd4_repro.zip through build_nodata (cached probabilities, exact
durations, no backbones needed) and compares it with the original zip event by event.

    python kaggle_t1/install_retrain.py            # install + build + compare
    python kaggle_t1/install_retrain.py --install  # install only
"""
import os
import pickle
import shutil
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CACHE = os.path.join(ROOT, "cache")
RT = os.path.join(ROOT, "kaggle_t1", "out_retrain")
FROZEN = os.path.join(ROOT, "kaggle_t1", "out_frozen", "cache")
TAGS = ["wsA", "wlP", "wlH", "wlH2", "wlH3", "wlHS", "aH", "bH", "dH"]
ORIG = os.path.join(ROOT, "submission", "submission_v20_Q1_bFSadd4.zip")


def copy(src, dst):
    assert os.path.exists(src) and os.path.getsize(src) > 0, src
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
        shutil.copy2(src, dst)
    return dst


def install():
    for t in TAGS:
        copy(os.path.join(FROZEN, f"test_sed_probs_{t}.pkl"), os.path.join(CACHE, f"test_sed_probs_{t}.pkl"))
    for kind, tag in (("ast", "aFG"), ("beats", "bFG")):
        for k in range(5):
            copy(os.path.join(RT, kind, "cache", f"test_probs_{tag}_f{k}.pkl"),
                 os.path.join(CACHE, f"test_probs_{tag}_f{k}.pkl"))
    ref = pickle.load(open(os.path.join(CACHE, "test_sed_probs_aH.pkl"), "rb"))["probs"]
    probs = pickle.load(open(os.path.join(RT, "beats", "cache", "test_probs_bFS_full.pkl"), "rb"))
    bad = [k for k in ref if k not in probs or probs[k].shape != ref[k].shape]
    print(f"bFS test probs: {len(probs)} clips | shape mismatches vs aH: {len(bad)}", flush=True)
    for k in bad:                                   # same padding rule as the original install_bfs.py
        p = probs.get(k, np.zeros_like(ref[k]))
        T = ref[k].shape[0]
        probs[k] = p[:T] if len(p) >= T else np.concatenate([p, np.repeat(p[-1:], T - len(p), 0)], 0)
    pickle.dump({"heads": [], "probs": probs}, open(os.path.join(CACHE, "test_sed_probs_bFS.pkl"), "wb"))
    for t in TAGS:
        n = len(pickle.load(open(os.path.join(CACHE, f"test_sed_probs_{t}.pkl"), "rb"))["probs"])
        assert n == len(ref), (t, n)
    print("installed: 9 frozen + aFG/bFG folds + bFS, all", len(ref), "clips", flush=True)


def build_and_compare():
    import build_nodata as bn
    out = os.path.join(ROOT, "submission", "submission_v20_Q1_bFSadd4.zip")
    assert os.path.exists(ORIG), ORIG
    keep = os.path.join(ROOT, "orig_submission", "submission_v20_Q1_bFSadd4.zip")
    if not os.path.exists(keep):                     # the file that was uploaded as 994586: never overwrite it
        shutil.copy2(ORIG, keep)
    bn.build("Q1_bFSadd4")
    repro = os.path.join(ROOT, "submission", "submission_v20_Q1_bFSadd4_repro.zip")
    shutil.move(out, repro)
    shutil.copy2(keep, ORIG)
    bn.compare(keep, repro)
    import scorer
    ev = {}
    for name, z in (("orig", keep), ("repro", repro)):
        jl = os.path.join(ROOT, "submission", f"predictions_{name}.jsonl")
        open(jl, "wb").write(zipfile.ZipFile(z).read("predictions.jsonl"))
        ev[name] = scorer.load_jsonl(jl)
    a, b = ev["orig"], ev["repro"]
    same = sum(a[k] == b.get(k) for k in a)
    print(f"clips {len(a)} vs {len(b)}; identical clips {same} ({100 * same / len(a):.2f}%); "
          f"events {sum(map(len, a.values()))} vs {sum(map(len, b.values()))}", flush=True)
    print("agreement (original as reference, metric scale 0-2):", scorer.score(a, b), flush=True)


if __name__ == "__main__":
    install()
    if "--install" not in sys.argv[1:]:
        build_and_compare()
