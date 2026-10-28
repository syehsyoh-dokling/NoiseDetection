"""Leak-free pseudo-labels for the natural test clips (semi-supervised training).

The organisers name "semi-supervised approaches exploiting unlabelled Vaani audio" as a
novelty direction; the 3,353 test clips with no noise-library hit are ~6.6 h of exactly
the target distribution (unseen districts and speakers), but without labels.

Leak-free CV: the student for validation fold f is trained on pseudo-labels produced by
the fold-f teachers only, i.e. models that never saw fold f. So the out-of-fold score of
the student is still an honest estimate.

    python pseudo.py dump   --tags wsA,wmA,wlA --store-feats wlA   # per-fold teacher probs
    python pseudo.py build  --spec wsA,wmA,wlA:2 --name p1        # weighted pseudo-labels
"""
import argparse
import glob
import json
import os
import pickle
import time

import numpy as np
import soundfile as sf
import torch

from run import CACHE, ROOT, VAL, load_library, parse_tags, test_clips
from sed import Head, compact, predict_probs, val_items
from template_match import null_levels, select_events


def dump(args):
    """cache/test_perfold_<tag>.pkl: clip -> float16 (n_folds, T, 8); optionally the
    compact test features of one tag's backbone under cache/feats_<backbone>_test/."""
    from features import make_extractor
    clips = test_clips()
    groups = {}
    for t in args.tags.split(","):
        heads = sorted(glob.glob(os.path.join(ROOT, "models", f"sed_head_{t}_f*.pt")))
        assert len(heads) == 5, (t, heads)
        ck = [torch.load(h, map_location="cpu") for h in heads]
        groups.setdefault((ck[0]["backbone"], ck[0]["layers"]), []).append((t, ck))
    for (backbone, layers), group in groups.items():
        todo = [(t, ck) for t, ck in group if not os.path.exists(os.path.join(CACHE, f"test_perfold_{t}.pkl"))]
        feat_dir = None
        if args.store_feats in [t for t, _ in group]:
            feat_dir = os.path.join(CACHE, "feats_" + os.path.basename(os.path.normpath(backbone)) + "_test")
            os.makedirs(feat_dir, exist_ok=True)
        if not todo and feat_dir is None:
            continue
        ext = make_extractor(backbone, tuple(int(x) for x in layers.split(",")))
        heads = []
        for t, ck in todo:
            ms = []
            for c in ck:
                m = Head(ext.dim, hidden=c.get("hidden", 192)).cuda()
                m.load_state_dict(c["state"])
                ms.append(m)
            heads.append((t, ms, {}))
        t0 = time.time()
        for i, (k, f) in enumerate(clips):
            fp = os.path.join(feat_dir, k[:-4] + ".npy") if feat_dir else None
            if fp and os.path.exists(fp):
                feat = np.load(fp)
            else:
                x, _ = sf.read(f, dtype="float32")
                feat = compact(ext(x).cpu().numpy()).astype(np.float16)
                if fp:
                    np.save(fp, feat)
            for _, ms, probs in heads:
                probs[k] = np.stack([predict_probs(m, [feat])[0] for m in ms]).astype(np.float16)
            if i % 500 == 0:
                print(f"  {backbone} {i}/{len(clips)} {time.time() - t0:.0f}s", flush=True)
        for t, _, probs in heads:
            pickle.dump(probs, open(os.path.join(CACHE, f"test_perfold_{t}.pkl"), "wb"))
        del ext
        torch.cuda.empty_cache()


def build(args):
    """cache/pseudo_<name>/<stem>.npy: float16 (n_folds, T, 8) weighted teacher average,
    for the test clips the matched filter does not flag as synthetic."""
    tags, w = parse_tags(args.spec)
    per = [pickle.load(open(os.path.join(CACHE, f"test_perfold_{t}.pkl"), "rb")) for t in tags]
    cfg = json.load(open(args.cfg))
    lib, lens = load_library()
    cands = pickle.load(open(os.path.join(CACHE, "test_template_cands.pkl"), "rb"))
    val_cands = pickle.load(open(os.path.join(CACHE, "val_template_cands.pkl"), "rb"))
    nat = [it["key"] for it in val_items(VAL) if not it["synthetic"]]
    null = null_levels(val_cands, nat, len(lib), cfg["null_q"])
    out = os.path.join(CACHE, f"pseudo_{args.name}")
    os.makedirs(out, exist_ok=True)
    kept = []
    for k, _ in test_clips():
        if select_events(cands[k], lib, lens, cfg["tpl_thr"], null=null):
            continue  # synthetic-like: already handled by the matched filter
        p = np.average([np.asarray(d[k], np.float32) for d in per], axis=0, weights=w)
        np.save(os.path.join(out, k[:-4] + ".npy"), p.astype(np.float16))
        kept.append(k)
    json.dump(kept, open(os.path.join(out, "clips.json"), "w"))
    print(f"pseudo-labelled natural-like test clips: {len(kept)} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dump", "build"])
    ap.add_argument("--tags", default="wsA,wmA,wlA")
    ap.add_argument("--store-feats", default="wlA", help="tag whose backbone's test features are kept")
    ap.add_argument("--spec", default="wsA,wmA,wlA:2")
    ap.add_argument("--cfg", default=os.path.join(ROOT, "models", "fusion_3Aw.json"))
    ap.add_argument("--name", default="p1")
    args = ap.parse_args()
    {"dump": dump, "build": build}[args.stage](args)


if __name__ == "__main__":
    main()
