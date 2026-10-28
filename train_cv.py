"""5-fold grouped cross-validation of the SED head on the validation release.

Folds are grouped by state/district because the hidden test set is split by
state/district/speaker. Out-of-fold probabilities are saved for post-processing tuning
and for the fusion with the template matcher.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

import sed
from postproc import fmt, grid_search
from scorer import score
from sed import build_memmap, predict_probs, train_head, val_items

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=os.path.join(ROOT, "models", "whisper-small"))
    ap.add_argument("--layers", default="3,6,9,12")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="ws")
    ap.add_argument("--final-seeds", type=int, default=0)
    ap.add_argument("--hf", action="store_true", help="add data/hf_gold (hf_data.py) to training")
    ap.add_argument("--aug", action="store_true", help="SpecAugment-style feature masking (sed.augment)")
    ap.add_argument("--seed", type=int, default=0, help="head init / batch order seed (folds stay fixed)")
    ap.add_argument("--silver", default="", help="folder under data/ with the Silver tier (hf_extract.py --quality unverified_timestamps)")
    ap.add_argument("--silver-weight", type=float, default=0.4, help="loss weight for Silver clips (their timestamps are unverified)")
    ap.add_argument("--pseudo", default="", help="name of cache/pseudo_<name> built by pseudo.py")
    ap.add_argument("--pseudo-weight", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=192, help="GRU width of the head")
    ap.add_argument("--only-fold", type=int, default=-1, help="train just this fold (quick experiment)")
    ap.add_argument("--val-repeat", type=int, default=1, help="times each validation training clip appears per epoch")
    ap.add_argument("--resume", action="store_true", help="reuse fold models already saved under this tag")
    ap.add_argument("--bs", type=int, default=16, help="clips per batch")
    ap.add_argument("--max-frames", type=int, default=600, help="random crop length in 20 ms frames")
    ap.add_argument("--dice-weight", type=float, default=0.0,
                    help="weight of the soft-Dice term on the agnostic channel (sed.soft_dice_loss)")
    ap.add_argument("--compact", action="store_true",
                    help="cache features already layer-averaged (4x less disk; needed for larger backbones)")
    args = ap.parse_args()
    frac = float(os.environ.get("INDOML_GPU_FRAC", "1"))
    if frac < 1.0 and sed.DEVICE == "cuda":
        torch.cuda.set_per_process_memory_fraction(frac)   # module-level torch (a local import here unbound it)
        print(f"GPU memory capped at {frac:.0%}", flush=True)

    items = val_items(os.path.join(ROOT, "data", "validation"))
    feat_dir = os.path.join(ROOT, "cache", "feats_" + os.path.basename(os.path.normpath(args.backbone)))
    indexed = os.path.exists(feat_dir + "_compact.npy") and os.path.exists(feat_dir + "_compact_index.json")
    if not indexed and not all(os.path.exists(os.path.join(feat_dir, it["key"][:-4] + ".npy")) for it in items):
        from features import make_extractor
        ext = make_extractor(args.backbone, tuple(int(x) for x in args.layers.split(",")))
        from sed import cache_features
        t0 = time.time()
        cache_features(items, ext, feat_dir, compact_only=args.compact)
        print(f"features done in {time.time() - t0:.0f}s", flush=True)
        del ext
    for it in items:
        it["feat"] = os.path.join(feat_dir, it["key"][:-4] + ".npy")
    t0 = time.time()
    store = build_memmap(items, feat_dir + "_compact.npy")
    print(f"compact store {store.shape} in {time.time() - t0:.0f}s", flush=True)

    n_val = len(items)
    extra = []
    if args.hf:
        from features import make_extractor
        from sed import cache_features, hf_items
        extra = hf_items(os.path.join(ROOT, "data", "hf_gold"))
        hf_dir = feat_dir + "_hf"
        if not all(os.path.exists(os.path.join(hf_dir, it["key"][:-4] + ".npy")) for it in extra):
            ext = make_extractor(args.backbone, tuple(int(x) for x in args.layers.split(",")))
            cache_features(extra, ext, hf_dir, compact_only=True, q8=True)
            del ext
        for it in extra:
            it["feat"] = os.path.join(hf_dir, it["key"][:-4] + ".npy")
        print(f"Hugging Face Gold clips added to every training fold: {len(extra)}", flush=True)
    if args.silver:
        from features import make_extractor
        from sed import cache_features, hf_items
        silver = hf_items(os.path.join(ROOT, "data", args.silver))
        sdir = feat_dir + "_" + args.silver
        if not all(os.path.exists(os.path.join(sdir, it["key"][:-4] + ".npy")) for it in silver):
            ext = make_extractor(args.backbone, tuple(int(x) for x in args.layers.split(",")))
            cache_features(silver, ext, sdir, compact_only=True, q8=True)
            del ext
        for it in silver:
            it["feat"] = os.path.join(sdir, it["key"][:-4] + ".npy")
            it["weight"] = args.silver_weight
        extra += silver
        hours = sum(it["dur"] for it in silver) / 3600
        print(f"Silver clips added to every training fold: {len(silver)} ({hours:.1f} h, weight {args.silver_weight})",
              flush=True)
    if args.pseudo:
        pdir = os.path.join(ROOT, "cache", f"pseudo_{args.pseudo}")
        tdir = os.path.join(ROOT, "cache", "feats_" + os.path.basename(os.path.normpath(args.backbone)) + "_test")
        assert os.path.isdir(tdir), f"test features for this backbone missing: {tdir} (pseudo.py dump --store-feats)"
        for k in json.load(open(os.path.join(pdir, "clips.json"))):
            extra.append({"key": k, "feat": os.path.join(tdir, k[:-4] + ".npy"),
                          "soft": os.path.join(pdir, k[:-4] + ".npy"), "weight": args.pseudo_weight,
                          "events": [], "synthetic": False, "group": "test"})
        print(f"pseudo-labelled test clips added to every training fold: "
              f"{sum('soft' in it for it in extra)} (weight {args.pseudo_weight})", flush=True)
    all_items = items + extra
    hf_idx = np.arange(n_val, len(all_items))

    groups = np.array([it["group"] for it in items])
    oof = {}
    for f, (tr, te) in enumerate(GroupKFold(n_splits=args.folds).split(items, groups=groups)):
        if args.only_fold >= 0 and f != args.only_fold:
            continue
        t0 = time.time()
        ck_path = os.path.join(ROOT, "models", f"sed_head_{args.tag}_f{f}.pt")
        if args.resume and os.path.exists(ck_path):
            # a crashed run left this fold finished: reuse it for the OOF predictions
            ck = torch.load(ck_path, map_location="cpu")
            model = sed.Head(store.shape[1], hidden=ck.get("hidden", 192)).to(sed.DEVICE)
            model.load_state_dict(ck["state"])
            for i, p in zip(te, predict_probs(model, (store[items[i]["off"]: items[i]["off"] + items[i]["T"]] for i in te))):
                oof[items[i]["key"]] = p
            print(f"fold {f}: reused finished model", flush=True)
            continue
        # validation hold-out stays pure; external clips only ever train, and pseudo-labels
        # come from the fold-f teachers, which never saw this fold
        sed.PSEUDO_FOLD = f
        # validation clips follow the test distribution exactly (annotation style, synthetic
        # mix); with 10k external clips they would be ~16% of a batch, so optionally repeat them
        model = train_head(all_items, store, np.concatenate([np.tile(tr, args.val_repeat), hf_idx]),
                           epochs=args.epochs, log=print,
                           aug=args.aug, seed=args.seed, hidden=args.hidden, bs=args.bs,
                           max_frames=args.max_frames, dice_weight=args.dice_weight)
        # each fold model (80% of the data) is also a member of the test-time ensemble
        torch.save({"state": model.state_dict(), "backbone": args.backbone, "layers": args.layers,
                    "hidden": args.hidden}, os.path.join(ROOT, "models", f"sed_head_{args.tag}_f{f}.pt"))
        probs = predict_probs(model, (store[items[i]["off"]: items[i]["off"] + items[i]["T"]] for i in te))
        for i, p in zip(te, probs):
            oof[items[i]["key"]] = p
        print(f"fold {f}: {len(tr)} train / {len(te)} held out, {time.time() - t0:.0f}s", flush=True)
    if args.only_fold >= 0:
        return  # single-fold experiment: no complete OOF file to write
    pickle.dump(oof, open(os.path.join(ROOT, "cache", f"oof_{args.tag}.pkl"), "wb"))

    ref = {it["key"]: [(s, e) for s, e, _ in it["events"]] for it in items}
    durs = {it["key"]: it["dur"] for it in items}
    nat = [it["key"] for it in items if not it["synthetic"]]
    syn = [it["key"] for it in items if it["synthetic"]]
    # quick sanity number; the real decoding/fusion search lives in `run.py tune`
    grid = {"mode": ["agnostic"], "thr": [0.4, 0.5], "med": [3], "min_dur": [0.06], "gap": [0.04]}
    for name, keys in (("ALL", list(ref)), ("natural", nat), ("synthetic", syn)):
        print(f"[{name}] SED-only OOF")
        grid_search(oof, ref, durs, grid, keys)

    # Optional extra heads on every validation clip (the fold models already form an ensemble).
    for seed in range(args.final_seeds):
        model = train_head(items, store, np.arange(len(items)), epochs=args.epochs, seed=seed, log=print)
        path = os.path.join(ROOT, "models", f"sed_head_{args.tag}_s{seed}.pt")
        torch.save({"state": model.state_dict(), "backbone": args.backbone, "layers": args.layers}, path)
        print("saved", path)


if __name__ == "__main__":
    main()
