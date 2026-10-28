"""End-to-end Track 1 pipeline: validation release + test audio -> submission.zip.

    python run.py tune       # pick post-processing / fusion settings on validation OOF
    python run.py predict    # run on data/test/audio and write submission.zip

Prerequisites (see README): data/validation and data/test extracted, SED heads trained
by train_cv.py, template candidates for validation from eval_templates.py.
"""
import argparse
import glob
import json
import os
import pickle
import time
import zipfile

import numpy as np
import soundfile as sf
import torch

from eval_templates import run_matcher
from fusion import fuse, tune_fusion
from postproc import grid_search
from scorer import load_jsonl, score
from sed import Head, compact, predict_probs, val_items
from template_match import build_library, null_levels, select_events

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
VAL = os.path.join(ROOT, "data", "validation")
TEST = os.path.join(ROOT, "data", "test", "audio")
CFG = os.path.join(ROOT, "models", "fusion_cfg.json")


def load_library():
    p = os.path.join(CACHE, "template_library.pkl")
    if not os.path.exists(p):
        os.makedirs(CACHE, exist_ok=True)
        pickle.dump(build_library(VAL), open(p, "wb"))
    lib = pickle.load(open(p, "rb"))
    return lib, np.array([len(t["wave"]) for t in lib])


def tune(args):
    items = val_items(VAL)
    ref = {it["key"]: [(s, e) for s, e, _ in it["events"]] for it in items}
    durs = {it["key"]: it["dur"] for it in items}
    tags, w = parse_tags(args.tag)
    oofs = [pickle.load(open(os.path.join(CACHE, f"oof_{t}.pkl"), "rb")) for t in tags]
    oof = {k: np.average([o[k] for o in oofs], axis=0, weights=w) for k in oofs[0]}
    cands = pickle.load(open(os.path.join(CACHE, "val_template_cands.pkl"), "rb"))
    lib, lens = load_library()
    nat = [it["key"] for it in items if not it["synthetic"]]
    base = {"thr": [0.3, 0.4, 0.5, 0.6], "med": [1, 3, 5, 7], "min_dur": [0.0, 0.06, 0.1],
            "gap": [0.0, 0.04, 0.1]}
    print("[natural] SED decode")
    s_a, cfg_a = grid_search(oof, ref, durs, {"mode": ["agnostic"], **base}, nat)
    s_p, cfg_p = grid_search(oof, ref, durs, {"mode": ["percat"], "iou": [0.3, 0.5], **base}, nat)
    # split decode: tight human_non_speech branch + merged/snapped branch for the
    # word-wrapping categories (see postproc.decode_split)
    split = {"mode": ["split"], "thr": [0.3, 0.4, 0.5], "med": [5], "min_dur": [0.1], "gap": [0.1], "iou": [0.5],
             "o_thr": [0.15, 0.2, 0.3], "o_med": [5], "o_min_dur": [0.3, 0.5, 0.8], "o_gap": [1.5, 2.5],
             "o_dil": [0.0], "o_snap": [0.3, 0.6]}
    s_s, cfg_s = grid_search(oof, ref, durs, split, nat)
    sed_nat = max((s_a, cfg_a), (s_p, cfg_p), (s_s, cfg_s), key=lambda t: t[0]["combined_score"])[1]
    # gap filling inside library-hit clips: stricter agnostic decodes, or the natural
    # decode itself (5% of natural clips also hit the library and still need their
    # other events found)
    sed_syn_grid = [{"mode": "agnostic", "thr": t, "med": 3, "min_dur": 0.1, "gap": 0.04} for t in (0.5, 0.7, 0.85)]
    sed_syn_grid.append(dict(sed_nat))
    print("[all] template + SED fusion (leave-one-clip-out library, cross-fitted null)")
    syn = [it["key"] for it in items if it["synthetic"]]
    s, cfg, nulls = tune_fusion(cands, lib, lens, oof, ref, durs, nat, syn, sed_nat, sed_syn_grid)
    for name, keys in (("natural", nat), ("synthetic", syn)):
        pred = {k: fuse(select_events(cands[k], lib, lens, cfg["tpl_thr"], exclude_src=k, null=nulls[k]),
                        oof[k], cfg, durs[k]) for k in keys}
        print(f"  {name}: " + json.dumps({k: round(v, 4) for k, v in score({k: ref[k] for k in keys}, pred).items()}))
    cfg["val_oof_score"] = s
    cfg["tags"] = args.tag
    json.dump(cfg, open(args.cfg, "w"), indent=1)
    print("wrote", args.cfg)


def test_clips():
    files = sorted(glob.glob(os.path.join(TEST, "*.wav")))
    return [(os.path.basename(f), f) for f in files]


def parse_tags(spec):
    """'ws,wm,wlA:2' -> (['ws', 'wm', 'wlA'], [1.0, 1.0, 2.0])"""
    tags, w = [], []
    for part in spec.split(","):
        t, _, x = part.partition(":")
        tags.append(t)
        w.append(float(x) if x else 1.0)
    return tags, w


def sed_test_probs(clips, spec):
    """Weighted average of per-model SED probabilities; `spec` e.g. 'ws,wm,wlA:2'
    selects models/sed_head_<tag>_*.pt, weighted exactly as `tune` weights the
    out-of-fold predictions."""
    tags, w = parse_tags(spec)
    per_tag = sed_test_probs_tags(clips, tags)
    return {k: np.average([per_tag[t][k] for t in tags], axis=0, weights=w) for k, _ in clips}


def sed_test_probs_one(clips, head_glob, tag):
    return sed_test_probs_tags(clips, [tag])[tag]


def sed_test_probs_tags(clips, tags):
    """tag -> {clip: probs}. Cached per tag; tags whose cache is stale are computed
    together, one encoder pass per backbone shared by every head on that backbone."""
    out, todo = {}, {}
    for t in tags:
        heads = sorted(glob.glob(os.path.join(ROOT, "models", f"sed_head_{t}_*.pt")))
        assert heads, f"no SED heads match models/sed_head_{t}_*.pt"
        stamp = [os.path.basename(h) for h in heads]
        p = os.path.join(CACHE, f"test_sed_probs_{t}.pkl")
        if os.path.exists(p):
            d = pickle.load(open(p, "rb"))
            if d["heads"] == stamp:
                out[t] = d["probs"]
                continue
        ck = [torch.load(h, map_location="cpu") for h in heads]
        todo.setdefault((ck[0]["backbone"], ck[0]["layers"]), []).append((t, stamp, ck))
    from features import make_extractor
    for (backbone, layers), group in todo.items():
        # compact test features saved by `pseudo.py dump --store-feats` skip the encoder
        fdir = os.path.join(CACHE, "feats_" + os.path.basename(os.path.normpath(backbone)) + "_test")
        cached = os.path.isdir(fdir) and all(os.path.exists(os.path.join(fdir, k[:-4] + ".npy")) for k, _ in clips)
        ext = None if cached else make_extractor(backbone, tuple(int(x) for x in layers.split(",")))
        dim = np.load(os.path.join(fdir, clips[0][0][:-4] + ".npy"), mmap_mode="r").shape[1] if cached else ext.dim
        heads = []
        for t, stamp, ck in group:
            ms = []
            for c in ck:
                m = Head(dim, hidden=c.get("hidden", 192)).to(os.environ.get("INDOML_DEVICE", "cuda"))
                m.load_state_dict(c["state"])
                ms.append(m)
            heads.append((t, stamp, ms, {}))
        t0 = time.time()
        for i, (k, f) in enumerate(clips):
            if cached:
                feat = np.load(os.path.join(fdir, k[:-4] + ".npy"))
            else:
                x, _ = sf.read(f, dtype="float32")
                # float16 round trip matches the training store exactly
                feat = compact(ext(x).cpu().numpy()).astype(np.float16)
            for _, _, ms, probs in heads:
                probs[k] = np.mean([predict_probs(m, [feat])[0] for m in ms], axis=0)
            if i % 500 == 0:
                print(f"  sed {backbone} {i}/{len(clips)} {time.time() - t0:.0f}s", flush=True)
        for t, stamp, _, probs in heads:
            pickle.dump({"heads": stamp, "probs": probs}, open(os.path.join(CACHE, f"test_sed_probs_{t}.pkl"), "wb"))
            out[t] = probs
        del ext
        torch.cuda.empty_cache()
    return out


def predict(args):
    cfg = json.load(open(args.cfg))
    clips = test_clips()
    print("test clips", len(clips))
    lib, lens = load_library()
    cp = os.path.join(CACHE, "test_template_cands.pkl")
    if not os.path.exists(cp):
        pickle.dump(run_matcher(lib, clips), open(cp, "wb"))
    cands = pickle.load(open(cp, "rb"))
    assert cfg.get("tags", args.tag) == args.tag, f"fusion_cfg.json was tuned for {cfg.get('tags')}, not {args.tag}"
    probs = sed_test_probs(clips, args.tag)
    # test-time null: every natural validation clip calibrates the template thresholds
    val_cands = pickle.load(open(os.path.join(CACHE, "val_template_cands.pkl"), "rb"))
    nat = [it["key"] for it in val_items(VAL) if not it["synthetic"]]
    null = null_levels(val_cands, nat, len(lib), cfg["null_q"])

    preds, n_tpl = {}, 0
    for k, f in clips:
        dur = sf.info(f).duration
        tpl = select_events(cands[k], lib, lens, cfg["tpl_thr"], null=null)
        n_tpl += bool(tpl)
        cid = k[:-4] if args.clip_id == "stem" else k
        preds[cid] = fuse(tpl, probs[k], cfg, dur)
    print(f"clips with library hits (synthetic-like): {n_tpl}/{len(clips)}; "
          f"events {sum(len(v) for v in preds.values())}; empty clips {sum(not v for v in preds.values())}")
    write_submission(preds, args.out)


def write_submission(preds, out_zip):
    """predictions.jsonl at the ZIP root, one object per clip, exactly as the Evaluation page shows."""
    jl = os.path.join(os.path.dirname(os.path.abspath(out_zip)), "predictions.jsonl")
    with open(jl, "w", encoding="utf-8", newline="\n") as f:
        for cid, evs in preds.items():
            rec = {"clip_id": cid, "events": [{"onset": float(a), "offset": float(b)} for a, b in evs]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jl, "predictions.jsonl")
    # read the archive back and re-check what the scorer will see
    with zipfile.ZipFile(out_zip) as z:
        assert z.namelist() == ["predictions.jsonl"], z.namelist()
    back = load_jsonl(jl)
    assert len(back) == len(preds) == len(test_clips()), (len(back), len(preds))
    assert all(isinstance(v, list) for v in back.values())
    print(f"wrote {out_zip} ({os.path.getsize(out_zip)} bytes), {len(back)} clips, validated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "predict"])
    ap.add_argument("--cfg", default=CFG, help="fusion settings file written by tune, read by predict")
    ap.add_argument("--tag", default="ws",
                    help="SED model tag(s), comma-separated for a backbone ensemble, e.g. ws,wm")
    ap.add_argument("--clip-id", choices=["stem", "filename"], default="stem",
                    help="clip_id written to the JSONL: file name without .wav (stem) or with it")
    ap.add_argument("--out", default=os.path.join(ROOT, "submission", "submission.zip"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    {"tune": tune, "predict": predict}[args.stage](args)


if __name__ == "__main__":
    main()
