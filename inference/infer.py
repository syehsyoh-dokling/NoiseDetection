"""End-to-end Track 1 inference: test WAVs -> predictions.jsonl in a zip (the Phase 2 submission format).

Reproduces submission 994586 (configuration models/fusion_Q1_bFSadd4.json) from the released artefacts:
  models/sed_head_<tag>_f<k>.pt          9 frozen members x 5 folds (in this repository)
  models/ast_ftgru_aFG_f<k>.pt           fine-tuned AST, 5 folds        (release v1.0 assets)
  models/beats_ftgru_bFG_f<k>.pt         fine-tuned BEATs, 5 folds      (release v1.0 assets)
  models/beats_ftgru_bFS_full.pt         fine-tuned BEATs, full data    (release v1.0 assets)
  models/BEATs_iter3_plus_AS2M.pt        BEATs backbone (microsoft/unilm, see README)
  cache/template_library.pkl, cache/val_template_cands.pkl   built by `python eval_templates.py`
Hugging Face backbones are downloaded on first use at the revisions listed in HF_MODELS.

The frozen-member step is the code that ran on Kaggle for the released reproduction
(kaggle/frozen_test_probs.py): backbones are resolved by their folder name, so the heads load on any machine.

    python inference/infer.py --test-dir /path/to/test_wavs --out work/submission_track1.zip
"""
import argparse
import glob
import os
import pickle
import shutil
import sys
import time

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

CONFIG = "Q1_bFSadd4"
FROZEN = ["wsA", "aH", "bH", "dH", "wlP", "wlH", "wlH2", "wlH3", "wlHS"]
HF_MODELS = {  # folder name under models/ -> (Hugging Face repo, revision used for the released reproduction)
    "whisper-small": ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d"),
    "whisper-large-v3": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"),
    "ast-audioset": ("MIT/ast-finetuned-audioset-10-10-0.4593", "f826b80d28226b62986cc218e5cec390b1096902"),
    "dasheng-base": ("mispeech/dasheng-base", "e0b0a27818689e59010bdd7739475e33090dff11"),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def backbone_dir(name):
    """models/<basename> for the backbone a head was trained on; Hugging Face models are fetched on first use."""
    base = os.path.basename(os.path.normpath(name.replace("\\", "/")))
    local = os.path.join(ROOT, "models", base)
    if base.endswith(".pt"):
        assert os.path.exists(local), f"place {base} in models/ (see README)"
        return local
    if not glob.glob(os.path.join(local, "*.safetensors")):
        from huggingface_hub import snapshot_download
        repo, rev = HF_MODELS[base]
        log("downloading", repo, rev[:7])
        snapshot_download(repo, revision=rev, local_dir=local, allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt"])
    return local


def frozen_members(clips):
    """cache/test_sed_probs_<tag>.pkl for the nine frozen members: one encoder pass per backbone, mean of 5 heads."""
    from features import make_extractor
    from sed import Head, compact, predict_probs
    groups = {}
    for t in FROZEN:
        out = os.path.join(ROOT, "cache", f"test_sed_probs_{t}.pkl")
        heads = sorted(glob.glob(os.path.join(ROOT, "models", f"sed_head_{t}_f*.pt")))
        assert len(heads) == 5, (t, heads)
        stamp = [os.path.basename(h) for h in heads]
        if os.path.exists(out) and pickle.load(open(out, "rb"))["heads"] == stamp:
            continue
        ck = [torch.load(h, map_location="cpu", weights_only=False) for h in heads]
        key = (os.path.basename(os.path.normpath(ck[0]["backbone"].replace("\\", "/"))), ck[0]["layers"])
        groups.setdefault(key, []).append((t, stamp, ck))
    for (base, layers), group in groups.items():
        ext = make_extractor(backbone_dir(base), tuple(int(x) for x in layers.split(",")))
        heads = []
        for t, stamp, ck in group:
            ms = []
            for c in ck:
                m = Head(ext.dim, hidden=c.get("hidden", 192)).to("cuda")
                m.load_state_dict(c["state"])
                m.eval()
                ms.append(m)
            heads.append((t, stamp, ms, {}))
        log(f"{base}: {[t for t, _, _, _ in heads]} on {len(clips)} clips")
        for k, f in clips:
            x, _ = sf.read(f, dtype="float32")
            feat = compact(ext(x).cpu().numpy()).astype(np.float16)
            for _, _, ms, probs in heads:
                probs[k] = np.mean([predict_probs(m, [feat])[0] for m in ms], axis=0)
        for t, stamp, _, probs in heads:
            pickle.dump({"heads": stamp, "probs": probs},
                        open(os.path.join(ROOT, "cache", f"test_sed_probs_{t}.pkl"), "wb"))
        del ext, heads
        torch.cuda.empty_cache()


def bfs_member(clips):
    """cache/test_sed_probs_bFS.pkl from the full-data BEATs fine-tune, cropped/padded to aH's frame count."""
    out = os.path.join(ROOT, "cache", "test_sed_probs_bFS.pkl")
    if os.path.exists(out):
        return
    import beats_ft_gru as B
    model = B.BEATsGru(backbone_dir("BEATs_iter3_plus_AS2M.pt")).to("cuda")
    ck = torch.load(os.path.join(ROOT, "models", "beats_ftgru_bFS_full.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state"])
    model.eval()
    ref = pickle.load(open(os.path.join(ROOT, "cache", "test_sed_probs_aH.pkl"), "rb"))["probs"]
    probs = {}
    for k, f in clips:
        p = B.predict_clip(model, f, "cuda")
        T = ref[k].shape[0]
        probs[k] = p[:T] if len(p) >= T else np.concatenate([p, np.repeat(p[-1:], T - len(p), 0)], 0)
    pickle.dump({"heads": [], "probs": probs}, open(out, "wb"))
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True, help="folder with the test WAVs (16 kHz mono; searched recursively)")
    ap.add_argument("--out", default=os.path.join(ROOT, "work", "submission_track1.zip"))
    args = ap.parse_args()
    import run
    os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
    for f in ("template_library.pkl", "val_template_cands.pkl"):
        assert os.path.exists(os.path.join(ROOT, "cache", f)), f"cache/{f} missing: run `python eval_templates.py`"
    os.makedirs(run.TEST, exist_ok=True)                      # run.py reads data/test/audio
    wavs = sorted(glob.glob(os.path.join(args.test_dir, "**", "*.wav"), recursive=True))
    assert wavs, args.test_dir
    for w in wavs:
        dst = os.path.join(run.TEST, os.path.basename(w))
        if not os.path.exists(dst):
            shutil.copy2(w, dst)
    clips = run.test_clips()
    log(f"test clips: {len(clips)}")

    backbone_dir("ast-audioset")                              # fine-tuned AST loads its config from here
    frozen_members(clips)
    bfs_member(clips)

    def cached(cl, tags):                                     # every member's probabilities are cached by now
        return {t: pickle.load(open(os.path.join(ROOT, "cache", f"test_sed_probs_{t}.pkl"), "rb"))["probs"]
                for t in tags}
    run.sed_test_probs_tags = cached
    import build_ft_candidate as b                            # aFG / bFG: mean of the 5 fold models
    sys.argv = ["build_ft_candidate.py", CONFIG]
    b.main()
    built = os.path.join(ROOT, "submission", f"submission_v20_{CONFIG}.zip")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    shutil.copy2(built, args.out)
    log("wrote", args.out)


if __name__ == "__main__":
    main()
