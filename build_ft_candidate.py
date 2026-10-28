"""Build a submission whose members include the fine-tuned models (aFG, wFG) from any fusion config.

Frozen members come from run.sed_test_probs_tags as usual. Fine-tuned members have no sed_head files,
so their test probabilities are the mean of the five fold models, cached per fold in
cache/test_probs_<tag>_f<k>.pkl (aFG's were written by build_aFG_candidate.py).

    python build_ft_candidate.py W8        # models/fusion_W8.json -> submission/submission_v20_W8.zip
"""
import json
import os
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

import run

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")


def fold_probs(tag, fold, clips):
    path = os.path.join(CACHE, f"test_probs_{tag}_f{fold}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    if tag.startswith("aFG"):
        import ast_ft_gru as m
        from transformers import ASTFeatureExtractor
        backbone = os.path.join(ROOT, "models", "ast-audioset")
        fe = ASTFeatureExtractor.from_pretrained(backbone)
        model = m.ASTGru(backbone).to("cuda")
        ck = torch.load(os.path.join(ROOT, "models", f"ast_ftgru_{tag}_f{fold}.pt"), map_location="cpu")
        model.load_state_dict(ck["state"])
    elif tag.startswith(("bFG", "dFG")):
        if tag.startswith("bFG"):
            import beats_ft_gru as m
            model = m.BEATsGru(os.path.join(ROOT, "models", "BEATs_iter3_plus_AS2M.pt")).to("cuda")
            ck = torch.load(os.path.join(ROOT, "models", f"beats_ftgru_{tag}_f{fold}.pt"), map_location="cpu")
        else:
            import dasheng_ft_gru as m
            model = m.DashengGru(os.path.join(ROOT, "models", "dasheng-base")).to("cuda")
            ck = torch.load(os.path.join(ROOT, "models", f"dasheng_ftgru_{tag}_f{fold}.pt"), map_location="cpu")
        model.load_state_dict(ck["state"])
        model.eval()
        probs, t0 = {}, time.time()
        for i, (k, f) in enumerate(clips):
            probs[k] = m.predict_clip(model, f, "cuda")
            if i % 1000 == 0:
                print(f"  {tag} fold {fold} test {i}/{len(clips)} {time.time() - t0:.0f}s", flush=True)
        pickle.dump(probs, open(path, "wb"))
        del model
        torch.cuda.empty_cache()
        return probs
    elif tag == "wFG":
        import whisper_ft_gru as m
        from transformers import WhisperFeatureExtractor
        backbone = os.path.join(ROOT, "models", "whisper-large-v3")
        fe = WhisperFeatureExtractor.from_pretrained(backbone)
        ck = torch.load(os.path.join(ROOT, "models", f"whisper_ftgru_wFG_f{fold}.pt"), map_location="cpu")
        model = m.WhisperGru(backbone, n_train=ck["n_train"]).to("cuda")
        missing, unexpected = model.load_state_dict(ck["state"], strict=False)
        assert not unexpected, unexpected[:3]
        assert all(not k.startswith("head.") for k in missing), "head weights missing"
    else:
        raise SystemExit(f"unknown fine-tuned member {tag}")
    model.eval()
    probs, t0 = {}, time.time()
    for i, (k, f) in enumerate(clips):
        probs[k] = m.predict_clip(fe, model, f, "cuda")
        if i % 1000 == 0:
            print(f"  {tag} fold {fold} test {i}/{len(clips)} {time.time() - t0:.0f}s", flush=True)
    pickle.dump(probs, open(path, "wb"))
    del model
    torch.cuda.empty_cache()
    return probs


def main():
    name = sys.argv[1]
    cfg_path = os.path.join(ROOT, "models", f"fusion_{name}.json")
    tags_spec = json.load(open(cfg_path))["tags"]
    tags, _ = run.parse_tags(tags_spec)
    fine = [t for t in tags if t.startswith(("aFG", "bFG", "dFG")) or t == "wFG"]
    clips = run.test_clips()
    ref = pickle.load(open(os.path.join(CACHE, "test_sed_probs_aH.pkl"), "rb"))["probs"]
    extra = {}
    for t in fine:
        folds = [fold_probs(t, f, clips) for f in range(5)]
        extra[t] = {k: np.mean([p[k] for p in folds], axis=0) for k, _ in clips}
        bad = [k for k, _ in clips if extra[t][k].shape != ref[k].shape]
        assert not bad, f"{t}: {len(bad)} clips differ in shape from aH, e.g. {bad[0]}"

    orig = run.sed_test_probs_tags

    def with_fine(cl, tg):
        out = orig(cl, [t for t in tg if t not in extra])
        out.update({t: extra[t] for t in tg if t in extra})
        return out

    run.sed_test_probs_tags = with_fine
    run.predict(SimpleNamespace(cfg=cfg_path, tag=tags_spec, clip_id="stem",
                                out=os.path.join(ROOT, "submission", f"submission_v20_{name}.zip")))


if __name__ == "__main__":
    main()
