"""Kaggle kernel: test probabilities of the nine frozen-encoder members of the final Track 1 submission.

The laptop lost its backbone weights (Whisper small / large-v3, Dasheng) and every cached test probability;
only the GRU heads (models/sed_head_<tag>_f<k>.pt) survived. This kernel downloads the public backbones,
runs each encoder once over the 5,517 test clips and every head that sits on it, and writes
cache/test_sed_probs_<tag>.pkl in exactly the format run.sed_test_probs_tags caches (so the laptop builder
accepts them as fresh). Each backbone's pickles are copied to /kaggle/working as soon as it finishes.
"""
import glob
import os
import pickle
import shutil
import sys
import time
import zipfile

import numpy as np
import soundfile as sf
import torch

WS = "/kaggle/temp/ws"
OUT = "/kaggle/working"
T0 = time.time()
TAGS = ["wsA", "aH", "bH", "dH", "wlP", "wlH", "wlH2", "wlH3", "wlHS"]      # small backbones first
HF = {"whisper-small": "openai/whisper-small", "whisper-large-v3": "openai/whisper-large-v3",
      "dasheng-base": "mispeech/dasheng-base", "ast-audioset": "MIT/ast-finetuned-audioset-10-10-0.4593"}


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 3600:.2f} h]", *a, flush=True)


def one(pattern):
    hits = glob.glob(pattern, recursive=True)
    assert hits, pattern
    return hits[0]


# ---- workspace: code, heads, test audio
os.makedirs(os.path.join(WS, "models"), exist_ok=True)
os.makedirs(os.path.join(WS, "cache"), exist_ok=True)
code = os.path.dirname(one("/kaggle/input/**/code/features.py"))
for f in glob.glob(os.path.join(code, "*.py")):
    shutil.copy2(f, WS)
shutil.copytree(os.path.dirname(one("/kaggle/input/**/beats/BEATs.py")), os.path.join(WS, "beats"), dirs_exist_ok=True)
for f in glob.glob("/kaggle/input/**/sed_head_*.pt", recursive=True):
    shutil.copy2(f, os.path.join(WS, "models"))
log("heads:", len(glob.glob(os.path.join(WS, "models", "sed_head_*.pt"))))

td = os.path.join(WS, "data", "test")
os.makedirs(td, exist_ok=True)
wavs = [d for d in glob.glob("/kaggle/input/**/audio", recursive=True)       # Kaggle extracts the zip on upload
        if len(glob.glob(os.path.join(d, "*.wav"))) > 5000]
if wavs:
    os.symlink(wavs[0], os.path.join(td, "audio"))
else:
    z = zipfile.ZipFile(one("/kaggle/input/**/t1_test.zip"))
    z.extractall(td, members=[m for m in z.namelist() if m.startswith("audio/") and m.endswith(".wav")])
clips = [(os.path.basename(f), f) for f in sorted(glob.glob(os.path.join(td, "audio", "*.wav")))]
log("test clips:", len(clips))
assert len(clips) == 5517, len(clips)

# ---- backbones: AST and BEATs shipped in the datasets, the rest from Hugging Face
shipped = glob.glob("/kaggle/input/**/ast-audioset/model.safetensors", recursive=True)
if shipped:
    shutil.copytree(os.path.dirname(shipped[0]), os.path.join(WS, "models", "ast-audioset"), dirs_exist_ok=True)
bw = one("/kaggle/input/**/BEATs_iter3_plus_AS2M.pt")
os.symlink(bw, os.path.join(WS, "models", "BEATs_iter3_plus_AS2M.pt"))


def backbone_dir(name):
    """Local directory (or file) for the backbone a head was trained on; downloads it on first use."""
    base = os.path.basename(os.path.normpath(name.replace("\\", "/")))
    local = os.path.join(WS, "models", base)
    if os.path.exists(local) and (base.endswith(".pt") or glob.glob(os.path.join(local, "*.safetensors"))):
        return local
    from huggingface_hub import snapshot_download
    log("downloading", HF[base])
    snapshot_download(HF[base], local_dir=local, allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt"])
    log("backbone ready:", base, sorted(os.listdir(local)))
    return local


os.chdir(WS)
sys.path.insert(0, WS)
from features import make_extractor  # noqa: E402
from sed import Head, compact, predict_probs  # noqa: E402

# ---- group heads by backbone (same grouping as run.sed_test_probs_tags), one encoder pass per group
groups = {}
for t in TAGS:
    heads = sorted(glob.glob(os.path.join(WS, "models", f"sed_head_{t}_*.pt")))
    assert len(heads) == 5, (t, heads)
    ck = [torch.load(h, map_location="cpu", weights_only=False) for h in heads]
    key = (os.path.basename(os.path.normpath(ck[0]["backbone"].replace("\\", "/"))), ck[0]["layers"])
    groups.setdefault(key, []).append((t, [os.path.basename(h) for h in heads], ck))
log("groups:", {k: [t for t, _, _ in v] for k, v in groups.items()})

for (base, layers), group in groups.items():
    bdir = backbone_dir(base)
    ext = make_extractor(bdir, tuple(int(x) for x in layers.split(",")))
    heads = []
    for t, stamp, ck in group:
        ms = []
        for c in ck:
            m = Head(ext.dim, hidden=c.get("hidden", 192)).to("cuda")
            m.load_state_dict(c["state"])
            m.eval()
            ms.append(m)
        heads.append((t, stamp, ms, {}))
    t0 = time.time()
    for i, (k, f) in enumerate(clips):
        x, _ = sf.read(f, dtype="float32")
        feat = compact(ext(x).cpu().numpy()).astype(np.float16)        # float16 round trip, as on the laptop
        for _, _, ms, probs in heads:
            probs[k] = np.mean([predict_probs(m, [feat])[0] for m in ms], axis=0)
        if i % 500 == 0:
            log(f"  {base} {i}/{len(clips)} {time.time() - t0:.0f}s")
    for t, stamp, _, probs in heads:
        p = os.path.join(WS, "cache", f"test_sed_probs_{t}.pkl")
        pickle.dump({"heads": stamp, "probs": probs}, open(p, "wb"))
        os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
        shutil.copy2(p, os.path.join(OUT, "cache"))
        log("saved", t, f"{os.path.getsize(p) / 1e6:.1f} MB")
    del ext, heads
    torch.cuda.empty_cache()
    if base in HF and not base.startswith("ast"):
        shutil.rmtree(bdir, ignore_errors=True)                        # keep /kaggle/temp small
log("DONE")
