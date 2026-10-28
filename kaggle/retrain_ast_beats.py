"""Kaggle kernel: re-train the Track 1 fine-tuned members whose checkpoints were lost (aFG, bFG, bFS), same recipe.

MODE "ast"   -> aFG folds 0-4 (AST fine-tune + BiGRU, 3 epochs, seed 0), GPU0 folds 0,2,4 / GPU1 folds 1,3.
MODE "beats" -> bFG folds 0-3 on GPU0; Silver extraction + bFS (full data, Silver weight 0.4) + bFG fold 4 on GPU1.

Every finished job copies its checkpoint, out-of-fold pickle and test probabilities to /kaggle/working at once,
so a session killed at the 12 h limit keeps everything that finished.
"""
import glob
import io
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile

import numpy as np
import soundfile as sf

MODE = "ast"
WS = "/kaggle/temp/ws"
OUT = "/kaggle/working"
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 3600:.2f} h]", *a, flush=True)


def one(pattern):
    hits = glob.glob(pattern, recursive=True)
    assert hits, pattern
    return hits[0]


def save(rel):
    p = os.path.join(WS, rel)
    if os.path.exists(p):
        d = os.path.join(OUT, os.path.dirname(rel))
        os.makedirs(d, exist_ok=True)
        shutil.copy2(p, d)
        log("saved", rel, f"{os.path.getsize(p) / 1e6:.1f} MB")
        return True
    log("MISSING", rel)
    return False


# ---- workspace: code, validation, Gold, test (same layout as the laptop repo)
os.makedirs(WS, exist_ok=True)
code = os.path.dirname(one("/kaggle/input/**/code/ast_ft_gru.py"))
for f in glob.glob(os.path.join(code, "*.py")):
    shutil.copy2(f, WS)
for d in ("models", "cache"):
    os.makedirs(os.path.join(WS, d), exist_ok=True)

vmeta = one("/kaggle/input/**/val/validationMetadata.json")
vdir = os.path.dirname(vmeta)
dv = os.path.join(WS, "data", "validation"); os.makedirs(dv, exist_ok=True)
shutil.copy2(vmeta, dv)
n = 0
for src, dst in (("natural", "naturalNoisyAudio"), ("synthetic", "syntheticNoiseAudio")):
    od = os.path.join(dv, dst); os.makedirs(od, exist_ok=True)
    for f in glob.glob(os.path.join(vdir, src, "*.flac")):
        o = os.path.join(od, os.path.basename(f)[:-5] + ".wav")
        if not os.path.exists(o):
            x, sr = sf.read(f, dtype="int16"); sf.write(o, x, sr, subtype="PCM_16"); n += 1
log(f"validation wavs written: {n}")

gd = os.path.join(WS, "data", "hf_gold"); os.makedirs(gd, exist_ok=True)
shutil.copy2(one("/kaggle/input/**/gold_metadata.json"), os.path.join(gd, "metadata.json"))
ga = os.path.dirname(one("/kaggle/input/**/gold_audio/*.flac"))
if not os.path.exists(os.path.join(gd, "audio")):
    os.symlink(ga, os.path.join(gd, "audio"))
log(f"gold clips: {len(os.listdir(ga))}")

td = os.path.join(WS, "data", "test"); os.makedirs(td, exist_ok=True)
wavs = [d for d in glob.glob("/kaggle/input/**/audio", recursive=True)
        if len(glob.glob(os.path.join(d, "*.wav"))) > 5000]
if wavs:
    if not os.path.exists(os.path.join(td, "audio")):
        os.symlink(wavs[0], os.path.join(td, "audio"))
else:
    z = zipfile.ZipFile(one("/kaggle/input/**/t1_test.zip"))
    z.extractall(td, members=[m for m in z.namelist() if m.startswith("audio/") and m.endswith(".wav")])
log(f"test clips: {len(glob.glob(os.path.join(td, 'audio', '*.wav')))}")

# test probabilities of one fold model, through the same function the laptop builder used
open(os.path.join(WS, "ft_test.py"), "w").write(
    "import sys\nimport build_ft_candidate as b\nimport run\n"
    "b.fold_probs(sys.argv[1], int(sys.argv[2]), run.test_clips())\n")

if MODE == "ast":
    shipped = glob.glob("/kaggle/input/**/ast-audioset/model.safetensors", recursive=True)
    if shipped:                                       # MIT/ast-finetuned-audioset-10-10-0.4593, shipped in the dataset
        shutil.copytree(os.path.dirname(shipped[0]), os.path.join(WS, "models", "ast-audioset"), dirs_exist_ok=True)
    else:
        from huggingface_hub import snapshot_download
        snapshot_download("MIT/ast-finetuned-audioset-10-10-0.4593",
                          local_dir=os.path.join(WS, "models", "ast-audioset"),
                          allow_patterns=["*.json", "*.safetensors", "*.bin"])
    log("AST backbone:", sorted(os.listdir(os.path.join(WS, "models", "ast-audioset"))))
else:
    bdir = os.path.dirname(one("/kaggle/input/**/beats/BEATs.py"))
    shutil.copytree(bdir, os.path.join(WS, "beats"), dirs_exist_ok=True)
    for f in glob.glob(os.path.join(os.path.dirname(one("/kaggle/input/**/code_beats/t1_beats_full.py")), "*.py")):
        shutil.copy2(f, WS)
    bw = one("/kaggle/input/**/BEATs_iter3_plus_AS2M.pt")
    if not os.path.exists(os.path.join(WS, "models", "BEATs_iter3_plus_AS2M.pt")):
        os.symlink(bw, os.path.join(WS, "models", "BEATs_iter3_plus_AS2M.pt"))


def extract_silver(max_silver=12000):
    """Silver (unverified_timestamps) rows from the public Kaggle copy of the HF dataset, as in the original bFS run."""
    import pyarrow as pa
    from scipy.signal import resample_poly
    sd = os.path.join(WS, "data", "hf_silver"); os.makedirs(os.path.join(sd, "audio"), exist_ok=True)
    vnames = {m["segmentFileName"].rsplit(".", 1)[0] for m in json.load(open(vmeta, encoding="utf-8"))}
    tnames = {os.path.basename(p).rsplit(".", 1)[0] for p in glob.glob(os.path.join(td, "audio", "*.wav"))}
    arrows = sorted(glob.glob("/kaggle/input/**/vaani_noise_dataset/train/*.arrow", recursive=True))
    smeta, t0 = [], time.time()

    def open_arrow(p):
        src = pa.memory_map(p, "r")
        try:
            return pa.ipc.open_stream(src)
        except Exception:
            return pa.ipc.open_file(src)

    for ai, p in enumerate(arrows):
        rd = open_arrow(p)
        batches = rd if hasattr(rd, "__iter__") and not hasattr(rd, "num_record_batches") else \
            (rd.get_batch(i) for i in range(rd.num_record_batches))
        for bi, b in enumerate(batches):
            cols = b.to_pydict()
            nr = b.num_rows
            for i in range(nr):
                q = cols.get("annotationQuality", [None] * nr)[i]
                if q != "unverified_timestamps" or len(smeta) >= max_silver:
                    continue
                au = cols["audio"][i]
                orig = os.path.basename((au.get("path") if isinstance(au, dict) else "") or "")
                stem = orig[:-4] if orig.endswith(".wav") else f"hf_{ai:02d}_{bi:04d}_{i:04d}"
                if stem in vnames or stem in tnames:
                    continue
                ts = cols.get("NoiseSubCategoryTimeStamp", [None] * nr)[i]
                if isinstance(ts, str):
                    try:
                        ts = json.loads(ts)
                    except Exception:
                        ts = []
                try:
                    x, sr0 = sf.read(io.BytesIO(au["bytes"]), dtype="float32")
                except Exception:
                    continue
                if x.ndim > 1:
                    x = x.mean(1)
                if sr0 != 16000:
                    g = math.gcd(int(sr0), 16000)
                    x = resample_poly(x, 16000 // g, int(sr0) // g).astype(np.float32)
                fn = stem + ".flac"
                sf.write(os.path.join(sd, "audio", fn), np.clip(x, -1, 1), 16000, subtype="PCM_16")
                smeta.append({"segmentFileName": fn, "NoiseSubCategoryTimeStamp": ts or [], "duration": len(x) / 16000.0,
                              "state": cols.get("state", [None] * nr)[i], "district": cols.get("district", [None] * nr)[i]})
        log(f"silver shard {ai + 1}/{len(arrows)}: {len(smeta)} clips, {time.time() - t0:.0f}s")
        if len(smeta) >= max_silver:
            break
    json.dump(smeta, open(os.path.join(sd, "metadata.json"), "w", encoding="utf-8"))
    log(f"silver clips written: {len(smeta)} ({sum(m['duration'] for m in smeta) / 3600:.1f} h)")


def sh(cmd, gpu, logname, extra_env=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES=str(gpu),
               **(extra_env or {}))
    with open(os.path.join(OUT, "logs", logname), "a") as lf:
        rc = subprocess.call(cmd, cwd=WS, env=env, stdout=lf, stderr=subprocess.STDOUT)
    log(f"gpu{gpu} {' '.join(cmd[2:])} -> exit {rc}")
    return rc


def fold_job(gpu, tag, fold):
    script, prefix = ("ast_ft_gru.py", "ast_ftgru") if tag == "aFG" else ("beats_ft_gru.py", "beats_ftgru")
    rc = sh([sys.executable, "-u", script, "--tag", tag, "--fold", str(fold), "--epochs", "3", "--seed", "0",
             "--no-eval"], gpu, f"train_{tag}_f{fold}.log")
    save(f"models/{prefix}_{tag}_f{fold}.pt")
    save(f"cache/oof_{tag}_fold{fold}.pkl")
    if rc == 0:
        sh([sys.executable, "-u", "ft_test.py", tag, str(fold)], gpu, f"test_{tag}_f{fold}.log")
        save(f"cache/test_probs_{tag}_f{fold}.pkl")


def bfs_job(gpu):
    extract_silver()
    sh([sys.executable, "-u", "t1_beats_full.py"], gpu, "train_bFS.log",
       {"TAG": "bFS", "EPOCHS": "2", "SILVER_W": "0.4", "MAX_HOURS": "4.0"})
    save("models/beats_ftgru_bFS_full.pt")
    save("cache/test_probs_bFS_full.pkl")
    shutil.rmtree(os.path.join(WS, "data", "hf_silver"), ignore_errors=True)


def lane(gpu, jobs):
    for j in jobs:
        try:
            j(gpu)
        except Exception as e:  # one failed job must not stop the rest of the lane
            log(f"gpu{gpu} job failed: {e!r}")


os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
if MODE == "ast":
    lanes = {0: [lambda g, k=k: fold_job(g, "aFG", k) for k in (0, 2, 4)],
             1: [lambda g, k=k: fold_job(g, "aFG", k) for k in (1, 3)]}
else:
    lanes = {0: [lambda g, k=k: fold_job(g, "bFG", k) for k in (0, 1, 2, 3)],
             1: [bfs_job, lambda g: fold_job(g, "bFG", 4)]}
threads = [threading.Thread(target=lane, args=(g, js)) for g, js in lanes.items()]
for t in threads:
    t.start()
for t in threads:
    t.join()
log("ALL DONE", sorted(os.path.relpath(p, OUT) for p in glob.glob(os.path.join(OUT, "**", "*.*"), recursive=True)))
