"""Leave-one-clip-out evaluation of the matched-filter detector on validation.

Writes cache/val_template_cands.pkl (raw candidates for every validation clip) so the
threshold sweep and the later fusion step need no GPU re-run.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import soundfile as sf

from scorer import score
from template_match import Matcher, build_library, select_events

ROOT = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.join(ROOT, "data", "validation")
CACHE = os.path.join(ROOT, "cache")


def val_audio_path(m):
    sub = "syntheticNoiseAudio" if m["syntheticData"] else "naturalNoisyAudio"
    return os.path.join(VAL, sub, m["segmentFileName"])


def run_matcher(lib, clips):
    """clips: list of (clip_key, wav_path). Returns clip_key -> candidates."""
    mt = Matcher(lib)
    sized = sorted(((mt.nfft_for(sf.info(p).frames), k, p) for k, p in clips))
    cands, t0, cur = {}, time.time(), None
    for i, (nfft, k, p) in enumerate(sized):
        if nfft != cur:
            mt.clear()
            cur = nfft
        x, _ = sf.read(p, dtype="float32")
        cands[k] = mt.match(x)
        if i % 500 == 0:
            print(f"  matcher {i}/{len(sized)} nfft={nfft} {time.time() - t0:.0f}s", flush=True)
    return cands


def main():
    os.makedirs(CACHE, exist_ok=True)
    lib_path = os.path.join(CACHE, "template_library.pkl")
    if os.path.exists(lib_path):
        lib = pickle.load(open(lib_path, "rb"))
    else:
        lib = build_library(VAL)
        pickle.dump(lib, open(lib_path, "wb"))
    print("library templates", len(lib), "total seconds", sum(len(t["wave"]) for t in lib) / 16000)

    meta = json.load(open(os.path.join(VAL, "validationMetadata.json"), encoding="utf-8"))
    cand_path = os.path.join(CACHE, "val_template_cands.pkl")
    if os.path.exists(cand_path):
        cands = pickle.load(open(cand_path, "rb"))
    else:
        cands = run_matcher(lib, [(m["segmentFileName"], val_audio_path(m)) for m in meta])
        pickle.dump(cands, open(cand_path, "wb"))

    lens = np.array([len(t["wave"]) for t in lib])
    ref = {m["segmentFileName"]: [(float(e["start"]), float(e["end"])) for e in m["NoiseSubCategoryTimeStamp"]]
           for m in meta}
    syn = {m["segmentFileName"] for m in meta if m["syntheticData"]}
    nat = set(ref) - syn
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        pred = {k: [(a, b) for a, b, _, _ in select_events(cands[k], lib, lens, thr, exclude_src=k)] for k in ref}
        s_syn = score({k: ref[k] for k in syn}, {k: pred[k] for k in syn})
        nat_clips_hit = sum(1 for k in nat if pred[k])
        nat_events = sum(len(pred[k]) for k in nat)
        print(f"thr={thr:.1f} synthetic-only {json.dumps({k: round(v, 4) for k, v in s_syn.items()})} "
              f"| natural clips with any hit {nat_clips_hit}/{len(nat)} ({nat_events} events)")


if __name__ == "__main__":
    sys.exit(main())
