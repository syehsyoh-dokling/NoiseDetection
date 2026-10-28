"""build_ft_candidate.py without the test audio on disk (the Drive copy streams at ~20 files/min).

The builder only reads the audio for the clip list and each clip's duration. The clip list comes from the cached
test probabilities, and a Vaani file name ends in _<startms>_<endms>.wav, so the duration is (end - start) / 1000.
Check first that this reproduces the real durations and the original F4_calib_plus zip:

    python build_nodata.py --check
    python build_nodata.py F4_gapiou
"""
import glob
import os
import pickle
import re
import sys
import zipfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import run  # noqa: E402

KEYS = sorted(pickle.load(open(os.path.join(ROOT, "cache", "test_sed_probs_aH.pkl"), "rb"))["probs"])
PAT = re.compile(r"_(\d+)_(\d+)\.wav$")


def name_dur(path):
    m = PAT.search(os.path.basename(path))
    return (int(m.group(2)) - int(m.group(1))) / 1000.0


def patch():
    run.test_clips = lambda: [(k, os.path.join(run.TEST, k)) for k in KEYS]
    import json
    durs = json.load(open(os.path.join(ROOT, "cache", "test_durs.json")))   # exact, read from the test-data zip
    assert len(durs) == len(KEYS), (len(durs), len(KEYS))
    run.sf.info = lambda f: SimpleNamespace(duration=durs[os.path.basename(f)])

    def cached_tags(clips, tags):
        # the restored workspace has the cached test probabilities but not the sed_head_*.pt files they were made from
        out = {}
        for t in tags:
            d = pickle.load(open(os.path.join(ROOT, "cache", f"test_sed_probs_{t}.pkl"), "rb"))
            out[t] = d["probs"]
        return out
    run.sed_test_probs_tags = cached_tags


def check():
    import soundfile as sf
    have = sorted(glob.glob(os.path.join(run.TEST, "*.wav")))
    diffs = [abs(sf.info(f).duration - name_dur(f)) for f in have]
    print(f"{len(have)} local test wavs: max |real - name| duration {max(diffs) * 1000:.2f} ms, "
          f"mean {sum(diffs) / len(diffs) * 1000:.2f} ms", flush=True)
    print(f"test keys {len(KEYS)}; all parse: {all(PAT.search(k) for k in KEYS)}", flush=True)


def build(name):
    patch()
    import build_ft_candidate as b
    sys.argv = ["build_ft_candidate.py", name]
    b.main()


def compare(a, b):
    za, zb = zipfile.ZipFile(a), zipfile.ZipFile(b)
    na, nb = sorted(za.namelist()), sorted(zb.namelist())
    same = na == nb and all(za.read(n) == zb.read(n) for n in na)
    print(f"compare {os.path.basename(a)} vs {os.path.basename(b)}: names equal {na == nb}, bytes equal {same}", flush=True)
    if not same and na == nb:
        for n in na:
            la, lb = za.read(n).decode("utf-8").splitlines(), zb.read(n).decode("utf-8").splitlines()
            diff = sum(x != y for x, y in zip(la, lb)) + abs(len(la) - len(lb))
            print(f"  {n}: {len(la)} vs {len(lb)} lines, {diff} differ", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check()
        build("F4_calib_plus")
        compare(os.path.join(ROOT, "submission", "submission_v20_F4_calib_plus.zip"),
                os.path.join(ROOT, "orig_submission", "submission_v20_F4_calib_plus.zip"))
    else:
        build(sys.argv[1])
