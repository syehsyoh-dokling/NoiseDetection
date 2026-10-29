"""Pack the cached model outputs (no audio) into two release assets of the public Track 1 repository.

  cache_oof_validation.zip   cache/oof_<tag>.pkl (out-of-fold validation probabilities) + val_template_cands.pkl
  cache_test_outputs.zip     cache/test_sed_probs_<tag>.pkl, test_probs_<tag>_f<k>.pkl, test_template_cands.pkl,
                             t1_syn_keys.json, test_durs.json
template_library.pkl is not packed: it holds waveforms cut from the validation audio, which may not be redistributed.
"""
import glob
import os
import zipfile

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(R, "cache")
OUT = os.path.join(R, "release_extra")
os.makedirs(OUT, exist_ok=True)
sets = {
    "cache_oof_validation.zip": sorted(glob.glob(os.path.join(C, "oof_*.pkl"))) + [os.path.join(C, "val_template_cands.pkl")],
    "cache_test_outputs.zip": sorted(glob.glob(os.path.join(C, "test_sed_probs_*.pkl")))
    + sorted(glob.glob(os.path.join(C, "test_probs_*_f*.pkl")))
    + [os.path.join(C, f) for f in ("test_template_cands.pkl", "t1_syn_keys.json", "test_durs.json")],
}
for name, files in sets.items():
    p = os.path.join(OUT, name)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, os.path.join("cache", os.path.basename(f)))
    print(name, len(files), "files", f"{os.path.getsize(p) / 1e6:.0f} MB", flush=True)
