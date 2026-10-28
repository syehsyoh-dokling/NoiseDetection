"""Build ../t1_retrain_data: everything the retrain kernels read, as one private Kaggle dataset (Siti account).

  val/{natural,synthetic}/*.flac + val/validationMetadata.json   (from the Track 2 Kaggle data folder)
  gold_audio/*.flac + gold_metadata.json                          (Gold tier)
  code/*.py, code_beats/*.py, beats/*.py                          (Track 1 code, BEATs model code)
  BEATs_iter3_plus_AS2M.pt                                        (downloaded beforehand)
  t1_test.zip                                                     (Track 1 test audio)
Files are hard-linked when possible (same NTFS volume), copied otherwise.
"""
import glob
import json
import os
import shutil

T1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.dirname(T1)
SRC = os.path.join(DOCS, "t2work", "kaggle_t2_data")
OUT = os.path.join(DOCS, "t1_retrain_data")


def put(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    n = 0
    for sub in ("natural", "synthetic"):
        for f in glob.glob(os.path.join(SRC, "val", sub, "*.flac")):
            put(f, os.path.join(OUT, "val", sub, os.path.basename(f))); n += 1
    put(os.path.join(SRC, "val", "validationMetadata.json"), os.path.join(OUT, "val", "validationMetadata.json"))
    g = 0
    for f in glob.glob(os.path.join(SRC, "gold_audio", "*.flac")):
        put(f, os.path.join(OUT, "gold_audio", os.path.basename(f))); g += 1
    put(os.path.join(SRC, "gold_metadata.json"), os.path.join(OUT, "gold_metadata.json"))
    for f in glob.glob(os.path.join(T1, "*.py")):
        put(f, os.path.join(OUT, "code", os.path.basename(f)))
    for f in glob.glob(os.path.join(T1, "kaggle_t1", "beats_data", "code_beats", "*.py")):
        put(f, os.path.join(OUT, "code_beats", os.path.basename(f)))
    for f in glob.glob(os.path.join(T1, "kaggle_t1", "beats_data", "beats", "*.py")):
        put(f, os.path.join(OUT, "beats", os.path.basename(f)))
    tz = os.path.join(OUT, "t1_test.zip")
    if not os.path.exists(tz):                        # raw/t1_test.zip is gone; rebuild it from the extracted audio
        import zipfile
        wavs = sorted(glob.glob(os.path.join(T1, "data", "test", "audio", "*.wav")))
        assert len(wavs) > 5000, len(wavs)
        with zipfile.ZipFile(tz + ".tmp", "w", zipfile.ZIP_STORED) as z:
            for w in wavs:
                z.write(w, "audio/" + os.path.basename(w))
        os.replace(tz + ".tmp", tz)
        print(f"test zip: {len(wavs)} clips")
    assert os.path.exists(os.path.join(OUT, "BEATs_iter3_plus_AS2M.pt")), "download the BEATs weights first"
    json.dump({"title": "indoml-t1-retrain", "id": "sitifitriah/indoml-t1-retrain", "licenses": [{"name": "other"}],
               "description": "Private working copy for Datathon@IndoML 2026 Track 1 retraining (not for redistribution)."},
              open(os.path.join(OUT, "dataset-metadata.json"), "w"), indent=1)
    tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) for f in fs)
    print(f"{OUT}: {n} validation + {g} Gold clips, {tot / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
