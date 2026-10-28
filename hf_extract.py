"""Extract the Gold (verified_timestamps) rows from Hugging Face parquet shards that were
downloaded by the browser, then delete each processed shard to keep disk use small.

The shards come from ARTPARK-IISc/Vaani-Noise-Event-Dataset (gated; accepted by the
account owner). Gold rows are spread evenly (~12% of every shard), so every shard is
needed, but only ~2.5 GB of the 17.7 GB is kept.

Leak guard: a clip whose original segment name matches any validation or test clip is
skipped, so adding this data can never contaminate validation scores or see test audio.

    python hf_extract.py --watch C:/Users/<you>/Downloads --expect 182
"""
import argparse
import glob
import io
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "hf_gold")
KEEP = ["state", "district", "duration", "language", "annotationQuality", "transcript",
        "NoiseCategory", "NoiseSubCategoryTimeStamp", "imageFileName"]


def to_16k_mono(raw):
    x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        g = np.gcd(int(sr), 16000)
        x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    return x


def forbidden_names():
    names = set()
    meta = json.load(open(os.path.join(ROOT, "data", "validation", "validationMetadata.json"), encoding="utf-8"))
    names |= {m["segmentFileName"] for m in meta}
    names |= {os.path.basename(f) for f in glob.glob(os.path.join(ROOT, "data", "test", "audio", "*.wav"))}
    return names


def extract(path, meta, forbidden, quality="verified_timestamps"):
    """Rows are committed to `meta` only after the whole shard succeeded, so a retry
    (e.g. after a disk-full error) can never double-count."""
    name = os.path.basename(path)
    pf = pq.ParquetFile(path)
    rows, leaked = [], 0
    for g in range(pf.num_row_groups):
        for rec in pf.read_row_group(g, columns=["audio"] + KEEP).to_pylist():
            if rec["annotationQuality"] != quality:
                continue
            orig = os.path.basename(rec["audio"].get("path") or "")
            if orig in forbidden:
                leaked += 1
                continue
            stem = orig[:-4] if orig.endswith(".wav") else f"hf_{name[:-8]}_{len(rows):04d}"
            fn = stem + ".flac"  # lossless, about half the size of PCM16 WAV
            x = to_16k_mono(rec["audio"]["bytes"])
            sf.write(os.path.join(OUT, "audio", fn), x, 16000, subtype="PCM_16")
            m = {k: rec[k] for k in KEEP}
            m.update({"segmentFileName": fn, "syntheticData": False, "_shard": name, "duration": len(x) / 16000})
            rows.append(m)
    meta.extend(rows)
    return len(rows), leaked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True, help="folder the browser downloads shards into")
    ap.add_argument("--expect", type=int, default=182)
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--quality", default="verified_timestamps",
                    help="annotation tier to keep: verified_timestamps (Gold) or unverified_timestamps (Silver)")
    ap.add_argument("--out", default="hf_gold", help="folder under data/ to write into")
    ap.add_argument("--max-hours", type=float, default=0.0, help="stop once this many hours are kept (0 = no limit)")
    args = ap.parse_args()
    global OUT
    OUT = os.path.join(ROOT, "data", args.out)
    os.makedirs(os.path.join(OUT, "audio"), exist_ok=True)
    meta_path = os.path.join(OUT, "metadata.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else []
    done = {m["_shard"] for m in meta}
    done_path = os.path.join(OUT, "shards_done.json")
    done |= set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    forbidden = forbidden_names()
    while len(done) < args.expect:
        for path in sorted(glob.glob(os.path.join(args.watch, "train-*-of-00182*.parquet"))):
            name = os.path.basename(path)
            canon = name.split(" ")[0] if " " in name else name  # "x.parquet" or "x (1).parquet"
            if canon in done:
                if not args.keep_shards:
                    os.remove(path)
                continue
            try:
                kept, leaked = extract(path, meta, forbidden, quality=args.quality)
            except Exception as e:  # file still being written, retry next pass
                print(f"skip {name} for now: {type(e).__name__}", flush=True)
                continue
            done.add(canon)
            json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)
            json.dump(sorted(done), open(done_path, "w"))
            if not args.keep_shards:
                os.remove(path)
            hours = sum(m.get("duration", 0) for m in meta) / 3600
            print(f"{canon}: kept {kept} clips, skipped {leaked} overlapping val/test; "
                  f"shards {len(done)}/{args.expect}, total {len(meta)} clips / {hours:.1f} h", flush=True)
            if args.max_hours and hours >= args.max_hours:
                print(f"reached {args.max_hours} h, stopping", flush=True)
                return
        time.sleep(5)
    print("all shards processed", flush=True)


if __name__ == "__main__":
    main()
