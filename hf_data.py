"""Pull the verified-timestamp (Gold) clips out of the Hugging Face training set.

ARTPARK-IISc/Vaani-Noise-Event-Dataset is 182 parquet shards / 17.7 GB across all three
annotation tiers, and it is gated: accept the conditions on the dataset page and run
`hf auth login` first. Only the Gold tier matches the test labels' quality, so this
reads the small `annotationQuality` column of each shard first and then fetches just the
row groups that hold Gold rows (HTTP range reads through HfFileSystem), instead of
downloading every shard whole.

Output mirrors the validation release: data/hf_gold/audio/*.wav (16 kHz mono PCM16) and
data/hf_gold/metadata.json with the same keys as validationMetadata.json.
"""
import argparse
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem
from scipy.signal import resample_poly

REPO = "datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="verified_timestamps")
    ap.add_argument("--max-shards", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, "audio"), exist_ok=True)
    meta_path = os.path.join(OUT, "metadata.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else []
    done_shards = {m["_shard"] for m in meta}

    fs = HfFileSystem()
    shards = sorted(fs.glob(f"{REPO}/data/train-*.parquet"))
    if args.max_shards:
        shards = shards[: args.max_shards]
    for si, path in enumerate(shards):
        name = os.path.basename(path)
        if name in done_shards:
            continue
        with fs.open(path, "rb", block_size=1 << 20) as f:
            pf = pq.ParquetFile(f)
            q = pf.read(columns=["annotationQuality"]).column(0).to_pylist()
            rows = [i for i, v in enumerate(q) if v == args.quality]
            if rows:
                # row groups holding at least one wanted row
                bounds = np.cumsum([0] + [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)])
                groups = sorted({int(np.searchsorted(bounds, r, side="right") - 1) for r in rows})
                for g in groups:
                    tab = pf.read_row_group(g, columns=["audio"] + KEEP).to_pylist()
                    for j, rec in enumerate(tab):
                        if rec["annotationQuality"] != args.quality:
                            continue
                        fn = f"hf_{name[:-8]}_{bounds[g] + j:06d}.wav"
                        x = to_16k_mono(rec["audio"]["bytes"])
                        sf.write(os.path.join(OUT, "audio", fn), x, 16000, subtype="PCM_16")
                        m = {k: rec[k] for k in KEEP}
                        m.update({"segmentFileName": fn, "syntheticData": False, "_shard": name,
                                  "duration": len(x) / 16000})
                        meta.append(m)
        # a shard with no wanted rows is still recorded so a restart skips it
        if not rows:
            meta.append({"_shard": name, "segmentFileName": None})
        json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)
        n = sum(1 for m in meta if m.get("segmentFileName"))
        print(f"shard {si + 1}/{len(shards)} {name}: {len(rows)} wanted rows, total clips {n}", flush=True)


if __name__ == "__main__":
    main()
