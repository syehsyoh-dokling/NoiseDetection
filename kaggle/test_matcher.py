"""Rebuild cache/test_template_cands.pkl (library matched-filter candidates on the 5,517 test clips), lost with the
other caches; run.predict needs it. CPU only, resumable: partial results are checkpointed every 250 clips in
cache/test_template_cands.partial.pkl and finished clips are skipped on restart (the first attempt segfaulted at
4,500/5,517 when the laptop ran out of RAM: the matcher's FFT cache at nfft 524288 is several GB).

    python kaggle_t1/test_matcher.py
"""
import os
import pickle
import sys
import time

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import run  # noqa: E402
from eval_templates import Matcher  # noqa: E402

out = os.path.join(run.CACHE, "test_template_cands.pkl")
part = os.path.join(run.CACHE, "test_template_cands.partial.pkl")
assert not os.path.exists(out), out
lib, lens = run.load_library()
clips = run.test_clips()
cands = pickle.load(open(part, "rb")) if os.path.exists(part) else {}
print(f"library {len(lib)} templates; test clips {len(clips)}; already done {len(cands)}", flush=True)

mt = Matcher(lib)
sized = sorted(((mt.nfft_for(sf.info(p).frames), k, p) for k, p in clips))   # same order as eval_templates.run_matcher
t0, cur = time.time(), None
for i, (nfft, k, p) in enumerate(sized):
    if k in cands:
        continue
    if nfft != cur:
        mt.clear()
        cur = nfft
    x, _ = sf.read(p, dtype="float32")
    cands[k] = mt.match(x)
    if i % 250 == 0:
        pickle.dump(cands, open(part, "wb"))
        print(f"  matcher {i}/{len(sized)} nfft={nfft} {time.time() - t0:.0f}s", flush=True)
assert len(cands) == len(clips), (len(cands), len(clips))
pickle.dump(cands, open(out, "wb"))
os.remove(part)
print(f"wrote {out}: {len(cands)} clips, {sum(bool(v) for v in cands.values())} with candidates, "
      f"{(time.time() - t0) / 60:.1f} min", flush=True)
