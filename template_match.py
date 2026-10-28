"""Matched-filter detector for the synthetic subset.

The synthetic clips are clean speech with noise samples pasted in from a finite library,
and the validation release ships both the noisy mix and the clean reference. So
residual = noisy - clean recovers every pasted sample exactly, and in validation ~94% of
pasted events reuse a waveform that also appears in another clip. Sliding each library
waveform over a test clip with normalised cross-correlation (NCC) finds the pasted
sample and its exact position.
"""
import json
import os

import numpy as np
import soundfile as sf
import torch

SR = 16000


def build_library(val_root, min_len_s=0.05, dedupe_ncc=0.995):
    """Cut every labelled synthetic event out of (noisy - clean); drop exact duplicates.

    Returns a list of dicts: wave (float32), src (clip file name), tag, category.
    """
    meta = json.load(open(os.path.join(val_root, "validationMetadata.json"), encoding="utf-8"))
    items = []
    for m in meta:
        if not m.get("syntheticData"):
            continue
        fn = m["segmentFileName"]
        x, _ = sf.read(os.path.join(val_root, "syntheticNoiseAudio", fn), dtype="float32")
        c, _ = sf.read(os.path.join(val_root, "syntheticCleanRefAudio", fn), dtype="float32")
        n = min(len(x), len(c))
        r = x[:n] - c[:n]
        for e in m["NoiseSubCategoryTimeStamp"]:
            a, b = int(round(float(e["start"]) * SR)), int(round(float(e["end"]) * SR))
            if b - a < min_len_s * SR:
                continue
            items.append({"wave": r[a:b].copy(), "src": fn, "tag": e["tag"], "category": e["category"]})
    # Exact-duplicate removal keeps the matcher fast; keep the list of every source clip
    # a waveform came from so leave-one-clip-out evaluation stays honest.
    items.sort(key=lambda d: (d["tag"], len(d["wave"])))
    kept = []
    for it in items:
        dup = None
        for k in reversed(kept[-60:]):
            if k["tag"] != it["tag"] or abs(len(k["wave"]) - len(it["wave"])) > 2:
                continue
            n = min(len(k["wave"]), len(it["wave"]))
            a, b = k["wave"][:n], it["wave"][:n]
            if float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)) > dedupe_ncc:
                dup = k
                break
        if dup is None:
            it["srcs"] = {it["src"]}
            kept.append(it)
        else:
            dup["srcs"].add(it["src"])
    return kept


class Matcher:
    """GPU NCC of every template against one clip, via batched FFT correlation.

    Template spectra are cached per FFT size; feed clips sorted by length so only one
    size is resident at a time (call clear() between sizes).
    """

    def __init__(self, library, device="cuda", batch=128):
        self.lib = library
        self.device = device
        self.batch = batch
        self.lens = np.array([len(t["wave"]) for t in library])
        self.maxlen = int(self.lens.max())
        self.norms = torch.tensor([np.linalg.norm(t["wave"]) + 1e-9 for t in library],
                                  dtype=torch.float32, device=device)
        self.lens_t = torch.tensor(self.lens, device=device)
        self._cache = {}

    def nfft_for(self, n_samples):
        return 1 << int(np.ceil(np.log2(n_samples + self.maxlen)))

    def clear(self):
        self._cache.clear()
        torch.cuda.empty_cache()

    def _tpl_fft(self, nfft, i0, i1, cache_max_nfft=1 << 18):
        key = (nfft, i0)
        if key in self._cache:
            return self._cache[key]
        m = torch.zeros((i1 - i0, nfft), dtype=torch.float32)
        for j, t in enumerate(self.lib[i0:i1]):
            w = t["wave"][::-1].copy()
            m[j, : len(w)] = torch.from_numpy(w)
        spec = torch.fft.rfft(m.to(self.device), n=nfft)
        # the whole library at 2^19 is ~2.5 GB of spectra: too much for an 8 GB card
        # shared with other jobs, and only ~10% of clips need it, so recompute those
        if nfft <= cache_max_nfft:
            self._cache[key] = spec
        return spec

    @torch.no_grad()
    def match(self, x, top_k=2, min_sep_s=0.3, floor=0.2):
        """Return candidate detections (ncc, template_index, start_sample) for clip x."""
        L = len(x)
        nfft = self.nfft_for(L)
        sig = torch.from_numpy(x.astype(np.float32)).to(self.device)
        S = torch.fft.rfft(sig, n=nfft)
        csum = torch.cat([torch.zeros(1, device=self.device, dtype=torch.float64),
                          torch.cumsum(sig.double() ** 2, 0)])
        s = torch.arange(L, device=self.device)
        sep = int(min_sep_s * SR)
        out = []
        batch = self.batch if nfft <= (1 << 18) else max(16, self.batch // 4)
        for i0 in range(0, len(self.lib), batch):
            i1 = min(i0 + batch, len(self.lib))
            n = self.lens_t[i0:i1, None]                               # (B,1)
            corr = torch.fft.irfft(self._tpl_fft(nfft, i0, i1) * S[None, :], n=nfft)
            # start s is valid when the template fits: s + n <= L; its value sits at s+n-1
            num = torch.gather(corr, 1, torch.clamp(s[None, :] + n - 1, max=nfft - 1))
            end = torch.clamp(s[None, :] + n, max=L)
            win = (csum[end] - csum[s[None, :].expand_as(end)]).float()
            ncc = num / (torch.sqrt(torch.clamp(win, min=1e-10)) * self.norms[i0:i1, None])
            ncc = torch.where(s[None, :] + n <= L, ncc, torch.full_like(ncc, -1.0))
            # 10 ms block maxima, then local maxima over +-sep worth of blocks, keep top_k
            blk = SR // 100
            nb = (L + blk - 1) // blk
            ncc = torch.nn.functional.pad(ncc, (0, nb * blk - L), value=-1.0).view(i1 - i0, nb, blk)
            bmax, barg = ncc.max(dim=2)
            sb = max(1, sep // blk)
            pooled = torch.nn.functional.max_pool1d(bmax[:, None, :], 2 * sb + 1, 1, sb)[:, 0, :]
            peaks = torch.where((bmax >= pooled) & (bmax > floor), bmax, torch.full_like(bmax, -1.0))
            vals, bidx = torch.topk(peaks, k=min(top_k, nb), dim=1)
            st = bidx * blk + torch.gather(barg, 1, bidx)
            vals, st = vals.cpu().numpy(), st.cpu().numpy()
            for j in range(i1 - i0):
                for v, s0 in zip(vals[j], st[j]):
                    if v > floor:
                        out.append((float(v), i0 + j, int(s0)))
        return out


def null_levels(cands, clips, n_templates, q=0.999, floor=0.2):
    """Per-template false-alarm level: the q-quantile of each template's best NCC over
    clips that cannot contain a pasted sample (natural clips).

    A tonal template (a horn) correlates with plenty of unrelated audio, a broadband one
    (breathing) with almost nothing, so one global NCC threshold is either too strict for
    the second or too loose for the first. On validation this lifts synthetic recall
    from 85% to 98% at the same false-alarm rate. Candidates below `floor` were never
    stored, so absent values count as `floor`.
    """
    best = [[] for _ in range(n_templates)]
    for k in clips:
        per = {}
        for v, ti, _ in cands[k]:
            per[ti] = max(per.get(ti, floor), v)
        for ti, v in per.items():
            best[ti].append(v)
    null = np.full(n_templates, floor)
    for ti, vals in enumerate(best):
        if vals:
            null[ti] = np.quantile(np.array(vals + [floor] * (len(clips) - len(vals))), q)
    return null


def select_events(cands, library, lens, thr, exclude_src=None, max_overlap=0.3, null=None):
    """Greedy NMS over candidates -> list of (onset_s, offset_s, ncc, template_idx).

    With `null`, a candidate's score is ncc - null[template] and `thr` applies to that
    margin; without it, `thr` is a plain NCC threshold."""
    margin = (lambda v, ti: v - null[ti]) if null is not None else (lambda v, ti: v)
    cands = sorted(((margin(v, ti), v, ti, s) for v, ti, s in cands if margin(v, ti) >= thr), reverse=True)
    chosen = []
    for _, v, ti, s in cands:
        # leave-one-clip-out: a waveform seen ONLY in this clip would not be in a
        # library built without it; one also seen elsewhere would be.
        if exclude_src is not None and library[ti]["srcs"] == {exclude_src}:
            continue
        e = s + int(lens[ti])
        ok = True
        for _, _, cs, ce in chosen:
            inter = min(e, ce) - max(s, cs)
            if inter > max_overlap * min(e - s, ce - cs):
                ok = False
                break
        if ok:
            chosen.append((v, ti, s, e))
    return [(s / SR, e / SR, v, ti) for v, ti, s, e in sorted(chosen, key=lambda c: c[2])]
