"""Frame-level noise-event detector: frozen Whisper features -> BiGRU head.

Channel 0 is class-agnostic "any noise"; channels 1..7 are the seven top-level
categories. The per-category channels exist because 12% of natural reference events
overlap another event of a different category, and the metric scores each reference
event separately, so a single merged mask can never recover both.
"""
import json
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

FPS = 50
HOP = 1.0 / FPS
CATEGORIES = ["animal", "vehicle_traffic", "baby_child", "singing_music",
              "phone_signal_alarm", "appliance_machine", "human_non_speech"]
N_OUT = 1 + len(CATEGORIES)
# "cpu" runs head training/prediction without the GPU (GPU-free probes)
DEVICE = os.environ.get("INDOML_DEVICE", "cuda")


def val_items(val_root):
    meta = json.load(open(os.path.join(val_root, "validationMetadata.json"), encoding="utf-8"))
    items = []
    for m in meta:
        sub = "syntheticNoiseAudio" if m["syntheticData"] else "naturalNoisyAudio"
        evs = [(float(e["start"]), float(e["end"]), e["category"]) for e in m["NoiseSubCategoryTimeStamp"]
               if "start" in e and "end" in e]
        items.append({"key": m["segmentFileName"], "path": os.path.join(val_root, sub, m["segmentFileName"]),
                      "dur": float(m["duration"]), "events": evs, "synthetic": bool(m["syntheticData"]),
                      "group": f'{m["state"]}/{m["district"]}'})
    return items


def hf_items(hf_root):
    """Items for the Gold clips written by hf_data.py (all natural)."""
    p = os.path.join(hf_root, "metadata.json")
    if not os.path.exists(p):
        return []
    items = []
    for m in json.load(open(p, encoding="utf-8")):
        if not m.get("segmentFileName"):
            continue
        evs = []
        for e in m["NoiseSubCategoryTimeStamp"] or []:
            try:
                s, t = float(e["start"]), float(e["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if t > s:
                evs.append((s, t, e.get("category")))
        items.append({"key": m["segmentFileName"], "path": os.path.join(hf_root, "audio", m["segmentFileName"]),
                      "dur": float(m["duration"]), "events": evs, "synthetic": False,
                      "group": f'{m.get("state")}/{m.get("district")}'})
    return items


def frame_targets(events, n_frames):
    y = np.zeros((n_frames, N_OUT), np.float32)
    centers = (np.arange(n_frames) + 0.5) * HOP
    for s, e, cat in events:
        on = (centers >= s) & (centers < e)
        y[on, 0] = 1.0
        if cat in CATEGORIES:
            y[on, 1 + CATEGORIES.index(cat)] = 1.0
    return y


Q8_SCALE = 16.0  # int8 feature storage: value * 16, clipped to +-127 (range +-7.9, step 0.0625)


def cache_features(items, extractor, cache_dir, compact_only=False, q8=False):
    """One .npy per clip: (L, T, D) layers, or (T, D) already compacted when
    compact_only (4x smaller). q8 stores the compact features as int8 (another 2x):
    they are layer-normalised, so +-7.9 in steps of 1/16 loses nothing the head uses,
    and it is what lets 21 h of large-v3 features fit on this disk."""
    os.makedirs(cache_dir, exist_ok=True)
    for i, it in enumerate(items):
        p = os.path.join(cache_dir, it["key"][:-4] + ".npy")
        it["feat"] = p
        if os.path.exists(p):
            continue
        x, _ = sf.read(it["path"], dtype="float32")
        f = extractor(x).cpu().numpy()
        if compact_only:
            c = compact(f)
            f = (np.clip(np.round(c * Q8_SCALE), -127, 127).astype(np.int8) if q8 else c.astype(np.float16))
        np.save(p, f)
        if i % 500 == 0:
            print(f"  features {i}/{len(items)}", flush=True)


def compact(f):
    """(L, T, D) encoder layers -> (T, D): per-frame layer norm of each layer, then the
    mean. The layers differ ~10x in scale, and one averaged stream is 4x less IO than
    keeping them apart, which matters on an 11 GB-RAM laptop."""
    f = np.asarray(f, dtype=np.float32)
    f = (f - f.mean(-1, keepdims=True)) / (f.std(-1, keepdims=True) + 1e-5)
    return f.mean(0)


def build_memmap(items, out_path):
    """Pack compact features of all items into one float16 file; sets it['off'], it['T']."""
    # clip lengths are indexed next to the store, so once it is built the per-clip
    # feature files are no longer needed (they can be deleted to free disk)
    idx_path = out_path[:-4] + "_index.json"
    if os.path.exists(out_path) and os.path.exists(idx_path):
        index = json.load(open(idx_path))
        if all(it["key"] in index for it in items):
            offs, T = [], 0
            for it in items:
                offs.append(T); T += index[it["key"]]
            store = np.load(out_path, mmap_mode="r")
            if store.shape[0] == T:
                for it, o in zip(items, offs):
                    it["off"], it["T"] = o, index[it["key"]]
                return store
    offs, T = [], 0
    for it in items:
        n = np.load(it["feat"], mmap_mode="r").shape[-2]
        offs.append(T); T += n
    D = np.load(items[0]["feat"], mmap_mode="r").shape[-1]
    ok = os.path.exists(out_path) and np.load(out_path, mmap_mode="r").shape == (T, D)
    if not ok:
        mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(T, D))
        for it, o in zip(items, offs):
            f = np.load(it["feat"])
            c = compact(f) if f.ndim == 3 else f
            mm[o:o + len(c)] = c
        mm.flush(); del mm
    for it, o, n in zip(items, offs, np.diff(offs + [T])):
        it["off"], it["T"] = o, int(n)
    json.dump({it["key"]: it["T"] for it in items}, open(idx_path, "w"))
    return np.load(out_path, mmap_mode="r")


class Head(nn.Module):
    def __init__(self, dim, hidden=192, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Linear(dim, 2 * hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(2 * hidden, hidden, num_layers=2, batch_first=True, bidirectional=True,
                          dropout=dropout)
        self.out = nn.Linear(2 * hidden, N_OUT)

    def forward(self, feats):
        """feats: (B, T, D) compact features -> logits (B, T, N_OUT)"""
        h, _ = self.gru(self.proj(self.norm(feats)))
        return self.out(h)


# Which fold's teacher supplies the soft targets of pseudo-labelled items (see pseudo.py);
# train_cv sets it before training each fold so the held-out fold never leaks back in.
PSEUDO_FOLD = 0


def _batch(items, store, idx, max_frames=None, rng=None, pad_to=None):
    feats, ys, lens, ws = [], [], [], []
    for i in idx:
        it = items[i]
        if "off" in it:                       # packed store (validation)
            src, a, n = store, it["off"], it["T"]
        else:                                 # per-clip compact file (HF Gold, pseudo-labelled test)
            src, a = np.load(it["feat"], mmap_mode="r"), 0
            n = src.shape[0]
        soft = np.load(it["soft"], mmap_mode="r")[PSEUDO_FOLD] if "soft" in it else None
        if soft is not None:
            n = min(n, soft.shape[0])
        full = n
        s = 0
        if max_frames and n > max_frames and rng is not None:
            s = int(rng.integers(0, n - max_frames + 1)); n = max_frames
        f = np.asarray(src[a + s: a + s + n], dtype=np.float32)
        if src.dtype == np.int8:
            f /= Q8_SCALE
        y = (np.asarray(soft[s:s + n], dtype=np.float32) if soft is not None
             else frame_targets(it["events"], full)[s:s + n])
        feats.append(f); ys.append(y); lens.append(n); ws.append(it.get("weight", 1.0))
    T = pad_to or max(lens)   # constant shape lets the CUDA allocator reuse its blocks
    F = np.zeros((len(idx), T, feats[0].shape[1]), np.float32)
    Y = np.zeros((len(idx), T, N_OUT), np.float32)
    M = np.zeros((len(idx), T), np.float32)
    for j, (f, y, n) in enumerate(zip(feats, ys, lens)):
        F[j, :n] = f; Y[j, :n] = y; M[j, :n] = 1
    return (torch.from_numpy(F), torch.from_numpy(Y), torch.from_numpy(M),
            torch.tensor(ws, dtype=torch.float32))


def augment(F, M, rng, n_masks=2, max_w=25, feat_drop=0.1):
    """SpecAugment-style regularisation on encoder features: zero a few random time
    spans (<= max_w frames) and randomly drop feature channels. Labels are unchanged,
    so the head must infer activity from context, not from single frames."""
    B, T, D = F.shape
    for j in range(B):
        n = int(M[j].sum())
        for _ in range(n_masks):
            w = int(rng.integers(0, max_w + 1))
            if n > w > 0:
                a = int(rng.integers(0, n - w))
                F[j, a:a + w] = 0
    if feat_drop:
        F = F * (torch.rand(B, 1, D) > feat_drop).float() / (1 - feat_drop)
    return F


def soft_dice_loss(logits, Y, M, eps=1.0):
    """1 - soft Dice on the agnostic channel, per clip, averaged over the batch.

    Half the competition metric is segment Dice, yet the head is trained on plain BCE,
    which optimises per-frame calibration and is indifferent to how much of the reference
    span a clip ends up covering. On validation that shows up as systematic
    under-coverage: 1.16 s of predicted active time against 1.55 s of reference. This
    term is the overlap itself, so the loss finally contains the thing being scored.
    """
    p = torch.sigmoid(logits[..., 0]) * M
    y = Y[..., 0] * M
    inter = (p * y).sum(-1)
    denom = p.sum(-1) + y.sum(-1)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def train_head(items, store, train_idx, epochs=30, lr=1e-3, bs=16, seed=0, device=DEVICE, log=None,
               aug=False, dropout=0.3, hidden=192, max_frames=600, dice_weight=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Head(store.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(train_idx) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    # channel 0 carries the decision; categories are auxiliary
    ch_w = torch.tensor([1.0] + [0.5] * len(CATEGORIES), device=device)
    skipped = 0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(train_idx)
        tot = 0.0
        for b in range(0, len(order), bs):
            F, Y, M, W = _batch(items, store, order[b:b + bs], max_frames=max_frames, rng=rng,
                                pad_to=max_frames)
            if aug:
                F = augment(F, M, rng)
            ok = False
            try:
                F, Y, M, W = F.to(device), Y.to(device), M.to(device), W.to(device)
                logits = model(F)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, Y, reduction="none")
                MW = M * W[:, None]  # pseudo-labelled clips count less than labelled ones
                loss = ((loss * ch_w).sum(-1) * MW).sum() / MW.sum()
                if dice_weight:
                    loss = loss + dice_weight * soft_dice_loss(logits, Y, MW)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                step_loss = loss.item()
                ok = True
            except RuntimeError as e:
                # On this shared laptop, host-memory pressure from other jobs surfaces as
                # sporadic cuDNN/CUDA internal errors; skip the batch instead of losing the
                # fold, but give up if it keeps happening.
                if "cuDNN" not in str(e) and "CUDA" not in str(e):
                    raise
                skipped += 1
                opt.zero_grad(set_to_none=True)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass  # the allocator itself can fail while memory is exhausted
                if log:
                    log(f"    skipped a batch after a transient GPU error ({skipped} so far)")
                if skipped > 300:
                    raise
            sched.step()
            if ok:
                tot += step_loss * len(order[b:b + bs])
        if log and (ep % 5 == 4 or ep == epochs - 1):
            log(f"    epoch {ep + 1}/{epochs} loss {tot / len(order):.4f}")
    return model


@torch.no_grad()
def predict_probs(model, feats_list, device=DEVICE):
    """feats_list: iterable of (T, D) compact arrays -> list of (T, N_OUT) sigmoid probs."""
    model.eval()
    out = []
    for f in feats_list:
        x = torch.from_numpy(np.asarray(f, dtype=np.float32))[None].to(device)
        out.append(torch.sigmoid(model(x))[0].cpu().numpy())
    return out
